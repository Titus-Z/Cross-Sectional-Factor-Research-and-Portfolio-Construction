from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.auto_factor_mining import (
    build_seed_nodes,
    evaluate_oos_survivors,
    evaluate_train_candidate,
    node_metadata,
)
from factor_mining_workspace.formula_language import DEFAULT_BLEND_WEIGHTS, FormulaNode
from factor_mining_workspace.single_factor_case_study import dataframe_to_markdown
from factor_mining_workspace.single_factor_case_study import load_or_build_preprocessed_train_test
from src.runtime_config import (
    DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)


DEFAULT_OUTPUT_ROOT = "factor_mining_workspace/rl_mining_outputs"
DEFAULT_HISTORY_RUN_DIR = "factor_mining_workspace/heuristic_search_outputs"
RL_ACTIONS = (
    "rank",
    "abs",
    "neg",
    "tanh",
    "signed_sq",
    "blend",
    "spread",
    "ratio",
    "confirm",
    "interaction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contextual-bandit factor mining.")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始日频数据路径。")
    parser.add_argument("--model-dir", default=DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR, help="模型与特征选择产物目录。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="RL-style 因子挖掘输出目录。")
    parser.add_argument("--history-run-dir", default=DEFAULT_HISTORY_RUN_DIR, help="旧 heuristic 搜索产物目录，用于 warm seed。")
    parser.add_argument("--cache-dir", default=".cache", help="缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="预测目标周期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--n-groups", type=int, default=5, help="单因子分组数量。")
    parser.add_argument("--min-cross-section", type=int, default=30, help="每个日期最少股票数。")
    parser.add_argument("--seed-top-k", type=int, default=16, help="从当前模型上下文取多少个 seed 特征。")
    parser.add_argument("--population-size", type=int, default=20, help="兼容 warm seed 读取逻辑时保留多少历史候选。")
    parser.add_argument("--episodes", type=int, default=80, help="bandit policy 交互轮数。")
    parser.add_argument("--survivor-ratio", type=float, default=0.20, help="训练期进入 OOS 验证的候选比例。")
    parser.add_argument("--final-top-k", type=int, default=10, help="报告展示的候选数量。")
    parser.add_argument("--max-depth", type=int, default=4, help="公式 AST 最大深度。")
    parser.add_argument("--max-complexity", type=int, default=9, help="公式 AST 最大复杂度。")
    parser.add_argument("--max-fields", type=int, default=4, help="单个公式最多使用多少个字段。")
    parser.add_argument("--epsilon-start", type=float, default=0.60, help="初始探索率。")
    parser.add_argument("--epsilon-end", type=float, default=0.10, help="最终探索率。")
    parser.add_argument("--learning-rate", type=float, default=0.25, help="Q 值更新步长。")
    parser.add_argument("--reward-complexity-penalty", type=float, default=0.01, help="复杂度惩罚。")
    parser.add_argument("--reward-logic-bonus", type=float, default=0.05, help="财务逻辑分数奖励权重。")
    parser.add_argument(
        "--include-alpha-seeds",
        action="store_true",
        help="允许 canonical 价格尺度不变 Alpha 子集进入 seed pool；不会加载全部 Alpha191。",
    )
    parser.add_argument("--include-raw-market-seeds", action="store_true", help="允许原始量价列进入 seed pool。")
    parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭横截面预处理缓存。")
    parser.add_argument("--random-seed", type=int, default=13, help="随机种子。")
    parser.add_argument("--run-name", default=None, help="输出目录名。")
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def bounded_epsilon(args: argparse.Namespace, episode_index: int) -> float:
    """线性衰减探索率。

    前期多试不同变异动作，后期更多利用已经学到的高 reward 动作。
    这是最小版 policy learning，不是深度强化学习。
    """

    if args.episodes <= 1:
        return float(args.epsilon_end)
    progress = episode_index / max(args.episodes - 1, 1)
    return float(args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start))


def state_from_node(node: FormulaNode) -> str:
    """把公式压缩成 bandit state。

    state 不需要很复杂。这里用 family + depth bucket，让策略能学到：
    对波动率类、量价类、通道类公式，哪些 mutation action 更容易得到好 reward。
    """

    depth_bucket = "deep" if node.depth >= 3 else "shallow"
    return f"{node.infer_family()}|{depth_bucket}"


def choose_action(
    state: str,
    q_table: dict[tuple[str, str], float],
    rng: random.Random,
    epsilon: float,
) -> str:
    if rng.random() < epsilon:
        return rng.choice(RL_ACTIONS)
    values = [(q_table[(state, action)], action) for action in RL_ACTIONS]
    max_value = max(value for value, _ in values)
    best_actions = [action for value, action in values if math.isclose(value, max_value, rel_tol=1e-12, abs_tol=1e-12)]
    return rng.choice(best_actions)


def apply_rl_action(
    action: str,
    parent: FormulaNode,
    seed_nodes: list[FormulaNode],
    rng: random.Random,
) -> FormulaNode:
    """把一个 policy action 变成新公式。

    unary action 只改造当前公式。
    binary action 会从 seed pool 里抽一个 partner，把当前公式和 partner 组合。
    """

    if action in {"rank", "abs", "neg", "tanh", "signed_sq"}:
        return FormulaNode.unary(action, parent)

    partner = rng.choice(seed_nodes)
    weight = rng.choice(DEFAULT_BLEND_WEIGHTS)
    if action == "blend":
        return FormulaNode.binary("blend", parent, partner, weight=weight)
    if action == "spread":
        return FormulaNode.binary("spread", parent, partner, weight=weight)
    if action == "ratio":
        return FormulaNode.binary("ratio", parent, partner, weight=weight)
    if action == "confirm":
        return FormulaNode.binary("confirm", parent, partner, weight=weight)
    if action == "interaction":
        return FormulaNode.binary("interaction", parent, partner, weight=weight)
    raise ValueError(f"Unsupported RL action: {action}")


def reward_from_record(
    node: FormulaNode,
    train_record: dict[str, object] | None,
    args: argparse.Namespace,
) -> float:
    """把候选训练表现转换成 policy reward。

    这里的 reward 不是只看 IC：
    - train_score 来自项目已有的 composite score；
    - financial_logic_score 奖励可解释公式；
    - complexity penalty 防止策略一直生成复杂表达式。
    """

    if train_record is None:
        return -0.05
    train_score = float(train_record.get("train_score", 0.0) or 0.0)
    logic_bonus = float(args.reward_logic_bonus) * node.financial_logic_score()
    complexity_penalty = float(args.reward_complexity_penalty) * max(node.complexity - 1, 0)
    filter_bonus = 0.03 if bool(train_record.get("passes_train_filter", False)) else 0.0
    return float(train_score + logic_bonus + filter_bonus - complexity_penalty)


def select_parent(
    seed_nodes: list[FormulaNode],
    train_records: dict[str, dict[str, object]],
    nodes_by_id: dict[str, FormulaNode],
    rng: random.Random,
) -> FormulaNode:
    """从 seed 和历史高 reward 候选中选 parent。

    这一步让策略不是每轮都从原始 seed 开始，而是可以沿着已经表现较好的公式继续探索。
    """

    if not train_records or rng.random() < 0.35:
        return rng.choice(seed_nodes)

    ranked_records = sorted(
        train_records.values(),
        key=lambda record: float(record.get("rl_reward", record.get("train_score", 0.0)) or 0.0),
        reverse=True,
    )
    top_count = max(1, int(math.ceil(len(ranked_records) * 0.30)))
    parent_record = rng.choice(ranked_records[:top_count])
    return nodes_by_id[str(parent_record["candidate_id"])]


def run_contextual_bandit_search(
    train_df: pd.DataFrame,
    seed_nodes: list[FormulaNode],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, FormulaNode]]:
    available_columns = set(train_df.columns)
    q_table: dict[tuple[str, str], float] = defaultdict(float)
    action_counts: dict[tuple[str, str], int] = defaultdict(int)
    formula_to_id: dict[str, str] = {}
    nodes_by_id: dict[str, FormulaNode] = {}
    candidate_records: dict[str, dict[str, object]] = {}
    train_records: dict[str, dict[str, object]] = {}
    trace_records: list[dict[str, object]] = []

    for episode in range(args.episodes):
        parent = select_parent(seed_nodes, train_records, nodes_by_id, rng)
        state = state_from_node(parent)
        epsilon = bounded_epsilon(args, episode)
        action = choose_action(state, q_table, rng, epsilon)
        child = apply_rl_action(action, parent, seed_nodes, rng)
        formula = child.to_formula()
        eval_status = "pending"
        train_record: dict[str, object] | None = None
        candidate_id = formula_to_id.get(formula)

        if not child.is_legal(available_columns, args.max_depth, args.max_complexity, args.max_fields):
            eval_status = "illegal_formula"
        elif candidate_id is not None:
            eval_status = "duplicate_formula"
            train_record = train_records.get(candidate_id)
        else:
            candidate_id = f"rl_{len(candidate_records) + 1:04d}"
            formula_to_id[formula] = candidate_id
            nodes_by_id[candidate_id] = child
            candidate_records[candidate_id] = {
                "candidate_id": candidate_id,
                "episode": episode,
                "search_source": "contextual_bandit",
                "parent_state": state,
                "policy_action": action,
                "epsilon": epsilon,
                "train_eval_status": "pending",
                **node_metadata(child),
            }
            train_record = evaluate_train_candidate(
                node=child,
                candidate_id=candidate_id,
                generation=episode,
                source="contextual_bandit",
                train_df=train_df,
                args=args,
            )
            if train_record is None:
                eval_status = "failed"
                candidate_records[candidate_id]["train_eval_status"] = "failed"
            else:
                eval_status = "ok"
                train_record = {
                    **train_record,
                    "episode": episode,
                    "parent_state": state,
                    "policy_action": action,
                    "epsilon": epsilon,
                }
                train_records[candidate_id] = train_record
                candidate_records[candidate_id]["train_eval_status"] = "ok"
                candidate_records[candidate_id]["train_score"] = train_record["train_score"]

        reward = reward_from_record(child, train_record, args)
        old_q = q_table[(state, action)]
        action_counts[(state, action)] += 1
        q_table[(state, action)] = old_q + float(args.learning_rate) * (reward - old_q)

        if train_record is not None and candidate_id in train_records:
            train_records[candidate_id]["rl_reward"] = reward
            candidate_records[candidate_id]["rl_reward"] = reward

        print(
            "[RL] "
            f"episode={episode + 1}/{args.episodes} "
            f"state={state} action={action} status={eval_status} "
            f"reward={reward:.4f} q={q_table[(state, action)]:.4f}",
            flush=True,
        )

        trace_records.append(
            {
                "episode": episode,
                "state": state,
                "action": action,
                "epsilon": epsilon,
                "candidate_id": candidate_id,
                "eval_status": eval_status,
                "reward": reward,
                "q_before": old_q,
                "q_after": q_table[(state, action)],
                "action_count_after": action_counts[(state, action)],
                "parent_formula": parent.to_formula(),
                "child_formula": formula,
                "child_family": child.infer_family(),
                "child_complexity": child.complexity,
                "child_depth": child.depth,
            }
        )

    candidate_df = pd.DataFrame(candidate_records.values())
    if not candidate_df.empty:
        candidate_df = candidate_df.sort_values("candidate_id").reset_index(drop=True)
    train_metrics_df = pd.DataFrame(train_records.values())
    if not train_metrics_df.empty:
        train_metrics_df = train_metrics_df.sort_values(["rl_reward", "train_score"], ascending=False).reset_index(drop=True)
    trace_df = pd.DataFrame(trace_records)
    action_value_df = pd.DataFrame(
        [
            {
                "state": state,
                "action": action,
                "q_value": q_value,
                "action_count": action_counts[(state, action)],
            }
            for (state, action), q_value in q_table.items()
        ]
    )
    if not action_value_df.empty:
        action_value_df = action_value_df.sort_values(["state", "q_value"], ascending=[True, False]).reset_index(drop=True)
    return candidate_df, train_metrics_df, trace_df, action_value_df, nodes_by_id


def build_report_text(
    config: dict[str, Any],
    seed_df: pd.DataFrame,
    train_metrics_df: pd.DataFrame,
    oos_metrics_df: pd.DataFrame,
    action_value_df: pd.DataFrame,
    runtime_seconds: float,
) -> str:
    top_train = train_metrics_df.head(int(config["final_top_k"])).copy()
    top_oos = oos_metrics_df.head(int(config["final_top_k"])).copy()
    top_actions = action_value_df.head(20).copy()

    return f"""# RL-Style Factor Mining Report

## 1. What This Is

This script implements a **contextual-bandit factor miner**.

It is a lightweight reinforcement-learning style search:

- state = current formula family and depth bucket;
- action = formula mutation operator such as `rank`, `blend`, `spread`, `ratio`, `confirm`;
- reward = train composite score + financial logic bonus - complexity penalty;
- policy = epsilon-greedy Q-value update.

This is not Deep RL, DQN, PPO, or GFlowNet.
The purpose is to make the factor-mining part genuinely policy-driven while staying small enough to run locally.

## 2. Config

```json
{json.dumps(config, ensure_ascii=False, indent=2)}
```

Runtime seconds: `{runtime_seconds:.2f}`

## 3. Seed Library

- Seed count: `{len(seed_df)}`

{dataframe_to_markdown(seed_df.head(20))}

## 4. Learned Action Values

{dataframe_to_markdown(top_actions)}

## 5. Top Train Candidates

{dataframe_to_markdown(top_train)}

## 6. OOS Survivor Validation

{dataframe_to_markdown(top_oos)}

## 7. Interview-Safe Explanation

```text
I extended the formulaic factor mining layer with a contextual-bandit searcher.
Instead of sampling mutations uniformly, the searcher learns which formula actions
produce better train rewards under different formula families, then validates survivors
out-of-sample. This adds a lightweight reinforcement-learning component without
overclaiming full Deep RL.
```
"""


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    rng = random.Random(args.random_seed)

    output_root = resolve_path(args.output_dir)
    run_name = args.run_name or f"rl_bandit_{args.target_horizon}d_e{args.episodes}_s{args.random_seed}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    seed_df, seed_nodes = build_seed_nodes(
        train_df=train_df,
        model_dir=resolve_path(args.model_dir),
        args=args,
    )
    if not seed_nodes:
        raise ValueError("No seed nodes were available for RL-style factor mining.")

    candidate_df, train_metrics_df, trace_df, action_value_df, nodes_by_id = run_contextual_bandit_search(
        train_df=train_df,
        seed_nodes=seed_nodes,
        args=args,
        rng=rng,
    )
    oos_metrics_df = evaluate_oos_survivors(
        train_metrics_df=train_metrics_df,
        nodes_by_id=nodes_by_id,
        test_df=test_df,
        args=args,
    )

    config = {
        "searcher": "contextual_bandit",
        "target_column": target_column,
        "target_horizon": args.target_horizon,
        "episodes": args.episodes,
        "seed_top_k": args.seed_top_k,
        "survivor_ratio": args.survivor_ratio,
        "final_top_k": args.final_top_k,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "learning_rate": args.learning_rate,
        "reward_complexity_penalty": args.reward_complexity_penalty,
        "reward_logic_bonus": args.reward_logic_bonus,
        "random_seed": args.random_seed,
        "dataset_summary": dataset_summary,
    }

    runtime_seconds = time.perf_counter() - start_time
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    seed_df.to_csv(output_dir / "seed_library.csv", index=False)
    candidate_df.to_csv(output_dir / "candidate_formulas.csv", index=False)
    train_metrics_df.to_csv(output_dir / "train_metrics.csv", index=False)
    oos_metrics_df.to_csv(output_dir / "oos_metrics.csv", index=False)
    trace_df.to_csv(output_dir / "rl_policy_trace.csv", index=False)
    action_value_df.to_csv(output_dir / "action_value_table.csv", index=False)
    (output_dir / "report.md").write_text(
        build_report_text(
            config=config,
            seed_df=seed_df,
            train_metrics_df=train_metrics_df,
            oos_metrics_df=oos_metrics_df,
            action_value_df=action_value_df,
            runtime_seconds=runtime_seconds,
        ),
        encoding="utf-8",
    )

    print(f"[Done] RL-style factor mining report written to: {output_dir / 'report.md'}", flush=True)
    if not action_value_df.empty:
        print(action_value_df.head(10).to_string(index=False), flush=True)
    if not oos_metrics_df.empty:
        print(oos_metrics_df.head(min(args.final_top_k, 10)).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
