from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

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
from factor_mining_workspace.formula_language import (
    BINARY_OPERATORS,
    DEFAULT_BLEND_WEIGHTS,
    UNARY_OPERATORS,
    FormulaNode,
    is_forbidden_formula_field,
)
from factor_mining_workspace.single_factor_case_study import dataframe_to_markdown
from factor_mining_workspace.single_factor_case_study import load_or_build_preprocessed_train_test
from src.runtime_config import (
    DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)


DEFAULT_OUTPUT_ROOT = "factor_mining_workspace/generative_mining_outputs"
DEFAULT_HISTORY_RUN_DIR = "factor_mining_workspace/heuristic_search_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probabilistic grammar factor mining.")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始日频数据路径。")
    parser.add_argument("--model-dir", default=DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR, help="模型与特征选择产物目录。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="生成式因子挖掘输出目录。")
    parser.add_argument("--history-run-dir", default=DEFAULT_HISTORY_RUN_DIR, help="旧 heuristic 搜索产物目录，用于学习 warm prior。")
    parser.add_argument("--cache-dir", default=".cache", help="缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="预测目标周期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--n-groups", type=int, default=5, help="单因子分组数量。")
    parser.add_argument("--min-cross-section", type=int, default=30, help="每个日期最少股票数。")
    parser.add_argument("--seed-top-k", type=int, default=16, help="从当前模型上下文取多少个 seed 特征。")
    parser.add_argument("--population-size", type=int, default=30, help="兼容 warm seed 读取逻辑时保留多少历史候选。")
    parser.add_argument(
        "--num-samples",
        "--n-candidates",
        dest="num_samples",
        type=int,
        default=80,
        help="生成并评价多少个公式候选；--n-candidates 是更接近论文/实验口径的别名。",
    )
    parser.add_argument("--survivor-ratio", type=float, default=0.20, help="训练期进入 OOS 验证的候选比例。")
    parser.add_argument(
        "--train-top-k",
        type=int,
        default=None,
        help="训练期最多送入 OOS 验证的候选数量；设置后优先于 survivor-ratio。",
    )
    parser.add_argument("--final-top-k", type=int, default=10, help="报告展示的候选数量。")
    parser.add_argument("--max-depth", type=int, default=4, help="公式 AST 最大深度。")
    parser.add_argument(
        "--max-complexity",
        "--max-nodes",
        dest="max_complexity",
        type=int,
        default=9,
        help="公式 AST 最大复杂度，也可以理解为最大节点数；--max-nodes 是别名。",
    )
    parser.add_argument("--max-fields", type=int, default=4, help="单个公式最多使用多少个字段。")
    parser.add_argument(
        "--min-operator-count",
        type=int,
        default=0,
        help="候选公式至少需要包含多少个操作符；设为 1 可以排除单列原始特征，只保留派生公式。",
    )
    parser.add_argument("--terminal-probability", type=float, default=0.35, help="生成树时直接采样字段节点的概率。")
    parser.add_argument("--unary-probability", type=float, default=0.35, help="非终止节点里采样 unary operator 的概率。")
    parser.add_argument("--smoothing", type=float, default=1.0, help="学习字段/操作符分布时的拉普拉斯平滑。")
    parser.add_argument(
        "--include-alpha-seeds",
        action="store_true",
        help="允许 canonical 价格尺度不变 Alpha 子集进入 seed pool；不会加载全部 Alpha191。",
    )
    parser.add_argument("--include-raw-market-seeds", action="store_true", help="允许原始量价列进入 seed pool。")
    parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭横截面预处理缓存。")
    parser.add_argument("--random-seed", type=int, default=23, help="随机种子。")
    parser.add_argument("--run-name", default=None, help="输出目录名。")
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def weighted_choice(weight_map: dict[str, float], rng: random.Random) -> str:
    """从离散概率分布中采样一个 key。"""

    items = [(key, max(float(value), 0.0)) for key, value in weight_map.items()]
    total = sum(value for _, value in items)
    if total <= 0:
        return rng.choice([key for key, _ in items])

    threshold = rng.random() * total
    running = 0.0
    for key, value in items:
        running += value
        if running >= threshold:
            return key
    return items[-1][0]


def normalized_counter(counter: Counter[str], candidates: list[str], smoothing: float) -> dict[str, float]:
    """把计数转换成带平滑的概率权重。"""

    return {
        candidate: float(counter.get(candidate, 0.0) + smoothing)
        for candidate in candidates
    }


def get_safe_formula_columns(data: pd.DataFrame) -> set[str]:
    """返回生成式模型允许采样的字段集合。

    生成式模型的风险点在于：如果把 `y_10d`、`next_open`、`predicted_y`
    这类字段放进采样空间，即使后面再做过滤，也会让搜索过程变得混乱。
    因此这里先从源头收缩采样空间：

    1. 只允许数值列进入公式；
    2. 显式排除标签、未来价格、预测结果和元信息字段；
    3. 后续 `FormulaNode.is_legal()` 仍然会再检查一次，形成双保险。
    """

    safe_columns: set[str] = set()
    for column in data.columns:
        column_name = str(column)
        if is_forbidden_formula_field(column_name):
            continue
        if not pd.api.types.is_numeric_dtype(data[column]):
            continue
        safe_columns.add(column_name)
    return safe_columns


def learn_probabilistic_prior(
    seed_nodes: list[FormulaNode],
    available_columns: set[str],
    smoothing: float,
) -> dict[str, dict[str, float]]:
    """从已有 seed 公式学习字段和操作符分布。

    这个 prior 是生成式因子挖掘的核心：
    - 高频 seed 字段更容易被采样；
    - 过去出现过的操作符更容易被采样；
    - 但通过 smoothing 保留探索未充分出现操作符的机会。
    """

    field_counter: Counter[str] = Counter()
    unary_counter: Counter[str] = Counter()
    binary_counter: Counter[str] = Counter()

    for node in seed_nodes:
        field_counter.update(node.fields)
        for operator in node.operators:
            if operator in UNARY_OPERATORS:
                unary_counter[operator] += 1
            if operator in BINARY_OPERATORS:
                binary_counter[operator] += 1

    # 不把 `id` 当成真实 unary 采样动作，否则生成器会大量产出和 seed 完全一样的公式。
    unary_candidates = [operator for operator in UNARY_OPERATORS if operator != "id"]
    return {
        "field_weights": normalized_counter(field_counter, sorted(available_columns), smoothing=smoothing),
        "unary_weights": normalized_counter(unary_counter, unary_candidates, smoothing=smoothing),
        "binary_weights": normalized_counter(binary_counter, list(BINARY_OPERATORS), smoothing=smoothing),
    }


def sample_formula_node(
    prior: dict[str, dict[str, float]],
    args: argparse.Namespace,
    rng: random.Random,
    current_depth: int = 1,
) -> FormulaNode:
    """递归采样公式 AST。

    这是概率语法生成器，不是手写固定模板：
    每次生成会根据 learned prior 决定字段和操作符。
    """

    if current_depth >= args.max_depth or rng.random() < float(args.terminal_probability):
        return FormulaNode.column(weighted_choice(prior["field_weights"], rng))

    if rng.random() < float(args.unary_probability):
        operator = weighted_choice(prior["unary_weights"], rng)
        child = sample_formula_node(prior, args, rng, current_depth=current_depth + 1)
        return FormulaNode.unary(operator, child)

    operator = weighted_choice(prior["binary_weights"], rng)
    left = sample_formula_node(prior, args, rng, current_depth=current_depth + 1)
    right = sample_formula_node(prior, args, rng, current_depth=current_depth + 1)
    weight = rng.choice(DEFAULT_BLEND_WEIGHTS)
    return FormulaNode.binary(operator, left, right, weight=weight)


def generate_unique_candidates(
    train_df: pd.DataFrame,
    seed_nodes: list[FormulaNode],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FormulaNode], dict[str, dict[str, float]]]:
    available_columns = get_safe_formula_columns(train_df)
    if not available_columns:
        raise ValueError("No safe numeric columns are available for generative factor mining.")

    prior = learn_probabilistic_prior(
        seed_nodes=seed_nodes,
        available_columns=available_columns,
        smoothing=float(args.smoothing),
    )

    candidate_records: dict[str, dict[str, object]] = {}
    train_records: dict[str, dict[str, object]] = {}
    nodes_by_id: dict[str, FormulaNode] = {}
    seen_formulas: set[str] = set()
    attempts = 0
    max_attempts = max(args.num_samples * 80, 100)

    while len(candidate_records) < args.num_samples and attempts < max_attempts:
        attempts += 1
        node = sample_formula_node(prior, args, rng)
        formula = node.to_formula()
        if formula in seen_formulas:
            continue
        seen_formulas.add(formula)

        candidate_id = f"gm_{len(candidate_records) + 1:04d}"
        if len(node.operators) < int(args.min_operator_count):
            candidate_records[candidate_id] = {
                "candidate_id": candidate_id,
                "search_source": "probabilistic_grammar",
                "train_eval_status": "below_min_operator_count",
                **node_metadata(node),
            }
            continue

        if not node.is_legal(available_columns, args.max_depth, args.max_complexity, args.max_fields):
            candidate_records[candidate_id] = {
                "candidate_id": candidate_id,
                "search_source": "probabilistic_grammar",
                "train_eval_status": "illegal_formula",
                **node_metadata(node),
            }
            continue

        train_record = evaluate_train_candidate(
            node=node,
            candidate_id=candidate_id,
            generation=0,
            source="probabilistic_grammar",
            train_df=train_df,
            args=args,
        )
        nodes_by_id[candidate_id] = node
        candidate_records[candidate_id] = {
            "candidate_id": candidate_id,
            "search_source": "probabilistic_grammar",
            "train_eval_status": "failed" if train_record is None else "ok",
            **node_metadata(node),
        }
        if train_record is not None:
            train_record = {
                **train_record,
                "generator": "probabilistic_grammar",
                "sample_attempt": attempts,
            }
            train_records[candidate_id] = train_record
            candidate_records[candidate_id]["train_score"] = train_record["train_score"]

        print(
            "[Generative] "
            f"candidate={candidate_id} status={candidate_records[candidate_id]['train_eval_status']} "
            f"registered={len(candidate_records)}/{args.num_samples} attempts={attempts}",
            flush=True,
        )

    candidate_df = pd.DataFrame(candidate_records.values())
    train_metrics_df = pd.DataFrame(train_records.values())
    if not train_metrics_df.empty:
        train_metrics_df = train_metrics_df.sort_values("train_score", ascending=False).reset_index(drop=True)
    return candidate_df, train_metrics_df, nodes_by_id, prior


def compact_prior_for_json(prior: dict[str, dict[str, float]], top_n: int = 30) -> dict[str, list[dict[str, object]]]:
    """把 prior 压缩成可读 JSON，避免把所有字段权重完整写进报告。"""

    compact: dict[str, list[dict[str, object]]] = {}
    for name, weights in prior.items():
        total = sum(max(float(value), 0.0) for value in weights.values())
        top_items = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:top_n]
        compact[name] = [
            {
                "name": key,
                "weight": float(value),
                "probability": float(value / total) if total > 0 else 0.0,
            }
            for key, value in top_items
        ]
    return compact


def build_report_text(
    config: dict[str, Any],
    seed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    train_metrics_df: pd.DataFrame,
    oos_metrics_df: pd.DataFrame,
    prior: dict[str, dict[str, float]],
    runtime_seconds: float,
) -> str:
    top_train = train_metrics_df.head(int(config["final_top_k"])).copy()
    top_oos = oos_metrics_df.head(int(config["final_top_k"])).copy()
    prior_preview = compact_prior_for_json(prior, top_n=15)
    status_summary = (
        candidate_df["train_eval_status"]
        .value_counts(dropna=False)
        .rename_axis("train_eval_status")
        .reset_index(name="count")
        if "train_eval_status" in candidate_df.columns
        else pd.DataFrame()
    )

    return f"""# Generative Factor Mining Report

## 1. What This Is

This script implements a **probabilistic grammar factor generator**.

It is a lightweight generative model:

- learns field/operator sampling weights from model-selected seed factors and historical warm-start candidates;
- samples formula ASTs from the learned grammar;
- evaluates generated candidates with the same leakage-safe train/OOS factor diagnostics.

This is not an LLM, GAN, VAE, diffusion model, or Deep RL system.
It is a small, auditable generative baseline that makes the formula mining stack more complete.

## 2. Config

```json
{json.dumps(config, ensure_ascii=False, indent=2)}
```

Runtime seconds: `{runtime_seconds:.2f}`

## 3. Candidate Status Summary

{dataframe_to_markdown(status_summary)}

Interpretation:

- `ok` means the generated formula could be evaluated on the training period.
- `illegal_formula` means the AST violated field, depth, complexity, or field-count rules.
- `below_min_operator_count` means the formula was a raw field while this run required derived formulas.
- `failed` means the formula was legal but could not produce usable factor diagnostics.

## 4. Learned Prior Preview

```json
{json.dumps(prior_preview, ensure_ascii=False, indent=2)}
```

## 5. Seed Library

- Seed count: `{len(seed_df)}`

{dataframe_to_markdown(seed_df.head(20))}

## 6. Top Train Candidates

{dataframe_to_markdown(top_train)}

## 7. OOS Survivor Validation

{dataframe_to_markdown(top_oos)}

## 8. Output Artifacts

Each run writes:

- `config.json`: reproducible run configuration;
- `learned_prior.json`: learned field/operator sampling weights;
- `seed_library.csv`: seed formulas used to learn the prior;
- `candidate_formulas.csv`: all generated formula candidates and status;
- `train_metrics.csv`: train-valid candidates and training-period diagnostics;
- `oos_metrics.csv`: OOS survivor diagnostics;
- `report.md`: this human-readable summary.

## 9. Interview-Safe Explanation

```text
I added a probabilistic grammar generator for formulaic factor mining. It learns
field and operator priors from validated seed factors, samples new formula ASTs,
and sends them through the same train/OOS evaluation gates. This provides a
generative baseline before moving to LLM-based alpha proposal.
```
"""


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    rng = random.Random(args.random_seed)

    output_root = resolve_path(args.output_dir)
    run_name = args.run_name or f"generative_grammar_{args.target_horizon}d_n{args.num_samples}_s{args.random_seed}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    seed_df, seed_nodes = build_seed_nodes(
        train_df=train_df,
        model_dir=resolve_path(args.model_dir),
        args=args,
    )
    if not seed_nodes:
        raise ValueError("No seed nodes were available for generative factor mining.")

    candidate_df, train_metrics_df, nodes_by_id, prior = generate_unique_candidates(
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

    safe_formula_columns = get_safe_formula_columns(train_df)
    config = {
        "searcher": "probabilistic_grammar",
        "target_column": target_column,
        "target_horizon": args.target_horizon,
        "num_samples": args.num_samples,
        "seed_top_k": args.seed_top_k,
        "survivor_ratio": args.survivor_ratio,
        "train_top_k": args.train_top_k,
        "final_top_k": args.final_top_k,
        "min_operator_count": args.min_operator_count,
        "terminal_probability": args.terminal_probability,
        "unary_probability": args.unary_probability,
        "smoothing": args.smoothing,
        "random_seed": args.random_seed,
        "raw_train_column_count": len(train_df.columns),
        "safe_formula_column_count": len(safe_formula_columns),
        "excluded_from_formula_sampling_count": len(train_df.columns) - len(safe_formula_columns),
        "generated_candidate_count": int(len(candidate_df)),
        "train_valid_candidate_count": int(len(train_metrics_df)),
        "oos_survivor_count": int(len(oos_metrics_df)),
        "oos_pass_count": (
            int(pd.to_numeric(oos_metrics_df["passes_oos_filter"], errors="coerce").fillna(False).astype(bool).sum())
            if "passes_oos_filter" in oos_metrics_df.columns
            else 0
        ),
        "field_filter_rule": "numeric columns only; excludes labels, future fields, prediction fields, and metadata",
        "dataset_summary": dataset_summary,
    }

    runtime_seconds = time.perf_counter() - start_time
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "learned_prior.json").write_text(
        json.dumps(compact_prior_for_json(prior, top_n=100), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    seed_df.to_csv(output_dir / "seed_library.csv", index=False)
    candidate_df.to_csv(output_dir / "candidate_formulas.csv", index=False)
    train_metrics_df.to_csv(output_dir / "train_metrics.csv", index=False)
    oos_metrics_df.to_csv(output_dir / "oos_metrics.csv", index=False)
    (output_dir / "report.md").write_text(
        build_report_text(
            config=config,
            seed_df=seed_df,
            candidate_df=candidate_df,
            train_metrics_df=train_metrics_df,
            oos_metrics_df=oos_metrics_df,
            prior=prior,
            runtime_seconds=runtime_seconds,
        ),
        encoding="utf-8",
    )

    print(f"[Done] Generative factor mining report written to: {output_dir / 'report.md'}", flush=True)
    if not train_metrics_df.empty:
        print(train_metrics_df.head(min(args.final_top_k, 10)).to_string(index=False), flush=True)
    if not oos_metrics_df.empty:
        print(oos_metrics_df.head(min(args.final_top_k, 10)).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
