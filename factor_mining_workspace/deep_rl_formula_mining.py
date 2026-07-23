from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.auto_factor_mining import build_seed_nodes, node_metadata
from factor_mining_workspace.auto_alpha_reward import REWARD_MODES, compute_reward
from factor_mining_workspace.formula_language import (
    BINARY_OPERATORS,
    DEFAULT_BLEND_WEIGHTS,
    UNARY_OPERATORS,
    FormulaNode,
    is_forbidden_formula_field,
)
from factor_mining_workspace.heuristic_factor_search import (
    compute_composite_score,
    evaluate_candidate,
    passes_oos_filter,
    passes_train_filter,
    standardize_candidate_cross_sectionally,
)
from factor_mining_workspace.single_factor_case_study import dataframe_to_markdown
from factor_mining_workspace.single_factor_case_study import load_or_build_preprocessed_train_test
from src.data_loader import PRICE_ADJUSTMENT_MODES
from src.runtime_config import (
    DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)
from src.provenance import build_data_fingerprint, dumps_strict_json, project_relative_path
from src.time_series_pipeline import purge_training_label_overlap


DEFAULT_OUTPUT_ROOT = "factor_mining_workspace/deep_rl_mining_outputs"
DEFAULT_HISTORY_RUN_DIR = "factor_mining_workspace/heuristic_search_outputs"
FAMILY_NAMES = ("volatility", "channel", "momentum", "liquidity", "vwap", "alpha191", "size", "other")
UNARY_ACTIONS = tuple(operator for operator in UNARY_OPERATORS if operator != "id")
STEP_PENALTY = -0.005


def finite_float(value: object, default: float = 0.0) -> float:
    """把任意数值安全转成有限 float。

    金融指标经常因为某天横截面不足、分组失败或标准差为 0 产生 NaN。
    PPO 对 NaN 极其敏感，一个 NaN reward 就可能让 policy logits 全部变成 NaN。
    """

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return numeric


def safe_composite_score(metrics: dict[str, float]) -> float:
    """PPO reward 使用的稳定版 composite score。

    保留原项目的指标权重，但把缺失项当成 0，而不是让 NaN 传进神经网络训练。
    """

    pearson_ic = finite_float(metrics.get("pearson_ic_mean"), 0.0)
    spearman_ic = finite_float(metrics.get("spearman_ic_mean"), 0.0)
    long_short = finite_float(metrics.get("long_short_spread"), 0.0)
    monotonic = finite_float(metrics.get("group_monotonic_spearman"), 0.0)
    pearson_positive = finite_float(metrics.get("pearson_ic_positive_ratio"), 0.5)
    spearman_positive = finite_float(metrics.get("spearman_ic_positive_ratio"), 0.5)
    return float(
        pearson_ic
        + 0.5 * spearman_ic
        + 0.75 * long_short
        + 0.02 * monotonic
        + 0.05 * max(pearson_positive - 0.5, 0.0)
        + 0.03 * max(spearman_positive - 0.5, 0.0)
    )


def optional_progress(iterable, **kwargs):
    """在 tqdm 可用时显示进度条；不可用时保持普通 iterable。

    因子挖掘一次可能评估几百个公式。进度条不是为了美观，
    而是为了让使用者知道时间花在 PPO rollout、OOS 审计还是报告生成。
    """

    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def format_duration(seconds: float) -> str:
    """把秒数格式化成适合报告阅读的短文本。"""

    seconds = float(max(seconds, 0.0))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def resolve_path(path_like: str | Path) -> Path:
    """把相对路径解析到项目根目录下，保证从 VS Code 或命令行运行结果一致。"""

    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO-based Deep RL formula factor mining.")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始日频数据路径。")
    parser.add_argument("--model-dir", default=DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR, help="seed 特征来源模型目录。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="Deep RL 因子挖掘输出目录。")
    parser.add_argument("--history-run-dir", default=DEFAULT_HISTORY_RUN_DIR, help="历史候选目录，用于 warm seed。")
    parser.add_argument("--cache-dir", default=".cache", help="特征和预处理缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="预测目标周期。")
    parser.add_argument(
        "--price-adjustment-mode",
        choices=list(PRICE_ADJUSTMENT_MODES),
        default="vendor_adjusted",
        help="价格口径；必须与 validation selector 和严格消融保持一致。",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--validation-fraction", type=float, default=0.25, help="训练期尾部多少日期作为 PPO reward validation。")
    parser.add_argument("--reward-validation-date-count", type=int, default=40, help="PPO reward 使用多少个 validation 日期做抽样评价。")
    parser.add_argument("--n-groups", type=int, default=5, help="单因子分组数量。")
    parser.add_argument("--min-cross-section", type=int, default=30, help="每个日期最少股票数。")
    parser.add_argument("--seed-top-k", type=int, default=20, help="从当前模型上下文取多少个 seed 特征。")
    parser.add_argument("--population-size", type=int, default=30, help="兼容 build_seed_nodes 的历史候选读取逻辑。")
    parser.add_argument("--max-depth", type=int, default=4, help="公式 AST 最大深度。")
    parser.add_argument("--max-complexity", type=int, default=9, help="公式 AST 最大复杂度。")
    parser.add_argument("--max-fields", type=int, default=4, help="单个公式最多字段数。")
    parser.add_argument("--max-steps", type=int, default=4, help="每个 episode 最多 mutation 步数。")
    parser.add_argument("--total-updates", type=int, default=20, help="PPO 更新轮数。")
    parser.add_argument("--episodes-per-update", type=int, default=32, help="每次 PPO 更新采样多少 episode。")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="每批 rollout 反复训练多少 epoch。")
    parser.add_argument("--minibatch-size", type=int, default=128, help="PPO minibatch 大小。")
    parser.add_argument("--hidden-dim", type=int, default=128, help="policy/value 网络隐藏层宽度。")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO Adam 学习率。")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子。")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda。")
    parser.add_argument("--clip-ratio", type=float, default=0.20, help="PPO ratio clipping。")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy bonus 权重。")
    parser.add_argument("--value-coef", type=float, default=0.50, help="value loss 权重。")
    parser.add_argument("--max-grad-norm", type=float, default=0.50, help="梯度裁剪阈值。")
    parser.add_argument("--reward-complexity-penalty", type=float, default=0.01, help="复杂度惩罚。")
    parser.add_argument("--reward-logic-bonus", type=float, default=0.05, help="财务逻辑奖励。")
    parser.add_argument("--reward-correlation-penalty", type=float, default=0.10, help="与历史候选相关性惩罚权重。")
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="predictive_ic",
        help=(
            "PPO 终止公式的 reward 口径。predictive_ic 偏单因子 IC；"
            "incremental_proxy 更重视 RankIC、long-short、去冗余和复杂度控制。"
        ),
    )
    parser.add_argument("--correlation-sample-size", type=int, default=8000, help="计算 reward 相关性惩罚时抽样多少行。")
    parser.add_argument("--duplicate-penalty", type=float, default=0.03, help="重复公式惩罚。")
    parser.add_argument("--illegal-penalty", type=float, default=0.10, help="非法公式惩罚。")
    parser.add_argument("--raw-seed-penalty", type=float, default=0.03, help="直接停在原始 seed 上的惩罚。")
    parser.add_argument("--selected-top-k", type=int, default=10, help="最终 factor zoo 最大数量。")
    parser.add_argument("--oos-eval-top-k", type=int, default=100, help="进入 OOS 审计的 validation 候选上限。")
    parser.add_argument("--max-factor-corr", type=float, default=0.80, help="factor zoo 内最大允许 OOS 相关性。")
    parser.add_argument(
        "--include-alpha-seeds",
        action="store_true",
        help="允许 canonical 价格尺度不变 Alpha 子集作为 seed；不会加载全部 Alpha191。",
    )
    parser.add_argument("--include-raw-market-seeds", action="store_true", help="允许原始量价 seed。")
    parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭预处理缓存。")
    parser.add_argument("--random-seed", type=int, default=31, help="随机种子。")
    parser.add_argument("--torch-threads", type=int, default=0, help="PyTorch CPU 线程数；0 表示不手动设置。")
    parser.add_argument("--run-name", default=None, help="输出目录名。")
    return parser.parse_args()


def split_train_validation_by_time(
    train_df: pd.DataFrame,
    validation_fraction: float,
    purge_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """在训练期内部再切出 validation。

    PPO 的 reward 只能来自 OOS 之前的数据。这里把 2022-2025 的训练期尾部
    留作 validation，PPO 训练过程中完全不读取 2026 OOS。
    """

    dates = sorted(pd.to_datetime(train_df["date"]).dropna().unique())
    if len(dates) < 30:
        raise ValueError("Not enough train dates for PPO train/validation split.")
    validation_fraction = min(max(float(validation_fraction), 0.05), 0.50)
    split_index = int(math.floor(len(dates) * (1.0 - validation_fraction)))
    split_index = min(max(split_index, 1), len(dates) - 1)
    fit_date_list = list(dates[:split_index])
    if purge_days < 0:
        raise ValueError("purge_days must be non-negative.")
    fit_dates = set(fit_date_list)
    validation_dates = set(dates[split_index:])
    date_series = pd.to_datetime(train_df["date"])
    fit_df = train_df[date_series.isin(fit_dates)].copy()
    validation_df = train_df[date_series.isin(validation_dates)].copy()
    fit_df, purge_summary = purge_training_label_overlap(fit_df, target_horizon=purge_days)
    return fit_df, validation_df, purge_summary


@dataclass(frozen=True)
class ActionSpec:
    """PPO 离散动作表中的一个动作。

    PPO 只能输出整数 action_id。这个类把 action_id 翻译成人能理解的公式操作。
    """

    kind: str
    operator: str = ""
    partner_index: int | None = None
    weight: float | None = None

    def label(self) -> str:
        if self.kind == "stop":
            return "STOP"
        if self.kind == "unary":
            return self.operator
        return f"{self.operator}:seed{self.partner_index}:w{self.weight}"


def build_action_catalog(seed_nodes: list[FormulaNode], blend_weights: tuple[float, ...]) -> list[ActionSpec]:
    """构造固定大小 action space。

    action space 必须固定，因为 policy network 的输出维度固定。
    binary action 通过 partner seed 和 blend weight 展开，避免在 PPO 内部再做复杂采样。
    """

    actions = [ActionSpec(kind="stop")]
    actions.extend(ActionSpec(kind="unary", operator=operator) for operator in UNARY_ACTIONS)
    for operator in BINARY_OPERATORS:
        for partner_index in range(len(seed_nodes)):
            for weight in blend_weights:
                actions.append(
                    ActionSpec(
                        kind="binary",
                        operator=operator,
                        partner_index=partner_index,
                        weight=float(weight),
                    )
                )
    return actions


def family_atoms(family: str) -> set[str]:
    atoms = {item.strip() for item in str(family).split("+") if item.strip()}
    return atoms or {"other"}


def family_vector(family: str) -> np.ndarray:
    atoms = family_atoms(family)
    return np.asarray([1.0 if family_name in atoms else 0.0 for family_name in FAMILY_NAMES], dtype=np.float32)


def field_family_vector(node: FormulaNode) -> np.ndarray:
    """把字段名映射成 family histogram。

    这个向量让 policy 能区分当前公式主要来自波动率、动量、流动性还是 size。
    """

    counts = np.zeros(len(FAMILY_NAMES), dtype=np.float32)
    lowered_fields = [field.lower() for field in node.fields]
    for index, family_name in enumerate(FAMILY_NAMES):
        if family_name == "other":
            continue
        if any(family_name in field_name for field_name in lowered_fields):
            counts[index] += 1.0
    if float(counts.sum()) == 0.0:
        counts[-1] = 1.0
    return counts / max(float(counts.sum()), 1.0)


def operator_histogram(node: FormulaNode) -> np.ndarray:
    operators = list(UNARY_ACTIONS) + list(BINARY_OPERATORS)
    counts = np.zeros(len(operators), dtype=np.float32)
    node_operators = list(node.operators)
    for index, operator in enumerate(operators):
        counts[index] = float(node_operators.count(operator))
    if float(counts.sum()) > 0.0:
        counts = counts / float(counts.sum())
    return counts


def state_to_vector(
    node: FormulaNode,
    step_index: int,
    max_steps: int,
    action_count: int,
    last_action_id: int | None,
) -> np.ndarray:
    """把公式 AST 压成 PPO policy 可以读取的定长向量。"""

    scalar_features = np.asarray(
        [
            node.depth / max(max_steps + 1, 1),
            node.complexity / 12.0,
            len(node.fields) / 6.0,
            len(node.operators) / 8.0,
            node.financial_logic_score(),
            step_index / max(max_steps, 1),
        ],
        dtype=np.float32,
    )
    last_action = np.zeros(action_count, dtype=np.float32)
    if last_action_id is not None and 0 <= int(last_action_id) < action_count:
        last_action[int(last_action_id)] = 1.0
    return np.concatenate(
        [
            scalar_features,
            family_vector(node.infer_family()),
            operator_histogram(node),
            field_family_vector(node),
            last_action,
        ]
    ).astype(np.float32)


def max_abs_signal_corr(candidate_signal: pd.Series, previous_signals: list[pd.Series]) -> float:
    """计算候选与已有候选的最大绝对相关性。

    PPO 很容易反复生成同一类 volatility/range 公式。相关性惩罚能让 reward
    更偏向互补信号，而不是只追逐单一强因子。
    """

    if not previous_signals:
        return 0.0
    candidate = pd.to_numeric(candidate_signal, errors="coerce")
    correlations: list[float] = []
    for signal in previous_signals:
        aligned = pd.concat(
            [candidate.rename("candidate"), pd.to_numeric(signal, errors="coerce").rename("peer")],
            axis=1,
        ).dropna()
        if len(aligned) < 20:
            continue
        corr_value = aligned["candidate"].corr(aligned["peer"], method="pearson")
        if pd.notna(corr_value):
            correlations.append(abs(float(corr_value)))
    return float(max(correlations)) if correlations else 0.0


class CandidateEvaluator:
    """负责把一个公式变成 reward、validation 指标和 OOS 指标。

    这个类是防泄露边界：
    - `evaluate_validation` 只读取训练期内部 validation；
    - `evaluate_oos_candidates` 只在 PPO 训练完成后调用。
    """

    def __init__(self, validation_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace):
        self.validation_df = validation_df
        self.test_df = test_df
        self.args = args
        self.reward_df = self.build_reward_validation_df()
        self.formula_to_candidate_id: dict[str, str] = {}
        self.nodes_by_id: dict[str, FormulaNode] = {}
        self.validation_records_by_id: dict[str, dict[str, object]] = {}
        self.validation_signal_by_id: dict[str, pd.Series] = {}
        self.validation_corr_vectors: list[np.ndarray] = []
        self.correlation_sample_index = self.build_correlation_sample_index()

    def build_correlation_sample_index(self) -> pd.Index:
        """固定抽样 validation 行用于 reward 相关性惩罚。

        全量 validation 大约 7.5 万行。若每个新公式都和所有历史公式做全量
        pandas corr，复杂度会快速失控。这里用固定抽样行做近似相关性惩罚：
        它足够判断“是否高度重复”，同时把正式 PPO run 控制在可运行时间内。
        """

        sample_size = int(max(self.args.correlation_sample_size, 1000))
        if len(self.reward_df) <= sample_size:
            return self.reward_df.index
        rng = np.random.default_rng(int(self.args.random_seed))
        sampled_positions = np.sort(rng.choice(len(self.reward_df), size=sample_size, replace=False))
        return self.reward_df.index[sampled_positions]

    def build_reward_validation_df(self) -> pd.DataFrame:
        """构造 PPO reward 使用的 validation 日期抽样。

        训练期 validation 全量有 251 个交易日、7.5 万行。PPO 会反复评估数百个公式，
        如果每次都跑全量因子诊断，正式 run 会非常慢。这里按时间均匀抽取日期，
        用来训练 policy；完整 2026 OOS 审计仍然不抽样。
        """

        unique_dates = sorted(pd.to_datetime(self.validation_df["date"]).dropna().unique())
        date_count = int(self.args.reward_validation_date_count)
        if date_count <= 0 or len(unique_dates) <= date_count:
            return self.validation_df
        selected_positions = np.linspace(0, len(unique_dates) - 1, num=date_count, dtype=int)
        selected_dates = {unique_dates[position] for position in selected_positions}
        date_series = pd.to_datetime(self.validation_df["date"])
        return self.validation_df[date_series.isin(selected_dates)].copy()

    def signal_to_corr_vector(self, signal: pd.Series) -> np.ndarray:
        sampled = pd.to_numeric(signal.reindex(self.correlation_sample_index), errors="coerce")
        values = sampled.to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return np.zeros(len(values), dtype=np.float32)
        fill_value = float(np.nanmean(values[finite_mask]))
        values = np.where(finite_mask, values, fill_value)
        std_value = float(np.std(values))
        if math.isclose(std_value, 0.0, abs_tol=1e-12):
            return np.zeros(len(values), dtype=np.float32)
        return ((values - float(np.mean(values))) / std_value).astype(np.float32)

    def max_abs_corr_to_existing_vectors(self, candidate_vector: np.ndarray) -> float:
        if not self.validation_corr_vectors:
            return 0.0
        candidate_norm = float(np.linalg.norm(candidate_vector))
        if math.isclose(candidate_norm, 0.0, abs_tol=1e-12):
            return 0.0
        matrix = np.vstack(self.validation_corr_vectors)
        denominators = np.linalg.norm(matrix, axis=1) * candidate_norm
        valid_mask = denominators > 1e-12
        if not valid_mask.any():
            return 0.0
        correlations = matrix[valid_mask] @ candidate_vector / denominators[valid_mask]
        return float(np.nanmax(np.abs(correlations))) if correlations.size else 0.0

    def next_candidate_id(self) -> str:
        return f"ppo_{len(self.formula_to_candidate_id) + 1:04d}"

    def evaluate_validation(self, node: FormulaNode) -> tuple[dict[str, object] | None, bool]:
        formula = node.to_formula()
        existing_id = self.formula_to_candidate_id.get(formula)
        if existing_id is not None:
            return self.validation_records_by_id.get(existing_id), True

        candidate_id = self.next_candidate_id()
        self.formula_to_candidate_id[formula] = candidate_id
        self.nodes_by_id[candidate_id] = node

        try:
            candidate_series = node.evaluate(self.reward_df)
            metrics, _ = evaluate_candidate(
                data=self.reward_df,
                candidate_name=candidate_id,
                candidate_series=candidate_series,
                n_groups=self.args.n_groups,
                min_cross_section=self.args.min_cross_section,
            )
        except Exception as exc:
            record = {
                "candidate_id": candidate_id,
                **node_metadata(node),
                "validation_status": "failed",
                "validation_error": str(exc),
                "validation_score": float("nan"),
                "validation_reward": -float(self.args.illegal_penalty),
            }
            self.validation_records_by_id[candidate_id] = record
            return record, False

        if not metrics:
            record = {
                "candidate_id": candidate_id,
                **node_metadata(node),
                "validation_status": "empty_metrics",
                "validation_score": float("nan"),
                "validation_reward": -float(self.args.illegal_penalty),
            }
            self.validation_records_by_id[candidate_id] = record
            return record, False

        standardized_signal = standardize_candidate_cross_sectionally(self.reward_df, candidate_series)
        corr_vector = self.signal_to_corr_vector(standardized_signal)
        max_corr = self.max_abs_corr_to_existing_vectors(corr_vector)
        validation_score = safe_composite_score(metrics)
        raw_seed_penalty = float(self.args.raw_seed_penalty) if len(node.operators) == 0 else 0.0
        predictive_reward = compute_reward(
            metrics,
            reward_mode="predictive_ic",
            financial_logic_score=node.financial_logic_score(),
            complexity=node.complexity,
            max_signal_corr_abs=max_corr,
        )
        incremental_proxy_reward = compute_reward(
            metrics,
            reward_mode="incremental_proxy",
            financial_logic_score=node.financial_logic_score(),
            complexity=node.complexity,
            max_signal_corr_abs=max_corr,
        )
        base_reward = compute_reward(
            metrics,
            reward_mode=str(self.args.reward_mode),
            financial_logic_score=node.financial_logic_score(),
            complexity=node.complexity,
            max_signal_corr_abs=max_corr,
        )
        # 命令行中的 penalty 参数保留给 PPO 实验调参。因为 base_reward 已经内置了
        # 基础复杂度/相关性惩罚，这里只叠加“相对默认值的额外惩罚”，避免新口径
        # 与旧参数完全脱钩。
        extra_complexity_penalty = max(float(self.args.reward_complexity_penalty) - 0.01, 0.0) * max(
            node.complexity - 1,
            0,
        )
        extra_correlation_penalty = max(float(self.args.reward_correlation_penalty) - 0.10, 0.0) * max_corr
        logic_bonus_adjustment = max(float(self.args.reward_logic_bonus) - 0.05, 0.0) * node.financial_logic_score()
        reward = base_reward + logic_bonus_adjustment - extra_complexity_penalty - extra_correlation_penalty - raw_seed_penalty
        reward = float(np.clip(finite_float(reward, -float(self.args.illegal_penalty)), -1.0, 1.0))
        record = {
            "candidate_id": candidate_id,
            **node_metadata(node),
            "validation_status": "ok",
            **{f"validation_{key}": value for key, value in metrics.items()},
            "validation_score": validation_score,
            "validation_predictive_ic_reward": predictive_reward,
            "validation_incremental_proxy_reward": incremental_proxy_reward,
            "validation_reward_mode": str(self.args.reward_mode),
            "validation_max_signal_corr_abs": finite_float(max_corr, 0.0),
            "validation_reward": reward,
            "passes_validation_filter": passes_train_filter(metrics),
        }
        self.validation_records_by_id[candidate_id] = record
        self.validation_signal_by_id[candidate_id] = standardized_signal
        self.validation_corr_vectors.append(corr_vector)
        return record, False

    def validation_metrics_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.validation_records_by_id.values())
        if df.empty:
            return df
        return df.sort_values(["validation_reward", "validation_score"], ascending=False).reset_index(drop=True)

    def evaluate_oos_candidates(self) -> pd.DataFrame:
        validation_df = self.validation_metrics_df()
        if validation_df.empty:
            return pd.DataFrame()
        top_df = validation_df[validation_df["validation_status"] == "ok"].head(int(self.args.oos_eval_top_k)).copy()
        if top_df.empty:
            return pd.DataFrame()

        oos_records: list[dict[str, object]] = []
        iterator = optional_progress(top_df.itertuples(index=False), total=len(top_df), desc="Deep RL OOS audit")
        for row in iterator:
            validation_record = row._asdict()
            candidate_id = str(validation_record["candidate_id"])
            node = self.nodes_by_id.get(candidate_id)
            if node is None:
                continue
            try:
                candidate_series = node.evaluate(self.test_df)
                metrics, _ = evaluate_candidate(
                    data=self.test_df,
                    candidate_name=candidate_id,
                    candidate_series=candidate_series,
                    n_groups=self.args.n_groups,
                    min_cross_section=self.args.min_cross_section,
                    rebalance_step=self.args.target_horizon,
                    include_spread_metrics=True,
                )
            except Exception as exc:
                oos_records.append(
                    {
                        **validation_record,
                        "oos_status": "failed",
                        "oos_error": str(exc),
                    }
                )
                continue
            if not metrics:
                continue
            oos_record = {
                **validation_record,
                "oos_status": "ok",
                **{f"oos_{key}": value for key, value in metrics.items()},
            }
            oos_record["oos_score"] = safe_composite_score(metrics)
            oos_record["passes_oos_filter"] = passes_oos_filter(metrics)
            oos_records.append(oos_record)

        oos_df = pd.DataFrame(oos_records)
        if not oos_df.empty and "oos_score" in oos_df.columns:
            oos_df = oos_df.sort_values(["oos_score", "oos_pearson_ic_mean"], ascending=False).reset_index(drop=True)
        return oos_df

    def build_selected_factor_zoo(self, oos_df: pd.DataFrame) -> pd.DataFrame:
        if oos_df.empty:
            return pd.DataFrame()

        eligible = oos_df[oos_df.get("oos_status", "ok") == "ok"].copy()
        if "operator_count" in eligible.columns:
            derived = eligible[pd.to_numeric(eligible["operator_count"], errors="coerce").fillna(0) > 0].copy()
            if not derived.empty:
                eligible = derived
        if "passes_oos_filter" in eligible.columns:
            passed = eligible[eligible["passes_oos_filter"].astype(bool)].copy()
            if not passed.empty:
                eligible = passed
        if eligible.empty:
            return pd.DataFrame()

        selected_rows: list[pd.Series] = []
        selected_signals: list[pd.Series] = []
        for _, row in eligible.iterrows():
            candidate_id = str(row["candidate_id"])
            node = self.nodes_by_id.get(candidate_id)
            if node is None:
                continue
            try:
                signal = standardize_candidate_cross_sectionally(self.test_df, node.evaluate(self.test_df))
            except Exception:
                continue
            corr_to_selected = max_abs_signal_corr(signal, selected_signals)
            if corr_to_selected > float(self.args.max_factor_corr):
                continue
            selected = row.copy()
            selected["selected_max_corr_to_previous"] = corr_to_selected
            selected_rows.append(selected)
            selected_signals.append(signal)
            if len(selected_rows) >= int(self.args.selected_top_k):
                break

        if not selected_rows:
            return eligible.head(int(self.args.selected_top_k)).copy()
        return pd.DataFrame(selected_rows).reset_index(drop=True)


class DeepRLFormulaEnv:
    """PPO 使用的公式生成环境。

    这个环境只负责生成公式，不负责直接交易。
    一个 episode 的过程是：选 seed -> 多步 mutation -> STOP 或达到 max_steps -> 评价终止公式。
    """

    def __init__(
        self,
        seed_nodes: list[FormulaNode],
        action_catalog: list[ActionSpec],
        evaluator: CandidateEvaluator,
        args: argparse.Namespace,
        rng: random.Random,
    ):
        self.seed_nodes = seed_nodes
        self.action_catalog = action_catalog
        self.evaluator = evaluator
        self.args = args
        self.rng = rng
        self.available_columns = set(evaluator.validation_df.columns)
        self.current_node: FormulaNode | None = None
        self.step_index = 0
        self.last_action_id: int | None = None

    @property
    def state_dim(self) -> int:
        example_node = self.seed_nodes[0]
        return len(
            state_to_vector(
                node=example_node,
                step_index=0,
                max_steps=self.args.max_steps,
                action_count=len(self.action_catalog),
                last_action_id=None,
            )
        )

    def reset(self) -> np.ndarray:
        self.current_node = self.rng.choice(self.seed_nodes)
        self.step_index = 0
        self.last_action_id = None
        return self.state()

    def state(self) -> np.ndarray:
        if self.current_node is None:
            raise RuntimeError("Environment was not reset.")
        return state_to_vector(
            node=self.current_node,
            step_index=self.step_index,
            max_steps=self.args.max_steps,
            action_count=len(self.action_catalog),
            last_action_id=self.last_action_id,
        )

    def action_mask(self) -> np.ndarray:
        """返回当前状态下哪些 action 合法。

        没有 action mask 时，PPO 会在几百个动作里大量采样会立刻超出
        `max_depth/max_complexity/max_fields` 的公式，训练信号会被非法惩罚淹没。
        这里把明显非法的动作提前屏蔽，让 RL 学习重点回到“哪些合法公式更有 reward”。
        """

        if self.current_node is None:
            raise RuntimeError("Environment was not reset.")

        mask = np.zeros(len(self.action_catalog), dtype=np.float32)
        # 初始 seed 直接 STOP 会退化成 raw feature，不利于“生成新公式”。
        # 但如果当前公式已经至少变异过，STOP 是合法终止动作。
        if len(self.current_node.operators) > 0 or self.step_index > 0:
            mask[0] = 1.0

        for action_id, action in enumerate(self.action_catalog[1:], start=1):
            try:
                candidate_node = self.apply_action(action)
            except Exception:
                continue
            if candidate_node.is_legal(
                self.available_columns,
                self.args.max_depth,
                self.args.max_complexity,
                self.args.max_fields,
            ):
                mask[action_id] = 1.0

        # 极端情况下没有任何 mutation 合法，就强制允许 STOP，避免 episode 死锁。
        if float(mask.sum()) == 0.0:
            mask[0] = 1.0
        return mask

    def apply_action(self, action: ActionSpec) -> FormulaNode:
        if self.current_node is None:
            raise RuntimeError("Environment was not reset.")
        if action.kind == "unary":
            return FormulaNode.unary(action.operator, self.current_node)
        if action.kind == "binary":
            partner_index = 0 if action.partner_index is None else int(action.partner_index)
            partner = self.seed_nodes[partner_index % len(self.seed_nodes)]
            return FormulaNode.binary(action.operator, self.current_node, partner, weight=action.weight)
        raise ValueError(f"Unsupported non-terminal action: {action}")

    def terminal_reward(self, node: FormulaNode) -> tuple[float, dict[str, object]]:
        if not node.is_legal(self.available_columns, self.args.max_depth, self.args.max_complexity, self.args.max_fields):
            return -float(self.args.illegal_penalty), {
                "terminal_status": "illegal_formula",
                **node_metadata(node),
            }
        if any(is_forbidden_formula_field(field) for field in node.fields):
            return -float(self.args.illegal_penalty), {
                "terminal_status": "forbidden_field",
                **node_metadata(node),
            }

        record, is_duplicate = self.evaluator.evaluate_validation(node)
        if record is None:
            return -float(self.args.illegal_penalty), {
                "terminal_status": "failed_validation",
                **node_metadata(node),
            }
        if is_duplicate:
            return -float(self.args.duplicate_penalty), {
                "terminal_status": "duplicate_formula",
                **record,
            }
        return float(record.get("validation_reward", -float(self.args.illegal_penalty))), {
            "terminal_status": str(record.get("validation_status", "unknown")),
            **record,
        }

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        action = self.action_catalog[int(action_id)]
        if self.current_node is None:
            raise RuntimeError("Environment was not reset.")

        if action.kind == "stop":
            reward, info = self.terminal_reward(self.current_node)
            info.update(
                {
                    "action_id": int(action_id),
                    "action_label": action.label(),
                    "steps_used": self.step_index,
                }
            )
            return self.state(), reward, True, info

        candidate_node = self.apply_action(action)
        self.step_index += 1
        self.last_action_id = int(action_id)
        self.current_node = candidate_node

        if not candidate_node.is_legal(self.available_columns, self.args.max_depth, self.args.max_complexity, self.args.max_fields):
            info = {
                "terminal_status": "illegal_formula",
                "action_id": int(action_id),
                "action_label": action.label(),
                "steps_used": self.step_index,
                **node_metadata(candidate_node),
            }
            return self.state(), -float(self.args.illegal_penalty), True, info

        if self.step_index >= int(self.args.max_steps):
            reward, info = self.terminal_reward(candidate_node)
            info.update(
                {
                    "action_id": int(action_id),
                    "action_label": action.label(),
                    "steps_used": self.step_index,
                }
            )
            return self.state(), reward, True, info

        return self.state(), STEP_PENALTY, False, {
            "terminal_status": "in_progress",
            "action_id": int(action_id),
            "action_label": action.label(),
            "steps_used": self.step_index,
            **node_metadata(candidate_node),
        }


class PPOPolicyNet(nn.Module):
    """共享 backbone 的 policy/value 网络。

    policy head 输出离散动作 logits，value head 估计当前公式状态的未来 reward。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values


@dataclass
class RolloutBatch:
    states: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    rewards: np.ndarray
    episode_records: list[dict[str, object]]


class PPOTrainer:
    """最小 PPO 训练器。

    这里没有使用第三方 RL 框架，目的是让代码可以被逐行解释：
    rollout -> GAE -> clipped policy update -> value update。
    """

    def __init__(self, env: DeepRLFormulaEnv, args: argparse.Namespace):
        self.env = env
        self.args = args
        self.device = torch.device("cpu")
        self.model = PPOPolicyNet(
            state_dim=env.state_dim,
            action_dim=len(env.action_catalog),
            hidden_dim=int(args.hidden_dim),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(args.learning_rate))

    def masked_logits(self, logits: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
        """把非法动作的 logit 置为极小值。

        采样和 PPO 更新都必须使用同一套 mask。否则训练时会给当时不可选的动作
        分配概率，old/new log-prob 的比较就不再对应同一个行为策略。
        """

        safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return safe_logits.masked_fill(action_masks <= 0.0, -1.0e9)

    def sample_action(self, state: np.ndarray, action_mask: np.ndarray) -> tuple[int, float, float]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            distribution = Categorical(logits=self.masked_logits(logits, mask_tensor))
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def collect_rollouts(self) -> RolloutBatch:
        states: list[np.ndarray] = []
        action_masks: list[np.ndarray] = []
        actions: list[int] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        values: list[float] = []
        dones: list[bool] = []
        episode_records: list[dict[str, object]] = []

        for episode_index in range(int(self.args.episodes_per_update)):
            state = self.env.reset()
            episode_reward = 0.0
            step_count = 0
            final_info: dict[str, object] = {}

            while True:
                action_mask = self.env.action_mask()
                action_id, log_prob, value = self.sample_action(state, action_mask)
                next_state, reward, done, info = self.env.step(action_id)
                states.append(state)
                action_masks.append(action_mask)
                actions.append(action_id)
                log_probs.append(log_prob)
                rewards.append(finite_float(reward, -float(self.args.illegal_penalty)))
                values.append(value)
                dones.append(bool(done))
                episode_reward += float(reward)
                step_count += 1
                state = next_state
                final_info = info
                if done:
                    break

            episode_records.append(
                {
                    "episode_in_update": episode_index,
                    "episode_reward": episode_reward,
                    "episode_steps": step_count,
                    **final_info,
                }
            )

        advantages, returns = self.compute_gae(rewards=rewards, values=values, dones=dones)
        return RolloutBatch(
            states=np.asarray(states, dtype=np.float32),
            action_masks=np.asarray(action_masks, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            old_log_probs=np.asarray(log_probs, dtype=np.float32),
            returns=returns.astype(np.float32),
            advantages=advantages.astype(np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            episode_records=episode_records,
        )

    def compute_gae(self, rewards: list[float], values: list[float], dones: list[bool]) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        for index in reversed(range(len(rewards))):
            if index == len(rewards) - 1 or dones[index]:
                next_value = 0.0
                next_non_terminal = 0.0
            else:
                next_value = values[index + 1]
                next_non_terminal = 1.0
            delta = rewards[index] + float(self.args.gamma) * next_value * next_non_terminal - values[index]
            last_gae = delta + float(self.args.gamma) * float(self.args.gae_lambda) * next_non_terminal * last_gae
            advantages[index] = float(last_gae)
        returns = advantages + np.asarray(values, dtype=np.float32)
        advantage_std = float(advantages.std())
        if advantage_std > 1e-8:
            advantages = (advantages - float(advantages.mean())) / (advantage_std + 1e-8)
        return advantages, returns

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        states = torch.nan_to_num(torch.as_tensor(batch.states, dtype=torch.float32, device=self.device), nan=0.0)
        action_masks = torch.as_tensor(batch.action_masks, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.nan_to_num(torch.as_tensor(batch.old_log_probs, dtype=torch.float32, device=self.device), nan=0.0)
        returns = torch.nan_to_num(torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device), nan=0.0)
        advantages = torch.nan_to_num(torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device), nan=0.0)
        sample_count = states.shape[0]
        indices = np.arange(sample_count)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_values: list[float] = []
        approx_kls: list[float] = []

        for _ in range(int(self.args.ppo_epochs)):
            np.random.shuffle(indices)
            for start in range(0, sample_count, int(self.args.minibatch_size)):
                batch_index = indices[start : start + int(self.args.minibatch_size)]
                batch_states = states[batch_index]
                batch_masks = action_masks[batch_index]
                batch_actions = actions[batch_index]
                batch_old_log_probs = old_log_probs[batch_index]
                batch_returns = returns[batch_index]
                batch_advantages = advantages[batch_index]

                logits, values = self.model(batch_states)
                distribution = Categorical(logits=self.masked_logits(logits, batch_masks))
                new_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - float(self.args.clip_ratio), 1.0 + float(self.args.clip_ratio))
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                value_loss = torch.mean((values - batch_returns) ** 2)
                loss = (
                    policy_loss
                    + float(self.args.value_coef) * value_loss
                    - float(self.args.entropy_coef) * entropy
                )
                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.args.max_grad_norm))
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_values.append(float(entropy.item()))
                approx_kls.append(float(approx_kl.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else float("nan"),
            "value_loss": float(np.mean(value_losses)) if value_losses else float("nan"),
            "entropy": float(np.mean(entropy_values)) if entropy_values else float("nan"),
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else float("nan"),
        }

    def train(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        curve_records: list[dict[str, object]] = []
        episode_records: list[dict[str, object]] = []
        iterator = optional_progress(range(int(self.args.total_updates)), desc="PPO updates")

        for update_index in iterator:
            rollout_start = time.perf_counter()
            batch = self.collect_rollouts()
            update_metrics = self.update(batch)
            elapsed = time.perf_counter() - rollout_start

            rewards = [float(record.get("episode_reward", 0.0)) for record in batch.episode_records]
            ok_count = sum(1 for record in batch.episode_records if str(record.get("terminal_status")) == "ok")
            duplicate_count = sum(1 for record in batch.episode_records if str(record.get("terminal_status")) == "duplicate_formula")
            illegal_count = sum(1 for record in batch.episode_records if "illegal" in str(record.get("terminal_status")))
            curve_record = {
                "update": update_index + 1,
                "episodes": int(self.args.episodes_per_update),
                "mean_episode_reward": float(np.mean(rewards)) if rewards else float("nan"),
                "max_episode_reward": float(np.max(rewards)) if rewards else float("nan"),
                "ok_terminal_count": ok_count,
                "duplicate_terminal_count": duplicate_count,
                "illegal_terminal_count": illegal_count,
                "unique_candidate_count": len(self.env.evaluator.validation_records_by_id),
                "elapsed_seconds": elapsed,
                **update_metrics,
            }
            curve_records.append(curve_record)

            for record in batch.episode_records:
                episode_records.append({"update": update_index + 1, **record})

            if tqdm is not None:
                iterator.set_postfix(
                    {
                        "mean_reward": f"{curve_record['mean_episode_reward']:.4f}",
                        "unique": curve_record["unique_candidate_count"],
                        "ok": ok_count,
                    }
                )
            else:
                print(
                    "[PPO] "
                    f"update={update_index + 1}/{self.args.total_updates} "
                    f"mean_reward={curve_record['mean_episode_reward']:.4f} "
                    f"unique={curve_record['unique_candidate_count']} "
                    f"elapsed={format_duration(elapsed)}",
                    flush=True,
                )

        return pd.DataFrame(curve_records), pd.DataFrame(episode_records)


def summarize_existing_miner(csv_path: Path, miner_name: str) -> dict[str, object]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return {"miner": miner_name, "source_file": str(csv_path), "status": f"failed: {exc}"}
    if df.empty:
        return {"miner": miner_name, "source_file": str(csv_path), "status": "empty"}

    score_column = "oos_score" if "oos_score" in df.columns else None
    if score_column is None and "alphaeval_style_score" in df.columns:
        score_column = "alphaeval_style_score"
    if score_column is None and "oos_pearson_ic_mean" in df.columns:
        score_column = "oos_pearson_ic_mean"
    if score_column is not None:
        df = df.sort_values(score_column, ascending=False)
    top = df.iloc[0].to_dict()
    formulas = df["formula"].astype(str) if "formula" in df.columns else pd.Series(dtype=str)
    duplicate_ratio = 1.0 - (formulas.nunique() / len(formulas)) if len(formulas) else float("nan")
    return {
        "miner": miner_name,
        "source_file": str(csv_path),
        "status": "ok",
        "candidate_count": len(df),
        "duplicate_formula_ratio": duplicate_ratio,
        "top_formula": top.get("formula", ""),
        "top_oos_pearson_ic_mean": top.get("oos_pearson_ic_mean", np.nan),
        "top_oos_spearman_ic_mean": top.get("oos_spearman_ic_mean", np.nan),
        "top_oos_long_short_spread": top.get("oos_long_short_spread", np.nan),
        "top_score": top.get(score_column, np.nan) if score_column else np.nan,
    }


def newest_matching_file(patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(PROJECT_ROOT.glob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_existing_miner_comparison(deep_rl_oos_path: Path) -> pd.DataFrame:
    sources = {
        "ppo_deep_rl": deep_rl_oos_path,
        "contextual_bandit": newest_matching_file(["factor_mining_workspace/rl_mining_outputs/*/oos_metrics.csv"]),
        "probabilistic_grammar": newest_matching_file(["factor_mining_workspace/generative_mining_outputs/*/oos_metrics.csv"]),
        "warm_gp": newest_matching_file(
            [
                "factor_mining_workspace/auto_mining_outputs/*/oos_metrics.csv",
                "factor_mining_workspace/auto_mining_outputs/*/alphaeval_scores.csv",
            ]
        ),
    }
    rows: list[dict[str, object]] = []
    for miner_name, path in sources.items():
        if path is None:
            rows.append({"miner": miner_name, "status": "not_found"})
            continue
        rows.append(summarize_existing_miner(path, miner_name))
    return pd.DataFrame(rows)


def dataframe_head_to_markdown(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    return dataframe_to_markdown(df.head(max_rows))


def build_report_text(
    config: dict[str, Any],
    seed_df: pd.DataFrame,
    training_curve_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    runtime_seconds: float,
) -> str:
    best_validation = validation_df.head(10).copy()
    best_oos = oos_df.head(10).copy()
    selected = selected_df.copy()
    return f"""# PPO Deep RL Formula Factor Mining Report

## 1. What This Is

This run implements a real PPO-based formula factor generator.

- `state`: current formula AST summary, family information, operator histogram, field-family histogram, last action and step progress.
- `action`: `STOP`, unary mutation, or binary formula composition with a partner seed.
- `reward`: controlled by `reward_mode`.
- `predictive_ic`: emphasizes single-factor IC, RankIC and long-short.
- `incremental_proxy`: emphasizes RankIC, long-short, de-correlation, lower complexity and portfolio-like stability.
- `reward sample`: PPO reward uses a deterministic sample of train-period validation dates for speed.
- `OOS`: 2026 data is used only after PPO training for final audit.

This is still a local research MVP. It should be described as `PPO-based Deep RL formula mining`, not as a production trading RL system.

## 2. Config

```json
{json.dumps(config, ensure_ascii=False, indent=2)}
```

Runtime: `{format_duration(runtime_seconds)}`

## 3. Seed Library

Seed count: `{len(seed_df)}`

{dataframe_head_to_markdown(seed_df, max_rows=15)}

## 4. PPO Training Curve

{dataframe_head_to_markdown(training_curve_df, max_rows=20)}

## 5. Top Validation Candidates

{dataframe_head_to_markdown(best_validation, max_rows=10)}

## 6. OOS Audit

{dataframe_head_to_markdown(best_oos, max_rows=10)}

## 7. Selected Deep RL Factor Zoo

{dataframe_head_to_markdown(selected, max_rows=10)}

## 8. Comparison With Existing Miners

{dataframe_head_to_markdown(comparison_df, max_rows=10)}

## 9. Conservative Interpretation

The correct interpretation is:

- PPO now controls multi-step formula generation through a neural policy.
- PPO reward does not use 2026 OOS.
- OOS improvement is an empirical result, not an assumption.
- If PPO candidates do not beat warm-GP / bandit / grammar, the engineering contribution is still valid, but the research conclusion must say the extra search complexity did not yet produce stable incremental alpha.
"""


def write_comparison_report(output_path: Path, comparison_df: pd.DataFrame) -> None:
    text = f"""# Deep RL vs Existing Factor Miners

This report compares the new PPO Deep RL factor miner with existing MyQuant factor-mining baselines.

{dataframe_head_to_markdown(comparison_df, max_rows=20)}

## Reading Rule

Use this table as a search-method comparison, not as a trading-performance claim.
The fair question is whether a more complex generator produces more stable, less redundant, and more OOS-useful formulas than warm-GP, contextual bandit, and probabilistic grammar.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    random.seed(int(args.random_seed))
    np.random.seed(int(args.random_seed))
    torch.manual_seed(int(args.random_seed))
    rng = random.Random(int(args.random_seed))
    start_time = time.perf_counter()

    output_root = resolve_path(args.output_dir)
    run_name = args.run_name or f"ppo_formula_{args.target_horizon}d_u{args.total_updates}_e{args.episodes_per_update}_s{args.random_seed}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Info] Loading strict time-split feature data for PPO factor mining...", flush=True)
    full_train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    policy_fit_df, validation_df, internal_purge_summary = split_train_validation_by_time(
        full_train_df,
        args.validation_fraction,
        purge_days=int(args.target_horizon),
    )
    print(
        "[Info] PPO reward validation window: "
        f"{pd.to_datetime(validation_df['date']).min().date()} to {pd.to_datetime(validation_df['date']).max().date()} "
        f"({len(validation_df)} rows)",
        flush=True,
    )
    print(
        "[Info] OOS audit window: "
        f"{pd.to_datetime(test_df['date']).min().date()} to {pd.to_datetime(test_df['date']).max().date()} "
        f"({len(test_df)} rows)",
        flush=True,
    )

    seed_df, seed_nodes = build_seed_nodes(
        train_df=policy_fit_df,
        model_dir=resolve_path(args.model_dir),
        args=args,
    )
    if not seed_nodes:
        raise ValueError("No seed nodes were available for PPO formula mining.")

    action_catalog = build_action_catalog(seed_nodes=seed_nodes, blend_weights=tuple(DEFAULT_BLEND_WEIGHTS))
    evaluator = CandidateEvaluator(validation_df=validation_df, test_df=test_df, args=args)
    env = DeepRLFormulaEnv(seed_nodes=seed_nodes, action_catalog=action_catalog, evaluator=evaluator, args=args, rng=rng)
    trainer = PPOTrainer(env=env, args=args)

    resolved_data_path = resolve_path(args.data_path)
    config = {
        "searcher": "ppo_deep_rl_formula_mining",
        "target_column": target_column,
        "target_horizon": args.target_horizon,
        "data_path": project_relative_path(resolved_data_path, PROJECT_ROOT),
        "data_fingerprint": build_data_fingerprint(resolved_data_path, PROJECT_ROOT),
        "sample_start_date": args.sample_start_date,
        "oos_start_date": args.oos_start_date,
        "price_adjustment_mode": args.price_adjustment_mode,
        "seed_top_k": args.seed_top_k,
        "seed_count": len(seed_nodes),
        "action_count": len(action_catalog),
        "state_dim": env.state_dim,
        "total_updates": args.total_updates,
        "episodes_per_update": args.episodes_per_update,
        "max_steps": args.max_steps,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_ratio": args.clip_ratio,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "selected_top_k": args.selected_top_k,
        "max_factor_corr": args.max_factor_corr,
        "reward_mode": args.reward_mode,
        "random_seed": args.random_seed,
        "dataset_summary": dataset_summary,
        "policy_fit_rows": len(policy_fit_df),
        "internal_validation_purge": internal_purge_summary,
        "validation_rows": len(validation_df),
        "reward_validation_rows": len(evaluator.reward_df),
        "reward_validation_date_count": int(pd.to_datetime(evaluator.reward_df["date"]).nunique()),
        "oos_rows": len(test_df),
    }
    (output_dir / "config.json").write_text(dumps_strict_json(config), encoding="utf-8")
    seed_df.to_csv(output_dir / "seed_library.csv", index=False)
    pd.DataFrame([{"action_id": index, **action.__dict__, "label": action.label()} for index, action in enumerate(action_catalog)]).to_csv(
        output_dir / "action_catalog.csv",
        index=False,
    )

    training_curve_df, episode_trace_df = trainer.train()
    validation_metrics_df = evaluator.validation_metrics_df()
    oos_metrics_df = evaluator.evaluate_oos_candidates()
    selected_factor_zoo_df = evaluator.build_selected_factor_zoo(oos_metrics_df)

    runtime_seconds = time.perf_counter() - start_time
    training_curve_df.to_csv(output_dir / "ppo_training_curve.csv", index=False)
    episode_trace_df.to_csv(output_dir / "episode_trace.csv", index=False)
    validation_metrics_df.to_csv(output_dir / "candidate_formulas.csv", index=False)
    validation_metrics_df.to_csv(output_dir / "policy_validation_metrics.csv", index=False)
    oos_metrics_df.to_csv(output_dir / "oos_metrics.csv", index=False)
    selected_factor_zoo_df.to_csv(output_dir / "selected_factor_zoo.csv", index=False)
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "config": config,
            "action_catalog": [action.__dict__ for action in action_catalog],
        },
        output_dir / "ppo_model.pt",
    )

    comparison_df = build_existing_miner_comparison(output_dir / "oos_metrics.csv")
    comparison_df.to_csv(output_dir / "deep_rl_vs_existing_miners.csv", index=False)
    write_comparison_report(output_dir / "deep_rl_vs_existing_miners_report.md", comparison_df)
    (output_dir / "report.md").write_text(
        build_report_text(
            config=config,
            seed_df=seed_df,
            training_curve_df=training_curve_df,
            validation_df=validation_metrics_df,
            oos_df=oos_metrics_df,
            selected_df=selected_factor_zoo_df,
            comparison_df=comparison_df,
            runtime_seconds=runtime_seconds,
        ),
        encoding="utf-8",
    )

    print(f"[Done] PPO Deep RL factor mining report written to: {output_dir / 'report.md'}", flush=True)
    print(f"[Done] Runtime: {format_duration(runtime_seconds)}", flush=True)
    if not selected_factor_zoo_df.empty:
        print(selected_factor_zoo_df.head(min(5, len(selected_factor_zoo_df))).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
