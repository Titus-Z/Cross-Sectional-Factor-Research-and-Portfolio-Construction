from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.auto_factor_mining import build_seed_nodes
from factor_mining_workspace.generative_factor_mining import generate_unique_candidates
from factor_mining_workspace.heuristic_factor_search import standardize_candidate_cross_sectionally
from factor_mining_workspace.mined_factor_model_ablation import get_numeric_feature_columns
from factor_mining_workspace.rl_factor_mining import run_contextual_bandit_search
from factor_mining_workspace.single_factor_case_study import (
    dataframe_to_markdown,
    load_or_build_preprocessed_train_test,
    sanitize_name,
)
from src.model import build_model
from src.reporting import calculate_prediction_metrics
from src.runtime_config import (
    DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)
from src.time_series_pipeline import purge_training_label_overlap


DEFAULT_OUTPUT_ROOT = "factor_mining_workspace/rl_generative_incremental_outputs"

FUNDAMENTAL_AND_CONTEXT_COLUMNS = {
    "eps",
    "pe",
    "pb",
    "ps",
    "roe",
    "roa",
    "yoy",
    "qoq",
    "earnings_yield",
    "book_to_price",
    "sales_to_price",
    "profitability_combo",
    "growth_combo",
    "earnings_yield_rank",
    "book_to_price_rank",
    "sales_to_price_rank",
    "profitability_combo_rank",
    "growth_combo_rank",
}

CONTEXT_PREFIXES = (
    "macro_",
    "sector_",
    "market_",
    "stock_excess_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search RL/generative formula factors and test model-layer incremental value."
    )
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始日频数据路径。")
    parser.add_argument("--model-dir", default=DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR, help="seed 特征来源模型目录。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="输出目录。")
    parser.add_argument("--cache-dir", default=".cache", help="缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="预测目标周期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--n-groups", type=int, default=5, help="单因子诊断分组数。")
    parser.add_argument("--min-cross-section", type=int, default=30, help="每个日期最少股票数。")
    parser.add_argument("--seed-top-k", type=int, default=20, help="从当前模型上下文取多少个 seed 特征。")
    parser.add_argument("--population-size", type=int, default=30, help="兼容已有 warm seed 读取逻辑。")
    parser.add_argument(
        "--history-run-dir",
        default="factor_mining_workspace/heuristic_search_outputs",
        help="可选历史候选目录；用于把以前搜索过的有效公式加入 seed pool。",
    )
    parser.add_argument("--rl-episodes", type=int, default=160, help="RL-style contextual bandit 轮数。")
    parser.add_argument("--generative-samples", type=int, default=220, help="概率语法生成候选数量。")
    parser.add_argument("--candidate-top-k", type=int, default=120, help="进入模型层筛选的候选上限。")
    parser.add_argument("--max-selected", type=int, default=5, help="最多选择多少个增量因子。")
    parser.add_argument("--validation-fraction", type=float, default=0.25, help="训练期内部最后多少日期作为验证。")
    parser.add_argument(
        "--validation-window-count",
        type=int,
        default=1,
        help="训练期尾部切成多少个连续 validation window；大于 1 时启用多窗口稳健选择。",
    )
    parser.add_argument(
        "--validation-stability-penalty",
        type=float,
        default=0.0,
        help="多窗口选择时对 fold score 标准差的惩罚系数。",
    )
    parser.add_argument(
        "--min-window-positive-ratio",
        type=float,
        default=0.0,
        help="多窗口选择时，候选相对当前组合改善的最小窗口占比。",
    )
    parser.add_argument(
        "--max-candidate-corr",
        type=float,
        default=1.0,
        help="候选与已选候选的最大允许绝对相关系数；小于 1 时启用相关性过滤。",
    )
    parser.add_argument(
        "--candidate-corr-penalty",
        type=float,
        default=0.0,
        help="候选与已选候选相关性越高，selection score 扣分越多。",
    )
    parser.add_argument(
        "--family-repeat-penalty",
        type=float,
        default=0.0,
        help="候选 family 与已选 family 有重叠时的扣分。",
    )
    parser.add_argument(
        "--max-selected-per-family",
        type=int,
        default=0,
        help="同一个精确 family 最多选择多少个候选；0 表示不限制。",
    )
    parser.add_argument("--model", default="ridge", help="增量筛选使用的模型，默认 Ridge。")
    parser.add_argument(
        "--selection-mode",
        choices=["forward", "individual_validation", "rank_rule"],
        default="forward",
        help=(
            "候选选择模式：forward 为累计贪心；individual_validation 先按单候选多窗口增量排序；"
            "rank_rule 只按训练期候选质量分数排序，用于避免 validation regime 失配。"
        ),
    )
    parser.add_argument(
        "--rank-rule-score-column",
        default="train_score",
        help="selection-mode=rank_rule 时使用的候选元数据分数字段，默认 train_score。",
    )
    parser.add_argument("--selection-metric", default="pearson_ic_mean", help="forward selection 主指标。")
    parser.add_argument(
        "--baseline-feature-mode",
        choices=["all_numeric", "technical_only"],
        default="all_numeric",
        help="baseline 特征口径：all_numeric 使用全部已生成数值特征；technical_only 排除基本面/行业/宏观/市场状态上下文。",
    )
    parser.add_argument("--min-validation-delta", type=float, default=0.0, help="每一步验证集最小提升。")
    parser.add_argument("--min-oos-delta", type=float, default=0.0, help="审计时每一步 OOS 最小提升。")
    parser.add_argument(
        "--exclude-candidate-families",
        default="",
        help="逗号分隔的候选 family 黑名单，例如 size。用于避免自动搜索反复选择不稳定的市值代理。",
    )
    parser.add_argument(
        "--exclude-candidate-fields",
        default="",
        help="逗号分隔的候选字段黑名单，例如 shares_outstanding_proxy。命中任一字段的公式会被过滤。",
    )
    parser.add_argument(
        "--min-candidate-coverage",
        type=float,
        default=0.0,
        help="候选训练期覆盖率下限。基本面字段覆盖不足时容易过拟合，可用该参数过滤。",
    )
    parser.add_argument(
        "--individual-oos-audit-top-k",
        type=int,
        default=0,
        help="大于 0 时，对进入筛选池的前 K 个候选逐个做 OOS 单独增量审计；只用于诊断，不参与选择。",
    )
    parser.add_argument(
        "--save-candidate-matrices",
        action="store_true",
        help="保存 train/test 候选特征矩阵，便于后处理诊断；文件较大，默认关闭。",
    )
    parser.add_argument(
        "--residual-prefilter-top-k",
        type=int,
        default=0,
        help="大于 0 时，先按 validation residual 解释力保留前 K 个候选，再做 forward selection。",
    )
    parser.add_argument(
        "--residual-prefilter-min-score",
        type=float,
        default=-1e9,
        help="validation residual composite score 最低门槛；默认不设门槛。",
    )
    parser.add_argument("--max-depth", type=int, default=4, help="公式 AST 最大深度。")
    parser.add_argument("--max-complexity", type=int, default=9, help="公式 AST 最大复杂度。")
    parser.add_argument("--max-fields", type=int, default=4, help="单个公式最多字段数。")
    parser.add_argument("--survivor-ratio", type=float, default=0.35, help="候选预筛保留比例。")
    parser.add_argument("--epsilon-start", type=float, default=0.70, help="RL 初始探索率。")
    parser.add_argument("--epsilon-end", type=float, default=0.10, help="RL 最终探索率。")
    parser.add_argument("--learning-rate", type=float, default=0.25, help="RL Q 值学习率。")
    parser.add_argument("--reward-complexity-penalty", type=float, default=0.01, help="RL 复杂度惩罚。")
    parser.add_argument("--reward-logic-bonus", type=float, default=0.05, help="RL 逻辑奖励。")
    parser.add_argument("--terminal-probability", type=float, default=0.30, help="生成式采样直接终止概率。")
    parser.add_argument("--unary-probability", type=float, default=0.35, help="生成式采样 unary 概率。")
    parser.add_argument("--smoothing", type=float, default=1.0, help="生成式 prior 平滑。")
    parser.add_argument(
        "--include-alpha-seeds",
        action="store_true",
        help="允许 canonical 价格尺度不变 Alpha 子集作为 seed；不会加载全部 Alpha191。",
    )
    parser.add_argument("--include-raw-market-seeds", action="store_true", help="允许原始量价 seed。")
    parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭预处理缓存。")
    parser.add_argument("--random-seed", type=int, default=31, help="随机种子。")
    parser.add_argument("--run-name", default=None, help="输出目录名。")
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def optional_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def parse_csv_arg(value: str | None) -> set[str]:
    """把命令行里的逗号分隔参数转成小写集合。

    自动因子搜索会产生大量公式。我们需要能快速排除某些不想研究的
    family 或字段，例如 `size`、`shares_outstanding_proxy`。
    """

    if not value:
        return set()
    return {item.strip().lower() for item in str(value).split(",") if item.strip()}


def split_metadata_list(value: object) -> set[str]:
    """解析候选元数据里的 `fields` 或 `family` 字段。

    - `fields` 通常是 `a,b,c`；
    - `family` 通常是 `momentum+volatility`。

    这里同时支持逗号和 `+`，方便统一做黑名单匹配。
    """

    if value is None or pd.isna(value):
        return set()
    text = str(value).replace("+", ",")
    return {item.strip().lower() for item in text.split(",") if item.strip()}


def metric_score(metrics: dict[str, float], metric_name: str) -> float:
    """返回用于 selection 的指标值。

    这里默认只用 `pearson_ic_mean`，因为它是项目当前最稳定的横截面模型层指标。
    如果以后要更保守，可以把这里改成多指标 composite。
    """

    if metric_name == "composite_ic_rank_ls":
        pearson_ic = float(metrics.get("pearson_ic_mean", 0.0) or 0.0)
        spearman_ic = float(metrics.get("spearman_ic_mean", 0.0) or 0.0)
        long_short = float(metrics.get("long_short_return", 0.0) or 0.0)
        # 这是模型层 selection 的保守综合分：
        # Pearson IC 衡量线性横截面相关，RankIC 衡量排序，long-short 衡量分组收益方向。
        return float(0.50 * pearson_ic + 0.30 * spearman_ic + 0.20 * long_short)

    value = metrics.get(metric_name, float("nan"))
    if value is None or pd.isna(value):
        return -1e9
    if metric_name in {"rmse", "mae", "portfolio_max_drawdown"}:
        return -float(value)
    return float(value)


def split_train_validation_by_time(
    train_df: pd.DataFrame,
    validation_fraction: float,
    purge_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(train_df["date"]).dropna().unique())
    if len(dates) < 20:
        raise ValueError("Not enough train dates for internal validation split.")
    validation_fraction = min(max(float(validation_fraction), 0.05), 0.50)
    split_index = int(math.floor(len(dates) * (1.0 - validation_fraction)))
    split_index = min(max(split_index, 1), len(dates) - 1)
    fit_dates = set(dates[:split_index])
    validation_dates = set(dates[split_index:])
    date_series = pd.to_datetime(train_df["date"])
    fit_df = train_df[date_series.isin(fit_dates)].copy()
    validation_df = train_df[date_series.isin(validation_dates)].copy()
    fit_df, _ = purge_training_label_overlap(fit_df, target_horizon=purge_days)
    return fit_df, validation_df


def build_train_validation_windows(
    train_df: pd.DataFrame,
    validation_fraction: float,
    window_count: int,
    purge_days: int,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """构造训练期内部的 expanding-window validation 切分。

    单窗口 validation 很容易只适配某一段市场风格。多窗口切分把训练期尾部
    分成几个连续窗口，每个窗口都只用它之前的数据训练模型。这样 selection
    更接近 walk-forward 思路，但仍然完全限制在 OOS 之前。
    """

    window_count = max(int(window_count), 1)
    if window_count <= 1:
        fit_df, validation_df = split_train_validation_by_time(
            train_df,
            validation_fraction,
            purge_days=purge_days,
        )
        return [("validation_tail_1", fit_df, validation_df)]

    dates = sorted(pd.to_datetime(train_df["date"]).dropna().unique())
    if len(dates) < window_count * 10:
        raise ValueError("Not enough train dates for multi-window validation split.")

    validation_fraction = min(max(float(validation_fraction), 0.10), 0.60)
    total_validation_dates = int(math.floor(len(dates) * validation_fraction))
    total_validation_dates = max(total_validation_dates, window_count * 5)
    total_validation_dates = min(total_validation_dates, len(dates) - 1)
    validation_tail = dates[-total_validation_dates:]
    validation_chunks = [chunk for chunk in np.array_split(np.array(validation_tail), window_count) if len(chunk) > 0]
    date_series = pd.to_datetime(train_df["date"])

    windows: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for index, chunk in enumerate(validation_chunks, start=1):
        validation_dates = set(chunk.tolist())
        first_validation_date = min(validation_dates)
        fit_dates = {date for date in dates if date < first_validation_date}
        if not fit_dates:
            continue
        fit_df = train_df[date_series.isin(fit_dates)].copy()
        validation_df = train_df[date_series.isin(validation_dates)].copy()
        if fit_df.empty or validation_df.empty:
            continue
        fit_df, _ = purge_training_label_overlap(fit_df, target_horizon=purge_days)
        windows.append((f"validation_window_{index}", fit_df, validation_df))

    if not windows:
        raise ValueError("No usable validation windows were created.")
    return windows


def prepare_xy_frames(
    fit_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    all_feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """用 fit 段 median 填充缺失值，然后返回可反复切片的 X 矩阵。"""

    fit_x = fit_df[all_feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    predict_x = predict_df[all_feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = fit_x.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_fit = pd.to_numeric(fit_df["y"], errors="coerce")
    valid_mask = y_fit.notna()
    return fit_x.fillna(medians).loc[valid_mask], predict_x.fillna(medians), y_fit.loc[valid_mask]


def select_baseline_feature_columns(data: pd.DataFrame, mode: str) -> list[str]:
    """按实验口径选择 baseline 特征列。

    - `all_numeric`：强 baseline，包含当前已生成的全部数值特征；
    - `technical_only`：技术 baseline，排除基本面、行业、市场状态、宏观上下文。

    第二种口径用于回答一个更清晰的问题：
    “自动挖出的 context 交互因子是否能相对传统技术模型提供增量？”
    """

    columns = get_numeric_feature_columns(data)
    if mode == "all_numeric":
        return columns
    if mode != "technical_only":
        raise ValueError(f"Unsupported baseline feature mode: {mode}")

    filtered_columns: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if lowered in FUNDAMENTAL_AND_CONTEXT_COLUMNS:
            continue
        if any(lowered.startswith(prefix) for prefix in CONTEXT_PREFIXES):
            continue
        filtered_columns.append(column)
    return filtered_columns


def fit_predict_with_feature_subset(
    fit_x: pd.DataFrame,
    predict_x: pd.DataFrame,
    y_fit: pd.Series,
    predict_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    random_seed: int,
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    model = build_model(model_name=model_name, random_state=random_seed)
    model.fit(fit_x[feature_columns], y_fit)
    prediction = np.asarray(model.predict(predict_x[feature_columns]), dtype=float)
    elapsed = time.perf_counter() - start
    prediction_df = predict_df[["date", "instrument_id", "y"]].copy()
    prediction_df["predicted_y"] = prediction
    return prediction_df, elapsed


def evaluate_feature_subset(
    fit_x: pd.DataFrame,
    predict_x: pd.DataFrame,
    y_fit: pd.Series,
    predict_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    random_seed: int,
    experiment_name: str,
) -> tuple[dict[str, float | str], float]:
    prediction_df, elapsed = fit_predict_with_feature_subset(
        fit_x=fit_x,
        predict_x=predict_x,
        y_fit=y_fit,
        predict_df=predict_df,
        feature_columns=feature_columns,
        model_name=model_name,
        random_seed=random_seed,
    )
    metrics = calculate_prediction_metrics(prediction_df)
    metrics["experiment"] = experiment_name
    return metrics, elapsed


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, float | str]],
    fold_scores: list[float],
    experiment_name: str,
    stability_penalty: float,
) -> dict[str, float | str]:
    """把多个 validation window 的结果合成一个选择指标。

    这里保留每个指标的平均值，同时把 fold score 的均值、标准差、最小值写入
    报告。selection score 使用 `mean - penalty * std`，避免只在某一个窗口暴涨
    的候选被选中。
    """

    numeric_values: dict[str, list[float]] = {}
    for metrics in fold_metrics:
        for key, value in metrics.items():
            if key == "experiment":
                continue
            if isinstance(value, (int, float, np.floating)) and pd.notna(value):
                numeric_values.setdefault(key, []).append(float(value))

    aggregated: dict[str, float | str] = {
        key: float(np.nanmean(values)) for key, values in numeric_values.items() if values
    }
    score_array = np.asarray(fold_scores, dtype=float)
    aggregated["experiment"] = experiment_name
    aggregated["validation_window_count"] = float(len(fold_scores))
    aggregated["validation_score_mean"] = float(np.nanmean(score_array)) if len(score_array) else np.nan
    aggregated["validation_score_std"] = float(np.nanstd(score_array)) if len(score_array) else np.nan
    aggregated["validation_score_min"] = float(np.nanmin(score_array)) if len(score_array) else np.nan
    aggregated["validation_stability_adjusted_score"] = float(
        aggregated["validation_score_mean"] - float(stability_penalty) * aggregated["validation_score_std"]
    )
    return aggregated


def family_overlap_penalty(candidate_family: object, selected_families: list[object], penalty: float) -> float:
    """如果候选 family 与已选 family 重叠，则扣一个固定分。

    这样做的目的：控制 5 个因子都来自同一种市场逻辑的风险。
    当前项目里动量、波动率、流动性信号高度相关，family diversity 是必要约束。
    """

    penalty = float(penalty or 0.0)
    if penalty <= 0 or not selected_families:
        return 0.0
    candidate_atoms = split_metadata_list(candidate_family)
    if not candidate_atoms:
        return 0.0
    for selected_family in selected_families:
        if candidate_atoms & split_metadata_list(selected_family):
            return penalty
    return 0.0


def max_abs_corr_to_selected(
    candidate_corr: pd.DataFrame,
    candidate_column: str,
    selected_features: list[str],
) -> float:
    """返回候选与已选候选之间的最大绝对相关系数。"""

    if not selected_features or candidate_column not in candidate_corr.index:
        return 0.0
    values: list[float] = []
    for selected_feature in selected_features:
        if selected_feature not in candidate_corr.columns:
            continue
        value = candidate_corr.loc[candidate_column, selected_feature]
        if pd.notna(value):
            values.append(abs(float(value)))
    return max(values) if values else 0.0


def calculate_signal_target_metrics(
    data: pd.DataFrame,
    signal_column: str,
    target_column: str,
    n_groups: int,
    min_cross_section: int,
) -> dict[str, float]:
    """计算候选信号对任意目标列的横截面诊断指标。

    这里的 `target_column` 可以是原始收益 `y`，也可以是 validation residual。
    residual-aware 预筛使用后者，目的是优先找 baseline 尚未解释的部分。
    """

    pearson_values: list[float] = []
    spearman_values: list[float] = []
    spread_values: list[float] = []

    for _, day_df in data.groupby("date"):
        signal = pd.to_numeric(day_df[signal_column], errors="coerce")
        target = pd.to_numeric(day_df[target_column], errors="coerce")
        valid = pd.DataFrame({"signal": signal, "target": target}).dropna()
        if len(valid) < int(min_cross_section):
            continue

        pearson = valid["signal"].corr(valid["target"], method="pearson")
        spearman = valid["signal"].corr(valid["target"], method="spearman")
        if pd.notna(pearson):
            pearson_values.append(float(pearson))
        if pd.notna(spearman):
            spearman_values.append(float(spearman))

        try:
            groups = pd.qcut(valid["signal"].rank(method="first"), q=int(n_groups), labels=False, duplicates="drop")
        except ValueError:
            continue
        grouped = valid.assign(group=groups).dropna(subset=["group"]).groupby("group")["target"].mean()
        if len(grouped) >= 2:
            spread_values.append(float(grouped.iloc[-1] - grouped.iloc[0]))

    pearson_mean = float(np.nanmean(pearson_values)) if pearson_values else np.nan
    spearman_mean = float(np.nanmean(spearman_values)) if spearman_values else np.nan
    long_short = float(np.nanmean(spread_values)) if spread_values else np.nan

    # residual prefilter 的 score 只用于训练期内部筛选，不碰 OOS。
    # 这里对 NaN 做 0 处理，避免单项缺失把整个候选直接变成不可排序。
    score = (
        0.50 * (0.0 if pd.isna(pearson_mean) else pearson_mean)
        + 0.30 * (0.0 if pd.isna(spearman_mean) else spearman_mean)
        + 0.20 * (0.0 if pd.isna(long_short) else long_short)
    )

    return {
        "residual_pearson_ic_mean": pearson_mean,
        "residual_spearman_ic_mean": spearman_mean,
        "residual_long_short_return": long_short,
        "residual_composite_score": float(score),
        "residual_ic_days": float(len(pearson_values)),
    }


def score_candidates_against_validation_residual(
    fit_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    baseline_columns: list[str],
    candidate_columns: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """用 validation residual 评估候选因子的增量方向。

    流程：
    1. 只用 baseline 特征训练模型；
    2. 在 validation 日期上预测；
    3. 构造 residual = actual y - baseline prediction；
    4. 逐个计算候选信号与 residual 的横截面 IC / RankIC / long-short。

    这一步只使用训练期内部切出来的 validation，不使用 OOS。
    """

    fit_x, valid_x, y_fit = prepare_xy_frames(fit_df, validation_df, baseline_columns)
    baseline_prediction_df, _ = fit_predict_with_feature_subset(
        fit_x=fit_x,
        predict_x=valid_x,
        y_fit=y_fit,
        predict_df=validation_df,
        feature_columns=baseline_columns,
        model_name=args.model,
        random_seed=args.random_seed,
    )

    scored_df = validation_df[["date", "instrument_id", "y", *candidate_columns]].copy()
    scored_df["_baseline_pred"] = baseline_prediction_df["predicted_y"].to_numpy(dtype=float)
    scored_df["_validation_residual"] = pd.to_numeric(scored_df["y"], errors="coerce") - scored_df["_baseline_pred"]

    rows: list[dict[str, object]] = []
    iterator = optional_progress(candidate_columns, desc="Scoring candidates vs validation residual", leave=False)
    for candidate_column in iterator:
        metrics = calculate_signal_target_metrics(
            data=scored_df,
            signal_column=candidate_column,
            target_column="_validation_residual",
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
        )
        rows.append({"mined_feature": candidate_column, **metrics})

    score_df = pd.DataFrame(rows)
    if not score_df.empty:
        score_df = score_df.sort_values("residual_composite_score", ascending=False).reset_index(drop=True)
    return score_df


def candidate_column_name(source: str, candidate_id: str) -> str:
    return f"mined_{sanitize_name(source)}_{sanitize_name(candidate_id)}"


def combine_candidate_records(
    rl_train_df: pd.DataFrame,
    rl_nodes_by_id: dict[str, Any],
    gen_train_df: pd.DataFrame,
    gen_nodes_by_id: dict[str, Any],
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """合并 RL 和生成式候选，只按训练期指标预筛，不使用 OOS。"""

    frames: list[pd.DataFrame] = []
    nodes_by_global_id: dict[str, Any] = {}

    if not rl_train_df.empty:
        rl_view = rl_train_df.copy()
        rl_view["candidate_source"] = "rl"
        frames.append(rl_view)
        for candidate_id, node in rl_nodes_by_id.items():
            nodes_by_global_id[f"rl::{candidate_id}"] = node

    if not gen_train_df.empty:
        gen_view = gen_train_df.copy()
        gen_view["candidate_source"] = "generative"
        frames.append(gen_view)
        for candidate_id, node in gen_nodes_by_id.items():
            nodes_by_global_id[f"generative::{candidate_id}"] = node

    if not frames:
        return pd.DataFrame(), {}

    candidates = pd.concat(frames, ignore_index=True)
    candidates["global_candidate_id"] = candidates["candidate_source"].astype(str) + "::" + candidates["candidate_id"].astype(str)
    candidates = candidates.drop_duplicates("formula").copy()

    # 新因子必须是“公式变换或组合”，不能把已有单列 seed 直接包装成 mined factor。
    # 否则 `price_range`、`return_std_20` 这类原本就在 baseline 里的技术指标
    # 可能因为重新标准化而看起来有增量，解释上不成立。
    if "operator_count" in candidates.columns:
        operator_count = pd.to_numeric(candidates["operator_count"], errors="coerce").fillna(0)
        candidates = candidates.loc[operator_count > 0].copy()
    if candidates.empty:
        return pd.DataFrame(), {}

    candidates["prefilter_score"] = pd.to_numeric(candidates.get("train_score"), errors="coerce").fillna(-1e9)
    if "rl_reward" in candidates.columns:
        candidates["prefilter_score"] = candidates[["prefilter_score", "rl_reward"]].max(axis=1, skipna=True)
    candidates = candidates.sort_values("prefilter_score", ascending=False).head(top_k).reset_index(drop=True)
    return candidates, {key: value for key, value in nodes_by_global_id.items() if key in set(candidates["global_candidate_id"])}


def apply_candidate_filters(
    candidate_df: pd.DataFrame,
    nodes_by_global_id: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    """按训练期可观察的元数据过滤候选公式。

    这一步的目的，是把明显解释不稳的候选先挡住，降低选择器过拟合。
    例子：
    - `size` family 在 validation 中很容易有效，但 OOS 可能受阶段性风格切换影响；
    - 覆盖率很低的基本面字段可能只是少量样本上的噪声；
    - 某些字段如果被识别为复权/事后代理，也可以通过字段黑名单禁用。
    """

    if candidate_df.empty:
        return candidate_df, nodes_by_global_id, {"before": 0, "after": 0}

    filtered = candidate_df.copy()
    before_count = len(filtered)
    excluded_families = parse_csv_arg(getattr(args, "exclude_candidate_families", ""))
    excluded_fields = parse_csv_arg(getattr(args, "exclude_candidate_fields", ""))
    min_coverage = float(getattr(args, "min_candidate_coverage", 0.0) or 0.0)

    removed_by_family = 0
    removed_by_field = 0
    removed_by_coverage = 0

    if excluded_families and "family" in filtered.columns:
        family_mask = filtered["family"].apply(lambda value: bool(split_metadata_list(value) & excluded_families))
        removed_by_family = int(family_mask.sum())
        filtered = filtered.loc[~family_mask].copy()

    if excluded_fields and "fields" in filtered.columns:
        field_mask = filtered["fields"].apply(lambda value: bool(split_metadata_list(value) & excluded_fields))
        removed_by_field = int(field_mask.sum())
        filtered = filtered.loc[~field_mask].copy()

    if min_coverage > 0 and "train_coverage_ratio" in filtered.columns:
        coverage = pd.to_numeric(filtered["train_coverage_ratio"], errors="coerce").fillna(0.0)
        coverage_mask = coverage < min_coverage
        removed_by_coverage = int(coverage_mask.sum())
        filtered = filtered.loc[~coverage_mask].copy()

    filtered = filtered.reset_index(drop=True)
    allowed_ids = set(filtered["global_candidate_id"].astype(str)) if "global_candidate_id" in filtered.columns else set()
    filtered_nodes = {key: value for key, value in nodes_by_global_id.items() if key in allowed_ids}

    summary = {
        "before": before_count,
        "after": len(filtered),
        "removed_by_family": removed_by_family,
        "removed_by_field": removed_by_field,
        "removed_by_coverage": removed_by_coverage,
        "excluded_families": sorted(excluded_families),
        "excluded_fields": sorted(excluded_fields),
        "min_candidate_coverage": min_coverage,
    }
    return filtered, filtered_nodes, summary


def materialize_candidate_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    nodes_by_global_id: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """把候选公式变成 train/test 上的标准化特征列。"""

    train_enhanced = train_df.copy()
    test_enhanced = test_df.copy()
    train_candidate_columns: dict[str, pd.Series] = {}
    test_candidate_columns: dict[str, pd.Series] = {}
    materialized_records: list[dict[str, object]] = []

    iterator = optional_progress(candidate_df.itertuples(index=False), total=len(candidate_df), desc="Materializing RL/generative factors")
    for row in iterator:
        row_dict = row._asdict()
        global_id = str(row_dict["global_candidate_id"])
        node = nodes_by_global_id.get(global_id)
        if node is None:
            continue

        source = str(row_dict["candidate_source"])
        candidate_id = str(row_dict["candidate_id"])
        column = candidate_column_name(source, candidate_id)
        try:
            train_raw = node.evaluate(train_enhanced)
            test_raw = node.evaluate(test_enhanced)
            train_candidate_columns[column] = standardize_candidate_cross_sectionally(train_enhanced, train_raw).rename(column)
            test_candidate_columns[column] = standardize_candidate_cross_sectionally(test_enhanced, test_raw).rename(column)
        except Exception as exc:
            materialized_records.append(
                {
                    **row_dict,
                    "mined_feature": column,
                    "materialize_status": "failed",
                    "materialize_error": str(exc),
                }
            )
            continue

        materialized_records.append(
            {
                **row_dict,
                "mined_feature": column,
                "materialize_status": "ok",
                "materialize_error": "",
            }
        )

    if train_candidate_columns:
        train_enhanced = pd.concat([train_enhanced, pd.DataFrame(train_candidate_columns, index=train_enhanced.index)], axis=1).copy()
    if test_candidate_columns:
        test_enhanced = pd.concat([test_enhanced, pd.DataFrame(test_candidate_columns, index=test_enhanced.index)], axis=1).copy()

    materialized_df = pd.DataFrame(materialized_records)
    return train_enhanced, test_enhanced, materialized_df


def forward_select_candidates(
    fit_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    baseline_columns: list[str],
    candidate_columns: list[str],
    candidate_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_columns = baseline_columns + candidate_columns
    fit_x, valid_x, y_fit = prepare_xy_frames(fit_df, validation_df, all_columns)

    baseline_metrics, baseline_runtime = evaluate_feature_subset(
        fit_x=fit_x,
        predict_x=valid_x,
        y_fit=y_fit,
        predict_df=validation_df,
        feature_columns=baseline_columns,
        model_name=args.model,
        random_seed=args.random_seed,
        experiment_name="validation_baseline",
    )
    current_columns = list(baseline_columns)
    current_score = metric_score(baseline_metrics, args.selection_metric)
    selected_rows: list[dict[str, object]] = []
    step_metric_rows: list[dict[str, object]] = [
        {
            "step": 0,
            "added_feature": "",
            "candidate_id": "",
            "candidate_source": "",
            "selection_score": current_score,
            "delta_selection_score": 0.0,
            "runtime_seconds": baseline_runtime,
            **baseline_metrics,
        }
    ]

    remaining = list(candidate_columns)
    metadata_by_feature = candidate_metadata.set_index("mined_feature").to_dict("index")

    for step in range(1, int(args.max_selected) + 1):
        best_feature = None
        best_metrics: dict[str, float | str] | None = None
        best_runtime = 0.0
        best_score = -1e9

        iterator = optional_progress(remaining, desc=f"Forward selection step {step}", leave=False)
        for candidate_column in iterator:
            trial_columns = current_columns + [candidate_column]
            trial_metrics, trial_runtime = evaluate_feature_subset(
                fit_x=fit_x,
                predict_x=valid_x,
                y_fit=y_fit,
                predict_df=validation_df,
                feature_columns=trial_columns,
                model_name=args.model,
                random_seed=args.random_seed,
                experiment_name=f"validation_add_{candidate_column}",
            )
            trial_score = metric_score(trial_metrics, args.selection_metric)
            if trial_score > best_score:
                best_feature = candidate_column
                best_metrics = trial_metrics
                best_runtime = trial_runtime
                best_score = trial_score

        delta_score = best_score - current_score
        if best_feature is None or best_metrics is None or delta_score <= float(args.min_validation_delta):
            break

        meta = metadata_by_feature.get(best_feature, {})
        selected_rows.append(
            {
                "selection_step": step,
                "mined_feature": best_feature,
                "candidate_source": meta.get("candidate_source"),
                "candidate_id": meta.get("candidate_id"),
                "formula": meta.get("formula"),
                "family": meta.get("family"),
                "hypothesis": meta.get("hypothesis"),
                "validation_selection_score_before": current_score,
                "validation_selection_score_after": best_score,
                "validation_delta_selection_score": delta_score,
                **{f"validation_{key}": value for key, value in best_metrics.items() if key != "experiment"},
            }
        )
        step_metric_rows.append(
            {
                "step": step,
                "added_feature": best_feature,
                "candidate_id": meta.get("candidate_id"),
                "candidate_source": meta.get("candidate_source"),
                "selection_score": best_score,
                "delta_selection_score": delta_score,
                "runtime_seconds": best_runtime,
                **best_metrics,
            }
        )
        current_columns.append(best_feature)
        remaining.remove(best_feature)
        current_score = best_score

    return pd.DataFrame(selected_rows), pd.DataFrame(step_metric_rows)


def forward_select_candidates_multi_window(
    train_df: pd.DataFrame,
    validation_windows: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    baseline_columns: list[str],
    candidate_columns: list[str],
    candidate_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """多验证窗口稳健 forward selection。

    选择逻辑：
    1. 每个候选必须在多个 validation window 上同时接受检验；
    2. score 使用各窗口均值，并可扣除窗口间波动；
    3. 候选与已选候选相关性过高时跳过或扣分；
    4. family 重复时扣分，避免选出一组高度相似的公式。

    OOS 仍然只在选择冻结后用于审计，不参与这里的选择。
    """

    all_columns = baseline_columns + candidate_columns
    prepared_windows: list[dict[str, object]] = []
    for window_name, fit_df, validation_df in validation_windows:
        fit_x, valid_x, y_fit = prepare_xy_frames(fit_df, validation_df, all_columns)
        prepared_windows.append(
            {
                "name": window_name,
                "fit_x": fit_x,
                "valid_x": valid_x,
                "y_fit": y_fit,
                "validation_df": validation_df,
            }
        )

    baseline_fold_metrics: list[dict[str, float | str]] = []
    baseline_fold_scores: list[float] = []
    baseline_runtime = 0.0
    for window in prepared_windows:
        metrics, runtime = evaluate_feature_subset(
            fit_x=window["fit_x"],
            predict_x=window["valid_x"],
            y_fit=window["y_fit"],
            predict_df=window["validation_df"],
            feature_columns=baseline_columns,
            model_name=args.model,
            random_seed=args.random_seed,
            experiment_name=f"{window['name']}_baseline",
        )
        baseline_fold_metrics.append(metrics)
        baseline_fold_scores.append(metric_score(metrics, args.selection_metric))
        baseline_runtime += runtime

    baseline_metrics = aggregate_fold_metrics(
        fold_metrics=baseline_fold_metrics,
        fold_scores=baseline_fold_scores,
        experiment_name="validation_baseline_multi_window",
        stability_penalty=float(args.validation_stability_penalty),
    )
    current_columns = list(baseline_columns)
    current_score = float(baseline_metrics["validation_stability_adjusted_score"])
    current_fold_scores = list(baseline_fold_scores)
    selected_rows: list[dict[str, object]] = []
    selected_features: list[str] = []
    selected_families: list[object] = []
    step_metric_rows: list[dict[str, object]] = [
        {
            "step": 0,
            "added_feature": "",
            "candidate_id": "",
            "candidate_source": "",
            "selection_score": current_score,
            "delta_selection_score": 0.0,
            "window_positive_ratio": 0.0,
            "max_abs_corr_to_selected": 0.0,
            "family_penalty": 0.0,
            "correlation_penalty": 0.0,
            "runtime_seconds": baseline_runtime,
            **baseline_metrics,
        }
    ]

    remaining = list(candidate_columns)
    metadata_by_feature = candidate_metadata.set_index("mined_feature").to_dict("index")
    if len(candidate_columns) >= 2:
        candidate_corr = (
            train_df[candidate_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .corr()
            .abs()
        )
    else:
        candidate_corr = pd.DataFrame()

    family_counts: dict[str, int] = {}
    max_per_family = int(getattr(args, "max_selected_per_family", 0) or 0)
    max_corr_allowed = float(getattr(args, "max_candidate_corr", 1.0) or 1.0)
    min_positive_ratio = float(getattr(args, "min_window_positive_ratio", 0.0) or 0.0)

    for step in range(1, int(args.max_selected) + 1):
        best_feature = None
        best_metrics: dict[str, float | str] | None = None
        best_runtime = 0.0
        best_score = -1e9
        best_raw_score = -1e9
        best_fold_scores: list[float] = []
        best_positive_ratio = 0.0
        best_max_corr = 0.0
        best_family_penalty = 0.0
        best_corr_penalty = 0.0

        iterator = optional_progress(remaining, desc=f"Multi-window selection step {step}", leave=False)
        for candidate_column in iterator:
            meta = metadata_by_feature.get(candidate_column, {})
            candidate_family = meta.get("family", "")
            family_key = str(candidate_family)
            if max_per_family > 0 and family_counts.get(family_key, 0) >= max_per_family:
                continue

            max_corr = max_abs_corr_to_selected(candidate_corr, candidate_column, selected_features)
            if selected_features and max_corr > max_corr_allowed:
                continue

            trial_columns = current_columns + [candidate_column]
            fold_metrics: list[dict[str, float | str]] = []
            fold_scores: list[float] = []
            trial_runtime = 0.0
            for window in prepared_windows:
                metrics, runtime = evaluate_feature_subset(
                    fit_x=window["fit_x"],
                    predict_x=window["valid_x"],
                    y_fit=window["y_fit"],
                    predict_df=window["validation_df"],
                    feature_columns=trial_columns,
                    model_name=args.model,
                    random_seed=args.random_seed,
                    experiment_name=f"{window['name']}_add_{candidate_column}",
                )
                fold_metrics.append(metrics)
                fold_scores.append(metric_score(metrics, args.selection_metric))
                trial_runtime += runtime

            fold_deltas = [score - current for score, current in zip(fold_scores, current_fold_scores)]
            positive_ratio = float(np.mean([delta > float(args.min_validation_delta) for delta in fold_deltas]))
            if positive_ratio < min_positive_ratio:
                continue

            aggregated_metrics = aggregate_fold_metrics(
                fold_metrics=fold_metrics,
                fold_scores=fold_scores,
                experiment_name=f"validation_add_{candidate_column}_multi_window",
                stability_penalty=float(args.validation_stability_penalty),
            )
            raw_score = float(aggregated_metrics["validation_stability_adjusted_score"])
            repeated_family_penalty = family_overlap_penalty(
                candidate_family=candidate_family,
                selected_families=selected_families,
                penalty=float(args.family_repeat_penalty),
            )
            corr_penalty = float(args.candidate_corr_penalty) * max_corr
            trial_score = raw_score - repeated_family_penalty - corr_penalty

            if trial_score > best_score:
                best_feature = candidate_column
                best_metrics = aggregated_metrics
                best_runtime = trial_runtime
                best_score = trial_score
                best_raw_score = raw_score
                best_fold_scores = fold_scores
                best_positive_ratio = positive_ratio
                best_max_corr = max_corr
                best_family_penalty = repeated_family_penalty
                best_corr_penalty = corr_penalty

        delta_score = best_score - current_score
        if best_feature is None or best_metrics is None or delta_score <= float(args.min_validation_delta):
            break

        meta = metadata_by_feature.get(best_feature, {})
        selected_rows.append(
            {
                "selection_step": step,
                "mined_feature": best_feature,
                "candidate_source": meta.get("candidate_source"),
                "candidate_id": meta.get("candidate_id"),
                "formula": meta.get("formula"),
                "family": meta.get("family"),
                "hypothesis": meta.get("hypothesis"),
                "validation_selection_score_before": current_score,
                "validation_selection_score_after": best_score,
                "validation_raw_score_after": best_raw_score,
                "validation_delta_selection_score": delta_score,
                "validation_window_positive_ratio": best_positive_ratio,
                "max_abs_corr_to_selected": best_max_corr,
                "family_penalty": best_family_penalty,
                "correlation_penalty": best_corr_penalty,
                **{f"validation_{key}": value for key, value in best_metrics.items() if key != "experiment"},
            }
        )
        step_metric_rows.append(
            {
                "step": step,
                "added_feature": best_feature,
                "candidate_id": meta.get("candidate_id"),
                "candidate_source": meta.get("candidate_source"),
                "selection_score": best_score,
                "raw_selection_score": best_raw_score,
                "delta_selection_score": delta_score,
                "window_positive_ratio": best_positive_ratio,
                "max_abs_corr_to_selected": best_max_corr,
                "family_penalty": best_family_penalty,
                "correlation_penalty": best_corr_penalty,
                "runtime_seconds": best_runtime,
                **best_metrics,
            }
        )
        current_columns.append(best_feature)
        selected_features.append(best_feature)
        selected_family = meta.get("family", "")
        selected_families.append(selected_family)
        family_counts[str(selected_family)] = family_counts.get(str(selected_family), 0) + 1
        remaining.remove(best_feature)
        current_score = best_score
        current_fold_scores = list(best_fold_scores)

    return pd.DataFrame(selected_rows), pd.DataFrame(step_metric_rows)


def select_candidates_by_individual_validation(
    train_df: pd.DataFrame,
    validation_windows: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    baseline_columns: list[str],
    candidate_columns: list[str],
    candidate_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """先做单候选 validation 增量审计，再按多样性约束挑选候选。

    这个模式针对当前发现的问题：很多候选单独加入 OOS 有效，但累计 forward
    在 validation 上会选出相互干扰的组合。这里先找“单独稳定有效”的候选，
    再用 family 和相关性约束做组合，避免过早受模型交互项影响。
    """

    all_columns = baseline_columns + candidate_columns
    prepared_windows: list[dict[str, object]] = []
    for window_name, fit_df, validation_df in validation_windows:
        fit_x, valid_x, y_fit = prepare_xy_frames(fit_df, validation_df, all_columns)
        prepared_windows.append(
            {
                "name": window_name,
                "fit_x": fit_x,
                "valid_x": valid_x,
                "y_fit": y_fit,
                "validation_df": validation_df,
            }
        )

    baseline_fold_metrics: list[dict[str, float | str]] = []
    baseline_fold_scores: list[float] = []
    baseline_runtime = 0.0
    for window in prepared_windows:
        metrics, runtime = evaluate_feature_subset(
            fit_x=window["fit_x"],
            predict_x=window["valid_x"],
            y_fit=window["y_fit"],
            predict_df=window["validation_df"],
            feature_columns=baseline_columns,
            model_name=args.model,
            random_seed=args.random_seed,
            experiment_name=f"{window['name']}_baseline",
        )
        baseline_fold_metrics.append(metrics)
        baseline_fold_scores.append(metric_score(metrics, args.selection_metric))
        baseline_runtime += runtime

    baseline_metrics = aggregate_fold_metrics(
        fold_metrics=baseline_fold_metrics,
        fold_scores=baseline_fold_scores,
        experiment_name="validation_baseline_individual_mode",
        stability_penalty=float(args.validation_stability_penalty),
    )
    baseline_score = float(baseline_metrics["validation_stability_adjusted_score"])
    metadata_by_feature = candidate_metadata.set_index("mined_feature").to_dict("index")
    audit_rows: list[dict[str, object]] = []

    iterator = optional_progress(candidate_columns, desc="Individual validation candidate audit", leave=False)
    for candidate_column in iterator:
        fold_metrics: list[dict[str, float | str]] = []
        fold_scores: list[float] = []
        runtime_total = 0.0
        for window in prepared_windows:
            metrics, runtime = evaluate_feature_subset(
                fit_x=window["fit_x"],
                predict_x=window["valid_x"],
                y_fit=window["y_fit"],
                predict_df=window["validation_df"],
                feature_columns=baseline_columns + [candidate_column],
                model_name=args.model,
                random_seed=args.random_seed,
                experiment_name=f"{window['name']}_individual_{candidate_column}",
            )
            fold_metrics.append(metrics)
            fold_scores.append(metric_score(metrics, args.selection_metric))
            runtime_total += runtime

        aggregated = aggregate_fold_metrics(
            fold_metrics=fold_metrics,
            fold_scores=fold_scores,
            experiment_name=f"validation_individual_{candidate_column}",
            stability_penalty=float(args.validation_stability_penalty),
        )
        fold_deltas = [score - baseline for score, baseline in zip(fold_scores, baseline_fold_scores)]
        positive_ratio = float(np.mean([delta > float(args.min_validation_delta) for delta in fold_deltas]))
        meta = metadata_by_feature.get(candidate_column, {})
        audit_rows.append(
            {
                "mined_feature": candidate_column,
                "candidate_source": meta.get("candidate_source"),
                "candidate_id": meta.get("candidate_id"),
                "formula": meta.get("formula"),
                "family": meta.get("family"),
                "hypothesis": meta.get("hypothesis"),
                "individual_validation_score": float(aggregated["validation_stability_adjusted_score"]),
                "individual_validation_raw_score": float(aggregated["validation_score_mean"]),
                "delta_vs_validation_baseline": float(aggregated["validation_stability_adjusted_score"]) - baseline_score,
                "window_positive_ratio": positive_ratio,
                "runtime_seconds": runtime_total,
                **{f"validation_{key}": value for key, value in aggregated.items() if key != "experiment"},
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    if audit_df.empty:
        return pd.DataFrame(), pd.DataFrame(), audit_df

    audit_df = audit_df.sort_values("individual_validation_score", ascending=False).reset_index(drop=True)
    if len(candidate_columns) >= 2:
        candidate_corr = (
            train_df[candidate_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .corr()
            .abs()
        )
    else:
        candidate_corr = pd.DataFrame()

    selected_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = [
        {
            "step": 0,
            "added_feature": "",
            "candidate_id": "",
            "candidate_source": "",
            "selection_score": baseline_score,
            "delta_selection_score": 0.0,
            "window_positive_ratio": 0.0,
            "max_abs_corr_to_selected": 0.0,
            "family_penalty": 0.0,
            "correlation_penalty": 0.0,
            "runtime_seconds": baseline_runtime,
            **baseline_metrics,
        }
    ]
    selected_features: list[str] = []
    selected_families: list[object] = []
    family_counts: dict[str, int] = {}
    max_per_family = int(getattr(args, "max_selected_per_family", 0) or 0)
    max_corr_allowed = float(getattr(args, "max_candidate_corr", 1.0) or 1.0)
    min_positive_ratio = float(getattr(args, "min_window_positive_ratio", 0.0) or 0.0)

    while len(selected_rows) < int(args.max_selected):
        best_row: pd.Series | None = None
        best_adjusted_score = -1e9
        best_max_corr = 0.0
        best_family_penalty = 0.0
        best_corr_penalty = 0.0

        for _, row in audit_df.iterrows():
            candidate_column = str(row["mined_feature"])
            if candidate_column in selected_features:
                continue
            if float(row["window_positive_ratio"]) < min_positive_ratio:
                continue
            if float(row["delta_vs_validation_baseline"]) <= float(args.min_validation_delta):
                continue
            family_key = str(row.get("family", ""))
            if max_per_family > 0 and family_counts.get(family_key, 0) >= max_per_family:
                continue
            max_corr = max_abs_corr_to_selected(candidate_corr, candidate_column, selected_features)
            if selected_features and max_corr > max_corr_allowed:
                continue
            repeated_family_penalty = family_overlap_penalty(
                candidate_family=row.get("family", ""),
                selected_families=selected_families,
                penalty=float(args.family_repeat_penalty),
            )
            corr_penalty = float(args.candidate_corr_penalty) * max_corr
            adjusted_score = float(row["individual_validation_score"]) - repeated_family_penalty - corr_penalty
            if adjusted_score > best_adjusted_score:
                best_row = row
                best_adjusted_score = adjusted_score
                best_max_corr = max_corr
                best_family_penalty = repeated_family_penalty
                best_corr_penalty = corr_penalty

        if best_row is None:
            break

        step = len(selected_rows) + 1
        candidate_column = str(best_row["mined_feature"])
        delta_score = best_adjusted_score - baseline_score
        selected_rows.append(
            {
                "selection_step": step,
                "mined_feature": candidate_column,
                "candidate_source": best_row.get("candidate_source"),
                "candidate_id": best_row.get("candidate_id"),
                "formula": best_row.get("formula"),
                "family": best_row.get("family"),
                "hypothesis": best_row.get("hypothesis"),
                "validation_selection_score_before": baseline_score,
                "validation_selection_score_after": best_adjusted_score,
                "validation_raw_score_after": best_row.get("individual_validation_score"),
                "validation_delta_selection_score": delta_score,
                "validation_window_positive_ratio": best_row.get("window_positive_ratio"),
                "max_abs_corr_to_selected": best_max_corr,
                "family_penalty": best_family_penalty,
                "correlation_penalty": best_corr_penalty,
                **{
                    key: value
                    for key, value in best_row.to_dict().items()
                    if key.startswith("validation_")
                },
            }
        )
        step_rows.append(
            {
                "step": step,
                "added_feature": candidate_column,
                "candidate_id": best_row.get("candidate_id"),
                "candidate_source": best_row.get("candidate_source"),
                "selection_score": best_adjusted_score,
                "raw_selection_score": best_row.get("individual_validation_score"),
                "delta_selection_score": delta_score,
                "window_positive_ratio": best_row.get("window_positive_ratio"),
                "max_abs_corr_to_selected": best_max_corr,
                "family_penalty": best_family_penalty,
                "correlation_penalty": best_corr_penalty,
                "runtime_seconds": best_row.get("runtime_seconds"),
                **{
                    key.replace("validation_", "", 1): value
                    for key, value in best_row.to_dict().items()
                    if key.startswith("validation_")
                },
            }
        )
        selected_features.append(candidate_column)
        selected_family = best_row.get("family", "")
        selected_families.append(selected_family)
        family_counts[str(selected_family)] = family_counts.get(str(selected_family), 0) + 1

    return pd.DataFrame(selected_rows), pd.DataFrame(step_rows), audit_df


def select_candidates_by_rank_rule(
    train_df: pd.DataFrame,
    baseline_columns: list[str],
    candidate_columns: list[str],
    candidate_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按训练期候选质量分数选择因子。

    这个模式是为当前实验暴露出来的问题准备的：内部 validation 和 2026 OOS 的
    regime 可能不一致，导致 validation 排名很高的候选在 OOS 上失效。

    注意：
    - 这里不读取 OOS 指标；
    - 只使用候选在训练期搜索阶段留下的元数据，例如 `train_score`；
    - 覆盖率过滤仍然沿用 `--min-candidate-coverage`，避免低覆盖基本面字段被选中。

    它不是最终最优选择器，但可以作为一个更朴素、可复现的 baseline selector。
    """

    if not candidate_columns:
        return pd.DataFrame(), pd.DataFrame()

    score_column = str(getattr(args, "rank_rule_score_column", "train_score") or "train_score")
    metadata = candidate_metadata[candidate_metadata["mined_feature"].isin(candidate_columns)].copy()
    if metadata.empty:
        return pd.DataFrame(), pd.DataFrame()
    if score_column not in metadata.columns:
        raise ValueError(f"rank_rule score column not found in candidate metadata: {score_column}")

    metadata["_rank_rule_score"] = pd.to_numeric(metadata[score_column], errors="coerce").fillna(-1e9)
    if "train_coverage_ratio" in metadata.columns:
        metadata["_coverage"] = pd.to_numeric(metadata["train_coverage_ratio"], errors="coerce").fillna(0.0)
    else:
        metadata["_coverage"] = 1.0

    min_coverage = float(getattr(args, "min_candidate_coverage", 0.0) or 0.0)
    metadata = metadata[metadata["_coverage"] >= min_coverage].copy()
    metadata = metadata.sort_values("_rank_rule_score", ascending=False).reset_index(drop=True)

    if len(candidate_columns) >= 2:
        candidate_corr = (
            train_df[candidate_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .corr()
            .abs()
        )
    else:
        candidate_corr = pd.DataFrame()

    selected_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = [
        {
            "step": 0,
            "added_feature": "",
            "candidate_id": "",
            "candidate_source": "",
            "selection_score": float("nan"),
            "delta_selection_score": 0.0,
            "rank_rule_score": float("nan"),
            "rank_rule_score_column": score_column,
            "train_coverage_ratio": float("nan"),
            "max_abs_corr_to_selected": 0.0,
            "family_penalty": 0.0,
            "correlation_penalty": 0.0,
            "runtime_seconds": 0.0,
        }
    ]
    selected_features: list[str] = []
    selected_families: list[object] = []
    family_counts: dict[str, int] = {}
    max_per_family = int(getattr(args, "max_selected_per_family", 0) or 0)
    max_corr_allowed = float(getattr(args, "max_candidate_corr", 1.0) or 1.0)

    for _, row in metadata.iterrows():
        if len(selected_rows) >= int(args.max_selected):
            break

        candidate_column = str(row["mined_feature"])
        family_key = str(row.get("family", ""))
        if max_per_family > 0 and family_counts.get(family_key, 0) >= max_per_family:
            continue

        max_corr = max_abs_corr_to_selected(candidate_corr, candidate_column, selected_features)
        if selected_features and max_corr > max_corr_allowed:
            continue

        repeated_family_penalty = family_overlap_penalty(
            candidate_family=row.get("family", ""),
            selected_families=selected_families,
            penalty=float(args.family_repeat_penalty),
        )
        corr_penalty = float(args.candidate_corr_penalty) * max_corr
        raw_score = float(row["_rank_rule_score"])
        adjusted_score = raw_score - repeated_family_penalty - corr_penalty

        step = len(selected_rows) + 1
        selected_rows.append(
            {
                "selection_step": step,
                "mined_feature": candidate_column,
                "candidate_source": row.get("candidate_source"),
                "candidate_id": row.get("candidate_id"),
                "formula": row.get("formula"),
                "family": row.get("family"),
                "hypothesis": row.get("hypothesis"),
                "rank_rule_score_column": score_column,
                "rank_rule_score": raw_score,
                "rank_rule_adjusted_score": adjusted_score,
                "train_coverage_ratio": float(row.get("_coverage", float("nan"))),
                "max_abs_corr_to_selected": max_corr,
                "family_penalty": repeated_family_penalty,
                "correlation_penalty": corr_penalty,
            }
        )
        step_rows.append(
            {
                "step": step,
                "added_feature": candidate_column,
                "candidate_id": row.get("candidate_id"),
                "candidate_source": row.get("candidate_source"),
                "selection_score": adjusted_score,
                "rank_rule_score": raw_score,
                "rank_rule_score_column": score_column,
                "train_coverage_ratio": float(row.get("_coverage", float("nan"))),
                "max_abs_corr_to_selected": max_corr,
                "family_penalty": repeated_family_penalty,
                "correlation_penalty": corr_penalty,
                "runtime_seconds": 0.0,
            }
        )
        selected_features.append(candidate_column)
        selected_family = row.get("family", "")
        selected_families.append(selected_family)
        family_counts[str(selected_family)] = family_counts.get(str(selected_family), 0) + 1

    return pd.DataFrame(selected_rows), pd.DataFrame(step_rows)


def audit_oos_steps(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_columns: list[str],
    selected_features: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    all_columns = baseline_columns + selected_features
    train_x, test_x, y_train = prepare_xy_frames(train_df, test_df, all_columns)

    rows: list[dict[str, object]] = []
    previous_score: float | None = None
    for step in range(0, len(selected_features) + 1):
        feature_columns = baseline_columns + selected_features[:step]
        metrics, runtime = evaluate_feature_subset(
            fit_x=train_x,
            predict_x=test_x,
            y_fit=y_train,
            predict_df=test_df,
            feature_columns=feature_columns,
            model_name=args.model,
            random_seed=args.random_seed,
            experiment_name="oos_baseline" if step == 0 else f"oos_step_{step}",
        )
        score = metric_score(metrics, args.selection_metric)
        delta = 0.0 if previous_score is None else score - previous_score
        rows.append(
            {
                "step": step,
                "added_feature": "" if step == 0 else selected_features[step - 1],
                "feature_count": len(feature_columns),
                "selection_metric": args.selection_metric,
                "oos_selection_score": score,
                "delta_oos_selection_score": delta,
                "oos_step_improves_model": bool(step > 0 and delta > float(args.min_oos_delta)),
                "runtime_seconds": runtime,
                **metrics,
            }
        )
        previous_score = score
    return pd.DataFrame(rows)


def audit_individual_oos_candidates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_columns: list[str],
    candidate_columns: list[str],
    candidate_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """逐个候选做 OOS 单独增量审计。

    重要：这个函数只在候选选择结束后生成诊断报告。
    它不能用于 forward selection，否则会把 OOS 信息泄露进选择过程。
    """

    top_k = int(getattr(args, "individual_oos_audit_top_k", 0) or 0)
    if top_k <= 0 or not candidate_columns:
        return pd.DataFrame()

    audited_candidates = candidate_columns[:top_k]
    all_columns = baseline_columns + audited_candidates
    train_x, test_x, y_train = prepare_xy_frames(train_df, test_df, all_columns)

    baseline_metrics, baseline_runtime = evaluate_feature_subset(
        fit_x=train_x,
        predict_x=test_x,
        y_fit=y_train,
        predict_df=test_df,
        feature_columns=baseline_columns,
        model_name=args.model,
        random_seed=args.random_seed,
        experiment_name="individual_oos_baseline",
    )
    baseline_score = metric_score(baseline_metrics, args.selection_metric)
    metadata_by_feature = candidate_metadata.set_index("mined_feature").to_dict("index")

    rows: list[dict[str, object]] = [
        {
            "rank": 0,
            "mined_feature": "",
            "candidate_id": "",
            "candidate_source": "",
            "formula": "",
            "family": "",
            "oos_selection_score": baseline_score,
            "delta_vs_baseline": 0.0,
            "improves_baseline": False,
            "runtime_seconds": baseline_runtime,
            **baseline_metrics,
        }
    ]

    iterator = optional_progress(audited_candidates, desc="Individual OOS candidate audit", leave=False)
    for rank, candidate_column in enumerate(iterator, start=1):
        metrics, runtime = evaluate_feature_subset(
            fit_x=train_x,
            predict_x=test_x,
            y_fit=y_train,
            predict_df=test_df,
            feature_columns=baseline_columns + [candidate_column],
            model_name=args.model,
            random_seed=args.random_seed,
            experiment_name=f"individual_oos_{candidate_column}",
        )
        score = metric_score(metrics, args.selection_metric)
        meta = metadata_by_feature.get(candidate_column, {})
        rows.append(
            {
                "rank": rank,
                "mined_feature": candidate_column,
                "candidate_id": meta.get("candidate_id", ""),
                "candidate_source": meta.get("candidate_source", ""),
                "formula": meta.get("formula", ""),
                "family": meta.get("family", ""),
                "oos_selection_score": score,
                "delta_vs_baseline": score - baseline_score,
                "improves_baseline": bool(score > baseline_score + float(args.min_oos_delta)),
                "runtime_seconds": runtime,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    config: dict[str, Any],
    seed_df: pd.DataFrame,
    materialized_candidates: pd.DataFrame,
    residual_score_df: pd.DataFrame,
    individual_validation_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    validation_steps_df: pd.DataFrame,
    oos_steps_df: pd.DataFrame,
    individual_oos_df: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    selected_view_cols = [
        "selection_step",
        "candidate_source",
        "candidate_id",
        "formula",
        "family",
        "rank_rule_score_column",
        "rank_rule_score",
        "rank_rule_adjusted_score",
        "train_coverage_ratio",
        "validation_delta_selection_score",
        "validation_window_positive_ratio",
        "max_abs_corr_to_selected",
        "family_penalty",
        "correlation_penalty",
        "validation_pearson_ic_mean",
        "validation_spearman_ic_mean",
        "validation_long_short_return",
    ]
    selected_view = selected_df[[column for column in selected_view_cols if column in selected_df.columns]].copy()

    oos_view_cols = [
        "step",
        "added_feature",
        "oos_selection_score",
        "delta_oos_selection_score",
        "oos_step_improves_model",
        "pearson_ic_mean",
        "spearman_ic_mean",
        "long_short_return",
    ]
    oos_view = oos_steps_df[[column for column in oos_view_cols if column in oos_steps_df.columns]].copy()
    residual_view_cols = [
        "mined_feature",
        "residual_composite_score",
        "residual_pearson_ic_mean",
        "residual_spearman_ic_mean",
        "residual_long_short_return",
        "residual_ic_days",
    ]
    residual_view = residual_score_df[[column for column in residual_view_cols if column in residual_score_df.columns]].head(20).copy()
    individual_validation_view_cols = [
        "candidate_source",
        "candidate_id",
        "formula",
        "family",
        "individual_validation_score",
        "delta_vs_validation_baseline",
        "window_positive_ratio",
        "validation_pearson_ic_mean",
        "validation_spearman_ic_mean",
        "validation_long_short_return",
    ]
    individual_validation_view = individual_validation_df[
        [column for column in individual_validation_view_cols if column in individual_validation_df.columns]
    ].head(25).copy()
    individual_view_cols = [
        "rank",
        "candidate_source",
        "candidate_id",
        "formula",
        "family",
        "oos_selection_score",
        "delta_vs_baseline",
        "improves_baseline",
        "pearson_ic_mean",
        "spearman_ic_mean",
        "long_short_return",
    ]
    individual_view = individual_oos_df[
        [column for column in individual_view_cols if column in individual_oos_df.columns]
    ].head(25).copy()

    if not materialized_candidates.empty and "candidate_source" in materialized_candidates.columns:
        source_counts = materialized_candidates["candidate_source"].value_counts().rename_axis("candidate_source").reset_index(name="count")
    else:
        source_counts = pd.DataFrame()

    oos_improving_count = int(oos_steps_df.get("oos_step_improves_model", pd.Series(dtype=bool)).sum())
    strict_step_target_achieved = bool(
        len(selected_df) >= int(config["max_selected"]) and oos_improving_count >= int(config["max_selected"])
    )
    if not oos_steps_df.empty and "oos_selection_score" in oos_steps_df.columns:
        baseline_oos_score = float(oos_steps_df["oos_selection_score"].iloc[0])
        final_oos_score = float(oos_steps_df["oos_selection_score"].iloc[-1])
    else:
        baseline_oos_score = float("nan")
        final_oos_score = float("nan")
    final_oos_delta = final_oos_score - baseline_oos_score
    final_set_improves_baseline = bool(len(selected_df) >= int(config["max_selected"]) and final_oos_delta > 0)
    selected_candidate_ids = set(selected_df.get("candidate_id", pd.Series(dtype=object)).dropna().astype(str))
    if selected_candidate_ids and not individual_oos_df.empty and "candidate_id" in individual_oos_df.columns:
        candidate_id_series = individual_oos_df["candidate_id"]
        selected_individual_oos_mask = candidate_id_series.notna() & candidate_id_series.astype(str).isin(selected_candidate_ids)
        selected_individual_oos = individual_oos_df[selected_individual_oos_mask].copy()
        selected_individual_oos_positive = int(
            (pd.to_numeric(selected_individual_oos.get("delta_vs_baseline", pd.Series(dtype=float)), errors="coerce") > 0).sum()
        )
    else:
        selected_individual_oos = pd.DataFrame()
        selected_individual_oos_positive = 0
    selected_individual_view = selected_individual_oos[
        [column for column in individual_view_cols if column in selected_individual_oos.columns]
    ].copy()

    report = f"""# RL + Generative Incremental Factor Selection Report

## 1. Goal

Target: find at least `{config["max_selected"]}` RL/generative formula factors that improve the model.

This report separates two claims:

- **Validation improvement**: selected using only the in-sample training period split into fit/validation dates.
- **OOS improvement audit**: evaluated on 2026 OOS after selection is frozen.

Target achieved under strict OOS step audit: `{strict_step_target_achieved}`
OOS improving steps: `{oos_improving_count}` / `{config["max_selected"]}`

Final selected-factor set improves OOS baseline: `{final_set_improves_baseline}`
Baseline OOS score: `{baseline_oos_score:.6f}`
Final OOS score: `{final_oos_score:.6f}`
Final OOS delta: `{final_oos_delta:.6f}`

Selected factors with positive individual OOS audit: `{selected_individual_oos_positive}` / `{len(selected_df)}`

## 2. Config

```json
{json.dumps(config, ensure_ascii=False, indent=2)}
```

Runtime seconds: `{runtime_seconds:.2f}`

## 3. Seed Library

- Seed count: `{len(seed_df)}`

{dataframe_to_markdown(seed_df.head(25))}

## 4. Candidate Pool

- Materialized candidates: `{int((materialized_candidates["materialize_status"] == "ok").sum()) if not materialized_candidates.empty else 0}`
- Candidate sources:

{dataframe_to_markdown(source_counts)}

## 5. Validation Residual Prefilter

{dataframe_to_markdown(residual_view)}

## 6. Individual Validation Candidate Audit

{dataframe_to_markdown(individual_validation_view)}

## 7. Selected Factors On Internal Validation

{dataframe_to_markdown(selected_view)}

## 8. Validation Steps

{dataframe_to_markdown(validation_steps_df)}

## 9. OOS Step Audit

{dataframe_to_markdown(oos_view)}

## 10. Individual OOS Candidate Audit

{dataframe_to_markdown(individual_view)}

## 11. Selected Factors Individual OOS Audit

{dataframe_to_markdown(selected_individual_view)}

## 12. Interpretation

- If a factor improves validation but fails OOS, it is not a confirmed resume claim.
- If at least five steps improve OOS sequentially, then this run satisfies the stated target.
- If fewer than five improve OOS, the framework works but the empirical target is not yet met.
- Individual OOS audit is diagnostic only. It must not be used as a training-time selector.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()

    # 这个一体化脚本的命令行参数会同时喂给 RL 子模块和生成式子模块。
    # 原子模块沿用了各自独立脚本里的命名：
    # - RL 脚本使用 args.episodes；
    # - 生成式脚本使用 args.num_samples；
    # - 两者报告展示使用 args.final_top_k。
    # 这里集中做一次兼容映射，避免修改原子模块，保持它们仍可独立运行。
    args.episodes = args.rl_episodes
    args.num_samples = args.generative_samples
    args.final_top_k = args.candidate_top_k

    total_start = time.perf_counter()
    rng = random.Random(args.random_seed)

    output_root = resolve_path(args.output_dir)
    run_name = args.run_name or (
        f"rlgen_{args.target_horizon}d_rl{args.rl_episodes}_gm{args.generative_samples}_"
        f"k{args.candidate_top_k}_s{args.random_seed}"
    )
    output_dir = output_root / sanitize_name(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    seed_df, seed_nodes = build_seed_nodes(train_df=train_df, model_dir=resolve_path(args.model_dir), args=args)
    if not seed_nodes:
        raise ValueError("No seed nodes were available.")

    print("[Info] Running RL-style candidate generation", flush=True)
    rl_candidate_df, rl_train_df, rl_trace_df, rl_action_value_df, rl_nodes_by_id = run_contextual_bandit_search(
        train_df=train_df,
        seed_nodes=seed_nodes,
        args=args,
        rng=rng,
    )

    print("[Info] Running generative candidate generation", flush=True)
    gen_candidate_df, gen_train_df, gen_nodes_by_id, gen_prior = generate_unique_candidates(
        train_df=train_df,
        seed_nodes=seed_nodes,
        args=args,
        rng=rng,
    )

    candidate_df, nodes_by_global_id = combine_candidate_records(
        rl_train_df=rl_train_df,
        rl_nodes_by_id=rl_nodes_by_id,
        gen_train_df=gen_train_df,
        gen_nodes_by_id=gen_nodes_by_id,
        top_k=args.candidate_top_k,
    )
    if candidate_df.empty:
        raise ValueError("No train-valid RL/generative candidates were generated.")

    candidate_df, nodes_by_global_id, candidate_filter_summary = apply_candidate_filters(
        candidate_df=candidate_df,
        nodes_by_global_id=nodes_by_global_id,
        args=args,
    )
    if candidate_df.empty:
        raise ValueError("Candidate filters removed all train-valid candidates.")

    train_enhanced, test_enhanced, materialized_df = materialize_candidate_columns(
        train_df=train_df,
        test_df=test_df,
        candidate_df=candidate_df,
        nodes_by_global_id=nodes_by_global_id,
    )
    usable_candidates = materialized_df[materialized_df["materialize_status"] == "ok"].copy()
    if usable_candidates.empty:
        raise ValueError("No candidates could be materialized as model features.")

    fit_df, validation_df = split_train_validation_by_time(
        train_enhanced,
        args.validation_fraction,
        purge_days=int(args.target_horizon),
    )
    validation_windows = build_train_validation_windows(
        train_df=train_enhanced,
        validation_fraction=args.validation_fraction,
        window_count=args.validation_window_count,
        purge_days=int(args.target_horizon),
    )
    baseline_columns = select_baseline_feature_columns(train_df, args.baseline_feature_mode)
    candidate_columns = usable_candidates["mined_feature"].astype(str).tolist()
    residual_score_df = pd.DataFrame()

    if int(args.residual_prefilter_top_k) > 0:
        residual_score_df = score_candidates_against_validation_residual(
            fit_df=fit_df,
            validation_df=validation_df,
            baseline_columns=baseline_columns,
            candidate_columns=candidate_columns,
            args=args,
        )
        residual_score_df = residual_score_df[
            pd.to_numeric(residual_score_df["residual_composite_score"], errors="coerce")
            >= float(args.residual_prefilter_min_score)
        ].copy()
        candidate_columns = (
            residual_score_df["mined_feature"]
            .astype(str)
            .head(int(args.residual_prefilter_top_k))
            .tolist()
        )
        usable_candidates = usable_candidates[usable_candidates["mined_feature"].isin(candidate_columns)].copy()
        # 把 residual prefilter 分数回填到候选元数据。
        # 这样 `selection-mode=rank_rule` 可以直接使用 `residual_composite_score`
        # 作为训练期可见的排序字段，而不需要读取 OOS 结果。
        residual_score_columns = [
            column
            for column in [
                "mined_feature",
                "residual_pearson_ic_mean",
                "residual_spearman_ic_mean",
                "residual_long_short_return",
                "residual_composite_score",
                "residual_ic_days",
            ]
            if column in residual_score_df.columns
        ]
        if len(residual_score_columns) > 1:
            reusable_residual_scores = residual_score_df[residual_score_columns].drop_duplicates("mined_feature")
            columns_to_drop = [
                column
                for column in residual_score_columns
                if column != "mined_feature" and column in usable_candidates.columns
            ]
            if columns_to_drop:
                usable_candidates = usable_candidates.drop(columns=columns_to_drop)
            usable_candidates = usable_candidates.merge(reusable_residual_scores, on="mined_feature", how="left")
        if not candidate_columns:
            raise ValueError("Residual prefilter removed all candidates. Lower the threshold or increase candidate pool.")

    individual_validation_df = pd.DataFrame()
    if args.selection_mode == "individual_validation":
        selected_df, validation_steps_df, individual_validation_df = select_candidates_by_individual_validation(
            train_df=train_enhanced,
            validation_windows=validation_windows,
            baseline_columns=baseline_columns,
            candidate_columns=candidate_columns,
            candidate_metadata=usable_candidates,
            args=args,
        )
    elif args.selection_mode == "rank_rule":
        selected_df, validation_steps_df = select_candidates_by_rank_rule(
            train_df=train_enhanced,
            baseline_columns=baseline_columns,
            candidate_columns=candidate_columns,
            candidate_metadata=usable_candidates,
            args=args,
        )
    elif int(args.validation_window_count) > 1:
        selected_df, validation_steps_df = forward_select_candidates_multi_window(
            train_df=train_enhanced,
            validation_windows=validation_windows,
            baseline_columns=baseline_columns,
            candidate_columns=candidate_columns,
            candidate_metadata=usable_candidates,
            args=args,
        )
    else:
        selected_df, validation_steps_df = forward_select_candidates(
            fit_df=fit_df,
            validation_df=validation_df,
            baseline_columns=baseline_columns,
            candidate_columns=candidate_columns,
            candidate_metadata=usable_candidates,
            args=args,
        )
    selected_features = selected_df["mined_feature"].astype(str).tolist() if not selected_df.empty else []

    oos_steps_df = audit_oos_steps(
        train_df=train_enhanced,
        test_df=test_enhanced,
        baseline_columns=baseline_columns,
        selected_features=selected_features,
        args=args,
    )
    individual_oos_df = audit_individual_oos_candidates(
        train_df=train_enhanced,
        test_df=test_enhanced,
        baseline_columns=baseline_columns,
        candidate_columns=candidate_columns,
        candidate_metadata=usable_candidates,
        args=args,
    )

    config = {
        "target_column": target_column,
        "dataset_summary": dataset_summary,
        "model": args.model,
        "selection_metric": args.selection_metric,
        "selection_mode": args.selection_mode,
        "baseline_feature_mode": args.baseline_feature_mode,
        "baseline_feature_count": len(baseline_columns),
        "max_selected": args.max_selected,
        "rl_episodes": args.rl_episodes,
        "generative_samples": args.generative_samples,
        "candidate_top_k": args.candidate_top_k,
        "candidate_filter_summary": candidate_filter_summary,
        "residual_prefilter_top_k": args.residual_prefilter_top_k,
        "residual_prefilter_min_score": args.residual_prefilter_min_score,
        "individual_oos_audit_top_k": args.individual_oos_audit_top_k,
        "validation_fraction": args.validation_fraction,
        "validation_window_count": args.validation_window_count,
        "validation_stability_penalty": args.validation_stability_penalty,
        "min_window_positive_ratio": args.min_window_positive_ratio,
        "max_candidate_corr": args.max_candidate_corr,
        "candidate_corr_penalty": args.candidate_corr_penalty,
        "family_repeat_penalty": args.family_repeat_penalty,
        "max_selected_per_family": args.max_selected_per_family,
        "random_seed": args.random_seed,
    }
    runtime_seconds = time.perf_counter() - total_start

    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    seed_df.to_csv(output_dir / "seed_library.csv", index=False)
    rl_candidate_df.to_csv(output_dir / "rl_candidate_formulas.csv", index=False)
    rl_train_df.to_csv(output_dir / "rl_train_metrics.csv", index=False)
    rl_trace_df.to_csv(output_dir / "rl_policy_trace.csv", index=False)
    rl_action_value_df.to_csv(output_dir / "rl_action_value_table.csv", index=False)
    gen_candidate_df.to_csv(output_dir / "generative_candidate_formulas.csv", index=False)
    gen_train_df.to_csv(output_dir / "generative_train_metrics.csv", index=False)
    (output_dir / "generative_prior.json").write_text(json.dumps(gen_prior, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_df.to_csv(output_dir / "combined_train_candidates.csv", index=False)
    materialized_df.to_csv(output_dir / "materialized_candidates.csv", index=False)
    residual_score_df.to_csv(output_dir / "validation_residual_candidate_scores.csv", index=False)
    individual_validation_df.to_csv(output_dir / "individual_validation_candidate_audit.csv", index=False)
    selected_df.to_csv(output_dir / "selected_incremental_factors.csv", index=False)
    validation_steps_df.to_csv(output_dir / "validation_forward_steps.csv", index=False)
    oos_steps_df.to_csv(output_dir / "oos_step_audit.csv", index=False)
    individual_oos_df.to_csv(output_dir / "individual_oos_candidate_audit.csv", index=False)
    if args.save_candidate_matrices:
        candidate_matrix_columns = ["date", "instrument_id", "y", *candidate_columns]
        train_enhanced[candidate_matrix_columns].to_csv(output_dir / "train_candidate_feature_matrix.csv", index=False)
        test_enhanced[candidate_matrix_columns].to_csv(output_dir / "test_candidate_feature_matrix.csv", index=False)

    write_report(
        output_path=output_dir / "report.md",
        config=config,
        seed_df=seed_df,
        materialized_candidates=materialized_df,
        residual_score_df=residual_score_df,
        individual_validation_df=individual_validation_df,
        selected_df=selected_df,
        validation_steps_df=validation_steps_df,
        oos_steps_df=oos_steps_df,
        individual_oos_df=individual_oos_df,
        runtime_seconds=runtime_seconds,
    )

    print(f"[Done] Report written to: {output_dir / 'report.md'}", flush=True)
    print(selected_df.to_string(index=False), flush=True)
    print(oos_steps_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
