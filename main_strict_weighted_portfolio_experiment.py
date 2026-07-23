"""Run full weighted-portfolio experiments on strict MyQuant predictions.

本脚本不重新训练模型，只使用上一阶段严格实验已经保存的预测文件：

    outputs/strict_mined_factor_experiment_oos202506/*/test_predictions_with_actual.csv

实验目标：

1. 比较旧强 baseline、Warm-GP、PPO、Warm-GP+PPO 四组预测；
2. 比较等权和三种不等权组合构建；
3. 判断不等权是否真正放大自挖因子的增量，而不是单纯改善所有模型。

输出是一个全量组合层实验，不是轻量 smoke test。
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm 只影响显示。
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.mined_factor_model_ablation import dataframe_to_markdown  # noqa: E402
from main_rolling_oos_backtest import (  # noqa: E402
    OOSWindow,
    build_run_dir_name,
    build_windows,
    summarize_window_predictions,
)
from src.long_short_backtest import LongShortBacktestConfig, run_long_short_backtest  # noqa: E402
from src.portfolio import load_market_snapshot_frame, load_prediction_frame, merge_predictions_with_market  # noqa: E402
from src.project_paths import resolve_project_path  # noqa: E402
from src.runtime_config import DEFAULT_PRIMARY_DATA_PATH  # noqa: E402


DEFAULT_STRICT_INPUT_DIR = "outputs/strict_mined_factor_experiment_oos202506"
DEFAULT_OUTPUT_DIR = "outputs/strict_weighted_portfolio_experiment_oos202506"
DEFAULT_OOS_START_DATE = "2025-06-01"


@dataclass(frozen=True)
class PredictionSpec:
    """一组已经训练好的预测文件。"""

    experiment: str
    feature_group: str
    prediction_path: Path


@dataclass(frozen=True)
class StrategySpec:
    """持有期和调仓频率配置。"""

    strategy_name: str
    hold_days: int
    step_days: int


def progress_iter(iterable, *, total: int | None = None, desc: str = "", position: int = 0, leave: bool = True):
    """统一进度条入口。"""

    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, position=position, leave=leave)


def read_path(path_like: str | Path) -> Path:
    """把相对路径解析为项目内路径。"""

    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict weighted portfolio experiments.")
    parser.add_argument("--strict-input-dir", default=DEFAULT_STRICT_INPUT_DIR, help="严格模型实验输出目录。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="本次组合权重实验输出目录。")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="真实日频数据 CSV。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 开始日期。")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["strict_baseline", "strict_warm_gp", "strict_ppo", "strict_warm_gp_ppo"],
        help="要纳入组合测试的预测实验组。",
    )
    parser.add_argument(
        "--weight-modes",
        nargs="+",
        default=["equal_weight", "rank_weight", "score_weight", "score_vol_weight"],
        help="组合权重方式。",
    )
    parser.add_argument("--top-k-list", nargs="+", type=int, default=[10, 20], help="Top-K 网格。")
    parser.add_argument("--cost-bps-list", nargs="+", type=float, default=[5.0, 10.0, 20.0], help="交易成本网格。")
    parser.add_argument(
        "--neutral-modes",
        nargs="+",
        default=["unconstrained", "sector_neutral"],
        choices=["unconstrained", "sector_neutral"],
        help="组合中性约束网格。",
    )
    parser.add_argument("--signal-delay-days", type=int, default=1, help="信号延迟天数。")
    parser.add_argument(
        "--holding-clock",
        choices=["signal_horizon", "execution_horizon"],
        default="signal_horizon",
        help="默认从信号日计算 forward-return horizon。",
    )
    parser.add_argument("--borrow-cost-bps", type=float, default=0.0, help="做空借券成本占位。")
    parser.add_argument(
        "--window-policy",
        choices=["rolling", "recent"],
        default="rolling",
        help="OOS窗口策略：rolling=完整滚动窗口；recent=full/recent_12m/recent_6m/recent_3m四个锚定窗口。",
    )
    parser.add_argument(
        "--max-abs-weight",
        type=float,
        default=None,
        help="单票绝对权重上限；不传时自动用 min(15%%, 2/top_k)。",
    )
    return parser.parse_args()


def build_recent_oos_windows(
    *,
    min_start: pd.Timestamp,
    max_date: pd.Timestamp,
) -> list[OOSWindow]:
    """构造给报告使用的四个锚定 OOS 窗口。

    rolling 窗口适合严谨研究，但会显著扩大组合回测次数。
    给老师或面试官看的第一版报告，更需要清楚回答：
    full、最近 12 个月、最近 6 个月、最近 3 个月是否方向一致。
    """

    max_date = pd.Timestamp(max_date)
    min_start = pd.Timestamp(min_start)
    window_specs = [
        ("full", "full", min_start, None),
        ("recent_12m", "12m", max(min_start, max_date - pd.DateOffset(months=12)), 12),
        ("recent_6m", "6m", max(min_start, max_date - pd.DateOffset(months=6)), 6),
        ("recent_3m", "3m", max(min_start, max_date - pd.DateOffset(months=3)), 3),
    ]
    windows: list[OOSWindow] = []
    for window_id, window_mode, start_date, calendar_months in window_specs:
        windows.append(
            OOSWindow(
                window_id=window_id,
                window_mode=window_mode,
                start_date=pd.Timestamp(start_date),
                end_date=max_date,
                calendar_months=calendar_months,
            )
        )
    return windows


def strategy_specs() -> list[StrategySpec]:
    """固定四组持有/调仓规则。"""

    return [
        StrategySpec("hold10_step10", 10, 10),
        StrategySpec("hold10_step5", 10, 5),
        StrategySpec("hold20_step20", 20, 20),
        StrategySpec("hold20_step10", 20, 10),
    ]


def load_prediction_specs(strict_input_dir: Path, experiment_names: list[str]) -> list[PredictionSpec]:
    """读取 A/B/C/D 的预测文件位置。"""

    specs: list[PredictionSpec] = []
    for experiment in experiment_names:
        prediction_path = strict_input_dir / experiment / "test_predictions_with_actual.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction file for {experiment}: {prediction_path}")
        if experiment == "strict_baseline":
            feature_group = "baseline"
        elif experiment == "strict_warm_gp":
            feature_group = "warm_gp"
        elif experiment == "strict_ppo":
            feature_group = "ppo"
        elif experiment == "strict_warm_gp_ppo":
            feature_group = "warm_gp_ppo"
        else:
            feature_group = experiment.replace("strict_", "")
        specs.append(PredictionSpec(experiment=experiment, feature_group=feature_group, prediction_path=prediction_path))
    return specs


def auto_max_abs_weight(top_k: int, explicit: float | None) -> float:
    """确定单票权重上限。

    `min(15%, 2/top_k)` 的含义：

    - top10 时上限 15%，比等权 10% 略宽；
    - top20 时上限 10%，比等权 5% 略宽；
    - 允许不等权表达强弱，但不允许单票过度支配组合。
    """

    if explicit is not None:
        return float(explicit)
    return float(min(0.15, 2.0 / float(top_k)))


def summarize_backtest_row(
    *,
    spec: PredictionSpec,
    strategy: StrategySpec,
    window: OOSWindow,
    weight_mode: str,
    top_k: int,
    cost_bps: float,
    neutral_mode: str,
    max_abs_weight: float,
    prediction_df: pd.DataFrame,
    result_metrics: dict[str, Any],
) -> dict[str, Any]:
    """把一次组合回测结果压平成一行。"""

    row: dict[str, Any] = {
        "experiment": spec.experiment,
        "feature_group": spec.feature_group,
        "weight_mode": weight_mode,
        "strategy_name": strategy.strategy_name,
        "window_id": window.window_id,
        "window_mode": window.window_mode,
        "window_start": str(window.start_date.date()),
        "window_end": str(window.end_date.date()),
        "calendar_months": window.calendar_months,
        "top_k": int(top_k),
        "cost_bps": float(cost_bps),
        "neutral_mode": neutral_mode,
        "max_abs_weight": float(max_abs_weight),
        "status": "ok",
        **summarize_window_predictions(prediction_df, window),
    }
    for field in [
        "hold_days",
        "holding_clock",
        "effective_holding_days",
        "step_days",
        "daily_count",
        "rebalance_count",
        "portfolio_total_return",
        "portfolio_annualized_return",
        "portfolio_annualized_vol",
        "portfolio_sharpe",
        "portfolio_max_drawdown",
        "portfolio_calmar",
        "hit_ratio",
        "average_gross_turnover",
        "average_net_turnover",
        "average_turnover_cost_bps",
        "total_turnover_cost",
        "benchmark_total_return",
        "excess_total_return_vs_benchmark",
        "average_long_weight",
        "average_short_weight_abs",
        "average_gross_exposure",
        "average_net_exposure",
        "average_max_abs_sector_net_weight",
        "average_total_abs_sector_net_weight",
        "is_short_sample_warning",
        "max_active_sleeves",
    ]:
        row[field] = result_metrics.get(field)
    row["error"] = ""
    return row


def run_weighted_backtests(
    *,
    prediction_specs: list[PredictionSpec],
    data_path: Path,
    output_dir: Path,
    oos_start_date: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """运行全量组合权重网格。"""

    market_snapshot_df = load_market_snapshot_frame(data_path)
    portfolio_root = output_dir / "portfolio_runs"
    rows: list[dict[str, Any]] = []

    for spec in progress_iter(prediction_specs, total=len(prediction_specs), desc="Prediction experiment groups"):
        prediction_df = load_prediction_frame(spec.prediction_path)
        prediction_df = prediction_df[prediction_df["date"] >= pd.Timestamp(oos_start_date)].copy()
        if prediction_df.empty:
            raise ValueError(f"No predictions remain for {spec.experiment} after {oos_start_date}.")

        if args.window_policy == "recent":
            windows = build_recent_oos_windows(
                min_start=pd.Timestamp(oos_start_date),
                max_date=pd.Timestamp(prediction_df["date"].max()),
            )
        else:
            windows = build_windows(
                min_start=pd.Timestamp(oos_start_date),
                max_date=pd.Timestamp(prediction_df["date"].max()),
                window_modes=["full", "3m", "6m", "12m"],
                include_partial_final_window=False,
            )
        window_cache: dict[str, dict[str, pd.DataFrame]] = {}
        grid_items = [
            (window, strategy, weight_mode, top_k, cost_bps, neutral_mode)
            for window in windows
            for strategy in strategy_specs()
            for weight_mode in args.weight_modes
            for top_k in args.top_k_list
            for cost_bps in args.cost_bps_list
            for neutral_mode in args.neutral_modes
        ]

        for window, strategy, weight_mode, top_k, cost_bps, neutral_mode in progress_iter(
            grid_items,
            total=len(grid_items),
            desc=f"Backtests: {spec.experiment}",
            position=1,
            leave=False,
        ):
            max_abs_weight = auto_max_abs_weight(int(top_k), args.max_abs_weight)
            run_dir_name = build_run_dir_name(
                base_run_name=f"{spec.experiment}_{weight_mode}",
                window=window,
                hold_days=strategy.hold_days,
                step_days=strategy.step_days,
                top_k=int(top_k),
                cost_bps=float(cost_bps),
                neutral_mode=neutral_mode,
                holding_clock=args.holding_clock,
            )
            run_dir_name = f"{run_dir_name}__w_{weight_mode}"
            if window.window_id not in window_cache:
                window_prediction_df = prediction_df[
                    (prediction_df["date"] >= window.start_date) & (prediction_df["date"] <= window.end_date)
                ].copy()
                window_market_df = market_snapshot_df[
                    (market_snapshot_df["date"] >= window.start_date) & (market_snapshot_df["date"] <= window.end_date)
                ].copy()
                window_cache[window.window_id] = {
                    "prediction_df": window_prediction_df,
                    "market_df": window_market_df,
                    "merged_df": merge_predictions_with_market(window_prediction_df, window_market_df),
                }
            cached_window = window_cache[window.window_id]
            config = LongShortBacktestConfig(
                run_name=run_dir_name,
                predictions_path=spec.prediction_path,
                data_path=data_path,
                output_dir=portfolio_root / spec.experiment / window.window_id / run_dir_name,
                hold_days=int(strategy.hold_days),
                step_days=int(strategy.step_days),
                top_k=int(top_k),
                cost_bps=float(cost_bps),
                neutral_mode=neutral_mode,
                signal_delay_days=int(args.signal_delay_days),
                holding_clock=args.holding_clock,
                borrow_cost_bps=float(args.borrow_cost_bps),
                weight_mode=weight_mode,
                max_abs_weight=max_abs_weight,
                write_outputs=False,
            )

            try:
                result = run_long_short_backtest(
                    config=config,
                    market_snapshot_df=cached_window["market_df"],
                    prediction_df=cached_window["prediction_df"],
                    merged_prediction_market_df=cached_window["merged_df"],
                )
                rows.append(
                    summarize_backtest_row(
                        spec=spec,
                        strategy=strategy,
                        window=window,
                        weight_mode=weight_mode,
                        top_k=int(top_k),
                        cost_bps=float(cost_bps),
                        neutral_mode=neutral_mode,
                        max_abs_weight=max_abs_weight,
                        prediction_df=prediction_df,
                        result_metrics=result["metrics"],
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "experiment": spec.experiment,
                        "feature_group": spec.feature_group,
                        "weight_mode": weight_mode,
                        "strategy_name": strategy.strategy_name,
                        "window_id": window.window_id,
                        "window_mode": window.window_mode,
                        "window_start": str(window.start_date.date()),
                        "window_end": str(window.end_date.date()),
                        "calendar_months": window.calendar_months,
                        "top_k": int(top_k),
                        "cost_bps": float(cost_bps),
                        "neutral_mode": neutral_mode,
                        "max_abs_weight": max_abs_weight,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_views(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """把滚动窗口明细汇总成核心组合视角。"""

    if portfolio_df.empty:
        return pd.DataFrame()
    ok = portfolio_df[portfolio_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_columns = [
        "experiment",
        "feature_group",
        "weight_mode",
        "strategy_name",
        "window_mode",
        "top_k",
        "cost_bps",
        "neutral_mode",
    ]
    numeric_columns = [
        "portfolio_total_return",
        "excess_total_return_vs_benchmark",
        "portfolio_sharpe",
        "portfolio_max_drawdown",
        "portfolio_calmar",
        "hit_ratio",
        "average_gross_turnover",
        "average_turnover_cost_bps",
        "rebalance_count",
        "average_max_abs_sector_net_weight",
    ]
    for column in numeric_columns:
        ok[column] = pd.to_numeric(ok[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for keys, frame in ok.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        total_return = frame["portfolio_total_return"].dropna()
        excess_return = frame["excess_total_return_vs_benchmark"].dropna()
        sharpe = frame["portfolio_sharpe"].dropna()
        row.update(
            {
                "window_count": int(frame["window_id"].nunique()),
                "ok_rows": int(len(frame)),
                "short_sample_warning_rows": int(frame["is_short_sample_warning"].fillna(False).sum()),
                "avg_total_return": float(total_return.mean()) if not total_return.empty else float("nan"),
                "min_total_return": float(total_return.min()) if not total_return.empty else float("nan"),
                "positive_total_return_windows": int((total_return > 0).sum()),
                "avg_excess_return": float(excess_return.mean()) if not excess_return.empty else float("nan"),
                "min_excess_return": float(excess_return.min()) if not excess_return.empty else float("nan"),
                "positive_excess_windows": int((excess_return > 0).sum()),
                "avg_sharpe": float(sharpe.mean()) if not sharpe.empty else float("nan"),
                "min_sharpe": float(sharpe.min()) if not sharpe.empty else float("nan"),
                "worst_max_drawdown": float(frame["portfolio_max_drawdown"].min()),
                "avg_calmar": float(frame["portfolio_calmar"].mean()),
                "avg_hit_ratio": float(frame["hit_ratio"].mean()),
                "avg_gross_turnover": float(frame["average_gross_turnover"].mean()),
                "avg_turnover_cost_bps": float(frame["average_turnover_cost_bps"].mean()),
                "avg_rebalance_count": float(frame["rebalance_count"].mean()),
                "avg_max_abs_sector_net_weight": float(frame["average_max_abs_sector_net_weight"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_weight_mode_delta(view_df: pd.DataFrame) -> pd.DataFrame:
    """同一模型下，不等权相对等权的增量。"""

    if view_df.empty:
        return pd.DataFrame()
    key_columns = ["experiment", "strategy_name", "window_mode", "top_k", "cost_bps", "neutral_mode"]
    baseline = view_df[view_df["weight_mode"] == "equal_weight"].set_index(key_columns)
    comparisons = view_df[view_df["weight_mode"] != "equal_weight"].copy()
    rows: list[dict[str, Any]] = []
    for _, row in comparisons.iterrows():
        key = tuple(row[column] for column in key_columns)
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        delta = {column: row[column] for column in key_columns}
        delta["weight_mode"] = row["weight_mode"]
        for column in [
            "avg_total_return",
            "avg_excess_return",
            "avg_sharpe",
            "min_excess_return",
            "worst_max_drawdown",
            "positive_excess_windows",
            "avg_gross_turnover",
            "avg_turnover_cost_bps",
        ]:
            delta[f"delta_{column}"] = float(row[column] - base[column])
        rows.append(delta)
    return pd.DataFrame(rows)


def build_mined_factor_delta(view_df: pd.DataFrame) -> pd.DataFrame:
    """同一组合口径下，自挖因子模型相对 strict_baseline 的增量。"""

    if view_df.empty:
        return pd.DataFrame()
    key_columns = ["weight_mode", "strategy_name", "window_mode", "top_k", "cost_bps", "neutral_mode"]
    baseline = view_df[view_df["experiment"] == "strict_baseline"].set_index(key_columns)
    comparisons = view_df[view_df["experiment"] != "strict_baseline"].copy()
    rows: list[dict[str, Any]] = []
    for _, row in comparisons.iterrows():
        key = tuple(row[column] for column in key_columns)
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        delta = {
            "experiment": row["experiment"],
            "feature_group": row["feature_group"],
            **{column: row[column] for column in key_columns},
        }
        for column in [
            "avg_total_return",
            "avg_excess_return",
            "avg_sharpe",
            "min_excess_return",
            "worst_max_drawdown",
            "positive_excess_windows",
            "avg_gross_turnover",
            "avg_turnover_cost_bps",
        ]:
            delta[f"delta_{column}"] = float(row[column] - base[column])
        rows.append(delta)
    return pd.DataFrame(rows)


def build_interaction_delta(weight_delta_df: pd.DataFrame) -> pd.DataFrame:
    """计算不等权是否特别放大了自挖因子。

    公式：

        interaction =
          (mined + weight_mode - mined + equal_weight)
          -
          (baseline + weight_mode - baseline + equal_weight)
    """

    if weight_delta_df.empty:
        return pd.DataFrame()
    key_columns = ["weight_mode", "strategy_name", "window_mode", "top_k", "cost_bps", "neutral_mode"]
    baseline = weight_delta_df[weight_delta_df["experiment"] == "strict_baseline"].set_index(key_columns)
    comparisons = weight_delta_df[weight_delta_df["experiment"] != "strict_baseline"].copy()
    rows: list[dict[str, Any]] = []
    for _, row in comparisons.iterrows():
        key = tuple(row[column] for column in key_columns)
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        item = {
            "experiment": row["experiment"],
            **{column: row[column] for column in key_columns},
        }
        for column in [
            "delta_avg_total_return",
            "delta_avg_excess_return",
            "delta_avg_sharpe",
            "delta_min_excess_return",
            "delta_positive_excess_windows",
        ]:
            item[f"interaction_{column.replace('delta_', '')}"] = float(row[column] - base[column])
        rows.append(item)
    return pd.DataFrame(rows)


def save_plot(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_figures(output_dir: Path, view_df: pd.DataFrame, weight_delta_df: pd.DataFrame, mined_delta_df: pd.DataFrame) -> list[Path]:
    """生成报告图表。"""

    sns.set_theme(style="whitegrid", context="talk")
    figure_dir = output_dir / "figures"
    paths: list[Path] = []

    summary = (
        view_df.groupby(["experiment", "weight_mode"], as_index=False)
        .agg(avg_excess=("avg_excess_return", "mean"), avg_total=("avg_total_return", "mean"), avg_sharpe=("avg_sharpe", "mean"))
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=summary, x="experiment", y="avg_excess", hue="weight_mode", ax=ax)
    ax.set_title("Average Excess Return by Model and Weight Mode")
    ax.set_xlabel("")
    ax.set_ylabel("Average Excess Return")
    ax.tick_params(axis="x", rotation=20)
    paths.append(save_plot(fig, figure_dir / "01_avg_excess_by_model_weight.png"))

    top = view_df.sort_values("avg_excess_return", ascending=False).head(30).copy()
    top["label"] = top["experiment"] + " | " + top["weight_mode"] + " | " + top["strategy_name"] + " | K" + top["top_k"].astype(str)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(data=top, y="label", x="avg_excess_return", ax=ax, color="#4C78A8")
    ax.set_title("Top 30 Portfolio Views by Average Excess Return")
    ax.set_xlabel("Average Excess Return")
    ax.set_ylabel("")
    paths.append(save_plot(fig, figure_dir / "02_top30_views.png"))

    if not weight_delta_df.empty and {"experiment", "weight_mode", "delta_avg_excess_return", "delta_avg_sharpe"}.issubset(weight_delta_df.columns):
        weight_summary = (
            weight_delta_df.groupby(["experiment", "weight_mode"], as_index=False)
            .agg(delta_excess=("delta_avg_excess_return", "mean"), delta_sharpe=("delta_avg_sharpe", "mean"))
        )
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(data=weight_summary, x="experiment", y="delta_excess", hue="weight_mode", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Non-Equal Weight Delta vs Equal Weight")
        ax.set_xlabel("")
        ax.set_ylabel("Average Excess Delta")
        ax.tick_params(axis="x", rotation=20)
        paths.append(save_plot(fig, figure_dir / "03_weight_delta_vs_equal.png"))

    if not mined_delta_df.empty and {"experiment", "weight_mode", "delta_avg_excess_return", "delta_avg_sharpe"}.issubset(mined_delta_df.columns):
        mined_summary = (
            mined_delta_df.groupby(["experiment", "weight_mode"], as_index=False)
            .agg(delta_excess=("delta_avg_excess_return", "mean"), delta_sharpe=("delta_avg_sharpe", "mean"))
        )
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(data=mined_summary, x="experiment", y="delta_excess", hue="weight_mode", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Mined-Factor Delta vs Strict Baseline")
        ax.set_xlabel("")
        ax.set_ylabel("Average Excess Delta")
        ax.tick_params(axis="x", rotation=20)
        paths.append(save_plot(fig, figure_dir / "04_mined_delta_vs_baseline.png"))

    heat_source = view_df[
        (view_df["cost_bps"] == view_df["cost_bps"].min())
        & (view_df["neutral_mode"] == "unconstrained")
        & (view_df["window_mode"].isin(["full", "6m"]))
    ].copy()
    if not heat_source.empty:
        pivot = heat_source.pivot_table(
            index=["experiment", "weight_mode"],
            columns=["strategy_name", "top_k", "window_mode"],
            values="avg_excess_return",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(18, 9))
        sns.heatmap(pivot, cmap="RdYlGn", center=0, ax=ax)
        ax.set_title("Excess Return Heatmap: 5bps, Unconstrained")
        ax.set_xlabel("Strategy / Top-K / Window")
        ax.set_ylabel("Experiment / Weight Mode")
        paths.append(save_plot(fig, figure_dir / "05_excess_heatmap.png"))

    return paths


def image_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def write_bilingual_reports(
    *,
    output_dir: Path,
    config: dict[str, Any],
    view_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    weight_delta_df: pd.DataFrame,
    mined_delta_df: pd.DataFrame,
    interaction_delta_df: pd.DataFrame,
    figure_paths: list[Path],
    runtime_seconds: float,
) -> tuple[Path, Path]:
    """生成中英双语 Markdown 和 HTML 报告。"""

    def sorted_head_or_empty(frame: pd.DataFrame, sort_columns: list[str], n: int = 30) -> pd.DataFrame:
        """在 smoke test 缺少某些对照组时，安全返回空表而不是中断报告生成。"""

        if frame.empty or not set(sort_columns).issubset(frame.columns):
            return pd.DataFrame()
        return frame.sort_values(sort_columns, ascending=False).head(n)

    overview = (
        view_df.groupby(["experiment", "weight_mode"], as_index=False)
        .agg(
            avg_total=("avg_total_return", "mean"),
            avg_excess=("avg_excess_return", "mean"),
            min_excess=("min_excess_return", "min"),
            avg_sharpe=("avg_sharpe", "mean"),
            positive_excess_windows=("positive_excess_windows", "sum"),
        )
        .sort_values("avg_excess", ascending=False)
    )
    top_views = sorted_head_or_empty(view_df, ["avg_excess_return", "avg_total_return"])
    top_weight_delta = sorted_head_or_empty(weight_delta_df, ["delta_avg_excess_return"])
    top_mined_delta = sorted_head_or_empty(mined_delta_df, ["delta_avg_excess_return"])
    top_interaction = sorted_head_or_empty(interaction_delta_df, ["interaction_avg_excess_return"])

    cn_conclusion = """
严格结论：

1. 不等权组合确实改变了收益分布，但它不自动等于“自挖因子更有效”。
2. 判断自挖因子是否有用，必须看 mined-factor delta；判断不等权是否放大自挖因子，必须看 interaction delta。
3. 如果 baseline 在同一个 weight_mode 下也同步提升，那么主要贡献来自组合构建，而不是自挖因子。
4. 当前回测仍然是 close-to-close 研究回测，不含真实盘口滑点、借券可得性和市场冲击。
"""
    en_conclusion = """
Strict conclusion:

1. Non-equal weighting changes the portfolio return distribution, but that does not automatically prove mined factors are better.
2. Mined-factor value should be judged by mined-factor delta; whether weighting amplifies mined factors should be judged by interaction delta.
3. If the baseline improves under the same weighting mode, the improvement mainly comes from portfolio construction rather than mined factors.
4. This is still a close-to-close research backtest. It does not include intraday slippage, short availability, or market impact.
"""

    md = f"""# Strict Weighted Portfolio Experiment / 严格不等权组合实验

## 1. Configuration / 实验配置

```json
{json.dumps(config, ensure_ascii=False, indent=2)}
```

Runtime: `{runtime_seconds:.2f}` seconds.

## 2. Chinese Summary / 中文总结

{cn_conclusion}

## 3. English Summary

{en_conclusion}

## 4. Overview Table / 总览表

{dataframe_to_markdown(overview)}

## 5. Top Portfolio Views / 最强组合视角

{dataframe_to_markdown(top_views)}

## 6. Weight Mode Delta vs Equal Weight / 不等权相对等权增量

{dataframe_to_markdown(top_weight_delta)}

## 7. Mined Factor Delta vs Strict Baseline / 自挖因子相对旧主线增量

{dataframe_to_markdown(top_mined_delta)}

## 8. Interaction Delta / 交互增量

{dataframe_to_markdown(top_interaction)}

## 9. Output Files / 输出文件

- `portfolio_detail.csv`
- `portfolio_view_summary.csv`
- `weight_mode_delta.csv`
- `mined_factor_delta.csv`
- `interaction_delta.csv`
- `figures/`
"""
    md_path = output_dir / "strict_weighted_experiment_report_bilingual.md"
    md_path.write_text(md, encoding="utf-8")

    image_html = "\n".join(
        f"<section><h3>{html.escape(path.stem)}</h3><img src='{image_to_data_uri(path)}' /></section>"
        for path in figure_paths
    )
    html_body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Strict Weighted Portfolio Experiment</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Segoe UI", sans-serif; margin: 36px; line-height: 1.55; color: #1f2933; }}
    h1, h2, h3 {{ color: #0f172a; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 16px 0; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code, pre {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
    img {{ width: 100%; max-width: 1400px; border: 1px solid #e5e7eb; margin: 12px 0 28px; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 16px; }}
  </style>
</head>
<body>
  <h1>Strict Weighted Portfolio Experiment / 严格不等权组合实验</h1>
  <h2>Configuration / 实验配置</h2>
  <pre>{html.escape(json.dumps(config, ensure_ascii=False, indent=2))}</pre>
  <p><b>Runtime:</b> {runtime_seconds:.2f} seconds</p>
  <h2>Chinese Summary / 中文总结</h2>
  <div class="note">{html.escape(cn_conclusion).replace(chr(10), '<br>')}</div>
  <h2>English Summary</h2>
  <div class="note">{html.escape(en_conclusion).replace(chr(10), '<br>')}</div>
  <h2>Overview Table / 总览表</h2>
  {overview.to_html(index=False, escape=False)}
  <h2>Top Portfolio Views / 最强组合视角</h2>
  {top_views.to_html(index=False, escape=False)}
  <h2>Weight Mode Delta vs Equal Weight / 不等权相对等权增量</h2>
  {top_weight_delta.to_html(index=False, escape=False)}
  <h2>Mined Factor Delta vs Strict Baseline / 自挖因子增量</h2>
  {top_mined_delta.to_html(index=False, escape=False)}
  <h2>Interaction Delta / 交互增量</h2>
  {top_interaction.to_html(index=False, escape=False)}
  <h2>Figures / 图表</h2>
  {image_html}
</body>
</html>
"""
    html_path = output_dir / "strict_weighted_experiment_report_bilingual.html"
    html_path.write_text(html_body, encoding="utf-8")
    return md_path, html_path


def zip_outputs(output_dir: Path) -> Path:
    """把核心报告文件打包，便于邮件发送。"""

    zip_path = output_dir / "strict_weighted_portfolio_experiment_package.zip"
    include_suffixes = {".csv", ".md", ".html", ".png", ".json"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path == zip_path or not path.is_file():
                continue
            if path.suffix.lower() in include_suffixes and "portfolio_runs" not in path.parts:
                zf.write(path, path.relative_to(output_dir))
    return zip_path


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    strict_input_dir = read_path(args.strict_input_dir)
    output_dir = read_path(args.output_dir)
    data_path = resolve_project_path(args.data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_specs = load_prediction_specs(strict_input_dir, args.experiments)
    config = {
        "strict_input_dir": str(strict_input_dir),
        "output_dir": str(output_dir),
        "data_path": str(data_path),
        "oos_start_date": args.oos_start_date,
        "experiments": args.experiments,
        "weight_modes": args.weight_modes,
        "top_k_list": args.top_k_list,
        "cost_bps_list": args.cost_bps_list,
        "neutral_modes": args.neutral_modes,
        "window_policy": args.window_policy,
        "strategies": [spec.__dict__ for spec in strategy_specs()],
        "signal_delay_days": args.signal_delay_days,
        "holding_clock": args.holding_clock,
        "borrow_cost_bps": args.borrow_cost_bps,
        "max_abs_weight_rule": "explicit or min(15%, 2/top_k)",
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_df = run_weighted_backtests(
        prediction_specs=prediction_specs,
        data_path=data_path,
        output_dir=output_dir,
        oos_start_date=args.oos_start_date,
        args=args,
    )
    detail_df.to_csv(output_dir / "portfolio_detail.csv", index=False)
    view_df = aggregate_views(detail_df)
    view_df.to_csv(output_dir / "portfolio_view_summary.csv", index=False)
    weight_delta_df = build_weight_mode_delta(view_df)
    weight_delta_df.to_csv(output_dir / "weight_mode_delta.csv", index=False)
    mined_delta_df = build_mined_factor_delta(view_df)
    mined_delta_df.to_csv(output_dir / "mined_factor_delta.csv", index=False)
    interaction_delta_df = build_interaction_delta(weight_delta_df)
    interaction_delta_df.to_csv(output_dir / "interaction_delta.csv", index=False)

    figure_paths = build_figures(output_dir, view_df, weight_delta_df, mined_delta_df)
    runtime_seconds = time.perf_counter() - start
    runtime_df = pd.DataFrame(
        [
            {"stage": "total_runtime", "runtime_seconds": runtime_seconds},
            {"stage": "detail_rows", "runtime_seconds": float(len(detail_df))},
            {"stage": "summary_rows", "runtime_seconds": float(len(view_df))},
        ]
    )
    runtime_df.to_csv(output_dir / "runtime.csv", index=False)

    md_path, html_path = write_bilingual_reports(
        output_dir=output_dir,
        config=config,
        view_df=view_df,
        detail_df=detail_df,
        weight_delta_df=weight_delta_df,
        mined_delta_df=mined_delta_df,
        interaction_delta_df=interaction_delta_df,
        figure_paths=figure_paths,
        runtime_seconds=runtime_seconds,
    )
    zip_path = zip_outputs(output_dir)

    print("[Info] Strict weighted portfolio experiment finished.", flush=True)
    print(f"[Info] Output dir: {output_dir}", flush=True)
    print(f"[Info] Markdown report: {md_path}", flush=True)
    print(f"[Info] HTML report: {html_path}", flush=True)
    print(f"[Info] Package: {zip_path}", flush=True)


if __name__ == "__main__":
    main()
