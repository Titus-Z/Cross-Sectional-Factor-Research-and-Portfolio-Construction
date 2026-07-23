"""Run mined-factor incremental model and portfolio experiments.

这个脚本专门回答一个问题：

    自己挖出来的公式因子，加入原有 MyQuant 特征体系后，是否真的带来增量？

它不替代 `main.py`，也不修改主训练入口。原因很简单：

1. 主线训练脚本负责产出一个标准模型；
2. 这里负责做受控实验，比较 baseline 与 baseline + mined factors；
3. 两者分开，后续复现实验和回滚都更清楚。

实验设计：

- 特征组：
  - F0: 原始 baseline 特征，包括技术指标、Alpha191、市场状态等；
  - F1: baseline + Warm-GP 挖出来的 factor zoo；
  - F2: baseline + PPO validation-selected factor zoo；
  - F3: baseline + Warm-GP + PPO。
- 模型组：
  - linear: Ridge + ElasticNet；
  - nonlinear: XGBoost + ExtraTrees。
- 组合策略：
  - hold10_step10: 10 天持有，10 天调仓；
  - hold10_step5: 10 天持有，5 天调仓，允许重叠持仓；
  - hold20_step20: 20 天持有，20 天调仓；
  - hold20_step10: 20 天持有，10 天调仓，允许重叠持仓。
- OOS 视角：
  - full、3m、6m、12m。

最终的 `view_96_summary.csv` 正好对应：

    6 个实验组 * 4 个策略 * 4 类 OOS 视角 = 96 行。

明细回测 `portfolio_metrics.csv` 会更多，因为 3m 和 6m 会展开成多个滚动子窗口。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - 进度条只影响显示，不影响实验逻辑。
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.mined_factor_model_ablation import (  # noqa: E402
    add_mined_factor_columns,
    dataframe_to_markdown,
    evaluate_prediction_frame,
    get_numeric_feature_columns,
    load_factor_zoo,
    train_and_predict,
)
from factor_mining_workspace.single_factor_case_study import (  # noqa: E402
    load_or_build_preprocessed_train_test,
    sanitize_name,
)
from main_rolling_oos_backtest import (  # noqa: E402
    OOSWindow,
    build_run_dir_name,
    build_windows,
    summarize_window_predictions,
    write_window_prediction_file,
)
from src.long_short_backtest import LongShortBacktestConfig, run_long_short_backtest  # noqa: E402
from src.pdf_report import PdfSection, write_pdf_report  # noqa: E402
from src.portfolio import load_market_snapshot_frame, load_prediction_frame  # noqa: E402
from src.project_paths import resolve_project_path  # noqa: E402
from src.runtime_config import (  # noqa: E402
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)


DEFAULT_OUTPUT_DIR = "outputs/mined_factor_incremental_experiment_oos202506"
DEFAULT_EXPERIMENT_OOS_START_DATE = "2025-06-01"
DEFAULT_WARM_GP_ZOO_PATH = (
    "factor_mining_workspace/auto_mining_outputs_oos202506/"
    "warm_gp_10d_g5_p80_c500_s7/factor_zoo.csv"
)
DEFAULT_PPO_ZOO_PATH = (
    "factor_mining_workspace/deep_rl_mining_outputs/"
    "ppo_formula_us300_10d_oos202506_v1/validation_selected_factor_zoo.csv"
)


@dataclass(frozen=True)
class ModelExperimentSpec:
    """一个模型实验组。

    `feature_group` 决定用哪些因子。
    `model_family` 决定它和哪个 baseline 对比。
    `model_names` 是真正传入 `src.model.build_model` 的模型名。
    """

    experiment_name: str
    feature_group: str
    model_family: str
    model_names: list[str]


@dataclass(frozen=True)
class StrategySpec:
    """一个组合执行规则。"""

    strategy_name: str
    hold_days: int
    step_days: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 96-view mined-factor incremental experiment: "
            "model ablation + rolling OOS long-short backtest."
        )
    )
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="us300 日频行情 CSV。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="实验输出目录。")
    parser.add_argument("--cache-dir", default=".cache", help="特征和预处理缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument(
        "--oos-start-date",
        default=DEFAULT_EXPERIMENT_OOS_START_DATE,
        help=(
            "OOS 起始日期。本实验固定默认 2025-06-01，"
            "因为要检验更长 OOS 窗口下自挖因子的增量。"
        ),
    )
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="目标收益周期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="没有显式 OOS 日期时的后段测试比例。")
    parser.add_argument("--warm-gp-zoo-path", default=DEFAULT_WARM_GP_ZOO_PATH, help="Warm-GP factor_zoo.csv。")
    parser.add_argument("--ppo-zoo-path", default=DEFAULT_PPO_ZOO_PATH, help="PPO validation-selected factor_zoo.csv。")
    parser.add_argument("--linear-models", nargs="+", default=["ridge", "elastic_net"], help="线性模型组。")
    parser.add_argument("--nonlinear-models", nargs="+", default=["xgboost", "extra_trees"], help="非线性模型组。")
    parser.add_argument("--top-k", type=int, default=20, help="每次做多 Top-K、做空 Bottom-K。")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="单边交易成本，单位 bps。")
    parser.add_argument(
        "--neutral-mode",
        default="unconstrained",
        choices=["unconstrained", "sector_neutral"],
        help="默认先固定一种组合约束，避免把策略网格和因子增量问题混在一起。",
    )
    parser.add_argument("--signal-delay-days", type=int, default=1, help="信号后延迟几个交易日执行。")
    parser.add_argument(
        "--holding-clock",
        choices=["signal_horizon", "execution_horizon"],
        default="signal_horizon",
        help="默认将组合终点与信号日 forward-return 标签终点对齐。",
    )
    parser.add_argument("--borrow-cost-bps", type=float, default=0.0, help="做空借券成本占位，默认 0。")
    parser.add_argument("--random-seed", type=int, default=42, help="模型随机种子。")
    parser.add_argument(
        "--disable-preprocessing-cache",
        action="store_true",
        help="关闭横截面预处理缓存。正式实验不建议打开这个开关。",
    )
    parser.add_argument(
        "--skip-nonlinear",
        action="store_true",
        help="只跑线性模型组。用于快速检查脚本，不建议作为正式结果。",
    )
    return parser.parse_args()


def progress_iter(iterable, *, total: int | None = None, desc: str, position: int = 0):
    """统一进度条入口。

    tqdm 不存在时退化成普通迭代，保证脚本在极简环境中也能运行。
    """

    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, position=position)


def read_path(path_like: str | Path) -> Path:
    """把相对路径解析到项目根目录。

    这个项目目录里有中文和空格，统一用 Path 处理能减少 shell 路径错误。
    """

    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_preprocessing_args(args: argparse.Namespace) -> argparse.Namespace:
    """构造可复用的特征缓存加载参数。

    这里显式设置 `include_alpha_seeds=True`。
    否则自动挖因子研究脚本为了省时间，会默认不生成完整 Alpha191。
    本实验的 baseline 必须包含完整原始特征体系，所以不能沿用那个轻量默认值。
    """

    return SimpleNamespace(
        data_path=str(args.data_path),
        model_dir="models",
        output_dir=str(args.output_dir),
        cache_dir=str(args.cache_dir),
        sample_start_date=str(args.sample_start_date),
        oos_start_date=str(args.oos_start_date),
        target_horizon=int(args.target_horizon),
        test_size=float(args.test_size),
        disable_preprocessing_cache=bool(args.disable_preprocessing_cache),
        include_alpha_seeds=True,
        factor="",
    )


def load_named_factor_zoo(path: Path, source_name: str) -> pd.DataFrame:
    """读取一个 factor zoo，并给 candidate_id 加来源前缀。

    Warm-GP 和 PPO 的候选 id 目前不会冲突，但显式加 source 字段能让报告更容易读。
    """

    factor_zoo = load_factor_zoo(path).copy()
    factor_zoo["factor_source"] = source_name
    factor_zoo["candidate_id"] = factor_zoo["candidate_id"].astype(str)
    return factor_zoo


def build_factor_groups(warm_gp_zoo: pd.DataFrame, ppo_zoo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """定义受控实验中的特征组。"""

    return {
        "baseline": pd.DataFrame(columns=["candidate_id", "formula", "factor_source"]),
        "warm_gp": warm_gp_zoo.copy(),
        "ppo": ppo_zoo.copy(),
        "warm_gp_ppo": pd.concat([warm_gp_zoo, ppo_zoo], ignore_index=True),
    }


def build_model_specs(args: argparse.Namespace) -> list[ModelExperimentSpec]:
    """定义 6 个模型实验组。

    这里刻意没有把所有特征组都配非线性模型。
    原因是这个实验的目标是判断“自挖因子是否有增量”，不是无限扩模型网格。
    """

    specs = [
        ModelExperimentSpec("baseline_linear", "baseline", "linear", list(args.linear_models)),
        ModelExperimentSpec("warm_gp_linear", "warm_gp", "linear", list(args.linear_models)),
        ModelExperimentSpec("ppo_linear", "ppo", "linear", list(args.linear_models)),
        ModelExperimentSpec("warm_gp_ppo_linear", "warm_gp_ppo", "linear", list(args.linear_models)),
    ]
    if not args.skip_nonlinear:
        specs.extend(
            [
                ModelExperimentSpec("baseline_nonlinear", "baseline", "nonlinear", list(args.nonlinear_models)),
                ModelExperimentSpec("warm_gp_ppo_nonlinear", "warm_gp_ppo", "nonlinear", list(args.nonlinear_models)),
            ]
        )
    return specs


def build_strategy_specs() -> list[StrategySpec]:
    """固定四个组合策略。

    `hold_days` 表示信号日 forward-return horizon。默认一日 close
    执行延迟会使实际可执行持有天数少 1。当 `step_days` 小于
    实际持有天数时，同一时间可能同时存在多个 sleeve。
    """

    return [
        StrategySpec("hold10_step10", hold_days=10, step_days=10),
        StrategySpec("hold10_step5", hold_days=10, step_days=5),
        StrategySpec("hold20_step20", hold_days=20, step_days=20),
        StrategySpec("hold20_step10", hold_days=20, step_days=10),
    ]


def materialize_feature_group(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    factor_zoo: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """把一个 factor zoo 变成 train/test 中的 mined_* 列。"""

    if factor_zoo.empty:
        return train_df, test_df, []

    train_with_mined, mined_columns = add_mined_factor_columns(train_df, factor_zoo)
    test_with_mined, _ = add_mined_factor_columns(test_df, factor_zoo)
    return train_with_mined, test_with_mined, mined_columns


def save_prediction_frame(prediction_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(path, index=False)


def run_model_experiments(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    factor_groups: dict[str, pd.DataFrame],
    model_specs: list[ModelExperimentSpec],
    output_dir: Path,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """训练模型实验组，并保存每组 OOS 预测。

    返回：
    - model_metrics_df: OOS 预测层指标；
    - model_runtime_df: 每个模型/阶段耗时；
    - feature_count_df: 每个实验组的特征数量；
    - prediction_paths: 每个实验组对应的预测 CSV。
    """

    materialized_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]] = {}
    model_metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    feature_count_rows: list[dict[str, Any]] = []
    prediction_paths: dict[str, Path] = {}

    prediction_dir = output_dir / "predictions"
    total_groups = len(model_specs)

    for spec in progress_iter(model_specs, total=total_groups, desc="Model experiment groups", position=0):
        group_start = time.perf_counter()

        if spec.feature_group not in materialized_cache:
            factor_zoo = factor_groups[spec.feature_group]
            materialize_start = time.perf_counter()
            materialized_cache[spec.feature_group] = materialize_feature_group(
                train_df=train_df,
                test_df=test_df,
                factor_zoo=factor_zoo,
            )
            runtime_rows.append(
                {
                    "experiment": spec.experiment_name,
                    "feature_group": spec.feature_group,
                    "stage": "materialize_feature_group",
                    "runtime_seconds": time.perf_counter() - materialize_start,
                }
            )

        group_train_df, group_test_df, mined_columns = materialized_cache[spec.feature_group]
        baseline_columns = get_numeric_feature_columns(group_train_df)
        feature_columns = get_numeric_feature_columns(group_train_df, mined_columns=mined_columns)

        feature_count_rows.append(
            {
                "experiment": spec.experiment_name,
                "feature_group": spec.feature_group,
                "model_family": spec.model_family,
                "model_names": ",".join(spec.model_names),
                "baseline_numeric_feature_count": int(len(baseline_columns)),
                "mined_feature_count": int(len(mined_columns)),
                "total_feature_count": int(len(feature_columns)),
            }
        )

        train_start = time.perf_counter()
        prediction_df, model_runtime_df = train_and_predict(
            train_df=group_train_df,
            test_df=group_test_df,
            feature_columns=feature_columns,
            model_names=spec.model_names,
            random_seed=random_seed,
        )
        train_seconds = time.perf_counter() - train_start

        metrics = evaluate_prediction_frame(prediction_df, spec.experiment_name)
        metrics.update(
            {
                "feature_group": spec.feature_group,
                "model_family": spec.model_family,
                "model_names": ",".join(spec.model_names),
                "feature_count": int(len(feature_columns)),
                "mined_feature_count": int(len(mined_columns)),
            }
        )
        model_metric_rows.append(metrics)

        prediction_path = prediction_dir / f"{sanitize_name(spec.experiment_name)}_predictions.csv"
        save_prediction_frame(prediction_df, prediction_path)
        prediction_paths[spec.experiment_name] = prediction_path

        runtime_rows.append(
            {
                "experiment": spec.experiment_name,
                "feature_group": spec.feature_group,
                "stage": "train_group_total",
                "runtime_seconds": train_seconds,
            }
        )
        for _, row in model_runtime_df.iterrows():
            runtime_rows.append(
                {
                    "experiment": spec.experiment_name,
                    "feature_group": spec.feature_group,
                    "stage": f"model_{row['model']}",
                    "runtime_seconds": float(row["runtime_seconds"]),
                    "feature_count": int(row["feature_count"]),
                    "train_rows": int(row["train_rows"]),
                    "oos_rows": int(row["oos_rows"]),
                }
            )
        runtime_rows.append(
            {
                "experiment": spec.experiment_name,
                "feature_group": spec.feature_group,
                "stage": "experiment_total",
                "runtime_seconds": time.perf_counter() - group_start,
            }
        )

    return (
        pd.DataFrame(model_metric_rows),
        pd.DataFrame(runtime_rows),
        pd.DataFrame(feature_count_rows),
        prediction_paths,
    )


def summarize_backtest_row(
    *,
    experiment: str,
    feature_group: str,
    model_family: str,
    strategy: StrategySpec,
    window: OOSWindow,
    prediction_df: pd.DataFrame,
    result_metrics: dict[str, Any],
) -> dict[str, Any]:
    """把一次组合回测结果压平成一行。"""

    row: dict[str, Any] = {
        "experiment": experiment,
        "feature_group": feature_group,
        "model_family": model_family,
        "strategy_name": strategy.strategy_name,
        "window_id": window.window_id,
        "window_mode": window.window_mode,
        "window_start": str(window.start_date.date()),
        "window_end": str(window.end_date.date()),
        "calendar_months": window.calendar_months,
        "status": "ok",
        **summarize_window_predictions(prediction_df, window),
    }
    metric_fields = [
        "hold_days",
        "holding_clock",
        "effective_holding_days",
        "step_days",
        "top_k",
        "cost_bps",
        "neutral_mode",
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
        "is_short_sample_warning",
        "max_active_sleeves",
    ]
    for field in metric_fields:
        row[field] = result_metrics.get(field)
    row["error"] = ""
    return row


def run_portfolio_experiments(
    *,
    model_specs: list[ModelExperimentSpec],
    prediction_paths: dict[str, Path],
    data_path: Path,
    output_dir: Path,
    oos_start_date: str,
    strategy_specs: list[StrategySpec],
    top_k: int,
    cost_bps: float,
    neutral_mode: str,
    signal_delay_days: int,
    holding_clock: str,
    borrow_cost_bps: float,
) -> pd.DataFrame:
    """对每组预测运行 rolling OOS 多空回测。"""

    market_snapshot_df = load_market_snapshot_frame(data_path)
    portfolio_root = output_dir / "portfolio_runs"
    portfolio_rows: list[dict[str, Any]] = []
    spec_map = {spec.experiment_name: spec for spec in model_specs}

    total_outer = len(prediction_paths)
    for experiment_name, prediction_path in progress_iter(
        prediction_paths.items(),
        total=total_outer,
        desc="Portfolio experiment groups",
        position=0,
    ):
        spec = spec_map[experiment_name]
        prediction_df = load_prediction_frame(prediction_path)
        prediction_df = prediction_df[prediction_df["date"] >= pd.Timestamp(oos_start_date)].copy()
        if prediction_df.empty:
            raise ValueError(f"No OOS predictions for {experiment_name} on or after {oos_start_date}.")

        max_prediction_date = pd.Timestamp(prediction_df["date"].max())
        windows = build_windows(
            min_start=pd.Timestamp(oos_start_date),
            max_date=max_prediction_date,
            window_modes=["full", "3m", "6m", "12m"],
            include_partial_final_window=False,
        )
        window_prediction_dir = portfolio_root / experiment_name / "_window_predictions"
        window_prediction_paths: dict[str, Path] = {}

        grid_items = [(window, strategy) for window in windows for strategy in strategy_specs]
        for window, strategy in progress_iter(
            grid_items,
            total=len(grid_items),
            desc=f"Backtests: {experiment_name}",
            position=1,
        ):
            if window.window_id not in window_prediction_paths:
                window_prediction_paths[window.window_id] = write_window_prediction_file(
                    window_prediction_dir,
                    prediction_df,
                    window,
                )

            run_dir_name = build_run_dir_name(
                base_run_name=experiment_name,
                window=window,
                hold_days=strategy.hold_days,
                step_days=strategy.step_days,
                top_k=top_k,
                cost_bps=cost_bps,
                neutral_mode=neutral_mode,
                holding_clock=holding_clock,
            )
            window_market_df = market_snapshot_df[
                (market_snapshot_df["date"] >= window.start_date) & (market_snapshot_df["date"] <= window.end_date)
            ].copy()
            config = LongShortBacktestConfig(
                run_name=run_dir_name,
                predictions_path=window_prediction_paths[window.window_id],
                data_path=data_path,
                output_dir=portfolio_root / experiment_name / window.window_id / run_dir_name,
                hold_days=int(strategy.hold_days),
                step_days=int(strategy.step_days),
                top_k=int(top_k),
                cost_bps=float(cost_bps),
                neutral_mode=str(neutral_mode),
                signal_delay_days=int(signal_delay_days),
                holding_clock=holding_clock,
                borrow_cost_bps=float(borrow_cost_bps),
            )

            try:
                result = run_long_short_backtest(config=config, market_snapshot_df=window_market_df)
                portfolio_rows.append(
                    summarize_backtest_row(
                        experiment=experiment_name,
                        feature_group=spec.feature_group,
                        model_family=spec.model_family,
                        strategy=strategy,
                        window=window,
                        prediction_df=prediction_df,
                        result_metrics=result["metrics"],
                    )
                )
            except Exception as exc:
                portfolio_rows.append(
                    {
                        "experiment": experiment_name,
                        "feature_group": spec.feature_group,
                        "model_family": spec.model_family,
                        "strategy_name": strategy.strategy_name,
                        "window_id": window.window_id,
                        "window_mode": window.window_mode,
                        "window_start": str(window.start_date.date()),
                        "window_end": str(window.end_date.date()),
                        "calendar_months": window.calendar_months,
                        "status": "failed",
                        "hold_days": strategy.hold_days,
                        "step_days": strategy.step_days,
                        "top_k": top_k,
                        "cost_bps": cost_bps,
                        "neutral_mode": neutral_mode,
                        "error": str(exc),
                    }
                )

    return pd.DataFrame(portfolio_rows)


def aggregate_96_views(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """把滚动子窗口明细汇总成 96 个结果视角。"""

    if portfolio_df.empty:
        return pd.DataFrame()

    df = portfolio_df.copy()
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_columns = ["experiment", "feature_group", "model_family", "strategy_name", "window_mode"]
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
                "short_sample_warning_rows": int(frame["is_short_sample_warning"].fillna(False).sum())
                if "is_short_sample_warning" in frame.columns
                else 0,
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
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_model_delta_table(model_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """计算模型预测层相对 baseline 的增量。"""

    rows: list[dict[str, Any]] = []
    if model_metrics_df.empty:
        return pd.DataFrame()

    metrics = model_metrics_df.copy()
    baseline_by_family = {
        "linear": metrics[metrics["experiment"] == "baseline_linear"],
        "nonlinear": metrics[metrics["experiment"] == "baseline_nonlinear"],
    }
    metric_columns = [
        column
        for column in [
            "pearson_corr",
            "spearman_corr",
            "rmse",
            "mae",
            "pearson_ic_mean",
            "spearman_ic_mean",
            "long_short_return",
        ]
        if column in metrics.columns
    ]
    for _, row in metrics.iterrows():
        family = str(row["model_family"])
        baseline_df = baseline_by_family.get(family, pd.DataFrame())
        if baseline_df.empty or str(row["experiment"]).startswith("baseline_"):
            continue
        baseline = baseline_df.iloc[0]
        delta_row: dict[str, Any] = {
            "experiment": row["experiment"],
            "feature_group": row["feature_group"],
            "model_family": family,
            "baseline_experiment": baseline["experiment"],
        }
        for column in metric_columns:
            delta_row[f"delta_{column}"] = float(row[column] - baseline[column])
        rows.append(delta_row)
    return pd.DataFrame(rows)


def build_portfolio_delta_table(view_96_df: pd.DataFrame) -> pd.DataFrame:
    """计算组合层相对同模型家族 baseline 的增量。"""

    if view_96_df.empty:
        return pd.DataFrame()

    baseline_rows = view_96_df[
        view_96_df["experiment"].isin(["baseline_linear", "baseline_nonlinear"])
    ].copy()
    comparison_rows = view_96_df[
        ~view_96_df["experiment"].isin(["baseline_linear", "baseline_nonlinear"])
    ].copy()
    if baseline_rows.empty or comparison_rows.empty:
        return pd.DataFrame()

    key_columns = ["model_family", "strategy_name", "window_mode"]
    baseline_lookup = baseline_rows.set_index(key_columns)
    rows: list[dict[str, Any]] = []
    delta_columns = [
        "avg_total_return",
        "min_total_return",
        "avg_excess_return",
        "min_excess_return",
        "avg_sharpe",
        "min_sharpe",
        "worst_max_drawdown",
        "positive_total_return_windows",
        "positive_excess_windows",
    ]

    for _, row in comparison_rows.iterrows():
        key = tuple(row[column] for column in key_columns)
        if key not in baseline_lookup.index:
            continue
        baseline = baseline_lookup.loc[key]
        if isinstance(baseline, pd.DataFrame):
            baseline = baseline.iloc[0]
        delta_row: dict[str, Any] = {
            "experiment": row["experiment"],
            "feature_group": row["feature_group"],
            "model_family": row["model_family"],
            "strategy_name": row["strategy_name"],
            "window_mode": row["window_mode"],
            "baseline_experiment": baseline["experiment"],
        }
        for column in delta_columns:
            if column in row.index and column in baseline.index:
                delta_row[f"delta_{column}"] = float(row[column] - baseline[column])
        rows.append(delta_row)
    return pd.DataFrame(rows)


def write_experiment_report(
    *,
    output_dir: Path,
    dataset_summary: dict[str, Any],
    factor_zoo_summary_df: pd.DataFrame,
    feature_count_df: pd.DataFrame,
    model_metrics_df: pd.DataFrame,
    model_delta_df: pd.DataFrame,
    view_96_df: pd.DataFrame,
    portfolio_delta_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    total_runtime_seconds: float,
) -> Path:
    """写入 Markdown + PDF 报告。"""

    top_model_metrics = model_metrics_df.sort_values("pearson_ic_mean", ascending=False).copy()
    top_portfolio_views = view_96_df.sort_values("avg_excess_return", ascending=False).head(20).copy()
    top_portfolio_delta = (
        portfolio_delta_df.sort_values("delta_avg_excess_return", ascending=False).head(20).copy()
        if not portfolio_delta_df.empty and "delta_avg_excess_return" in portfolio_delta_df.columns
        else pd.DataFrame()
    )
    runtime_summary = (
        runtime_df.groupby(["experiment", "stage"], as_index=False)["runtime_seconds"].sum()
        if not runtime_df.empty
        else pd.DataFrame()
    )

    report_text = f"""# Mined Factor Incremental Experiment: 96 Result Views

## 1. Purpose

这个实验回答：

```text
在原始 MyQuant 特征体系之上加入自挖公式因子，是否能稳定改善预测层和组合层表现？
```

关键判定标准：

- 预测层看 `Pearson IC`、`RankIC`、`RMSE/MAE`、模型 long-short proxy；
- 组合层看 `total return`、`excess return`、`Sharpe`、`Max Drawdown`、`turnover cost`；
- 简历和答辩更应该引用严格对照结果，不能只引用单因子 IC。

## 2. Dataset

```json
{json.dumps(dataset_summary, ensure_ascii=False, indent=2)}
```

## 3. Experiment Grid

- Model experiment groups: `{model_metrics_df["experiment"].nunique() if not model_metrics_df.empty else 0}`
- Strategies per group: `4`
- OOS view modes: `full, 3m, 6m, 12m`
- 96-view table rows: `{len(view_96_df)}`
- Portfolio detail rows: `{int(view_96_df["ok_rows"].sum()) if not view_96_df.empty and "ok_rows" in view_96_df.columns else 0}`
- Total runtime seconds: `{total_runtime_seconds:.2f}`

## 4. Factor Zoo Summary

{dataframe_to_markdown(factor_zoo_summary_df)}

## 5. Feature Counts

{dataframe_to_markdown(feature_count_df)}

## 6. Model Metrics

{dataframe_to_markdown(top_model_metrics)}

## 7. Model-Layer Delta

`delta = mined feature group - same-family baseline`

{dataframe_to_markdown(model_delta_df)}

## 8. Top Portfolio Views By Average Excess Return

{dataframe_to_markdown(top_portfolio_views)}

## 9. Portfolio-Layer Delta

`delta = mined feature group - same-family baseline`，按相同策略和相同窗口类型比较。

{dataframe_to_markdown(top_portfolio_delta)}

## 10. Runtime

{dataframe_to_markdown(runtime_summary)}

## 11. Reading Rule

- 如果只在 `full` 窗口好看，但 3m/6m 多数窗口不稳定，这不能说明因子稳定有效。
- 如果模型 IC 改善，但组合层 excess return 没改善，说明信号可能难以转成可交易组合。
- 如果组合层改善，但模型层 IC 没改善，要检查是否来自少数窗口或回测参数偶然性。
- 如果线性模型改善明显，非线性模型没有改善，说明新因子更像可解释线性增量，而不是复杂交互增量。
"""
    report_path = output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    pdf_path = output_dir / "report.pdf"
    sections = [
        PdfSection(
            "Scope",
            body=(
                f"Dataset: {dataset_summary.get('data_path')}\n"
                f"OOS start: {dataset_summary.get('oos_start_date_used')}\n"
                f"Target horizon: {dataset_summary.get('target_horizon')}\n"
                f"96-view rows: {len(view_96_df)}\n"
                f"Runtime seconds: {total_runtime_seconds:.2f}"
            ),
        ),
        PdfSection("Factor Zoo Summary", table=factor_zoo_summary_df, max_table_rows=10),
        PdfSection("Feature Counts", table=feature_count_df, max_table_rows=12),
        PdfSection("Model Metrics", table=top_model_metrics, max_table_rows=12),
        PdfSection("Model Delta", table=model_delta_df, max_table_rows=12),
        PdfSection("Top Portfolio Views", table=top_portfolio_views, max_table_rows=20),
        PdfSection("Portfolio Delta", table=top_portfolio_delta, max_table_rows=20),
    ]
    write_pdf_report(
        pdf_path,
        title="MyQuant Mined Factor Incremental Experiment",
        subtitle="96 result views: model ablation + rolling OOS portfolio diagnostics",
        sections=sections,
    )
    return report_path


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()

    output_dir = read_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = read_path(args.data_path)
    warm_gp_path = read_path(args.warm_gp_zoo_path)
    ppo_path = read_path(args.ppo_zoo_path)

    print("[Experiment] Loading factor zoos...", flush=True)
    warm_gp_zoo = load_named_factor_zoo(warm_gp_path, "warm_gp")
    ppo_zoo = load_named_factor_zoo(ppo_path, "ppo")
    factor_groups = build_factor_groups(warm_gp_zoo, ppo_zoo)
    factor_zoo_summary_df = pd.DataFrame(
        [
            {
                "feature_group": name,
                "factor_count": int(len(zoo)),
                "sources": ",".join(sorted(zoo["factor_source"].dropna().astype(str).unique())) if not zoo.empty else "",
            }
            for name, zoo in factor_groups.items()
        ]
    )

    print("[Experiment] Loading cached strict time-split features with full Alpha191...", flush=True)
    preprocessing_args = build_preprocessing_args(args)
    train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(preprocessing_args)
    dataset_summary = dict(dataset_summary)
    dataset_summary.update(
        {
            "target_column": target_column,
            "data_path": str(data_path),
            "warm_gp_zoo_path": str(warm_gp_path),
            "ppo_zoo_path": str(ppo_path),
            "top_k": int(args.top_k),
            "cost_bps": float(args.cost_bps),
            "neutral_mode": str(args.neutral_mode),
            "strategy_specs": [spec.__dict__ for spec in build_strategy_specs()],
        }
    )

    model_specs = build_model_specs(args)
    strategy_specs = build_strategy_specs()

    print("[Experiment] Running model-layer ablations...", flush=True)
    model_metrics_df, runtime_df, feature_count_df, prediction_paths = run_model_experiments(
        train_df=train_df,
        test_df=test_df,
        factor_groups=factor_groups,
        model_specs=model_specs,
        output_dir=output_dir,
        random_seed=int(args.random_seed),
    )
    model_delta_df = build_model_delta_table(model_metrics_df)

    print("[Experiment] Running portfolio-layer rolling OOS backtests...", flush=True)
    portfolio_df = run_portfolio_experiments(
        model_specs=model_specs,
        prediction_paths=prediction_paths,
        data_path=data_path,
        output_dir=output_dir,
        oos_start_date=str(args.oos_start_date),
        strategy_specs=strategy_specs,
        top_k=int(args.top_k),
        cost_bps=float(args.cost_bps),
        neutral_mode=str(args.neutral_mode),
        signal_delay_days=int(args.signal_delay_days),
        holding_clock=str(args.holding_clock),
        borrow_cost_bps=float(args.borrow_cost_bps),
    )

    view_96_df = aggregate_96_views(portfolio_df)
    portfolio_delta_df = build_portfolio_delta_table(view_96_df)

    total_runtime_seconds = time.perf_counter() - total_start
    runtime_df = pd.concat(
        [
            runtime_df,
            pd.DataFrame(
                [
                    {
                        "experiment": "ALL",
                        "feature_group": "ALL",
                        "stage": "total_experiment_runtime",
                        "runtime_seconds": total_runtime_seconds,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # 所有关键结果都落盘，避免后续只能靠聊天记录回忆。
    factor_zoo_summary_df.to_csv(output_dir / "factor_zoo_summary.csv", index=False)
    feature_count_df.to_csv(output_dir / "feature_counts.csv", index=False)
    model_metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    model_delta_df.to_csv(output_dir / "model_metric_delta.csv", index=False)
    runtime_df.to_csv(output_dir / "runtime.csv", index=False)
    portfolio_df.to_csv(output_dir / "portfolio_metrics.csv", index=False)
    view_96_df.to_csv(output_dir / "view_96_summary.csv", index=False)
    portfolio_delta_df.to_csv(output_dir / "portfolio_view_delta.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = write_experiment_report(
        output_dir=output_dir,
        dataset_summary=dataset_summary,
        factor_zoo_summary_df=factor_zoo_summary_df,
        feature_count_df=feature_count_df,
        model_metrics_df=model_metrics_df,
        model_delta_df=model_delta_df,
        view_96_df=view_96_df,
        portfolio_delta_df=portfolio_delta_df,
        runtime_df=runtime_df,
        total_runtime_seconds=total_runtime_seconds,
    )

    print(f"[Experiment] report={report_path}", flush=True)
    print(f"[Experiment] model_metrics={output_dir / 'model_metrics.csv'}", flush=True)
    print(f"[Experiment] view_96_summary={output_dir / 'view_96_summary.csv'}", flush=True)
    if not view_96_df.empty:
        print("[Experiment] Top 10 views by avg_excess_return:", flush=True)
        display_columns = [
            "experiment",
            "strategy_name",
            "window_mode",
            "avg_excess_return",
            "avg_sharpe",
            "positive_excess_windows",
            "window_count",
        ]
        print(
            view_96_df.sort_values("avg_excess_return", ascending=False)
            .head(10)[display_columns]
            .to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
