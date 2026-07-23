"""实验对比入口。

这个文件和原来的 `main.py` 是并行关系，不会替换或破坏原主入口。

它的设计目标很明确：

1. 保留原始 `main.py` 作为单次训练主流程；
2. 额外提供一个“批量实验对比”入口；
3. 一次性对比不同目标周期（5d / 10d）和不同模型组合；
4. 自动把每组实验的结果分目录保存，避免互相覆盖。

为什么单独写一个新入口，而不是继续往 `main.py` 里加条件分支？

- 你已经明确担心主入口越改越难回退；
- 实验对比逻辑本来就比单次训练更复杂；
- 分成两个入口后，学习和维护都会更清晰。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data_loader import activate_target_horizon, load_daily_data
from src.feature_cache import build_feature_cache_key, load_feature_cache, save_feature_cache
from src.feature_selector import FeatureSelector
from src.model import (
    ModelEnsemble,
    build_model,
    list_available_models,
    list_supported_models,
    normalize_feature_importance,
)
from src.model_params import (
    model_params_for_model,
    parse_hyperparameter_grid,
    parse_model_params_json,
    selected_model_params_from_summary,
)
from src.preprocessing import DEFAULT_WINSORIZE_QUANTILE, apply_cross_sectional_preprocessing
from src.preprocessing_cache import build_preprocessing_cache_key, load_preprocessing_cache, save_preprocessing_cache
from src.progress import create_progress_bar, format_duration
from src.project_paths import resolve_project_path
from src.reporting import calculate_prediction_metrics, write_training_report
from src.runtime_config import (
    CORE_MODEL_SUITE,
    DEFAULT_EXPERIMENT_MODEL_ROOT,
    DEFAULT_EXPERIMENT_OUTPUT_ROOT,
    DEFAULT_EXPERIMENT_PRESETS,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_UNIVERSE,
    DEFAULT_SAMPLE_START_DATE,
    LINEAR_MODEL_SUITE,
    RANDOM_FOREST_CONTROL_SUITE,
)
from src.time_series_pipeline import DEFAULT_HISTORY_WINDOW, strict_time_split_feature_engineering
from src.universe import get_symbol_sector_map, get_universe_symbols, list_supported_universes
from src.validation import run_walk_forward_validation
from src.validation_cache import build_validation_cache_key, load_validation_cache, save_validation_cache
from src.yfinance_loader import download_yfinance_to_csv


ALL_MODEL_SUITE = list(CORE_MODEL_SUITE)

EXPERIMENT_PRESETS = {
    "5d_all_models": {
        "target_horizon": 5,
        "model_names": ALL_MODEL_SUITE,
        "description": "5日目标 + 全模型组合",
    },
    "5d_linear_models": {
        "target_horizon": 5,
        "model_names": LINEAR_MODEL_SUITE,
        "description": "5日目标 + 线性模型组合",
    },
    "10d_all_models": {
        "target_horizon": 10,
        "model_names": ALL_MODEL_SUITE,
        "description": "10日目标 + 全模型组合",
    },
    "10d_linear_models": {
        "target_horizon": 10,
        "model_names": LINEAR_MODEL_SUITE,
        "description": "10日目标 + 线性模型组合",
    },
    "10d_random_forest_control": {
        "target_horizon": 10,
        "model_names": RANDOM_FOREST_CONTROL_SUITE,
        "description": "10日目标 + random_forest 对照组",
    },
}

EXPERIMENT_STAGE_LABELS = [
    ("activate_target", "Activate target"),
    ("feature_engineering", "Time split + feature engineering"),
    ("preprocess_train", "Cross-sectional preprocessing (train)"),
    ("preprocess_test", "Cross-sectional preprocessing (test)"),
    ("walk_forward", "Walk-forward validation"),
    ("final_feature_selection", "Final feature selection"),
    ("final_model_training", "Final model training"),
    ("prediction_and_report", "Prediction + report output"),
]


def parse_args() -> argparse.Namespace:
    """解析实验入口参数。

    这里故意不暴露太多复杂选项，重点是把“批量对比实验”做简单：

    - 你可以直接跑全部预设；
    - 也可以只挑其中几组跑；
    - 每一组实验都会自动落到自己的输出目录。
    """

    parser = argparse.ArgumentParser(description="Run multiple experiment presets without touching main.py.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_PRIMARY_DATA_PATH,
        help="实验使用的数据路径。",
    )
    parser.add_argument(
        "--model-root-dir",
        type=str,
        default=DEFAULT_EXPERIMENT_MODEL_ROOT,
        help="各组实验模型和特征文件的根目录。",
    )
    parser.add_argument(
        "--output-root-dir",
        type=str,
        default=DEFAULT_EXPERIMENT_OUTPUT_ROOT,
        help="各组实验预测结果和报告的根目录。",
    )
    parser.add_argument(
        "--experiment-names",
        nargs="+",
        default=list(DEFAULT_EXPERIMENT_PRESETS),
        help="要运行的实验名字列表，例如 5d_all_models 10d_linear_models。",
    )
    parser.add_argument("--sample-start-date", type=str, default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", type=str, default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--top-n", type=int, default=50, help="最终保留的特征数量。")
    parser.add_argument("--missing-threshold", type=float, default=0.5, help="缺失率过滤阈值。")
    parser.add_argument("--variance-threshold", type=float, default=0.001, help="低方差过滤阈值。")
    parser.add_argument("--correlation-threshold", type=float, default=0.95, help="高相关过滤阈值。")
    parser.add_argument(
        "--feature-score-method",
        type=str,
        choices=["correlation", "mutual_info"],
        default="correlation",
        help="特征打分方式。",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="walk-forward fold 数量。")
    parser.add_argument(
        "--validation-score-metric",
        type=str,
        choices=["pearson_corr", "pearson_ic_mean", "spearman_ic_mean", "long_short_return"],
        default="pearson_ic_mean",
        help="模型集成权重依据的验证指标。",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--model-params-json",
        type=str,
        default="",
        help='按模型名覆盖参数的 JSON，例如 {"ridge":{"alpha":2.0}}。',
    )
    parser.add_argument(
        "--hyperparameter-grid",
        type=str,
        default="",
        help="轻量超参网格，例如 alpha=0.1,1,10;elastic_net_l1_ratio=0.1,0.5。",
    )
    parser.add_argument("--max-grid-combinations", type=int, default=12, help="每个模型最多验证多少组超参组合。")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="批量实验软超时秒数；0 表示不限制。")
    parser.add_argument("--fetch-yfinance", action="store_true", help="运行前是否重新下载真实数据。")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="缓存目录。当前用于保存横截面预处理后的 train/test 表。",
    )
    parser.add_argument(
        "--disable-preprocessing-cache",
        action="store_true",
        help="关闭横截面预处理缓存。",
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="自定义下载股票列表。")
    parser.add_argument(
        "--universe",
        type=str,
        choices=list_supported_universes(),
        default=DEFAULT_PRIMARY_UNIVERSE,
        help="如果不手动给 --symbols，就下载这个股票池。",
    )
    parser.add_argument("--start-date", type=str, default=DEFAULT_SAMPLE_START_DATE, help="下载开始日期。")
    parser.add_argument("--end-date", type=str, default=None, help="下载结束日期。")
    parser.add_argument("--auto-adjust", action="store_true", help="下载时是否自动复权。")
    return parser.parse_args()


def configure_runtime_warning_display() -> None:
    """降低重复 warning 对进度条可读性的干扰。

    这里保留重要 warning，同时把某些已知会高频重复的 `pandas`
    数值 warning 收敛成“只显示一次”。

    这样做的目的主要是：

    - 避免进度条被同一类 warning 连续刷屏；
    - 让用户更容易判断当前到底跑到了哪个阶段；
    - 不改变计算结果本身。
    """

    warnings.filterwarnings(
        "once",
        message="overflow encountered in square",
        category=RuntimeWarning,
        module=r"pandas\.core\.nanops",
    )


def finish_stage_progress(
    progress_bar,
    stage_index: int,
    total_stages: int,
    stage_label: str,
    stage_elapsed: float,
    experiment_elapsed: float,
) -> None:
    """更新实验阶段进度条，并把 ETA 显示得更直观一些。"""

    average_stage_time = experiment_elapsed / max(stage_index, 1)
    estimated_remaining = average_stage_time * max(total_stages - stage_index, 0)
    progress_bar.update(1)
    progress_bar.set_postfix_str(
        (
            f"{stage_label} {format_duration(stage_elapsed)} | "
            f"total {format_duration(experiment_elapsed)} | "
            f"est left {format_duration(estimated_remaining)}"
        )
    )


def write_timing_artifacts(
    output_dir: Path,
    stage_timing_df: pd.DataFrame,
    final_model_timing_df: pd.DataFrame,
) -> None:
    """把阶段耗时与最终模型耗时额外落盘。

    这样即使训练已经结束，你也可以回头看：

    - 到底最慢的是哪一个大阶段；
    - 最终训练阶段里是哪一个模型最慢；
    - 下次如果想优化速度，应该优先动哪里。
    """

    if not stage_timing_df.empty:
        stage_timing_df.to_csv(output_dir / "stage_timing.csv", index=False)
        stage_markdown = "# Stage Timing\n\n" + stage_timing_df.to_markdown(index=False) + "\n"
        (output_dir / "stage_timing.md").write_text(stage_markdown, encoding="utf-8")

    if not final_model_timing_df.empty:
        final_model_timing_df.to_csv(output_dir / "final_model_timing.csv", index=False)
        model_markdown = "# Final Model Timing\n\n" + final_model_timing_df.to_markdown(index=False) + "\n"
        (output_dir / "final_model_timing.md").write_text(model_markdown, encoding="utf-8")


def summarize_weighted_feature_importance(importance_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """把多个模型的重要性按最终模型权重融合。"""

    if not importance_frames:
        return pd.DataFrame(columns=["feature", "importance"])

    merged = pd.concat(importance_frames, ignore_index=True)
    return (
        merged.groupby("feature", as_index=False)["weighted_importance"]
        .sum()
        .rename(columns={"weighted_importance": "importance"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def resolve_requested_models(requested_models: list[str]) -> list[str]:
    """过滤当前环境里真正可用的模型。"""

    supported_models = set(list_supported_models())
    available_models = set(list_available_models())
    unknown_models = [model_name for model_name in requested_models if model_name not in supported_models]

    if unknown_models:
        raise ValueError(
            f"Unsupported model names: {unknown_models}. "
            f"Supported models: {sorted(supported_models)}"
        )

    resolved_models = [model_name for model_name in requested_models if model_name in available_models]
    if not resolved_models:
        raise ValueError(
            "None of the requested models are available. "
            f"Available models in the current environment: {sorted(available_models)}"
        )
    return resolved_models


def check_deadline(deadline: float | None, stage: str) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError(f"Experiment timeout exceeded during {stage}.")


def finalize_model_weights(model_names: list[str], model_weights: dict[str, float]) -> dict[str, float]:
    """把验证期权重补成最终可直接集成的版本。"""

    final_weights = {model_name: float(model_weights.get(model_name, 0.0)) for model_name in model_names}
    total_weight = sum(final_weights.values())

    if total_weight <= 0:
        uniform_weight = 1.0 / len(model_names)
        return {model_name: uniform_weight for model_name in model_names}

    return {model_name: weight / total_weight for model_name, weight in final_weights.items()}


def prepare_raw_data(args: argparse.Namespace) -> pd.DataFrame:
    """准备实验共用的原始数据。

    这里先统一下载 / 读取一次数据，后面每组实验只切换目标周期和模型组合，
    不重复做无意义的数据 I/O。
    """

    data_path = resolve_project_path(args.data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    direct_run_mode = len(sys.argv) == 1
    auto_fetch_for_direct_run = direct_run_mode and (not data_path.exists()) and (not args.fetch_yfinance)

    if args.fetch_yfinance or auto_fetch_for_direct_run:
        download_symbols = list(args.symbols) if args.symbols else get_universe_symbols(args.universe)
        symbol_preview = ", ".join(download_symbols[:10])
        if len(download_symbols) > 10:
            symbol_preview += ", ..."
        print(f"[Info] Downloading experiment data for {len(download_symbols)} symbols: {symbol_preview}")
        download_yfinance_to_csv(
            symbols=download_symbols,
            output_path=data_path,
            start_date=args.start_date,
            end_date=args.end_date,
            auto_adjust=args.auto_adjust,
        )

    if not data_path.exists():
        raise FileNotFoundError(f"Experiment data file does not exist: {data_path}")

    raw_data = load_daily_data(data_path)
    raw_data["date"] = pd.to_datetime(raw_data["date"])

    if args.sample_start_date:
        sample_start_date = pd.Timestamp(args.sample_start_date)
        raw_data = raw_data[raw_data["date"] >= sample_start_date].copy()
        if raw_data.empty:
            raise ValueError("No rows remain after applying sample_start_date in experiment runner.")

    if "sector" not in raw_data.columns or raw_data["sector"].isna().all():
        sector_map = get_symbol_sector_map(sorted(raw_data["instrument_id"].dropna().unique()))
        if sector_map:
            raw_data["sector"] = raw_data["instrument_id"].map(sector_map).fillna("Unknown")

    return raw_data


def run_single_experiment(
    experiment_name: str,
    experiment_config: dict,
    raw_data_base: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    """执行单组实验。

    每组实验都会：

    1. 激活自己的目标周期；
    2. 用自己的模型组合做 walk-forward；
    3. 保存独立的模型目录、输出目录和训练报告；
    4. 返回一条摘要记录，供最终汇总对比。
    """

    model_dir = resolve_project_path(args.model_root_dir) / experiment_name
    output_dir = resolve_project_path(args.output_root_dir) / experiment_name
    cache_root = resolve_project_path(args.cache_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    target_horizon = int(experiment_config["target_horizon"])
    requested_models = list(experiment_config["model_names"])
    resolved_model_names = resolve_requested_models(requested_models)
    model_params_by_name = parse_model_params_json(args.model_params_json)
    hyperparameter_grid_by_name = parse_hyperparameter_grid(
        args.hyperparameter_grid,
        resolved_model_names,
        max_combinations_per_model=max(1, args.max_grid_combinations),
    )
    timeout_deadline = time.perf_counter() + args.timeout_seconds if args.timeout_seconds > 0 else None
    total_stage_count = len(EXPERIMENT_STAGE_LABELS)
    experiment_start_time = time.perf_counter()
    stage_timing_records: list[dict] = []
    final_model_timing_records: list[dict] = []
    experiment_stage_progress = create_progress_bar(
        total=total_stage_count,
        description=f"{experiment_name}: stages",
        enabled=True,
    )

    stage_start_time = time.perf_counter()
    raw_data, target_column = activate_target_horizon(raw_data_base, target_horizon=target_horizon)
    stage_elapsed = time.perf_counter() - stage_start_time
    experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 1,
            "stage_key": "activate_target",
            "stage_label": "Activate target",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": target_column,
        }
    )
    finish_stage_progress(
        experiment_stage_progress,
        stage_index=1,
        total_stages=total_stage_count,
        stage_label="activate target",
        stage_elapsed=stage_elapsed,
        experiment_elapsed=experiment_elapsed,
    )

    print(f"[Info] Running experiment: {experiment_name} | target={target_column} | models={resolved_model_names}")

    feature_cache_key = build_feature_cache_key(
        data_path=resolve_project_path(args.data_path),
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
    )
    stage_start_time = time.perf_counter()
    cached_features = load_feature_cache(cache_root=cache_root, cache_key=feature_cache_key)
    if cached_features is not None:
        train_df, test_df, feature_columns, feature_metadata = cached_features
        feature_cache_status = "hit"
        feature_detail = f"{len(feature_columns)} feature columns | cache hit"
    else:
        train_df, test_df, feature_columns, feature_metadata = strict_time_split_feature_engineering(
            raw_data=raw_data,
            test_size=args.test_size,
            test_start_date=args.oos_start_date,
            history_window=DEFAULT_HISTORY_WINDOW,
            target_horizon=target_horizon,
            show_progress=True,
        )
        save_feature_cache(
            cache_root=cache_root,
            cache_key=feature_cache_key,
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            feature_metadata=feature_metadata,
            metadata={
                "target_horizon": target_horizon,
                "history_window": DEFAULT_HISTORY_WINDOW,
            },
        )
        feature_cache_status = "miss_written"
        feature_detail = f"{len(feature_columns)} feature columns | cache miss"
    stage_elapsed = time.perf_counter() - stage_start_time
    experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 2,
            "stage_key": "feature_engineering",
            "stage_label": "Time split + feature engineering",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": feature_detail,
        }
    )
    finish_stage_progress(
        experiment_stage_progress,
        stage_index=2,
        total_stages=total_stage_count,
        stage_label="feature engineering",
        stage_elapsed=stage_elapsed,
        experiment_elapsed=experiment_elapsed,
    )

    preprocessing_cache_enabled = not args.disable_preprocessing_cache
    preprocessing_cache_key = build_preprocessing_cache_key(
        data_path=resolve_project_path(args.data_path),
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        feature_columns=feature_columns,
        apply_preprocessing=True,
        apply_neutralization=True,
        winsorize_quantile=DEFAULT_WINSORIZE_QUANTILE,
    )

    cached_preprocessing = None
    if preprocessing_cache_enabled:
        stage_start_time = time.perf_counter()
        cached_preprocessing = load_preprocessing_cache(cache_root=cache_root, cache_key=preprocessing_cache_key)
        stage_elapsed = time.perf_counter() - stage_start_time
    else:
        stage_elapsed = 0.0

    if cached_preprocessing is not None:
        train_df, test_df, preprocessing_summary = cached_preprocessing
        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = "hit"
        preprocessing_summary["cache_key"] = preprocessing_cache_key

        experiment_elapsed = time.perf_counter() - experiment_start_time
        stage_timing_records.append(
            {
                "stage_order": 3,
                "stage_key": "preprocess_train",
                "stage_label": "Cross-sectional preprocessing (train)",
                "elapsed_sec": float(stage_elapsed),
                "elapsed_readable": format_duration(stage_elapsed),
                "details": f"{len(train_df)} rows | cache hit",
            }
        )
        finish_stage_progress(
            experiment_stage_progress,
            stage_index=3,
            total_stages=total_stage_count,
            stage_label="preprocess train (cache hit)",
            stage_elapsed=stage_elapsed,
            experiment_elapsed=experiment_elapsed,
        )

        stage_timing_records.append(
            {
                "stage_order": 4,
                "stage_key": "preprocess_test",
                "stage_label": "Cross-sectional preprocessing (test)",
                "elapsed_sec": 0.0,
                "elapsed_readable": format_duration(0.0),
                "details": f"{len(test_df)} rows | cache hit",
            }
        )
        finish_stage_progress(
            experiment_stage_progress,
            stage_index=4,
            total_stages=total_stage_count,
            stage_label="preprocess test (cache hit)",
            stage_elapsed=0.0,
            experiment_elapsed=experiment_elapsed,
        )
    else:
        stage_start_time = time.perf_counter()
        train_df, preprocessing_summary = apply_cross_sectional_preprocessing(
            train_df,
            feature_columns=feature_columns,
            show_progress=True,
        )
        stage_elapsed = time.perf_counter() - stage_start_time
        experiment_elapsed = time.perf_counter() - experiment_start_time
        stage_timing_records.append(
            {
                "stage_order": 3,
                "stage_key": "preprocess_train",
                "stage_label": "Cross-sectional preprocessing (train)",
                "elapsed_sec": float(stage_elapsed),
                "elapsed_readable": format_duration(stage_elapsed),
                "details": f"{len(train_df)} rows" + (" | cache miss" if preprocessing_cache_enabled else " | cache disabled"),
            }
        )
        finish_stage_progress(
            experiment_stage_progress,
            stage_index=3,
            total_stages=total_stage_count,
            stage_label="preprocess train",
            stage_elapsed=stage_elapsed,
            experiment_elapsed=experiment_elapsed,
        )

        stage_start_time = time.perf_counter()
        test_df, _ = apply_cross_sectional_preprocessing(
            test_df,
            feature_columns=feature_columns,
            show_progress=True,
        )
        stage_elapsed = time.perf_counter() - stage_start_time
        experiment_elapsed = time.perf_counter() - experiment_start_time
        stage_timing_records.append(
            {
                "stage_order": 4,
                "stage_key": "preprocess_test",
                "stage_label": "Cross-sectional preprocessing (test)",
                "elapsed_sec": float(stage_elapsed),
                "elapsed_readable": format_duration(stage_elapsed),
                "details": f"{len(test_df)} rows" + (" | cache miss" if preprocessing_cache_enabled else " | cache disabled"),
            }
        )
        finish_stage_progress(
            experiment_stage_progress,
            stage_index=4,
            total_stages=total_stage_count,
            stage_label="preprocess test",
            stage_elapsed=stage_elapsed,
            experiment_elapsed=experiment_elapsed,
        )

        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = "disabled" if not preprocessing_cache_enabled else "miss_written"
        preprocessing_summary["cache_key"] = preprocessing_cache_key if preprocessing_cache_enabled else None

        if preprocessing_cache_enabled:
            save_preprocessing_cache(
                cache_root=cache_root,
                cache_key=preprocessing_cache_key,
                train_df=train_df,
                test_df=test_df,
                preprocessing_summary=preprocessing_summary,
                metadata={
                    "target_horizon": target_horizon,
                    "feature_count": len(feature_columns),
                    "winsorize_quantile": DEFAULT_WINSORIZE_QUANTILE,
                    "apply_neutralization": True,
                },
            )

    selector_config = {
        "missing_threshold": args.missing_threshold,
        "variance_threshold": args.variance_threshold,
        "correlation_threshold": args.correlation_threshold,
        "top_n": args.top_n,
        "score_method": args.feature_score_method,
        "random_state": args.random_state,
    }

    validation_context = {
        "model_params": model_params_by_name,
        "hyperparameter_grid": hyperparameter_grid_by_name,
        "max_grid_combinations": args.max_grid_combinations,
    }
    validation_cache_key = build_validation_cache_key(
        data_path=resolve_project_path(args.data_path),
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        feature_columns=feature_columns,
        model_names=resolved_model_names,
        selector_config=selector_config,
        n_splits=args.n_splits,
        random_state=args.random_state,
        score_metric=args.validation_score_metric,
        apply_preprocessing=True,
        apply_neutralization=True,
        winsorize_quantile=DEFAULT_WINSORIZE_QUANTILE,
        extra_context=validation_context,
    )
    stage_start_time = time.perf_counter()
    check_deadline(timeout_deadline, f"{experiment_name}: walk-forward validation")
    cached_validation = load_validation_cache(cache_root=cache_root, cache_key=validation_cache_key)
    if cached_validation is not None:
        fold_metrics_df, model_summary_df, model_weights = cached_validation
        validation_detail = f"{args.n_splits} folds × {len(resolved_model_names)} models | cache hit"
    else:
        fold_metrics_df, model_summary_df, model_weights = run_walk_forward_validation(
            train_df=train_df,
            feature_columns=feature_columns,
            model_names=resolved_model_names,
            selector_config=selector_config,
            model_params_by_name=model_params_by_name,
            hyperparameter_grid_by_name=hyperparameter_grid_by_name,
            random_state=args.random_state,
            n_splits=args.n_splits,
            purge_days=target_horizon,
            score_metric=args.validation_score_metric,
            show_progress=True,
        )
        save_validation_cache(
            cache_root=cache_root,
            cache_key=validation_cache_key,
            fold_metrics_df=fold_metrics_df,
            model_summary_df=model_summary_df,
            model_weights=model_weights,
            metadata={
                "target_horizon": target_horizon,
                "model_names": resolved_model_names,
                "n_splits": args.n_splits,
                "model_params": model_params_by_name,
                "hyperparameter_grid": hyperparameter_grid_by_name,
                "max_grid_combinations": args.max_grid_combinations,
            },
        )
        validation_detail = f"{args.n_splits} folds × {len(resolved_model_names)} models | cache miss"
    stage_elapsed = time.perf_counter() - stage_start_time
    experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 5,
            "stage_key": "walk_forward",
            "stage_label": "Walk-forward validation",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": validation_detail,
        }
    )
    finish_stage_progress(
        experiment_stage_progress,
        stage_index=5,
        total_stages=total_stage_count,
        stage_label="walk-forward",
        stage_elapsed=stage_elapsed,
        experiment_elapsed=experiment_elapsed,
    )

    stage_start_time = time.perf_counter()
    selector = FeatureSelector(**selector_config)
    selector.fit(
        train_df[feature_columns],
        train_df["y"],
        dates=train_df["date"],
    )

    X_train_full = selector.transform(train_df[feature_columns])
    y_train_full = train_df["y"].reset_index(drop=True)
    X_test = selector.transform(test_df[feature_columns])
    stage_elapsed = time.perf_counter() - stage_start_time
    experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 6,
            "stage_key": "final_feature_selection",
            "stage_label": "Final feature selection",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": f"{len(selector.selected_features_)} selected features",
        }
    )
    finish_stage_progress(
        experiment_stage_progress,
        stage_index=6,
        total_stages=total_stage_count,
        stage_label="final feature selection",
        stage_elapsed=stage_elapsed,
        experiment_elapsed=experiment_elapsed,
    )

    final_model_weights = finalize_model_weights(resolved_model_names, model_weights)
    ensemble = ModelEnsemble()
    model_params: Dict[str, Dict] = {}
    weighted_importance_frames: List[pd.DataFrame] = []

    stage_start_time = time.perf_counter()
    final_model_progress = create_progress_bar(
        total=len(resolved_model_names),
        description=f"{experiment_name}: final model training",
        enabled=True,
    )
    for model_index, model_name in enumerate(resolved_model_names, start=1):
        check_deadline(timeout_deadline, f"{experiment_name}: final model training {model_name}")
        model_start_time = time.perf_counter()
        selected_params = selected_model_params_from_summary(model_summary_df, model_name)
        model_wrapper = build_model(
            model_name=model_name,
            random_state=args.random_state,
            params=selected_params or model_params_for_model(model_name, model_params_by_name),
        )
        model_wrapper.fit(X_train_full, y_train_full)
        model_wrapper.save(model_dir / f"{model_name}_model.joblib")
        model_elapsed = time.perf_counter() - model_start_time

        current_weight = final_model_weights.get(model_name, 0.0)
        ensemble.add_model(model_name, model_wrapper, weight=current_weight)
        model_params[model_name] = model_wrapper.get_params()

        raw_importance_df = model_wrapper.get_feature_importance(selector.selected_features_, model_name=model_name)
        normalized_importance_df = normalize_feature_importance(raw_importance_df, model_weight=current_weight)
        weighted_importance_frames.append(normalized_importance_df)
        final_model_timing_records.append(
            {
                "model": model_name,
                "elapsed_sec": float(model_elapsed),
                "elapsed_readable": format_duration(model_elapsed),
                "ensemble_weight": float(current_weight),
            }
        )
        average_model_time = (time.perf_counter() - stage_start_time) / max(model_index, 1)
        estimated_model_remaining = average_model_time * max(len(resolved_model_names) - model_index, 0)
        final_model_progress.update(1)
        final_model_progress.set_postfix_str(
            (
                f"{model_name} {format_duration(model_elapsed)} | "
                f"est left {format_duration(estimated_model_remaining)}"
            )
        )
    final_model_progress.close()
    stage_elapsed = time.perf_counter() - stage_start_time
    experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 7,
            "stage_key": "final_model_training",
            "stage_label": "Final model training",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": f"{len(resolved_model_names)} models",
        }
    )
    finish_stage_progress(
        experiment_stage_progress,
        stage_index=7,
        total_stages=total_stage_count,
        stage_label="final model training",
        stage_elapsed=stage_elapsed,
        experiment_elapsed=experiment_elapsed,
    )

    stage_start_time = time.perf_counter()
    test_predictions = ensemble.predict(X_test)

    predictions_df = test_df[["date", "instrument_id"]].copy()
    predictions_df["date"] = pd.to_datetime(predictions_df["date"]).dt.strftime("%Y-%m-%d")
    predictions_df["predicted_y"] = test_predictions
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    predictions_with_actual_df = test_df[["date", "instrument_id", "y"]].copy()
    predictions_with_actual_df["date"] = pd.to_datetime(predictions_with_actual_df["date"]).dt.strftime("%Y-%m-%d")
    predictions_with_actual_df["predicted_y"] = test_predictions
    predictions_with_actual_df.to_csv(output_dir / "test_predictions_with_actual.csv", index=False)

    test_metrics = calculate_prediction_metrics(predictions_with_actual_df)

    fold_metrics_df.to_csv(output_dir / "walk_forward_fold_metrics.csv", index=False)
    model_summary_df.to_csv(output_dir / "walk_forward_model_summary.csv", index=False)
    pd.DataFrame(
        [{"model": model_name, "weight": weight} for model_name, weight in final_model_weights.items()]
    ).to_csv(model_dir / "model_weights.csv", index=False)

    pd.DataFrame({"feature": selector.selected_features_}).to_csv(model_dir / "selected_features.csv", index=False)
    selector.get_top_features(top_k=20).to_csv(model_dir / "selected_feature_scores.csv", index=False)
    selector.save(model_dir / "feature_selector.json")

    weighted_feature_importance_summary = summarize_weighted_feature_importance(weighted_importance_frames)
    if weighted_importance_frames:
        pd.concat(weighted_importance_frames, ignore_index=True).to_csv(
            model_dir / "feature_importance_by_model.csv",
            index=False,
        )
    if not weighted_feature_importance_summary.empty:
        weighted_feature_importance_summary.to_csv(model_dir / "feature_importance.csv", index=False)

    dataset_summary = {
        "data_path": str(resolve_project_path(args.data_path)),
        "min_date": str(pd.to_datetime(raw_data["date"]).min().date()),
        "max_date": str(pd.to_datetime(raw_data["date"]).max().date()),
        "instrument_count": int(raw_data["instrument_id"].nunique()),
        "n_rows": int(len(train_df) + len(test_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_min_date": str(pd.to_datetime(train_df["date"]).min().date()),
        "train_max_date": str(pd.to_datetime(train_df["date"]).max().date()),
        "test_min_date": str(pd.to_datetime(test_df["date"]).min().date()),
        "test_max_date": str(pd.to_datetime(test_df["date"]).max().date()),
        "n_splits": int(args.n_splits),
        "sample_start_date": args.sample_start_date,
        "oos_start_date_used": args.oos_start_date,
        "target_horizon": target_horizon,
        "target_column": target_column,
        "universe": args.universe if args.fetch_yfinance and not args.symbols else "custom_symbols_or_csv",
        "resolved_model_count": len(resolved_model_names),
        "hyperparameter_grid": hyperparameter_grid_by_name,
        "max_grid_combinations": args.max_grid_combinations,
        "timeout_seconds": args.timeout_seconds,
    }

    selector_report_summary = {
        "missing_threshold": selector.missing_threshold,
        "variance_threshold": selector.variance_threshold,
        "correlation_threshold": selector.correlation_threshold,
        "top_n": selector.top_n,
        "score_method": selector.score_method,
        "stage_feature_counts": selector.stage_feature_counts_,
        "selected_feature_count": len(selector.selected_features_),
        "validation_score_metric": args.validation_score_metric,
    }

    stage_elapsed = time.perf_counter() - stage_start_time
    total_experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 8,
            "stage_key": "prediction_and_report",
            "stage_label": "Prediction + report output",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": "predictions, reports, weights, importances",
        }
    )
    finish_stage_progress(
        experiment_stage_progress,
        stage_index=8,
        total_stages=total_stage_count,
        stage_label="prediction + report",
        stage_elapsed=stage_elapsed,
        experiment_elapsed=total_experiment_elapsed,
    )
    experiment_stage_progress.close()

    stage_timing_df = pd.DataFrame(stage_timing_records)
    final_model_timing_df = pd.DataFrame(final_model_timing_records)
    write_timing_artifacts(
        output_dir=output_dir,
        stage_timing_df=stage_timing_df,
        final_model_timing_df=final_model_timing_df,
    )

    write_training_report(
        output_path=output_dir / "training_report.md",
        dataset_summary=dataset_summary,
        feature_metadata=feature_metadata,
        preprocessing_summary=preprocessing_summary,
        selector_summary=selector_report_summary,
        test_metrics=test_metrics,
        model_params=model_params,
        top_score_features=selector.get_top_features(top_k=10),
        top_importance_features=weighted_feature_importance_summary.head(10),
        validation_summary_df=model_summary_df,
        model_weights=final_model_weights,
        stage_timing_df=stage_timing_df,
        final_model_timing_df=final_model_timing_df,
    )

    summary_record = {
        "experiment_name": experiment_name,
        "description": experiment_config["description"],
        "target_horizon": target_horizon,
        "models": ",".join(resolved_model_names),
        "selected_feature_count": len(selector.selected_features_),
        "total_runtime_sec": float(total_experiment_elapsed),
        "total_runtime_readable": format_duration(total_experiment_elapsed),
        **test_metrics,
    }
    return summary_record


def write_experiment_summary(summary_df: pd.DataFrame, output_root_dir: Path) -> None:
    """把全部实验摘要写成 CSV 和 Markdown。"""

    output_root_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_root_dir / "experiment_summary.csv", index=False)

    if summary_df.empty:
        markdown_text = "# Experiment Summary\n\n_No experiments were executed._\n"
    else:
        markdown_text = (
            "# Experiment Summary\n\n"
            "这份摘要表用于快速比较不同目标周期和不同模型组合的 OOS 表现。\n\n"
            + summary_df.to_markdown(index=False)
            + "\n"
        )

    with (output_root_dir / "experiment_summary.md").open("w", encoding="utf-8") as file:
        file.write(markdown_text)


def main() -> None:
    """执行多组实验。"""

    configure_runtime_warning_display()
    args = parse_args()
    unknown_experiments = [name for name in args.experiment_names if name not in EXPERIMENT_PRESETS]
    if unknown_experiments:
        raise ValueError(
            f"Unknown experiment names: {unknown_experiments}. "
            f"Supported names: {list(EXPERIMENT_PRESETS.keys())}"
        )

    raw_data_base = prepare_raw_data(args)
    output_root_dir = resolve_project_path(args.output_root_dir)

    summary_records: list[dict] = []
    experiment_timing_records: list[dict] = []
    all_experiments_start = time.perf_counter()
    experiment_progress = create_progress_bar(
        total=len(args.experiment_names),
        description="Experiment presets",
        enabled=True,
    )

    for experiment_index, experiment_name in enumerate(args.experiment_names, start=1):
        single_experiment_start = time.perf_counter()
        summary_records.append(
            run_single_experiment(
                experiment_name=experiment_name,
                experiment_config=EXPERIMENT_PRESETS[experiment_name],
                raw_data_base=raw_data_base,
                args=args,
            )
        )
        experiment_elapsed = time.perf_counter() - single_experiment_start
        total_elapsed = time.perf_counter() - all_experiments_start
        average_experiment_time = total_elapsed / max(experiment_index, 1)
        estimated_remaining = average_experiment_time * max(len(args.experiment_names) - experiment_index, 0)
        experiment_timing_records.append(
            {
                "experiment_name": experiment_name,
                "elapsed_sec": float(experiment_elapsed),
                "elapsed_readable": format_duration(experiment_elapsed),
            }
        )
        experiment_progress.update(1)
        experiment_progress.set_postfix_str(
            (
                f"last {experiment_name} {format_duration(experiment_elapsed)} | "
                f"total {format_duration(total_elapsed)} | "
                f"est left {format_duration(estimated_remaining)}"
            )
        )
    experiment_progress.close()

    summary_df = pd.DataFrame(summary_records).sort_values(
        ["target_horizon", "pearson_ic_mean", "long_short_return"],
        ascending=[True, False, False],
    )
    write_experiment_summary(summary_df, output_root_dir=output_root_dir)
    pd.DataFrame(experiment_timing_records).to_csv(output_root_dir / "experiment_timing_summary.csv", index=False)

    print("[Info] Experiment runner finished.")
    print(f"[Info] Summary saved to: {output_root_dir / 'experiment_summary.csv'}")
    print(f"[Info] Summary report saved to: {output_root_dir / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
