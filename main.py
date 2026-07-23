"""项目主入口。

这个版本不再只是一个“教学演示脚本”，而是更接近真实研究流程：

1. 读取真实数据或通过 yfinance 下载最新数据；
2. 明确划分 in-sample 训练期和 out-of-sample 测试期；
3. 在训练期内部使用 walk-forward validation；
4. 同时训练多种模型，包括树模型和线性 baseline；
5. 根据验证表现给模型分配权重；
6. 在最终 OOS 测试集上做加权集成预测；
7. 输出模型、预测结果、验证结果和特征重要性报告。
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.alpha191 import list_supported_alpha_factors
from src.data_quality import build_corporate_action_audit, build_universe_coverage_audit
from src.data_loader import PRICE_ADJUSTMENT_MODES, SUPPORTED_TARGET_HORIZONS, activate_target_horizon, load_daily_data
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
from src.progress import optional_progress
from src.project_paths import resolve_project_path
from src.project_paths import PROJECT_ROOT
from src.provenance import (
    build_data_fingerprint,
    collect_environment,
    project_relative_path,
    sanitize_arguments,
    sanitize_command,
    utc_now_iso,
    write_run_manifest,
)
from src.reporting import calculate_prediction_metrics, write_training_report
from src.runtime_config import (
    CORE_MODEL_SUITE,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_PRIMARY_UNIVERSE,
    DEFAULT_SAMPLE_START_DATE,
)
from src.time_series_pipeline import DEFAULT_HISTORY_WINDOW, strict_time_split_feature_engineering
from src.universe import get_symbol_sector_map, get_universe_symbols, list_supported_universes
from src.validation import run_walk_forward_validation
from src.validation_cache import build_validation_cache_key, load_validation_cache, save_validation_cache
from src.yfinance_loader import download_yfinance_to_csv


DEFAULT_MODELS = list(CORE_MODEL_SUITE)


def configure_runtime_warning_display() -> None:
    """收敛重复数值 warning，避免刷屏破坏进度条可读性。

    Alpha191 里有少数公式可能产生极端大数，`pandas` 在计算方差时会
    打印 `overflow encountered in square`。这类 warning 对结果有提示价值，
    但重复上百次没有额外信息，只会让用户看不清当前进度。
    """

    warnings.filterwarnings(
        "once",
        message="overflow encountered in square",
        category=RuntimeWarning,
        module=r"pandas\.core\.nanops",
    )
    warnings.filterwarnings(
        "once",
        message="overflow encountered in reduce",
        category=RuntimeWarning,
        module=r"numpy\.core\._methods",
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这次增加了几个更偏“实战研究”的参数：

    - `models`：选择参与比较和集成的模型列表；
    - `n_splits`：walk-forward fold 数量；
    - `validation_score_metric`：决定模型权重时采用哪种验证指标；
    - `oos_start_date`：显式指定 out-of-sample 测试集开始日期。
    """

    parser = argparse.ArgumentParser(description="Train a more practical stock return prediction pipeline.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_PRIMARY_DATA_PATH,
        help="原始数据 CSV 路径。直接运行 main.py 时，默认使用 us300 主线数据文件。",
    )
    parser.add_argument(
        "--universe-label",
        type=str,
        default=None,
        help="仅用于 provenance 的股票池标签；自备 CSV 时应显式填写，不能代替成分股证据。",
    )
    parser.add_argument("--model-dir", type=str, default="models", help="模型与特征文件的保存目录。")
    parser.add_argument("--output-dir", type=str, default="outputs", help="预测结果与报告的保存目录。")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="如果没有指定 --oos-start-date，就使用最后 test-size 比例的日期作为 OOS 测试集。",
    )
    parser.add_argument(
        "--sample-start-date",
        type=str,
        default=DEFAULT_SAMPLE_START_DATE,
        help="先对原始样本按起始日期截断；公开主线默认从 2022-01-01 开始。",
    )
    parser.add_argument(
        "--oos-start-date",
        type=str,
        default=None,
        help="显式指定 out-of-sample 测试集开始日期，例如 2025-06-01。优先级高于 --test-size。",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        choices=list(SUPPORTED_TARGET_HORIZONS),
        default=DEFAULT_PRIMARY_TARGET_HORIZON,
        help="选择当前训练要预测的未来收益周期，支持 1 / 5 / 10 个交易日。",
    )
    parser.add_argument(
        "--max-alpha",
        type=int,
        default=0,
        help="最多生成多少个 Alpha191 因子；0 表示使用全部已实现因子。可用于限制 smoke run 耗时。",
    )
    parser.add_argument(
        "--alpha-factors",
        nargs="*",
        default=None,
        help="显式指定 Alpha191 因子名；传 none/off/0 表示不生成 Alpha191。",
    )
    parser.add_argument("--top-n", type=int, default=50, help="最终保留的 Top N 特征数量。")
    parser.add_argument("--missing-threshold", type=float, default=0.5, help="缺失率过滤阈值。")
    parser.add_argument("--variance-threshold", type=float, default=0.001, help="低方差过滤阈值。")
    parser.add_argument("--correlation-threshold", type=float, default=0.95, help="高相关过滤阈值。")
    parser.add_argument(
        "--feature-score-method",
        type=str,
        choices=["correlation", "mutual_info"],
        default="correlation",
        help="Top N 特征的打分方式。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="参与对比与集成的模型列表，例如 lightgbm xgboost random_forest ridge lasso。",
    )
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
    parser.add_argument(
        "--max-grid-combinations",
        type=int,
        default=12,
        help="每个模型最多验证多少组超参组合。",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=0,
        help="可选的整次训练软超时秒数；0 表示不限制。",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="walk-forward validation 的 fold 数量。",
    )
    parser.add_argument(
        "--validation-score-metric",
        type=str,
        choices=["pearson_corr", "pearson_ic_mean", "spearman_ic_mean", "long_short_return"],
        default="pearson_ic_mean",
        help="根据哪种验证指标为模型分配集成权重。",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子。")
    parser.add_argument("--fetch-yfinance", action="store_true", help="通过 yfinance 下载真实股票数据。")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help=(
            "缓存目录。用于保存严格时间切分后的特征、"
            "横截面预处理结果和 walk-forward 验证结果。"
        ),
    )
    parser.add_argument(
        "--disable-preprocessing-cache",
        action="store_true",
        help="关闭横截面预处理缓存。",
    )
    parser.add_argument(
        "--refresh-caches",
        action="store_true",
        help=(
            "跳过现有特征、预处理和验证缓存并重新计算，然后写入新缓存。"
            "正式 canonical release run 使用此选项证明完整流水线已执行。"
        ),
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="yfinance 股票代码列表，例如 AAPL MSFT NVDA。")
    parser.add_argument(
        "--universe",
        type=str,
        choices=list_supported_universes(),
        default=DEFAULT_PRIMARY_UNIVERSE,
        help="如果没有手动提供 --symbols，就自动下载这个内置股票池。",
    )
    parser.add_argument("--start-date", type=str, default=DEFAULT_SAMPLE_START_DATE, help="下载开始日期。")
    parser.add_argument("--end-date", type=str, default=None, help="下载结束日期，默认到最近交易日。")
    parser.add_argument(
        "--auto-adjust",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "yfinance 下载时是否直接返回复权 OHLC；默认关闭以保留 raw OHLC "
            "和 Adj Close / Close 审计因子。"
        ),
    )
    parser.add_argument(
        "--price-adjustment-mode",
        choices=list(PRICE_ADJUSTMENT_MODES),
        default="vendor_adjusted",
        help="读取本地 CSV 时使用 vendor adjustment 复权，或显式保留 raw OHLC。",
    )
    return parser.parse_args()


def summarize_weighted_feature_importance(importance_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """把多个模型的“归一化后重要性”按模型权重融合。

    这一步比简单平均更贴近实战，因为：

    - 不同模型的重要性尺度不一样；
    - 模型本身的验证表现也不一样；
    - 所以先做模型内归一化，再按模型权重加权，会更合理。
    """

    if not importance_frames:
        return pd.DataFrame(columns=["feature", "importance"])

    merged = pd.concat(importance_frames, ignore_index=True)
    summary = (
        merged.groupby("feature", as_index=False)["weighted_importance"]
        .sum()
        .rename(columns={"weighted_importance": "importance"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def resolve_requested_models(requested_models: list[str]) -> list[str]:
    """过滤掉当前环境中不可用的模型。"""

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


def finalize_model_weights(model_names: list[str], model_weights: dict[str, float]) -> dict[str, float]:
    """把验证阶段得到的模型权重补齐成最终可直接用于集成的版本。"""

    final_weights: dict[str, float] = {}
    for model_name in model_names:
        candidate_weight = float(model_weights.get(model_name, 0.0))
        final_weights[model_name] = candidate_weight if pd.notna(candidate_weight) and candidate_weight > 0.0 else 0.0
    total_weight = sum(final_weights.values())

    if total_weight <= 0:
        uniform_weight = 1.0 / len(model_names)
        return {model_name: uniform_weight for model_name in model_names}

    return {model_name: weight / total_weight for model_name, weight in final_weights.items()}


def resolve_alpha_factor_names(alpha_factors: list[str] | None, max_alpha: int) -> list[str] | None:
    """解析本次运行需要生成的 Alpha191 范围。

    `None` 表示使用全部已实现 Alpha191；空列表表示显式关闭 Alpha191。
    将这个选择放进 CLI 和缓存键后，公开实验才能准确重现“明确的
    尺度不变 Alpha 清单”或“只用技术指标”等不同口径。
    """

    supported_alpha_names = list_supported_alpha_factors()
    supported_lookup = set(supported_alpha_names)

    if max_alpha < 0:
        raise ValueError("--max-alpha must be non-negative.")

    if alpha_factors is not None:
        requested_names: list[str] = []
        for raw_token in alpha_factors:
            requested_names.extend(token.strip() for token in raw_token.split(",") if token.strip())
        lowered = {name.lower() for name in requested_names}
        if lowered & {"none", "off", "0", "false"}:
            return []
        unknown_names = [name for name in requested_names if name not in supported_lookup]
        if unknown_names:
            raise ValueError(
                f"Unknown Alpha191 factor names: {unknown_names}. "
                f"Supported examples: {supported_alpha_names[:10]}"
            )
        alpha_names = requested_names
    elif max_alpha > 0:
        alpha_names = supported_alpha_names[:max_alpha]
    else:
        return None

    if max_alpha > 0:
        alpha_names = alpha_names[:max_alpha]
    return alpha_names


def check_deadline(deadline: float | None, stage: str) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError(f"Training timeout exceeded during {stage}.")


def main() -> None:
    """执行更接近实战研究流程的完整训练任务。"""

    run_started_at_utc = utc_now_iso()
    run_started_at_perf = time.perf_counter()
    stage_timing_records: list[dict[str, Any]] = []
    final_model_timing_records: list[dict[str, Any]] = []
    configure_runtime_warning_display()

    args = parse_args()
    timeout_deadline = time.perf_counter() + args.timeout_seconds if args.timeout_seconds > 0 else None
    direct_run_mode = len(sys.argv) == 1

    data_path = resolve_project_path(args.data_path)
    model_dir = resolve_project_path(args.model_dir)
    output_dir = resolve_project_path(args.output_dir)
    cache_root = resolve_project_path(args.cache_dir)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # 如果用户是直接点运行 main.py，而不是手动传命令行参数，
    # 这里启用一个更适合 VSCode / PyCharm 一键运行的默认行为：
    #
    # - 优先使用公开主线 `data/us_large_cap_300_daily.csv`
    # - 如果该文件不存在，就自动下载内置的 300 股票池
    #
    # 这样你直接点击“运行当前文件”时，也能得到一套更像实战研究的默认配置，
    # 这样不会在无参数时意外启动耗时很长、尚未形成公开结果的 US3000 实验。
    auto_fetch_for_direct_run = direct_run_mode and (not data_path.exists()) and (not args.fetch_yfinance)

    if direct_run_mode:
        print("[Info] No command-line arguments detected. Running the direct-run default profile.")

    download_stage_start = time.perf_counter()
    download_executed = bool(args.fetch_yfinance or auto_fetch_for_direct_run)
    if download_executed:
        download_symbols = list(args.symbols) if args.symbols else get_universe_symbols(args.universe)
        symbol_preview = ", ".join(download_symbols[:10])
        if len(download_symbols) > 10:
            symbol_preview += ", ..."
        print(f"[Info] Downloading Yahoo Finance data for {len(download_symbols)} symbols: {symbol_preview}")
        download_yfinance_to_csv(
            symbols=download_symbols,
            output_path=data_path,
            start_date=args.start_date,
            end_date=args.end_date,
            auto_adjust=args.auto_adjust,
        )
    stage_timing_records.append(
        {
            "stage": "download_market_data",
            "runtime_seconds": time.perf_counter() - download_stage_start,
            "status": "executed" if download_executed else "skipped_existing_input",
        }
    )

    # 这个项目现在只保留真实数据路径：
    # - 要么使用本地已有 CSV
    # - 要么通过 yfinance 下载真实数据
    # 不再自动生成 demo 数据，避免把教学数据和真实研究结果混在一起。
    if not data_path.exists() and not (args.fetch_yfinance or auto_fetch_for_direct_run):
        raise FileNotFoundError(
            f"Real data file does not exist: {data_path}. "
            "Please provide --data-path or use --fetch-yfinance."
        )

    data_stage_start = time.perf_counter()
    print(f"[Info] Loading data from: {data_path}")
    raw_data = load_daily_data(data_path, price_adjustment_mode=args.price_adjustment_mode)
    raw_data["date"] = pd.to_datetime(raw_data["date"])

    # 这里先把样本窗口截断到一个更贴近当前实战研究的起点。
    # 公开主线默认从 2024 年开始保留数据，原因是：
    #
    # - 太早的数据有时市场结构已经变化较大；
    # - 当前 canonical experiment 的训练和验证边界从 2024 年开始；
    # - 2026 年被单独保留为最终 OOS 审计。
    if args.sample_start_date:
        sample_start_date = pd.Timestamp(args.sample_start_date)
        raw_data = raw_data[raw_data["date"] >= sample_start_date].copy()
        if raw_data.empty:
            raise ValueError("No rows remain after applying --sample-start-date.")
        print(f"[Info] Keeping samples from {sample_start_date.date()} onward.")

    # 在这里显式激活“当前要训练的目标周期”。
    # 例如当你传 `--target-horizon 10` 时，项目会把 `y_10d` 映射到统一列名 `y`，
    # 这样后续特征筛选、训练、评估都能继续复用同一套代码接口。
    raw_data, target_column = activate_target_horizon(raw_data, target_horizon=args.target_horizon)
    print(f"[Info] Active target column: {target_column}")

    # 如果原始 CSV 没有直接给行业列，但股票代码又能在内置股票池映射表里找到，
    # 这里就自动补一个 sector 字段，供后面的横截面行业中性化使用。
    if "sector" not in raw_data.columns or raw_data["sector"].isna().all():
        sector_map = get_symbol_sector_map(sorted(raw_data["instrument_id"].dropna().unique()))
        if sector_map:
            raw_data["sector"] = raw_data["instrument_id"].map(sector_map).fillna("Unknown")

    # 数据质量审计放在耗时特征工程之前。即使后续模型训练失败，
    # 研究者仍然能看到公司行为调整和逐股历史覆盖的审计文件。
    corporate_action_audit_df, corporate_action_audit_summary = build_corporate_action_audit(
        raw_data,
        price_adjustment_mode=args.price_adjustment_mode,
    )
    corporate_action_audit_df.to_csv(output_dir / "corporate_action_audit.csv", index=False)
    universe_coverage_audit_df, universe_coverage_audit_summary = build_universe_coverage_audit(raw_data)
    universe_coverage_audit_df.to_csv(output_dir / "universe_coverage_audit.csv", index=False)

    # 如果用户没有显式给 OOS 起点，但当前样本已经覆盖到 2026 年，
    # 就自动把统一主线默认的 OOS 日期作为测试集开始日期。
    # 这样训练集对应 2024-2025，测试集对应 2026 年及之后。
    if args.oos_start_date is None:
        recommended_oos_start = pd.Timestamp(DEFAULT_OOS_START_DATE)
        if raw_data["date"].max() >= recommended_oos_start:
            effective_oos_start_date = str(recommended_oos_start.date())
            print(f"[Info] Auto-setting OOS start date to: {effective_oos_start_date}")
        else:
            effective_oos_start_date = None
    else:
        effective_oos_start_date = args.oos_start_date

    resolved_model_names = resolve_requested_models(args.models)
    print(f"[Info] Models to evaluate: {', '.join(resolved_model_names)}")
    alpha_factor_names = resolve_alpha_factor_names(args.alpha_factors, args.max_alpha)
    if alpha_factor_names is None:
        print("[Info] Alpha191 scope: full implemented library")
    elif alpha_factor_names:
        print(f"[Info] Alpha191 scope: {len(alpha_factor_names)} selected factor(s)")
    else:
        print("[Info] Alpha191 scope: disabled")
    model_params_by_name = parse_model_params_json(args.model_params_json)
    hyperparameter_grid_by_name = parse_hyperparameter_grid(
        args.hyperparameter_grid,
        resolved_model_names,
        max_combinations_per_model=max(1, args.max_grid_combinations),
    )
    if model_params_by_name:
        print(f"[Info] Model parameter overrides: {model_params_by_name}")
    if hyperparameter_grid_by_name:
        print(f"[Info] Hyperparameter grid candidates: {hyperparameter_grid_by_name}")
    stage_timing_records.append(
        {
            "stage": "load_clean_and_configure_data",
            "runtime_seconds": time.perf_counter() - data_stage_start,
            "status": "completed",
        }
    )

    # 这里显式区分 in-sample 和 OOS。
    # 如果用户给了 oos_start_date，就直接把该日期之后的数据当作真正的未来测试集；
    # 否则才退回到“最后 test_size 比例的日期作为测试集”。
    feature_stage_start = time.perf_counter()
    print("[Info] Building in-sample / OOS feature sets...")
    feature_cache_key = build_feature_cache_key(
        data_path=data_path,
        sample_start_date=args.sample_start_date,
        oos_start_date=effective_oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        alpha_factor_names=alpha_factor_names,
        price_adjustment_mode=args.price_adjustment_mode,
    )
    cached_features = (
        None
        if args.refresh_caches
        else load_feature_cache(cache_root=cache_root, cache_key=feature_cache_key)
    )
    if cached_features is not None:
        train_df, test_df, feature_columns, feature_metadata = cached_features
        print(f"[Info] Feature cache hit: {feature_cache_key}")
    else:
        train_df, test_df, feature_columns, feature_metadata = strict_time_split_feature_engineering(
            raw_data=raw_data,
            test_size=args.test_size,
            test_start_date=effective_oos_start_date,
            history_window=DEFAULT_HISTORY_WINDOW,
            target_horizon=args.target_horizon,
            alpha_factor_names=alpha_factor_names,
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
                "target_horizon": args.target_horizon,
                "history_window": DEFAULT_HISTORY_WINDOW,
                "alpha_factor_count": None if alpha_factor_names is None else len(alpha_factor_names),
                "price_adjustment_mode": args.price_adjustment_mode,
            },
        )
        print(f"[Info] Feature cache written: {feature_cache_key}")
    feature_cache_status = (
        "hit"
        if cached_features is not None
        else ("refresh_written" if args.refresh_caches else "miss_written")
    )
    stage_timing_records.append(
        {
            "stage": "strict_split_and_feature_engineering",
            "runtime_seconds": time.perf_counter() - feature_stage_start,
            "status": feature_cache_status,
        }
    )

    preprocessing_stage_start = time.perf_counter()
    preprocessing_cache_enabled = not args.disable_preprocessing_cache
    preprocessing_cache_key = build_preprocessing_cache_key(
        data_path=data_path,
        sample_start_date=args.sample_start_date,
        oos_start_date=effective_oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        feature_columns=feature_columns,
        apply_preprocessing=True,
        apply_neutralization=True,
        winsorize_quantile=DEFAULT_WINSORIZE_QUANTILE,
        price_adjustment_mode=args.price_adjustment_mode,
    )

    cached_preprocessing = None
    if preprocessing_cache_enabled and not args.refresh_caches:
        cached_preprocessing = load_preprocessing_cache(cache_root=cache_root, cache_key=preprocessing_cache_key)

    if cached_preprocessing is not None:
        train_df, test_df, preprocessing_summary = cached_preprocessing
        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = "hit"
        preprocessing_summary["cache_key"] = preprocessing_cache_key
        print(f"[Info] Preprocessing cache hit: {preprocessing_cache_key}")
    else:
        print("[Info] Applying cross-sectional preprocessing to train/test features...")
        train_df, preprocessing_summary = apply_cross_sectional_preprocessing(
            train_df,
            feature_columns=feature_columns,
            show_progress=True,
        )
        test_df, _ = apply_cross_sectional_preprocessing(
            test_df,
            feature_columns=feature_columns,
            show_progress=True,
        )
        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = (
            "disabled"
            if not preprocessing_cache_enabled
            else ("refresh_written" if args.refresh_caches else "miss_written")
        )
        preprocessing_summary["cache_key"] = preprocessing_cache_key if preprocessing_cache_enabled else None
        if preprocessing_cache_enabled:
            save_preprocessing_cache(
                cache_root=cache_root,
                cache_key=preprocessing_cache_key,
                train_df=train_df,
                test_df=test_df,
                preprocessing_summary=preprocessing_summary,
                metadata={
                    "target_horizon": args.target_horizon,
                    "feature_count": len(feature_columns),
                    "winsorize_quantile": DEFAULT_WINSORIZE_QUANTILE,
                    "apply_neutralization": True,
                    "price_adjustment_mode": args.price_adjustment_mode,
                },
            )
            print(f"[Info] Preprocessing cache written: {preprocessing_cache_key}")
    stage_timing_records.append(
        {
            "stage": "cross_sectional_preprocessing",
            "runtime_seconds": time.perf_counter() - preprocessing_stage_start,
            "status": str(preprocessing_summary.get("cache_status", "unknown")),
        }
    )

    selector_config = {
        "missing_threshold": args.missing_threshold,
        "variance_threshold": args.variance_threshold,
        "correlation_threshold": args.correlation_threshold,
        "top_n": args.top_n,
        "score_method": args.feature_score_method,
        "random_state": args.random_state,
    }

    validation_stage_start = time.perf_counter()
    print("[Info] Running walk-forward validation on the in-sample period...")
    validation_context = {
        "model_params": model_params_by_name,
        "hyperparameter_grid": hyperparameter_grid_by_name,
        "max_grid_combinations": args.max_grid_combinations,
    }
    validation_cache_key = build_validation_cache_key(
        data_path=data_path,
        sample_start_date=args.sample_start_date,
        oos_start_date=effective_oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
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
        price_adjustment_mode=args.price_adjustment_mode,
        extra_context=validation_context,
    )
    check_deadline(timeout_deadline, "walk-forward validation")
    cached_validation = (
        None
        if args.refresh_caches
        else load_validation_cache(cache_root=cache_root, cache_key=validation_cache_key)
    )
    if cached_validation is not None:
        fold_metrics_df, model_summary_df, model_weights = cached_validation
        print(f"[Info] Validation cache hit: {validation_cache_key}")
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
            purge_days=args.target_horizon,
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
                "target_horizon": args.target_horizon,
                "model_names": resolved_model_names,
                "n_splits": args.n_splits,
                "model_params": model_params_by_name,
                "hyperparameter_grid": hyperparameter_grid_by_name,
                "max_grid_combinations": args.max_grid_combinations,
            },
        )
        print(f"[Info] Validation cache written: {validation_cache_key}")
    validation_cache_status = (
        "hit"
        if cached_validation is not None
        else ("refresh_written" if args.refresh_caches else "miss_written")
    )
    stage_timing_records.append(
        {
            "stage": "purged_walk_forward_validation",
            "runtime_seconds": time.perf_counter() - validation_stage_start,
            "status": validation_cache_status,
        }
    )
    check_deadline(timeout_deadline, "final feature selection")

    selection_stage_start = time.perf_counter()
    print("[Info] Fitting final feature selector on the full in-sample period...")
    selector = FeatureSelector(**selector_config)
    selector.fit(
        train_df[feature_columns],
        train_df["y"],
        dates=train_df["date"],
    )

    X_train_full = selector.transform(train_df[feature_columns])
    y_train_full = train_df["y"].reset_index(drop=True)
    X_test = selector.transform(test_df[feature_columns])

    final_model_weights = finalize_model_weights(resolved_model_names, model_weights)
    stage_timing_records.append(
        {
            "stage": "final_train_only_feature_selection",
            "runtime_seconds": time.perf_counter() - selection_stage_start,
            "status": "completed",
        }
    )

    final_models_stage_start = time.perf_counter()
    print("[Info] Training final models on the full in-sample period...")
    ensemble = ModelEnsemble()
    model_params: Dict[str, Dict] = {}
    weighted_importance_frames: List[pd.DataFrame] = []

    for model_name in optional_progress(
        resolved_model_names,
        description="Final model training",
        enabled=True,
        total=len(resolved_model_names),
    ):
        check_deadline(timeout_deadline, f"final model training: {model_name}")
        current_model_start = time.perf_counter()
        selected_params = selected_model_params_from_summary(model_summary_df, model_name)
        model_wrapper = build_model(
            model_name=model_name,
            random_state=args.random_state,
            params=selected_params or model_params_for_model(model_name, model_params_by_name),
        )
        model_wrapper.fit(X_train_full, y_train_full)
        model_wrapper.save(model_dir / f"{model_name}_model.joblib")

        current_weight = final_model_weights.get(model_name, 0.0)
        ensemble.add_model(model_name, model_wrapper, weight=current_weight)
        model_params[model_name] = model_wrapper.get_params()

        raw_importance_df = model_wrapper.get_feature_importance(selector.selected_features_, model_name=model_name)
        normalized_importance_df = normalize_feature_importance(raw_importance_df, model_weight=current_weight)
        weighted_importance_frames.append(normalized_importance_df)
        final_model_timing_records.append(
            {
                "model": model_name,
                "runtime_seconds": time.perf_counter() - current_model_start,
                "ensemble_weight": float(current_weight),
                "selected_feature_count": int(len(selector.selected_features_)),
            }
        )
    stage_timing_records.append(
        {
            "stage": "fit_and_save_final_models",
            "runtime_seconds": time.perf_counter() - final_models_stage_start,
            "status": "completed",
        }
    )

    prediction_stage_start = time.perf_counter()
    print("[Info] Predicting on the out-of-sample test set...")
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
    stage_timing_records.append(
        {
            "stage": "oos_prediction_and_evaluation",
            "runtime_seconds": time.perf_counter() - prediction_stage_start,
            "status": "completed",
        }
    )
    stage_timing_df = pd.DataFrame(stage_timing_records)
    final_model_timing_df = pd.DataFrame(final_model_timing_records)
    stage_timing_df.to_csv(output_dir / "stage_timing.csv", index=False)
    final_model_timing_df.to_csv(output_dir / "final_model_timing.csv", index=False)

    # 这些中间文件对实战研究很有帮助，因为你可以单独查看：
    # - 每个 fold 的结果
    # - 每个模型的平均验证表现
    # - 最终用于集成的权重
    fold_metrics_df.to_csv(output_dir / "walk_forward_fold_metrics.csv", index=False)
    model_summary_df.to_csv(output_dir / "walk_forward_model_summary.csv", index=False)
    pd.DataFrame(
        [{"model": model_name, "weight": weight} for model_name, weight in final_model_weights.items()]
    ).to_csv(model_dir / "model_weights.csv", index=False)

    selected_feature_df = pd.DataFrame({"feature": selector.selected_features_})
    selected_feature_df.to_csv(model_dir / "selected_features.csv", index=False)
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

    core_quality_columns = [
        column for column in ["open", "high", "low", "close", "volume", "vwap"] if column in raw_data.columns
    ]
    non_unit_adjustment_ratio = 0.0
    missing_or_invalid_adjustment_ratio = 0.0
    adjustment_change_rows = 0
    large_adjustment_change_rows = 0
    adjustment_values = pd.Series(np.nan, index=raw_data.index, dtype=float)
    if "adjustment" in raw_data.columns:
        adjustment_values = pd.to_numeric(raw_data["adjustment"], errors="coerce")
        adjustment_values = adjustment_values.where(adjustment_values > 0.0, np.nan)
        valid_adjustment = adjustment_values.dropna()
        missing_or_invalid_adjustment_ratio = float(adjustment_values.isna().mean())
        non_unit_adjustment_ratio = (
            float((valid_adjustment.sub(1.0).abs() >= 1e-6).mean())
            if not valid_adjustment.empty
            else 0.0
        )
        adjustment_change = adjustment_values.groupby(raw_data["instrument_id"]).pct_change(fill_method=None)
        adjustment_change_rows = int(adjustment_change.abs().gt(1e-6).sum())
        large_adjustment_change_rows = int(adjustment_change.abs().gt(0.20).sum())

    close_values = pd.to_numeric(raw_data["close"], errors="coerce")
    volume_values = pd.to_numeric(raw_data["volume"], errors="coerce")
    turnover_values = pd.to_numeric(raw_data["turnover"], errors="coerce")
    if args.price_adjustment_mode == "vendor_adjusted":
        raw_close_for_turnover = close_values / adjustment_values
    else:
        raw_close_for_turnover = close_values
    expected_raw_turnover = raw_close_for_turnover * volume_values
    turnover_consistency_mask = (
        turnover_values.notna()
        & expected_raw_turnover.notna()
        & np.isfinite(turnover_values)
        & np.isfinite(expected_raw_turnover)
    )
    turnover_relative_error = (
        (turnover_values - expected_raw_turnover).abs()
        / expected_raw_turnover.abs().clip(lower=1.0)
    ).where(turnover_consistency_mask)
    turnover_consistent_mask = turnover_relative_error.le(1e-8)
    daily_close_return = close_values.groupby(raw_data["instrument_id"]).pct_change(fill_method=None)
    instrument_counts = raw_data.groupby("instrument_id").size()
    first_dates = pd.to_datetime(raw_data.groupby("instrument_id")["date"].min())
    last_dates = pd.to_datetime(raw_data.groupby("instrument_id")["date"].max())
    sector_values = (
        raw_data["sector"].fillna("Unknown").astype(str).str.strip()
        if "sector" in raw_data.columns
        else pd.Series("Unknown", index=raw_data.index)
    )
    unknown_sector_mask = sector_values.str.lower().isin({"", "unknown", "nan", "none", "n/a"})
    data_quality_summary = {
        "duplicate_instrument_date_rows": int(
            raw_data.duplicated(subset=["instrument_id", "date"], keep=False).sum()
        ),
        "core_missing_ratio": {
            column: float(raw_data[column].isna().mean()) for column in core_quality_columns
        },
        "nonpositive_close_rows": int((close_values <= 0.0).sum()),
        "unknown_sector_ratio": float(unknown_sector_mask.mean()),
        "market_cap_coverage_ratio": float(
            pd.to_numeric(raw_data["market_cap"], errors="coerce").gt(0.0).mean()
        ),
        "market_cap_source": (
            str(raw_data["market_cap_source"].iloc[0])
            if "market_cap_source" in raw_data.columns and not raw_data.empty
            else "unknown"
        ),
        "turnover_raw_close_evaluable_ratio": float(turnover_consistency_mask.mean()),
        "turnover_raw_close_consistency_ratio": (
            float(turnover_consistent_mask.loc[turnover_consistency_mask].mean())
            if turnover_consistency_mask.any()
            else None
        ),
        "turnover_raw_close_max_relative_error": (
            float(turnover_relative_error.max())
            if turnover_relative_error.notna().any()
            else None
        ),
        "turnover_definition": "raw_unadjusted_close_times_volume_or_equivalent_vendor_field",
        "invalid_ohlc_order_rows": int(
            (
                (pd.to_numeric(raw_data["high"], errors="coerce") < pd.to_numeric(raw_data["low"], errors="coerce"))
                | (pd.to_numeric(raw_data["high"], errors="coerce") < pd.to_numeric(raw_data["open"], errors="coerce"))
                | (pd.to_numeric(raw_data["high"], errors="coerce") < pd.to_numeric(raw_data["close"], errors="coerce"))
                | (pd.to_numeric(raw_data["low"], errors="coerce") > pd.to_numeric(raw_data["open"], errors="coerce"))
                | (pd.to_numeric(raw_data["low"], errors="coerce") > pd.to_numeric(raw_data["close"], errors="coerce"))
            ).sum()
        ),
        "negative_volume_rows": int((volume_values < 0.0).sum()),
        "zero_volume_rows": int(volume_values.eq(0.0).sum()),
        "stale_close_with_positive_volume_rows": int(
            (daily_close_return.abs().le(1e-12) & volume_values.gt(0.0)).sum()
        ),
        "extreme_abs_daily_return_gt_50pct_rows": int(daily_close_return.abs().gt(0.50).sum()),
        "extreme_abs_daily_return_gt_100pct_rows": int(daily_close_return.abs().gt(1.00).sum()),
        "max_abs_daily_return": (
            float(daily_close_return.abs().max()) if daily_close_return.notna().any() else None
        ),
        "non_unit_adjustment_ratio": non_unit_adjustment_ratio,
        "missing_or_invalid_adjustment_ratio": missing_or_invalid_adjustment_ratio,
        "adjustment_change_rows": adjustment_change_rows,
        "large_adjustment_change_gt_20pct_rows": large_adjustment_change_rows,
        "corporate_action_audit": corporate_action_audit_summary,
        "universe_coverage_audit": universe_coverage_audit_summary,
        "observations_per_instrument_min": int(instrument_counts.min()),
        "observations_per_instrument_median": float(instrument_counts.median()),
        "observations_per_instrument_max": int(instrument_counts.max()),
        "instruments_with_fewer_than_252_observations": int((instrument_counts < 252).sum()),
        "instrument_first_date_min": str(first_dates.min().date()),
        "instrument_first_date_max": str(first_dates.max().date()),
        "instrument_last_date_min": str(last_dates.min().date()),
        "instrument_last_date_max": str(last_dates.max().date()),
    }
    write_run_manifest(output_dir / "data_quality_summary.json", data_quality_summary)

    effective_universe_label = args.universe_label
    if effective_universe_label is None:
        effective_universe_label = (
            args.universe
            if (args.fetch_yfinance or auto_fetch_for_direct_run) and not args.symbols
            else "custom_symbols_or_csv"
        )

    dataset_summary = {
        "data_path": project_relative_path(data_path, PROJECT_ROOT),
        "min_date": str(pd.to_datetime(raw_data["date"]).min().date()),
        "max_date": str(pd.to_datetime(raw_data["date"]).max().date()),
        "instrument_count": int(raw_data["instrument_id"].nunique()),
        "n_rows": int(len(train_df) + len(test_df)),
        "raw_rows_after_sample_filter": int(len(raw_data)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_count": int(pd.to_datetime(train_df["date"]).nunique()),
        "test_date_count": int(pd.to_datetime(test_df["date"]).nunique()),
        "train_min_date": str(pd.to_datetime(train_df["date"]).min().date()),
        "train_max_date": str(pd.to_datetime(train_df["date"]).max().date()),
        "test_min_date": str(pd.to_datetime(test_df["date"]).min().date()),
        "test_max_date": str(pd.to_datetime(test_df["date"]).max().date()),
        "n_splits": int(args.n_splits),
        "resolved_model_count": int(len(resolved_model_names)),
        "experiment_name": output_dir.name,
        "experiment_description": "canonical cross-sectional forward-return ranking pipeline",
        "feature_subset_mode": (
            "all_implemented_alpha191" if alpha_factor_names is None else f"selected_alpha191_{len(alpha_factor_names)}"
        ),
        "preprocessing_mode": "daily_winsorize_zscore_sector_and_optional_size_neutralization",
        "sample_start_date": args.sample_start_date,
        "oos_start_date_used": effective_oos_start_date,
        "target_horizon": int(args.target_horizon),
        "target_column": target_column,
        "price_adjustment_mode": args.price_adjustment_mode,
        "label_purge_policy": "last_target_horizon_rows_per_instrument",
        # ``universe_label`` is the canonical manifest field used by the public
        # evidence exporter.  Keep the shorter legacy alias during the current
        # schema version so older local report readers do not break abruptly.
        "universe_label": effective_universe_label,
        "universe": effective_universe_label,
        "hyperparameter_grid": hyperparameter_grid_by_name,
        "max_grid_combinations": args.max_grid_combinations,
        "timeout_seconds": args.timeout_seconds,
        "data_quality": data_quality_summary,
    }

    selector_report_summary = {
        "missing_threshold": selector.missing_threshold,
        "variance_threshold": selector.variance_threshold,
        "correlation_threshold": selector.correlation_threshold,
        "top_n": selector.top_n,
        "score_method": selector.score_method,
        "score_basis": selector.score_basis_,
        "stage_feature_counts": selector.stage_feature_counts_,
        "selected_feature_count": len(selector.selected_features_),
        "validation_score_metric": args.validation_score_metric,
    }

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

    # Manifest 是公开复现的证据入口。它记录真实 commit、dirty 状态、
    # 依赖版本、输入文件 SHA256、完整参数、数据边界和总耗时。
    # 这里不会写入环境变量或 API key。
    run_finished_at_utc = utc_now_iso()
    total_runtime_seconds = time.perf_counter() - run_started_at_perf
    run_manifest = {
        "schema_version": 1,
        "status": "completed",
        "started_at_utc": run_started_at_utc,
        "finished_at_utc": run_finished_at_utc,
        "total_runtime_seconds": total_runtime_seconds,
        "command": sanitize_command([Path(sys.executable).name, *sys.argv]),
        "arguments": sanitize_arguments(vars(args)),
        "effective_configuration": {
            "oos_start_date": effective_oos_start_date,
            "target_column": target_column,
            "models": resolved_model_names,
            "alpha_factors": alpha_factor_names,
            "selector": selector_config,
            "history_window": DEFAULT_HISTORY_WINDOW,
            "outer_label_purge_days": int(args.target_horizon),
            "walk_forward_label_purge_days": int(args.target_horizon),
            "label_purge_policy": "last_target_horizon_rows_per_instrument",
            "winsorize_quantile": DEFAULT_WINSORIZE_QUANTILE,
            "neutralization": True,
            "price_adjustment_mode": args.price_adjustment_mode,
            "feature_cache_key": feature_cache_key,
            "preprocessing_cache_key": preprocessing_cache_key,
            "validation_cache_key": validation_cache_key,
        },
        "data": {
            **dataset_summary,
            "fingerprint": build_data_fingerprint(data_path, PROJECT_ROOT),
        },
        "features": {
            "candidate_count": len(feature_columns),
            "selected_count": len(selector.selected_features_),
            "selected_features": list(selector.selected_features_),
            "metadata": feature_metadata,
            "preprocessing": preprocessing_summary,
            "selection_stage_counts": selector.stage_feature_counts_,
            "selection_score_basis": selector.score_basis_,
        },
        "validation": {
            "fold_count": int(args.n_splits),
            "purge_days": int(args.target_horizon),
            "selection_metric": args.validation_score_metric,
            "model_weights": final_model_weights,
            "model_weight_policy": "50pct_validation_positive_score_plus_50pct_equal_weight",
        },
        "oos_metrics": test_metrics,
        "artifacts": {
            "predictions": build_data_fingerprint(output_dir / "predictions.csv", PROJECT_ROOT),
            "test_predictions_with_actual": build_data_fingerprint(
                output_dir / "test_predictions_with_actual.csv",
                PROJECT_ROOT,
            ),
            "walk_forward_fold_metrics": build_data_fingerprint(
                output_dir / "walk_forward_fold_metrics.csv",
                PROJECT_ROOT,
            ),
            "walk_forward_model_summary": build_data_fingerprint(
                output_dir / "walk_forward_model_summary.csv",
                PROJECT_ROOT,
            ),
            "stage_timing": build_data_fingerprint(output_dir / "stage_timing.csv", PROJECT_ROOT),
            "final_model_timing": build_data_fingerprint(
                output_dir / "final_model_timing.csv",
                PROJECT_ROOT,
            ),
            "data_quality_summary": build_data_fingerprint(
                output_dir / "data_quality_summary.json",
                PROJECT_ROOT,
            ),
            "corporate_action_audit": build_data_fingerprint(
                output_dir / "corporate_action_audit.csv",
                PROJECT_ROOT,
            ),
            "universe_coverage_audit": build_data_fingerprint(
                output_dir / "universe_coverage_audit.csv",
                PROJECT_ROOT,
            ),
            "selected_features": build_data_fingerprint(
                model_dir / "selected_features.csv",
                PROJECT_ROOT,
            ),
            "selected_feature_scores": build_data_fingerprint(
                model_dir / "selected_feature_scores.csv",
                PROJECT_ROOT,
            ),
            "model_weights": build_data_fingerprint(
                model_dir / "model_weights.csv",
                PROJECT_ROOT,
            ),
        },
        "runtime": {
            "stage_timing": stage_timing_df.to_dict("records"),
            "final_model_timing": final_model_timing_df.to_dict("records"),
        },
        "metric_definitions": {
            "pearson_ic_mean": "mean daily cross-sectional Pearson correlation",
            "spearman_ic_mean": "mean daily cross-sectional Spearman correlation",
            "long_short_spread": (
                "mean across dates of top predicted decile y minus bottom predicted decile y; "
                "not cumulative portfolio return"
            ),
            "long_short_return": "backward-compatible alias of long_short_spread",
        },
        "environment": collect_environment(PROJECT_ROOT),
    }
    write_run_manifest(output_dir / "run_manifest.json", run_manifest)

    print("[Info] Training finished.")
    print(f"[Info] Candidate features: {len(feature_columns)}")
    print(f"[Info] Selected features: {len(selector.selected_features_)}")
    print(f"[Info] OOS predictions saved to: {output_dir / 'predictions.csv'}")
    print(f"[Info] Walk-forward fold metrics saved to: {output_dir / 'walk_forward_fold_metrics.csv'}")
    print(f"[Info] Walk-forward model summary saved to: {output_dir / 'walk_forward_model_summary.csv'}")
    print(f"[Info] Training report saved to: {output_dir / 'training_report.md'}")
    print(f"[Info] Reproducibility manifest saved to: {output_dir / 'run_manifest.json'}")
    print(f"[Info] Total runtime: {total_runtime_seconds:.1f} seconds")
    print(f"[Info] Models saved to: {model_dir}")


if __name__ == "__main__":
    main()
