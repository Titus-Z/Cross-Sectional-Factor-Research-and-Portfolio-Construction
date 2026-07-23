"""因子诊断入口。

这个入口专门做一件事：

- 基于某个已经跑完的实验结果；
- 读取该实验最终保留下来的特征列表；
- 在同口径 OOS 数据上做单因子诊断；
- 输出 IC、分组收益和解释报告。

为什么要单独开一个入口，而不是塞进 `main_experiments.py`？

- 训练实验和因子诊断是两个不同问题；
- 训练脚本关注“模型整体是否有效”；
- 诊断脚本关注“单个特征到底贡献了什么”；
- 拆开以后，主流程不会继续膨胀。
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd

from src.data_loader import activate_target_horizon, load_daily_data
from src.factor_diagnostics import summarize_factor_diagnostics, write_factor_diagnostics_report
from src.feature_cache import build_feature_cache_key, load_feature_cache, save_feature_cache
from src.preprocessing import DEFAULT_WINSORIZE_QUANTILE, apply_cross_sectional_preprocessing
from src.preprocessing_cache import build_preprocessing_cache_key, load_preprocessing_cache, save_preprocessing_cache
from src.progress import create_progress_bar, format_duration
from src.project_paths import resolve_project_path
from src.runtime_config import (
    DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
    DEFAULT_FACTOR_DIAGNOSTICS_OUTPUT_DIR,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)
from src.time_series_pipeline import DEFAULT_HISTORY_WINDOW, strict_time_split_feature_engineering
from src.universe import get_symbol_sector_map


def parse_args() -> argparse.Namespace:
    """解析因子诊断入口参数。"""

    parser = argparse.ArgumentParser(description="Run OOS factor diagnostics for a finished experiment.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_PRIMARY_DATA_PATH,
        help="原始数据路径。",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
        help="训练完成实验对应的模型目录，里面需要有 selected_features.csv。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_FACTOR_DIAGNOSTICS_OUTPUT_DIR,
        help="因子诊断输出目录。",
    )
    parser.add_argument(
        "--feature-source",
        type=str,
        choices=["selected_features", "selected_scores", "importance"],
        default="selected_features",
        help="从哪个文件读取要诊断的特征列表。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="只诊断前 K 个特征。默认诊断来源文件中的全部特征。",
    )
    parser.add_argument("--sample-start-date", type=str, default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", type=str, default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="当前诊断的目标周期。")
    parser.add_argument("--n-groups", type=int, default=5, help="分组收益分析的分组数量。")
    parser.add_argument(
        "--min-cross-section",
        type=int,
        default=30,
        help="某个日期最少需要多少只股票，才会参与 IC 和分组收益计算。",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="缓存目录。当前用于复用横截面预处理结果。",
    )
    parser.add_argument(
        "--disable-preprocessing-cache",
        action="store_true",
        help="关闭横截面预处理缓存。",
    )
    return parser.parse_args()


def configure_runtime_warning_display() -> None:
    """降低重复 warning 对终端阅读的干扰。"""

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
    total_elapsed: float,
) -> None:
    """更新阶段进度和剩余时间估计。"""

    average_stage_time = total_elapsed / max(stage_index, 1)
    estimated_remaining = average_stage_time * max(total_stages - stage_index, 0)
    progress_bar.update(1)
    progress_bar.set_postfix_str(
        (
            f"{stage_label} {format_duration(stage_elapsed)} | "
            f"total {format_duration(total_elapsed)} | "
            f"est left {format_duration(estimated_remaining)}"
        )
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """把耗时表写成简单 Markdown。"""

    if df.empty:
        return "_No data available._"
    headers = " | ".join(df.columns)
    separators = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(str(value) for value in row) for row in df.astype(str).itertuples(index=False, name=None)]
    return "\n".join([f"| {headers} |", f"| {separators} |"] + [f"| {row} |" for row in rows])


def write_stage_timing(output_dir: Path, stage_timing_df: pd.DataFrame) -> None:
    """把诊断脚本本身的阶段耗时落盘。"""

    if stage_timing_df.empty:
        return
    stage_timing_df.to_csv(output_dir / "stage_timing.csv", index=False)
    markdown_text = "# Stage Timing\n\n" + dataframe_to_markdown(stage_timing_df) + "\n"
    (output_dir / "stage_timing.md").write_text(markdown_text, encoding="utf-8")


def load_feature_source(model_dir: Path, feature_source: str, top_k: int | None) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """从训练产物里读取要诊断的特征列表。"""

    auxiliary_tables: dict[str, pd.DataFrame] = {}

    selected_features_path = model_dir / "selected_features.csv"
    selected_scores_path = model_dir / "selected_feature_scores.csv"
    importance_path = model_dir / "feature_importance.csv"

    if selected_scores_path.exists():
        auxiliary_tables["selector_scores"] = pd.read_csv(selected_scores_path)
    if importance_path.exists():
        auxiliary_tables["importance_scores"] = pd.read_csv(importance_path)

    if feature_source == "selected_features":
        if not selected_features_path.exists():
            raise FileNotFoundError(f"Missing selected_features.csv in: {model_dir}")
        feature_df = pd.read_csv(selected_features_path)
        feature_list = feature_df["feature"].dropna().astype(str).tolist()
    elif feature_source == "selected_scores":
        if "selector_scores" not in auxiliary_tables:
            raise FileNotFoundError(f"Missing selected_feature_scores.csv in: {model_dir}")
        feature_list = auxiliary_tables["selector_scores"]["feature"].dropna().astype(str).tolist()
    else:
        if "importance_scores" not in auxiliary_tables:
            raise FileNotFoundError(f"Missing feature_importance.csv in: {model_dir}")
        feature_list = auxiliary_tables["importance_scores"]["feature"].dropna().astype(str).tolist()

    if top_k is not None:
        feature_list = feature_list[:top_k]

    deduplicated_feature_list = list(dict.fromkeys(feature_list))
    return deduplicated_feature_list, auxiliary_tables


def main() -> None:
    """执行 OOS 因子诊断。"""

    configure_runtime_warning_display()
    args = parse_args()

    data_path = resolve_project_path(args.data_path)
    model_dir = resolve_project_path(args.model_dir)
    output_dir = resolve_project_path(args.output_dir)
    cache_root = resolve_project_path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    total_stage_count = 6
    stage_timing_records: list[dict] = []
    stage_progress = create_progress_bar(total=total_stage_count, description="Factor diagnostics stages", enabled=True)
    workflow_start = time.perf_counter()

    stage_start = time.perf_counter()
    raw_data = load_daily_data(data_path)
    raw_data["date"] = pd.to_datetime(raw_data["date"])
    if args.sample_start_date:
        raw_data = raw_data[raw_data["date"] >= pd.Timestamp(args.sample_start_date)].copy()
    if raw_data.empty:
        raise ValueError("No rows remain after applying sample_start_date.")

    if "sector" not in raw_data.columns or raw_data["sector"].isna().all():
        sector_map = get_symbol_sector_map(sorted(raw_data["instrument_id"].dropna().unique()))
        if sector_map:
            raw_data["sector"] = raw_data["instrument_id"].map(sector_map).fillna("Unknown")

    stage_elapsed = time.perf_counter() - stage_start
    total_elapsed = time.perf_counter() - workflow_start
    stage_timing_records.append(
        {
            "stage_order": 1,
            "stage_key": "load_data",
            "stage_label": "Load daily data",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": f"{len(raw_data)} rows",
        }
    )
    finish_stage_progress(stage_progress, 1, total_stage_count, "load data", stage_elapsed, total_elapsed)

    stage_start = time.perf_counter()
    raw_data, target_column = activate_target_horizon(raw_data, target_horizon=args.target_horizon)
    stage_elapsed = time.perf_counter() - stage_start
    total_elapsed = time.perf_counter() - workflow_start
    stage_timing_records.append(
        {
            "stage_order": 2,
            "stage_key": "activate_target",
            "stage_label": "Activate target",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": target_column,
        }
    )
    finish_stage_progress(stage_progress, 2, total_stage_count, "activate target", stage_elapsed, total_elapsed)

    feature_cache_key = build_feature_cache_key(
        data_path=data_path,
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
    )
    stage_start = time.perf_counter()
    cached_features = load_feature_cache(cache_root=cache_root, cache_key=feature_cache_key)
    if cached_features is not None:
        train_df, test_df, feature_columns, feature_metadata = cached_features
        feature_detail = f"{len(feature_columns)} features | cache hit"
    else:
        train_df, test_df, feature_columns, feature_metadata = strict_time_split_feature_engineering(
            raw_data=raw_data,
            test_size=args.test_size,
            history_window=DEFAULT_HISTORY_WINDOW,
            test_start_date=args.oos_start_date,
            target_horizon=args.target_horizon,
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
            },
        )
        feature_detail = f"{len(feature_columns)} features | cache miss"
    stage_elapsed = time.perf_counter() - stage_start
    total_elapsed = time.perf_counter() - workflow_start
    stage_timing_records.append(
        {
            "stage_order": 3,
            "stage_key": "feature_engineering",
            "stage_label": "Time split + feature engineering",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": feature_detail,
        }
    )
    finish_stage_progress(stage_progress, 3, total_stage_count, "feature engineering", stage_elapsed, total_elapsed)

    stage_start = time.perf_counter()
    preprocessing_cache_enabled = not args.disable_preprocessing_cache
    preprocessing_cache_key = build_preprocessing_cache_key(
        data_path=data_path,
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        feature_columns=feature_columns,
        apply_preprocessing=True,
        apply_neutralization=True,
        winsorize_quantile=DEFAULT_WINSORIZE_QUANTILE,
    )

    cached_preprocessing = None
    if preprocessing_cache_enabled:
        cached_preprocessing = load_preprocessing_cache(cache_root=cache_root, cache_key=preprocessing_cache_key)

    if cached_preprocessing is not None:
        train_df, test_df, preprocessing_summary = cached_preprocessing
        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = "hit"
        preprocessing_summary["cache_key"] = preprocessing_cache_key
        details = f"{len(test_df)} OOS rows | cache hit"
    else:
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
        preprocessing_summary["cache_status"] = "disabled" if not preprocessing_cache_enabled else "miss_written"
        preprocessing_summary["cache_key"] = preprocessing_cache_key if preprocessing_cache_enabled else None
        details = f"{len(test_df)} OOS rows" + (" | cache miss" if preprocessing_cache_enabled else " | cache disabled")
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
                },
            )

    stage_elapsed = time.perf_counter() - stage_start
    total_elapsed = time.perf_counter() - workflow_start
    stage_timing_records.append(
        {
            "stage_order": 4,
            "stage_key": "preprocessing",
            "stage_label": "Cross-sectional preprocessing",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": details,
        }
    )
    finish_stage_progress(stage_progress, 4, total_stage_count, "preprocessing", stage_elapsed, total_elapsed)

    stage_start = time.perf_counter()
    feature_list, auxiliary_tables = load_feature_source(
        model_dir=model_dir,
        feature_source=args.feature_source,
        top_k=args.top_k,
    )
    available_feature_list = [feature_name for feature_name in feature_list if feature_name in test_df.columns]
    missing_feature_list = [feature_name for feature_name in feature_list if feature_name not in test_df.columns]
    if not available_feature_list:
        raise ValueError("No requested diagnostic features are present in the reconstructed OOS feature table.")

    feature_source_summary = {
        "model_dir": str(model_dir),
        "feature_source": args.feature_source,
        "requested_feature_count": len(feature_list),
        "diagnosed_feature_count": len(available_feature_list),
        "missing_feature_count": len(missing_feature_list),
        "top_k": args.top_k,
        "n_groups": args.n_groups,
        "min_cross_section": args.min_cross_section,
        "preprocessing_cache_status": preprocessing_summary.get("cache_status"),
    }
    stage_elapsed = time.perf_counter() - stage_start
    total_elapsed = time.perf_counter() - workflow_start
    stage_timing_records.append(
        {
            "stage_order": 5,
            "stage_key": "resolve_features",
            "stage_label": "Resolve diagnostic feature list",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": f"{len(available_feature_list)} usable | {len(missing_feature_list)} missing",
        }
    )
    finish_stage_progress(stage_progress, 5, total_stage_count, "resolve features", stage_elapsed, total_elapsed)

    stage_start = time.perf_counter()
    summary_df, daily_ic_df, group_returns_df, average_group_returns_df = summarize_factor_diagnostics(
        data=test_df,
        factor_columns=available_feature_list,
        target_column="y",
        n_groups=args.n_groups,
        min_cross_section=args.min_cross_section,
        selector_scores=auxiliary_tables.get("selector_scores"),
        importance_scores=auxiliary_tables.get("importance_scores"),
        show_progress=True,
    )

    summary_df.to_csv(output_dir / "factor_ic_summary.csv", index=False)
    daily_ic_df.to_csv(output_dir / "factor_daily_ic.csv", index=False)
    group_returns_df.to_csv(output_dir / "factor_group_returns.csv", index=False)
    average_group_returns_df.to_csv(output_dir / "factor_average_group_returns.csv", index=False)
    pd.DataFrame({"feature": available_feature_list}).to_csv(output_dir / "diagnosed_features.csv", index=False)

    stage_elapsed = time.perf_counter() - stage_start
    total_elapsed = time.perf_counter() - workflow_start
    stage_timing_records.append(
        {
            "stage_order": 6,
            "stage_key": "factor_diagnostics",
            "stage_label": "Run OOS factor diagnostics",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": f"{len(available_feature_list)} features",
        }
    )
    finish_stage_progress(stage_progress, 6, total_stage_count, "factor diagnostics", stage_elapsed, total_elapsed)
    stage_progress.close()

    stage_timing_df = pd.DataFrame(stage_timing_records)
    write_stage_timing(output_dir=output_dir, stage_timing_df=stage_timing_df)

    dataset_summary = {
        "data_path": str(data_path),
        "min_date": str(pd.to_datetime(raw_data["date"]).min().date()),
        "max_date": str(pd.to_datetime(raw_data["date"]).max().date()),
        "sample_start_date": args.sample_start_date,
        "oos_start_date_used": args.oos_start_date,
        "target_horizon": args.target_horizon,
        "target_column": target_column,
        "test_rows": int(len(test_df)),
        "test_date_count": int(pd.to_datetime(test_df["date"]).nunique()),
        "test_instrument_count": int(test_df["instrument_id"].nunique()),
    }

    write_factor_diagnostics_report(
        output_path=output_dir / "factor_report.md",
        dataset_summary=dataset_summary,
        feature_source_summary=feature_source_summary,
        summary_df=summary_df,
        average_group_returns_df=average_group_returns_df,
        stage_timing_df=stage_timing_df,
    )

    print("[Info] Factor diagnostics finished.")
    print(f"[Info] Summary saved to: {output_dir / 'factor_ic_summary.csv'}")
    print(f"[Info] Daily IC saved to: {output_dir / 'factor_daily_ic.csv'}")
    print(f"[Info] Group returns saved to: {output_dir / 'factor_group_returns.csv'}")
    print(f"[Info] Report saved to: {output_dir / 'factor_report.md'}")


if __name__ == "__main__":
    main()
