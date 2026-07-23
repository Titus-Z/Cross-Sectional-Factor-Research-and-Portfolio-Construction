"""Alpha191 专项诊断入口。

这个脚本不重新训练模型，职责是产出一套可审计的 Alpha 研究证据：

1. Alpha 家族分类；
2. 不同 horizon 的 IC / RankIC；
3. Alpha 相关性与冗余聚类；
4. Alpha 排名换手 proxy；
5. Alpha 按年份/季度的衰减检查；
6. 可直接改写成简历 bullet 的总结。

为什么单独做入口？

- 主训练入口应该保持清晰；
- Alpha 诊断属于研究解释层，不应该和模型训练强耦合；
- 后续自动挖掘因子也可以复用这里的评价口径。
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

from src.alpha191 import list_supported_alpha_factors
from src.alpha_diagnostics import (
    build_alpha_family_map,
    build_correlation_clusters,
    build_decay_table,
    build_horizon_match_table,
    compute_alpha_correlation,
    compute_alpha_turnover_proxy,
    compute_daily_alpha_ic,
    plot_alpha_correlation_heatmap,
    summarize_alpha_ic,
    write_alpha_diagnostics_report,
    write_resume_bullet_report,
)
from src.data_loader import activate_target_horizon, load_daily_data
from src.feature_cache import build_feature_cache_key, load_feature_cache, save_feature_cache
from src.progress import create_progress_bar, format_duration
from src.project_paths import resolve_project_path
from src.runtime_config import (
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)
from src.time_series_pipeline import DEFAULT_HISTORY_WINDOW, strict_time_split_feature_engineering
from src.universe import get_symbol_sector_map


DEFAULT_ALPHA_DIAGNOSTICS_OUTPUT_DIR = "outputs/alpha_diagnostics/us300"


def parse_horizon_list(raw_value: str) -> list[int]:
    """把 `1,5,10` 这种命令行字符串解析成 horizon 列表。"""

    horizons: list[int] = []
    for item in raw_value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        horizons.append(int(stripped))
    if not horizons:
        raise ValueError("At least one target horizon must be provided.")
    return sorted(set(horizons))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run Alpha191-specific diagnostics.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_PRIMARY_DATA_PATH, help="原始日线数据路径。")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_ALPHA_DIAGNOSTICS_OUTPUT_DIR, help="输出目录。")
    parser.add_argument("--sample-start-date", type=str, default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", type=str, default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的测试集比例。")
    parser.add_argument(
        "--target-horizons",
        type=str,
        default="1,5,10",
        help="需要诊断的目标周期，例如 1,5,10。",
    )
    parser.add_argument(
        "--main-horizon",
        type=int,
        default=DEFAULT_PRIMARY_TARGET_HORIZON,
        help="报告里重点排序和 decay 使用的主 horizon。",
    )
    parser.add_argument("--min-cross-section", type=int, default=30, help="单日横截面最少样本数。")
    parser.add_argument("--correlation-threshold", type=float, default=0.9, help="高相关 Alpha 聚类阈值。")
    parser.add_argument("--max-corr-rows", type=int, default=120_000, help="计算相关性时最多抽样多少行。")
    parser.add_argument("--heatmap-alpha-count", type=int, default=60, help="热力图最多展示多少个 Alpha。")
    parser.add_argument("--max-alpha", type=int, default=None, help="只诊断前 N 个 Alpha，用于快速 smoke test。")
    parser.add_argument("--cache-dir", type=str, default=".cache", help="特征缓存目录。")
    parser.add_argument("--disable-cache", action="store_true", help="关闭特征缓存。")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条。")
    return parser.parse_args()


def configure_warning_display() -> None:
    """减少 pandas/numpy 重复 warning 对进度条的干扰。"""

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
    warnings.filterwarnings(
        "once",
        message="invalid value encountered in divide",
        category=RuntimeWarning,
        module=r"numpy\.lib\.function_base",
    )


def finish_stage(progress_bar, stage_name: str, stage_start: float, workflow_start: float, stage_index: int, total_stages: int) -> None:
    """更新阶段进度条，显示当前阶段耗时和估计剩余时间。"""

    stage_elapsed = time.perf_counter() - stage_start
    total_elapsed = time.perf_counter() - workflow_start
    average_stage_time = total_elapsed / max(stage_index, 1)
    estimated_left = average_stage_time * max(total_stages - stage_index, 0)
    progress_bar.update(1)
    progress_bar.set_postfix_str(
        f"{stage_name} {format_duration(stage_elapsed)} | total {format_duration(total_elapsed)} | est left {format_duration(estimated_left)}"
    )


def add_sector_if_missing(data: pd.DataFrame) -> pd.DataFrame:
    """如果数据里没有 sector，尽量用内置 universe 映射补上。"""

    enriched = data.copy()
    if "sector" in enriched.columns and enriched["sector"].notna().any():
        return enriched

    sector_map = get_symbol_sector_map(sorted(enriched["instrument_id"].dropna().unique()))
    if sector_map:
        enriched["sector"] = enriched["instrument_id"].map(sector_map).fillna("Unknown")
    return enriched


def main() -> None:
    """执行 Alpha191 专项诊断。"""

    configure_warning_display()
    args = parse_args()
    show_progress = not args.no_progress

    data_path = resolve_project_path(args.data_path)
    output_dir = resolve_project_path(args.output_dir)
    cache_root = resolve_project_path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    target_horizons = parse_horizon_list(args.target_horizons)
    if args.main_horizon not in target_horizons:
        target_horizons.append(args.main_horizon)
        target_horizons = sorted(set(target_horizons))

    alpha_names = list_supported_alpha_factors()
    if args.max_alpha is not None:
        alpha_names = alpha_names[: args.max_alpha]

    total_stages = 7
    stage_progress = create_progress_bar(total=total_stages, description="Alpha diagnostics stages", enabled=show_progress)
    workflow_start = time.perf_counter()

    stage_start = time.perf_counter()
    raw_data = load_daily_data(data_path)
    raw_data["date"] = pd.to_datetime(raw_data["date"])
    if args.sample_start_date:
        raw_data = raw_data[raw_data["date"] >= pd.Timestamp(args.sample_start_date)].copy()
    raw_data = add_sector_if_missing(raw_data)
    if raw_data.empty:
        raise ValueError("No data remains after sample_start_date filtering.")
    finish_stage(stage_progress, "load data", stage_start, workflow_start, 1, total_stages)

    stage_start = time.perf_counter()
    # 使用最长 horizon 激活 `y`，确保 train/test 特征表尾部不会保留更长目标缺失的样本。
    feature_target_horizon = max(target_horizons)
    activated_data, _ = activate_target_horizon(raw_data, target_horizon=feature_target_horizon)
    target_columns = {horizon: f"y_{horizon}d" for horizon in target_horizons}
    finish_stage(stage_progress, "activate targets", stage_start, workflow_start, 2, total_stages)

    stage_start = time.perf_counter()
    cache_key = build_feature_cache_key(
        data_path=data_path,
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=feature_target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        alpha_factor_names=alpha_names,
        extra_context={"entry": "main_alpha_diagnostics"},
    )
    cached_features = None if args.disable_cache else load_feature_cache(cache_root=cache_root, cache_key=cache_key)
    if cached_features is not None:
        train_df, oos_df, feature_columns, feature_metadata = cached_features
        cache_status = "hit"
    else:
        train_df, oos_df, feature_columns, feature_metadata = strict_time_split_feature_engineering(
            raw_data=activated_data,
            test_size=args.test_size,
            history_window=DEFAULT_HISTORY_WINDOW,
            test_start_date=args.oos_start_date,
            target_horizon=args.target_horizon,
            alpha_factor_names=alpha_names,
            show_progress=show_progress,
        )
        cache_status = "disabled" if args.disable_cache else "miss_written"
        if not args.disable_cache:
            save_feature_cache(
                cache_root=cache_root,
                cache_key=cache_key,
                train_df=train_df,
                test_df=oos_df,
                feature_columns=feature_columns,
                feature_metadata=feature_metadata,
                metadata={
                    "entry": "main_alpha_diagnostics",
                    "target_horizons": target_horizons,
                    "alpha_count": len(alpha_names),
                },
            )
    finish_stage(stage_progress, f"features ({cache_status})", stage_start, workflow_start, 3, total_stages)

    stage_start = time.perf_counter()
    alpha_columns = [column for column in feature_metadata.get("alpha_feature_columns", []) if column in train_df.columns]
    if not alpha_columns:
        alpha_columns = [column for column in feature_columns if column.lower().startswith("alpha")]
    if not alpha_columns:
        raise ValueError("No Alpha columns were found in the generated feature matrix.")
    alpha_family_map = build_alpha_family_map(alpha_columns)
    alpha_family_map.to_csv(output_dir / "alpha_family_map.csv", index=False)
    finish_stage(stage_progress, "family map", stage_start, workflow_start, 4, total_stages)

    stage_start = time.perf_counter()
    train_ic_df = compute_daily_alpha_ic(
        data=train_df,
        alpha_columns=alpha_columns,
        target_columns=target_columns,
        subset_label="train",
        min_cross_section=args.min_cross_section,
        show_progress=show_progress,
    )
    oos_ic_df = compute_daily_alpha_ic(
        data=oos_df,
        alpha_columns=alpha_columns,
        target_columns=target_columns,
        subset_label="oos",
        min_cross_section=args.min_cross_section,
        show_progress=show_progress,
    )
    daily_ic_df = pd.concat([train_ic_df, oos_ic_df], ignore_index=True)
    ic_summary_df = summarize_alpha_ic(daily_ic_df, alpha_family_map=alpha_family_map)
    horizon_match_df = build_horizon_match_table(ic_summary_df, subset_label="train")
    daily_ic_df.to_csv(output_dir / "alpha_daily_ic.csv", index=False)
    ic_summary_df.to_csv(output_dir / "alpha_horizon_ic_summary.csv", index=False)
    horizon_match_df.to_csv(output_dir / "alpha_horizon_match.csv", index=False)
    finish_stage(stage_progress, "IC + horizon", stage_start, workflow_start, 5, total_stages)

    stage_start = time.perf_counter()
    turnover_df = compute_alpha_turnover_proxy(
        data=train_df,
        alpha_columns=alpha_columns,
        top_fraction=0.2,
        show_progress=show_progress,
    )
    turnover_df.to_csv(output_dir / "alpha_turnover_proxy.csv", index=False)
    finish_stage(stage_progress, "turnover", stage_start, workflow_start, 6, total_stages)

    stage_start = time.perf_counter()
    correlation_df = compute_alpha_correlation(
        data=train_df,
        alpha_columns=alpha_columns,
        max_rows=args.max_corr_rows,
    )
    correlation_df.to_csv(output_dir / "alpha_correlation_matrix.csv")
    cluster_df = build_correlation_clusters(
        correlation_df=correlation_df,
        alpha_family_map=alpha_family_map,
        ic_summary_df=ic_summary_df,
        subset_label="train",
        target_horizon=args.main_horizon,
        threshold=args.correlation_threshold,
    )
    cluster_df.to_csv(output_dir / "alpha_correlation_clusters.csv", index=False)

    train_main = ic_summary_df[
        (ic_summary_df["subset"] == "train")
        & (ic_summary_df["target_horizon"] == args.main_horizon)
    ].copy()
    heatmap_order = train_main.sort_values("rank_ic_mean", ascending=False)["alpha_name"].tolist()
    plot_alpha_correlation_heatmap(
        correlation_df=correlation_df,
        output_path=output_dir / "alpha_correlation_heatmap.png",
        alpha_order=heatmap_order,
        max_alpha_count=args.heatmap_alpha_count,
    )

    decay_df = build_decay_table(
        daily_ic_df=daily_ic_df,
        target_horizon=args.main_horizon,
        method="rank",
        period="Y",
    )
    decay_df.to_csv(output_dir / "alpha_decay_by_year.csv", index=False)

    dataset_summary = {
        "data_path": str(data_path),
        "min_date": str(pd.to_datetime(raw_data["date"]).min().date()),
        "max_date": str(pd.to_datetime(raw_data["date"]).max().date()),
        "train_rows": int(len(train_df)),
        "oos_rows": int(len(oos_df)),
        "train_min_date": str(pd.to_datetime(train_df["date"]).min().date()),
        "train_max_date": str(pd.to_datetime(train_df["date"]).max().date()),
        "oos_min_date": str(pd.to_datetime(oos_df["date"]).min().date()),
        "oos_max_date": str(pd.to_datetime(oos_df["date"]).max().date()),
        "target_horizons": ",".join(str(item) for item in target_horizons),
        "alpha_count": int(len(alpha_columns)),
        "feature_cache_status": cache_status,
    }
    write_alpha_diagnostics_report(
        output_path=output_dir / "alpha_diagnostics_report.md",
        dataset_summary=dataset_summary,
        alpha_family_map=alpha_family_map,
        ic_summary_df=ic_summary_df,
        horizon_match_df=horizon_match_df,
        turnover_df=turnover_df,
        cluster_df=cluster_df,
        decay_df=decay_df,
        main_horizon=args.main_horizon,
    )
    write_resume_bullet_report(output_dir / "alpha_resume_bullets.md")
    finish_stage(stage_progress, "correlation + reports", stage_start, workflow_start, 7, total_stages)
    stage_progress.close()

    total_elapsed = time.perf_counter() - workflow_start
    print("[Info] Alpha diagnostics finished.")
    print(f"[Info] Alpha count: {len(alpha_columns)}")
    print(f"[Info] Target horizons: {target_horizons}")
    print(f"[Info] Feature cache status: {cache_status}")
    print(f"[Info] Output dir: {output_dir}")
    print(f"[Info] Total runtime: {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
