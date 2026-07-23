from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alpha191 import CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS
from src.data_loader import PRICE_ADJUSTMENT_MODES, activate_target_horizon, load_daily_data
from src.factor_diagnostics import summarize_factor_diagnostics
from src.feature_cache import build_feature_cache_key, load_feature_cache, save_feature_cache
from src.preprocessing import DEFAULT_WINSORIZE_QUANTILE, apply_cross_sectional_preprocessing
from src.preprocessing_cache import build_preprocessing_cache_key, load_preprocessing_cache, save_preprocessing_cache
from src.project_paths import resolve_project_path
from src.runtime_config import (
    DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)
from src.time_series_pipeline import DEFAULT_HISTORY_WINDOW, strict_time_split_feature_engineering
from src.universe import get_symbol_sector_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reusable OOS single-factor case study.")
    parser.add_argument("--factor", required=True, help="单因子名称，例如 return_std_20。")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始数据路径。")
    parser.add_argument("--model-dir", default=DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR, help="用于读取 selected_features 和分数的模型目录。")
    parser.add_argument("--output-dir", default="factor_mining_workspace/outputs", help="研究输出根目录。")
    parser.add_argument("--cache-dir", default=".cache", help="缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="目标周期。")
    parser.add_argument(
        "--price-adjustment-mode",
        choices=list(PRICE_ADJUSTMENT_MODES),
        default="vendor_adjusted",
        help="价格口径；因子挖掘、选择和严格消融必须保持一致。",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--n-groups", type=int, default=5, help="分组数量。")
    parser.add_argument("--min-cross-section", type=int, default=30, help="每个日期最少参与诊断的股票数。")
    parser.add_argument(
        "--disable-preprocessing-cache",
        action="store_true",
        help="关闭横截面预处理缓存。",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    headers = " | ".join(df.columns)
    separators = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(str(value) for value in row) for row in df.astype(str).itertuples(index=False, name=None)]
    return "\n".join([f"| {headers} |", f"| {separators} |"] + [f"| {row} |" for row in rows])


def load_model_context(model_dir: Path, factor_name: str) -> tuple[dict[str, object], pd.DataFrame | None, pd.DataFrame | None]:
    selected_features_path = model_dir / "selected_features.csv"
    selected_scores_path = model_dir / "selected_feature_scores.csv"
    importance_path = model_dir / "feature_importance.csv"

    selected_features: list[str] = []
    selector_scores_df: pd.DataFrame | None = None
    importance_scores_df: pd.DataFrame | None = None

    if selected_features_path.exists():
        selected_features = pd.read_csv(selected_features_path)["feature"].dropna().astype(str).tolist()
    if selected_scores_path.exists():
        selector_scores_df = pd.read_csv(selected_scores_path)
    if importance_path.exists():
        importance_scores_df = pd.read_csv(importance_path)

    selector_score = float("nan")
    model_importance = float("nan")

    if selector_scores_df is not None and {"feature", "score"}.issubset(selector_scores_df.columns):
        matched = selector_scores_df.loc[selector_scores_df["feature"] == factor_name, "score"]
        if not matched.empty:
            selector_score = float(matched.iloc[0])

    if importance_scores_df is not None and {"feature", "importance"}.issubset(importance_scores_df.columns):
        matched = importance_scores_df.loc[importance_scores_df["feature"] == factor_name, "importance"]
        if not matched.empty:
            model_importance = float(matched.iloc[0])

    context = {
        "factor": factor_name,
        "selected_in_model": factor_name in set(selected_features),
        "selected_feature_count": len(selected_features),
        "selector_score": selector_score,
        "model_importance": model_importance,
    }
    return context, selector_scores_df, importance_scores_df


def resolve_alpha_factor_names_for_research(args: argparse.Namespace) -> list[str] | None:
    """决定单因子/自动挖因子研究是否需要生成 Alpha191。

    这里的默认策略是偏工程效率，而不是偏“特征越多越好”：

    - 自动挖因子的默认 seed 是技术指标和量价结构，不需要 176 个 Alpha191；
    - 如果用户显式传入 `--include-alpha-seeds`，只生成 canonical
      逐股价格尺度不变 Alpha 子集；
    - 如果单因子脚本研究的是 `alpha144` 这类具体 Alpha，只生成这个 Alpha；
    - 其他情况明确返回空列表，表示不生成任何 Alpha191。

    这个选择能避免一个很大的隐形浪费：搜索 20 个候选公式却先花十几分钟计算全量 Alpha。
    """

    if bool(getattr(args, "include_alpha_seeds", False)):
        return list(CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS)

    factor_name = str(getattr(args, "factor", "") or "").strip().lower()
    if factor_name.startswith("alpha"):
        return [factor_name]

    return []


def load_or_build_preprocessed_train_test(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object]]:
    data_path = resolve_project_path(args.data_path)
    cache_root = resolve_project_path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    price_adjustment_mode = str(getattr(args, "price_adjustment_mode", "vendor_adjusted"))
    raw_data = load_daily_data(data_path, price_adjustment_mode=price_adjustment_mode)
    raw_data["date"] = pd.to_datetime(raw_data["date"])
    if args.sample_start_date:
        raw_data = raw_data[raw_data["date"] >= pd.Timestamp(args.sample_start_date)].copy()
    if raw_data.empty:
        raise ValueError("No rows remain after applying sample_start_date.")

    if "sector" not in raw_data.columns or raw_data["sector"].isna().all():
        sector_map = get_symbol_sector_map(sorted(raw_data["instrument_id"].dropna().unique()))
        if sector_map:
            raw_data["sector"] = raw_data["instrument_id"].map(sector_map).fillna("Unknown")

    raw_data, target_column = activate_target_horizon(raw_data, target_horizon=args.target_horizon)
    alpha_factor_names = resolve_alpha_factor_names_for_research(args)

    feature_cache_key = build_feature_cache_key(
        data_path=data_path,
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        alpha_factor_names=alpha_factor_names,
        price_adjustment_mode=price_adjustment_mode,
    )
    cached_features = load_feature_cache(cache_root=cache_root, cache_key=feature_cache_key)
    if cached_features is None:
        alpha_scope = "all" if alpha_factor_names is None else len(alpha_factor_names)
        print(f"[Info] Feature cache miss; building features with Alpha191 scope: {alpha_scope}", flush=True)
        train_df, test_df, feature_columns, feature_metadata = strict_time_split_feature_engineering(
            raw_data=raw_data,
            test_size=args.test_size,
            history_window=DEFAULT_HISTORY_WINDOW,
            test_start_date=args.oos_start_date,
            target_horizon=args.target_horizon,
            alpha_factor_names=alpha_factor_names,
            show_progress=False,
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
                "alpha_factor_names": "all" if alpha_factor_names is None else list(alpha_factor_names),
                "price_adjustment_mode": price_adjustment_mode,
            },
        )
    else:
        print("[Info] Feature cache hit; reusing strict time-split features", flush=True)
        train_df, test_df, feature_columns, _ = cached_features

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
        price_adjustment_mode=price_adjustment_mode,
    )

    cached_preprocessing = None
    if not args.disable_preprocessing_cache:
        cached_preprocessing = load_preprocessing_cache(cache_root=cache_root, cache_key=preprocessing_cache_key)

    if cached_preprocessing is None:
        train_df, preprocessing_summary = apply_cross_sectional_preprocessing(
            train_df,
            feature_columns=feature_columns,
            show_progress=False,
        )
        test_df, _ = apply_cross_sectional_preprocessing(
            test_df,
            feature_columns=feature_columns,
            show_progress=False,
        )
        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = "disabled" if args.disable_preprocessing_cache else "miss_written"
        preprocessing_summary["cache_key"] = preprocessing_cache_key if not args.disable_preprocessing_cache else None
        if not args.disable_preprocessing_cache:
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
                    "price_adjustment_mode": price_adjustment_mode,
                },
            )
    else:
        train_df, test_df, preprocessing_summary = cached_preprocessing
        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = "hit"
        preprocessing_summary["cache_key"] = preprocessing_cache_key

    dataset_summary = {
        "data_path": str(data_path),
        "sample_start_date": args.sample_start_date,
        "oos_start_date_used": args.oos_start_date,
        "target_horizon": args.target_horizon,
        "target_column": target_column,
        "price_adjustment_mode": price_adjustment_mode,
        "oos_rows": int(len(test_df)),
        "oos_dates": int(pd.to_datetime(test_df["date"]).nunique()),
        "oos_instruments": int(test_df["instrument_id"].nunique()),
        "oos_min_date": str(pd.to_datetime(test_df["date"]).min().date()),
        "oos_max_date": str(pd.to_datetime(test_df["date"]).max().date()),
        "preprocessing_cache_status": preprocessing_summary.get("cache_status"),
        "alpha_factor_scope": "all" if alpha_factor_names is None else list(alpha_factor_names),
        "winsorize_quantile": DEFAULT_WINSORIZE_QUANTILE,
        "apply_neutralization": True,
        "candidate_feature_columns": list(feature_columns),
    }
    return train_df, test_df, target_column, dataset_summary


def load_or_build_preprocessed_oos(args: argparse.Namespace) -> tuple[pd.DataFrame, str, dict[str, object]]:
    _, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    return test_df, target_column, dataset_summary


def build_spread_outputs(
    group_returns_df: pd.DataFrame,
    n_groups: int,
    rebalance_step: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if group_returns_df.empty:
        empty_df = pd.DataFrame(columns=["date", "bottom_group_return", "top_group_return", "long_short_spread"])
        empty_proxy_df = pd.DataFrame(columns=["date", "long_short_spread"])
        return empty_df, empty_proxy_df, {
            "overlap_spread_mean": float("nan"),
            "overlap_spread_std": float("nan"),
            "overlap_spread_sharpe_sqrt252": float("nan"),
            "non_overlap_periods": 0,
            "non_overlap_spread_mean": float("nan"),
            "non_overlap_spread_std": float("nan"),
            "non_overlap_sharpe_horizon_adj": float("nan"),
            "non_overlap_cumulative_return": float("nan"),
        }

    pivot_df = (
        group_returns_df.pivot_table(index="date", columns="quantile", values="average_forward_return")
        .sort_index()
        .copy()
    )
    if 1 not in pivot_df.columns or n_groups not in pivot_df.columns:
        raise ValueError("Quantile return table is missing top or bottom group columns.")

    spread_df = pd.DataFrame(
        {
            "date": pd.to_datetime(pivot_df.index),
            "bottom_group_return": pd.to_numeric(pivot_df[1], errors="coerce"),
            "top_group_return": pd.to_numeric(pivot_df[n_groups], errors="coerce"),
        }
    )
    spread_df["long_short_spread"] = spread_df["top_group_return"] - spread_df["bottom_group_return"]
    overlap_series = spread_df["long_short_spread"].dropna()

    overlap_mean = float(overlap_series.mean()) if not overlap_series.empty else float("nan")
    overlap_std = float(overlap_series.std(ddof=0)) if not overlap_series.empty else float("nan")
    overlap_sharpe = float("nan")
    if not overlap_series.empty and not math.isclose(overlap_std, 0.0, abs_tol=1e-12):
        overlap_sharpe = float(overlap_mean / overlap_std * math.sqrt(252))

    non_overlap_df = spread_df.iloc[::max(rebalance_step, 1)].copy()
    non_overlap_series = non_overlap_df["long_short_spread"].dropna()

    non_overlap_mean = float(non_overlap_series.mean()) if not non_overlap_series.empty else float("nan")
    non_overlap_std = float(non_overlap_series.std(ddof=0)) if not non_overlap_series.empty else float("nan")
    non_overlap_sharpe = float("nan")
    if not non_overlap_series.empty and not math.isclose(non_overlap_std, 0.0, abs_tol=1e-12):
        non_overlap_sharpe = float(non_overlap_mean / non_overlap_std * math.sqrt(252 / max(rebalance_step, 1)))

    non_overlap_cumulative = float("nan")
    if not non_overlap_series.empty:
        non_overlap_cumulative = float((1.0 + non_overlap_series).prod() - 1.0)

    metrics = {
        "overlap_spread_mean": overlap_mean,
        "overlap_spread_std": overlap_std,
        "overlap_spread_sharpe_sqrt252": overlap_sharpe,
        "non_overlap_periods": int(non_overlap_series.shape[0]),
        "non_overlap_spread_mean": non_overlap_mean,
        "non_overlap_spread_std": non_overlap_std,
        "non_overlap_sharpe_horizon_adj": non_overlap_sharpe,
        "non_overlap_cumulative_return": non_overlap_cumulative,
    }
    return spread_df, non_overlap_df[["date", "long_short_spread"]].copy(), metrics


def write_report(
    output_path: Path,
    factor_context: dict[str, object],
    dataset_summary: dict[str, object],
    summary_df: pd.DataFrame,
    average_group_returns_df: pd.DataFrame,
    spread_metrics: dict[str, float],
) -> None:
    factor_view = summary_df[
        [
            "factor",
            "feature_family",
            "selector_score",
            "model_importance",
            "pearson_ic_mean",
            "pearson_ic_ir",
            "spearman_ic_mean",
            "spearman_ic_ir",
            "long_short_spread",
            "group_monotonic_spearman",
        ]
    ].copy()

    overlap_table = pd.DataFrame(
        [
            {
                "metric": "overlap_spread_mean",
                "value": spread_metrics["overlap_spread_mean"],
                "note": "基于每个信号日期的 forward return spread，属于单因子诊断，不是组合回测。",
            },
            {
                "metric": "overlap_spread_sharpe_sqrt252",
                "value": spread_metrics["overlap_spread_sharpe_sqrt252"],
                "note": "同样基于重叠 forward return 序列，只能当信号强弱参考。",
            },
            {
                "metric": "non_overlap_spread_mean",
                "value": spread_metrics["non_overlap_spread_mean"],
                "note": "每隔一个目标周期取一次信号，作为更保守的 long-short 代理。",
            },
            {
                "metric": "non_overlap_sharpe_horizon_adj",
                "value": spread_metrics["non_overlap_sharpe_horizon_adj"],
                "note": "仍然不是带约束、带成本的正式组合 Sharpe。",
            },
            {
                "metric": "non_overlap_cumulative_return",
                "value": spread_metrics["non_overlap_cumulative_return"],
                "note": "仅供单因子策略原型检查。",
            },
        ]
    )

    report_text = f"""# Single Factor Case Study

## Factor

```json
{json.dumps(factor_context, ensure_ascii=False, indent=2)}
```

## Dataset

```json
{json.dumps(dataset_summary, ensure_ascii=False, indent=2)}
```

## OOS Single-Factor Diagnostics

{dataframe_to_markdown(factor_view)}

## Average Quantile Returns

{dataframe_to_markdown(average_group_returns_df.copy())}

## Signal vs Strategy Boundary

- `IC`、`分组收益`、`top-bottom spread` 属于单因子研究层。
- `overlap_spread_*` 直接基于 forward return 标签，不能当正式组合回测收益。
- `non_overlap_*` 只是把信号按目标周期抽样后的 long-short 代理检查。
- 真正的组合回测仍然要回到 `src/portfolio.py` 那一层去做约束、成本和持有重叠处理。

## Long-Short Proxy Metrics

{dataframe_to_markdown(overlap_table)}
"""
    output_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    model_dir = resolve_project_path(args.model_dir)
    output_root = resolve_project_path(args.output_dir)
    output_dir = output_root / sanitize_name(args.factor)
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_context, selector_scores_df, importance_scores_df = load_model_context(model_dir=model_dir, factor_name=args.factor)
    test_df, target_column, dataset_summary = load_or_build_preprocessed_oos(args)

    if args.factor not in test_df.columns:
        raise ValueError(f"Factor '{args.factor}' is not present in the reconstructed OOS feature table.")

    summary_df, daily_ic_df, group_returns_df, average_group_returns_df = summarize_factor_diagnostics(
        data=test_df,
        factor_columns=[args.factor],
        target_column="y",
        n_groups=args.n_groups,
        min_cross_section=args.min_cross_section,
        selector_scores=selector_scores_df,
        importance_scores=importance_scores_df,
        show_progress=False,
    )

    spread_df, non_overlap_df, spread_metrics = build_spread_outputs(
        group_returns_df=group_returns_df,
        n_groups=args.n_groups,
        rebalance_step=args.target_horizon,
    )

    dataset_summary = dict(dataset_summary)
    dataset_summary["target_column"] = target_column

    summary_row = summary_df.iloc[0].to_dict()
    merged_summary = {
        **factor_context,
        **dataset_summary,
        **summary_row,
        **spread_metrics,
    }

    summary_df.to_csv(output_dir / "factor_summary.csv", index=False)
    daily_ic_df.to_csv(output_dir / "factor_daily_ic.csv", index=False)
    group_returns_df.to_csv(output_dir / "factor_group_returns.csv", index=False)
    average_group_returns_df.to_csv(output_dir / "factor_average_group_returns.csv", index=False)
    spread_df.to_csv(output_dir / "factor_daily_spread.csv", index=False)
    non_overlap_df.to_csv(output_dir / "factor_non_overlap_spread.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(merged_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        output_path=output_dir / "report.md",
        factor_context=factor_context,
        dataset_summary=dataset_summary,
        summary_df=summary_df,
        average_group_returns_df=average_group_returns_df,
        spread_metrics=spread_metrics,
    )

    print(f"[Info] Single-factor case study finished for: {args.factor}")
    print(f"[Info] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
