"""训练报告生成模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.provenance import dumps_strict_json


def safe_corr(series_a: pd.Series, series_b: pd.Series, method: str = "pearson") -> float:
    """安全计算相关系数。"""

    valid_mask = series_a.notna() & series_b.notna()
    if valid_mask.sum() < 2:
        return float("nan")

    valid_a = series_a[valid_mask]
    valid_b = series_b[valid_mask]
    if valid_a.nunique() <= 1 or valid_b.nunique() <= 1:
        return float("nan")
    return float(valid_a.corr(valid_b, method=method))


def compute_daily_ic(merged_df: pd.DataFrame, method: str = "pearson") -> pd.Series:
    """按日期计算每日横截面 IC。"""

    # Avoid DataFrameGroupBy.apply here. New pandas releases changed whether
    # grouping columns are included and emit a deprecation warning. Explicit
    # per-date records keep the metric definition stable across supported versions.
    daily_ic = pd.Series(
        {
            date_value: safe_corr(group["predicted_y"], group["y"], method=method)
            for date_value, group in merged_df.groupby("date", sort=True)
        },
        dtype=float,
    )
    return daily_ic.dropna()


def assign_quantile_group(values: pd.Series, n_groups: int) -> pd.Series:
    """按预测值做分组。"""

    ranked = values.rank(method="first", pct=True)
    return pd.Series(np.ceil(ranked * n_groups).clip(1, n_groups).astype(int), index=values.index)


def compute_group_long_short(merged_df: pd.DataFrame, n_groups: int = 10) -> tuple[pd.DataFrame, float]:
    """计算分组收益和多空收益。"""

    grouped_df = merged_df.copy()
    grouped_df["group"] = grouped_df.groupby("date")["predicted_y"].transform(
        lambda series: assign_quantile_group(series, n_groups)
    )
    group_returns = (
        grouped_df.groupby(["date", "group"])["y"]
        .mean()
        .groupby("group")
        .mean()
        .reset_index()
        .rename(columns={"y": "average_actual_return"})
    )
    long_short_return = float(
        group_returns.loc[group_returns["group"] == n_groups, "average_actual_return"].mean()
        - group_returns.loc[group_returns["group"] == 1, "average_actual_return"].mean()
    )
    return group_returns, long_short_return


def summarize_ic_series(ic_series: pd.Series, prefix: str) -> Dict[str, float]:
    """汇总每日横截面 IC 的稳定性，而不把日期当作独立股票行混在一起。"""

    clean = pd.to_numeric(ic_series, errors="coerce").dropna()
    if clean.empty:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_icir": float("nan"),
            f"{prefix}_positive_ratio": float("nan"),
            f"{prefix}_date_count": 0.0,
        }

    mean_value = float(clean.mean())
    std_value = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    icir = mean_value / std_value if pd.notna(std_value) and abs(std_value) > 1e-12 else float("nan")
    return {
        f"{prefix}_mean": mean_value,
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_std": std_value,
        # 这里报告非年化 mean/std。样本存在重叠标签和序列相关，强行乘
        # sqrt(252) 会制造过度精确的“年化 ICIR”。
        f"{prefix}_icir": float(icir),
        f"{prefix}_positive_ratio": float((clean > 0.0).mean()),
        f"{prefix}_date_count": float(len(clean)),
    }


def calculate_prediction_metrics(result_df: pd.DataFrame) -> Dict[str, float]:
    """根据真实值与预测值计算一组常用指标。"""

    required_columns = {"date", "instrument_id", "y", "predicted_y"}
    missing_columns = required_columns - set(result_df.columns)
    if missing_columns:
        raise ValueError(f"Prediction metrics require columns: {sorted(missing_columns)}")

    eligible_rows = int(pd.to_numeric(result_df["y"], errors="coerce").notna().sum())
    clean_df = result_df.dropna(subset=["predicted_y", "y"]).copy()
    if clean_df.empty:
        return {}

    pearson_corr = safe_corr(clean_df["predicted_y"], clean_df["y"], method="pearson")
    spearman_corr = safe_corr(clean_df["predicted_y"], clean_df["y"], method="spearman")
    rmse = float(np.sqrt(np.mean((clean_df["predicted_y"] - clean_df["y"]) ** 2)))
    mae = float(np.mean(np.abs(clean_df["predicted_y"] - clean_df["y"])))
    pearson_ic = compute_daily_ic(clean_df, method="pearson")
    spearman_ic = compute_daily_ic(clean_df, method="spearman")
    group_returns, long_short_return = compute_group_long_short(clean_df, n_groups=10)
    group_monotonicity = safe_corr(
        pd.to_numeric(group_returns["group"], errors="coerce"),
        pd.to_numeric(group_returns["average_actual_return"], errors="coerce"),
        method="spearman",
    )

    pearson_ic_summary = summarize_ic_series(pearson_ic, "pearson_ic")
    spearman_ic_summary = summarize_ic_series(spearman_ic, "spearman_ic")

    return {
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "rmse": rmse,
        "mae": mae,
        # 这是同日分组的平均 Top-Bottom 前瞻收益差，没有建模持仓、
        # 换手或交易成本。新报告使用 spread 命名，旧字段暂时保留以
        # 兼容已有分析脚本和历史输出。
        "long_short_spread": long_short_return,
        "long_short_return": long_short_return,
        "group_monotonic_spearman": group_monotonicity,
        "prediction_coverage_ratio": float(len(clean_df) / eligible_rows) if eligible_rows else float("nan"),
        "evaluation_rows": float(len(clean_df)),
        "evaluation_date_count": float(pd.to_datetime(clean_df["date"]).nunique()),
        "evaluation_instrument_count": float(clean_df["instrument_id"].nunique()),
        **pearson_ic_summary,
        **spearman_ic_summary,
    }


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """把小表格转成 Markdown。"""

    if df.empty:
        return "_No data available._"

    headers = " | ".join(df.columns)
    separators = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(str(value) for value in row) for row in df.astype(str).itertuples(index=False, name=None)]
    return "\n".join([f"| {headers} |", f"| {separators} |"] + [f"| {row} |" for row in rows])


def _model_params_to_markdown(model_params: Dict) -> str:
    """把模型参数字典转成 Markdown。

    这里使用动态生成而不是写死 LightGBM / XGBoost，
    因为当前项目已经支持多模型组合，用户还可以通过命令行参数
    自由决定到底启用哪些模型。
    """

    if not model_params:
        return "_No model parameters available._"

    sections = []
    for model_name, params in model_params.items():
        display_name = model_name.replace("_", " ").title()
        sections.append(
            "\n".join(
                [
                    f"### {display_name}",
                    "",
                    "```json",
                    dumps_strict_json(params),
                    "```",
                ]
            )
        )
    return "\n\n".join(sections)


def write_training_report(
    output_path: str | Path,
    dataset_summary: Dict,
    feature_metadata: Dict,
    preprocessing_summary: Dict,
    selector_summary: Dict,
    test_metrics: Dict,
    model_params: Dict,
    top_score_features: pd.DataFrame,
    top_importance_features: pd.DataFrame,
    validation_summary_df: pd.DataFrame | None = None,
    model_weights: dict[str, float] | None = None,
    stage_timing_df: pd.DataFrame | None = None,
    final_model_timing_df: pd.DataFrame | None = None,
) -> None:
    """生成训练结果 Markdown 报告。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_counts = feature_metadata.get("feature_counts", {})
    context_family_map = feature_metadata.get("context_feature_family_map", {}) or {}
    context_family_counts = {
        family_name: len(list(feature_list))
        for family_name, feature_list in context_family_map.items()
    }
    preprocessing_summary = preprocessing_summary or {}
    size_neutralization_used = bool(preprocessing_summary.get("size_neutralization_used", False))
    sector_neutralization_used = bool(preprocessing_summary.get("sector_neutralization_used", False))
    stage_counts = selector_summary.get("stage_feature_counts", {})
    validation_summary_df = validation_summary_df if validation_summary_df is not None else pd.DataFrame()
    model_weights = model_weights or {}
    stage_timing_df = stage_timing_df if stage_timing_df is not None else pd.DataFrame()
    final_model_timing_df = final_model_timing_df if final_model_timing_df is not None else pd.DataFrame()

    report = f"""# Training Report

## 1. Dataset Summary

- Data path: `{dataset_summary.get("data_path")}`
- Date range: `{dataset_summary.get("min_date")}` to `{dataset_summary.get("max_date")}`
- Sample start date used: `{dataset_summary.get("sample_start_date", "N/A")}`
- OOS start date used: `{dataset_summary.get("oos_start_date_used", "N/A")}`
- Active target horizon (days): `{dataset_summary.get("target_horizon", "N/A")}`
- Active target column: `{dataset_summary.get("target_column", "N/A")}`
- Price adjustment mode: `{dataset_summary.get("price_adjustment_mode", "N/A")}`
- Label purge policy: `{dataset_summary.get("label_purge_policy", "N/A")}`
- Universe label: `{dataset_summary.get("universe_label", dataset_summary.get("universe", "N/A"))}`
- Instruments: `{dataset_summary.get("instrument_count")}`
- Total rows: `{dataset_summary.get("n_rows")}`
- In-sample train rows: `{dataset_summary.get("train_rows")}`
- OOS test rows: `{dataset_summary.get("test_rows")}`
- In-sample train date range: `{dataset_summary.get("train_min_date", "N/A")}` to `{dataset_summary.get("train_max_date", "N/A")}`
- OOS test date range: `{dataset_summary.get("test_min_date", "N/A")}` to `{dataset_summary.get("test_max_date", "N/A")}`
- Walk-forward folds: `{dataset_summary.get("n_splits", "N/A")}`
- Resolved model count: `{dataset_summary.get("resolved_model_count", "N/A")}`
- Experiment name: `{dataset_summary.get("experiment_name", "N/A")}`
- Experiment description: `{dataset_summary.get("experiment_description", "N/A")}`
- Feature subset mode: `{dataset_summary.get("feature_subset_mode", "N/A")}`
- Preprocessing mode: `{dataset_summary.get("preprocessing_mode", "N/A")}`

## 2. Feature Inventory

- Raw market features: `{feature_counts.get("raw_feature_count", 0)}`
- Fundamental raw features already present in CSV: `{feature_counts.get("fundamental_raw_count", 0)}`
- Base technical features: `{feature_counts.get("base_feature_count", 0)}`
- Advanced technical indicators: `{feature_counts.get("advanced_feature_count", 0)}`
- Context features: `{feature_counts.get("context_feature_count", 0)}`
- Context feature families: `{dumps_strict_json(context_family_counts, indent=None)}`
- Alpha191 features generated: `{feature_counts.get("alpha_feature_count", 0)}`
- Total candidate features before selection: `{feature_counts.get("candidate_feature_count", 0)}`

## 3. Cross-Sectional Preprocessing

```json
{dumps_strict_json(preprocessing_summary)}
```

- `winsorize_quantile` 表示每天横截面会裁掉最极端的一小部分取值，降低异常点影响。
- `zscore_applied = true` 表示每天都会把因子重新标准化到相近尺度。
- `size_neutralization_used = {str(size_neutralization_used).lower()}` 记录本次运行是否实际使用了 `log_market_cap` 暴露。
- `sector_neutralization_used = {str(sector_neutralization_used).lower()}` 记录本次运行是否实际使用了行业 dummy 暴露。

## 4. Feature Selection Pipeline

| Stage | Remaining Features |
| --- | ---: |
| Initial candidate features | {stage_counts.get("initial", 0)} |
| After missing-ratio filter | {stage_counts.get("after_missing_filter", 0)} |
| After low-variance filter | {stage_counts.get("after_variance_filter", 0)} |
| After high-correlation filter | {stage_counts.get("after_correlation_filter", 0)} |
| Final selected Top N | {stage_counts.get("after_top_n_selection", 0)} |

### Why The Pipeline Can Still Run Fast

- Even though the project now supports many Alpha191 factors, most factors are implemented with vectorized `pandas` operations and rolling windows rather than slow Python row-by-row loops.
- The model does **not** train on all candidate variables forever. After feature selection, only `{stage_counts.get("after_top_n_selection", 0)}` features enter the final model training stage.
- Tree models in this project are trained on tabular daily data rather than extremely large intraday tensors, so runtime is often much shorter than people first expect.
- If your current dataset itself is not very large, the whole pipeline will naturally finish quickly.

## 5. Model Hyperparameters

{_model_params_to_markdown(model_params)}

### Feature Selector

```json
{dumps_strict_json(selector_summary)}
```

## 6. Walk-Forward Validation Summary

{_dataframe_to_markdown(validation_summary_df)}

### Ensemble Weights

```json
{dumps_strict_json(model_weights)}
```

## 7. Runtime Breakdown

### Stage Timing

{_dataframe_to_markdown(stage_timing_df)}

### Final Model Timing

{_dataframe_to_markdown(final_model_timing_df)}

## 8. OOS Test Metrics

```json
{dumps_strict_json(test_metrics)}
```

`long_short_spread` is the cross-date mean of the same-date Top-minus-Bottom
forward-label spread. `long_short_return` is retained only as a compatibility
alias. Neither field is cumulative portfolio return, annualized return, or a
cost-adjusted backtest result.

## 9. Best Features By Selector Score

{_dataframe_to_markdown(top_score_features)}

## 10. Best Features By Model Importance

{_dataframe_to_markdown(top_importance_features)}

## 11. Notes On Technical Indicators

- This project now includes many classic indicators in the feature pipeline, such as `EMA`, `MACD`, `DMA`, `VMA`, `RSI`, `WR`, `RSV`, `KDJ`, `UOS`, `BOLL`, `MIKE`, `XSChannel`, `OBV`, `AMT`-style turnover features, and `MarketCap`.
- Fundamental variables such as `EPS`, `PE`, `PB`, `PS`, `ROE`, `ROA`, `YoY`, and `QoQ` are supported when they are present in the training CSV. The optional FMP workflow can build those columns using filing/accepted-date availability rules before training.
- The canonical US300 experiment currently has zero fundamental-feature coverage. Missing accounting fields and market capitalization remain missing; the pipeline does not fabricate them from OHLCV.
"""

    with output_path.open("w", encoding="utf-8") as file:
        file.write(report)
