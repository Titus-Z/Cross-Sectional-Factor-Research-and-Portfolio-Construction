"""因子诊断模块。

这个模块不重新训练模型，职责是回答三个更基础的问题：

1. 某个单独特征在 OOS 上到底有没有横截面排序能力？
2. 这种能力是稳定的，还是只在少数日期里偶然出现？
3. 如果把股票按该特征分组，最高组和最低组之间有没有可解释的收益差？

这里默认把“因子诊断”理解得稍微宽一点：

- 既可以诊断 Alpha191 因子；
- 也可以诊断技术指标；
- 也可以诊断原始量价特征。

原因很简单：在实际研究里，最终进入模型的“有效信号”不一定都来自传统意义上的 Alpha 公式，
很多时候技术指标、波动率、通道宽度这类特征同样是值得单独诊断的。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.progress import create_progress_bar, format_duration
from src.reporting import assign_quantile_group, safe_corr


RAW_MARKET_FEATURES = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "adjustment",
    "market_cap",
    "turnover",
    "log_return",
}

FUNDAMENTAL_KEYWORDS = {
    "eps",
    "pe",
    "pb",
    "ps",
    "roe",
    "roa",
    "yoy",
    "qoq",
}

SECTOR_CONTEXT_KEYWORDS = {
    "sector_",
    "stock_excess_sector",
    "sector_relative",
}

MARKET_STATE_KEYWORDS = {
    "market_return",
    "market_volatility",
    "market_high_vol_regime",
    "market_cross_sectional_dispersion",
    "market_breadth",
    "stock_excess_market",
}

MACRO_CONTEXT_KEYWORDS = {
    "macro_",
}


def classify_feature_family(feature_name: str) -> str:
    """把特征粗分类，方便在诊断报告里快速看结构。

    这里不追求绝对严格，而是给一个稳定、好读的分组：

    - `alpha191`
    - `fundamental`
    - `sector_context`
    - `market_state`
    - `macro_context`
    - `raw_market`
    - `technical`
    """

    lowered = feature_name.lower()
    if lowered.startswith("alpha"):
        return "alpha191"
    if any(keyword in lowered for keyword in MACRO_CONTEXT_KEYWORDS):
        return "macro_context"
    if any(keyword in lowered for keyword in MARKET_STATE_KEYWORDS):
        return "market_state"
    if any(keyword in lowered for keyword in SECTOR_CONTEXT_KEYWORDS):
        return "sector_context"
    if lowered in RAW_MARKET_FEATURES:
        return "raw_market"
    if any(keyword in lowered for keyword in FUNDAMENTAL_KEYWORDS):
        return "fundamental"
    return "technical"


def _safe_quantile_assignment(values: pd.Series, n_groups: int) -> pd.Series:
    """对单个横截面做稳健分组。

    如果当天有效样本太少，或者因子值几乎没变化，
    强行做分组只会得到没有意义的结果。
    这里统一返回缺失，后面直接跳过这一天。
    """

    valid = pd.to_numeric(values, errors="coerce")
    if valid.notna().sum() < n_groups:
        return pd.Series(np.nan, index=values.index)
    if valid.nunique(dropna=True) <= 1:
        return pd.Series(np.nan, index=values.index)

    grouped = assign_quantile_group(valid, n_groups=n_groups).astype(float)
    grouped[valid.isna()] = np.nan
    return grouped


def compute_daily_factor_ic(
    data: pd.DataFrame,
    factor_column: str,
    target_column: str = "y",
    method: str = "pearson",
    min_cross_section: int = 20,
) -> pd.Series:
    """按日期计算某个单因子的横截面 IC 序列。"""

    if factor_column not in data.columns:
        raise ValueError(f"Factor column '{factor_column}' is not present in the input data.")
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the input data.")

    daily_ic: dict[pd.Timestamp, float] = {}

    for current_date, group in data.groupby("date"):
        valid_group = group[[factor_column, target_column]].dropna()
        if len(valid_group) < min_cross_section:
            continue
        ic_value = safe_corr(valid_group[factor_column], valid_group[target_column], method=method)
        if pd.notna(ic_value):
            daily_ic[pd.Timestamp(current_date)] = float(ic_value)

    return pd.Series(daily_ic, dtype=float).sort_index()


def compute_factor_turnover_proxies(
    data: pd.DataFrame,
    factor_column: str,
    top_fraction: float = 0.20,
    min_cross_section: int = 20,
) -> dict[str, float | int]:
    """用相邻交易日的横截面排名变化估算单因子换手。

    这里计算的是因子稳定性代理，还不是包含持仓、成本和成交限制的
    真实组合换手。两个输出指标分别回答：

    - ``rank_turnover_mean``：相邻日共同股票的百分位排名平均绝对变化；
      值越大，信号排名越不稳定。
    - ``top_retention_mean``：上一日 Top 20% 股票在下一日仍留在
      Top 20% 的比例；值越大，顶部股票集合越稳定。

    只比较相邻两天都存在的股票，避免 IPO、停牌或数据缺口被误认为
    因子排名变化。
    """

    if not 0.0 < float(top_fraction) < 1.0:
        raise ValueError("top_fraction must be between 0 and 1.")
    required_columns = {"date", "instrument_id", factor_column}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Turnover proxy input is missing columns: {sorted(missing_columns)}")

    rank_changes: list[float] = []
    top_retentions: list[float] = []
    previous_ranks: pd.Series | None = None
    previous_top: set[str] | None = None

    ordered = data[["date", "instrument_id", factor_column]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered[factor_column] = pd.to_numeric(ordered[factor_column], errors="coerce")

    for _, current_group in ordered.sort_values(["date", "instrument_id"]).groupby("date", sort=True):
        current_group = current_group.dropna(subset=["instrument_id", factor_column]).copy()
        # 同一日同一股票如果意外重复，只保留最后一条，避免索引非唯一。
        current_group = current_group.drop_duplicates(subset=["instrument_id"], keep="last")
        if len(current_group) < int(min_cross_section):
            continue

        current_group["instrument_id"] = current_group["instrument_id"].astype(str)
        current_ranks = current_group.set_index("instrument_id")[factor_column].rank(pct=True)
        top_threshold = 1.0 - float(top_fraction)
        current_top = set(current_ranks[current_ranks >= top_threshold].index.astype(str))

        if previous_ranks is not None:
            common_names = previous_ranks.index.intersection(current_ranks.index)
            if len(common_names) >= max(2, int(min_cross_section) // 2):
                rank_change = (current_ranks.loc[common_names] - previous_ranks.loc[common_names]).abs().mean()
                if pd.notna(rank_change):
                    rank_changes.append(float(rank_change))

        if previous_top:
            # 分母使用上一日 Top 集合，表示旧持仓的留存率。
            top_retentions.append(float(len(previous_top & current_top) / len(previous_top)))

        previous_ranks = current_ranks
        previous_top = current_top

    return {
        "rank_turnover_mean": float(np.mean(rank_changes)) if rank_changes else float("nan"),
        "top_retention_mean": (
            float(np.mean(top_retentions)) if top_retentions else float("nan")
        ),
        "top_retention_fraction": float(top_fraction),
        "turnover_transition_count": int(len(rank_changes)),
    }


def compute_factor_group_returns(
    data: pd.DataFrame,
    factor_column: str,
    target_column: str = "y",
    n_groups: int = 5,
    min_cross_section: int = 20,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """对单个因子做分组收益分析。

    输出有两部分：

    1. 每个分组的平均未来收益；
    2. 一组摘要指标，例如 top-bottom spread 和单调性。
    """

    if factor_column not in data.columns:
        raise ValueError(f"Factor column '{factor_column}' is not present in the input data.")

    group_return_records: list[dict[str, Any]] = []

    for current_date, group in data.groupby("date"):
        valid_group = group[[factor_column, target_column]].dropna().copy()
        if len(valid_group) < min_cross_section:
            continue

        quantile_series = _safe_quantile_assignment(valid_group[factor_column], n_groups=n_groups)
        valid_group["quantile"] = quantile_series.values
        valid_group = valid_group.dropna(subset=["quantile"])
        if valid_group.empty:
            continue

        date_group_returns = (
            valid_group.groupby("quantile", as_index=False)[target_column]
            .mean()
            .rename(columns={target_column: "average_forward_return"})
        )
        date_group_returns["date"] = pd.Timestamp(current_date)
        date_group_returns["factor"] = factor_column
        group_return_records.extend(date_group_returns.to_dict("records"))

    if not group_return_records:
        empty_group_df = pd.DataFrame(columns=["factor", "date", "quantile", "average_forward_return"])
        empty_summary = {
            "long_short_spread": float("nan"),
            "group_monotonic_spearman": float("nan"),
            "quantile_date_count": 0,
        }
        return empty_group_df, empty_summary

    group_returns_df = pd.DataFrame(group_return_records)
    average_group_returns = (
        group_returns_df.groupby("quantile", as_index=False)["average_forward_return"]
        .mean()
        .sort_values("quantile")
        .reset_index(drop=True)
    )

    long_short_spread = float(
        average_group_returns.loc[average_group_returns["quantile"] == n_groups, "average_forward_return"].mean()
        - average_group_returns.loc[average_group_returns["quantile"] == 1, "average_forward_return"].mean()
    )

    group_monotonic_spearman = safe_corr(
        pd.Series(average_group_returns["quantile"], dtype=float),
        pd.Series(average_group_returns["average_forward_return"], dtype=float),
        method="spearman",
    )

    summary = {
        "long_short_spread": long_short_spread,
        "group_monotonic_spearman": float(group_monotonic_spearman) if pd.notna(group_monotonic_spearman) else float("nan"),
        "quantile_date_count": int(group_returns_df["date"].nunique()),
    }
    return group_returns_df, summary


def _summarize_ic_series(ic_series: pd.Series, prefix: str) -> dict[str, float]:
    """把 IC 序列压缩成更适合排序和报告的摘要指标。"""

    if ic_series.empty:
        return {
            f"{prefix}_ic_mean": float("nan"),
            f"{prefix}_ic_std": float("nan"),
            f"{prefix}_ic_ir": float("nan"),
            f"{prefix}_ic_positive_ratio": float("nan"),
            f"{prefix}_ic_days": 0,
        }

    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std(ddof=0))
    ic_ir = float(ic_mean / ic_std) if abs(ic_std) > 1e-12 else float("nan")
    positive_ratio = float((ic_series > 0).mean())
    return {
        f"{prefix}_ic_mean": ic_mean,
        f"{prefix}_ic_std": ic_std,
        f"{prefix}_ic_ir": ic_ir,
        f"{prefix}_ic_positive_ratio": positive_ratio,
        f"{prefix}_ic_days": int(ic_series.shape[0]),
    }


def summarize_factor_diagnostics(
    data: pd.DataFrame,
    factor_columns: list[str],
    target_column: str = "y",
    n_groups: int = 5,
    min_cross_section: int = 20,
    selector_scores: pd.DataFrame | None = None,
    importance_scores: pd.DataFrame | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对一组因子做 OOS 诊断。

    返回四张表：

    1. `summary_df`：每个因子的诊断摘要；
    2. `daily_ic_df`：逐日 IC 明细；
    3. `group_returns_df`：逐组收益明细；
    4. `average_group_returns_df`：每个因子的平均分组收益概览。
    """

    summary_records: list[dict[str, Any]] = []
    daily_ic_records: list[dict[str, Any]] = []
    group_return_frames: list[pd.DataFrame] = []
    average_group_return_records: list[dict[str, Any]] = []

    score_map = {}
    if selector_scores is not None and not selector_scores.empty and {"feature", "score"}.issubset(selector_scores.columns):
        score_map = dict(zip(selector_scores["feature"], selector_scores["score"]))

    importance_map = {}
    if importance_scores is not None and not importance_scores.empty and {"feature", "importance"}.issubset(importance_scores.columns):
        importance_map = dict(zip(importance_scores["feature"], importance_scores["importance"]))

    progress_bar = create_progress_bar(
        total=len(factor_columns),
        description="Factor diagnostics",
        enabled=show_progress,
    )
    diagnostics_start_time = time.perf_counter()

    for factor_index, factor_name in enumerate(factor_columns, start=1):
        single_factor_start = time.perf_counter()
        valid_series = pd.to_numeric(data[factor_name], errors="coerce")
        coverage_ratio = float(valid_series.notna().mean())
        valid_rows = int(valid_series.notna().sum())

        pearson_ic = compute_daily_factor_ic(
            data=data,
            factor_column=factor_name,
            target_column=target_column,
            method="pearson",
            min_cross_section=min_cross_section,
        )
        spearman_ic = compute_daily_factor_ic(
            data=data,
            factor_column=factor_name,
            target_column=target_column,
            method="spearman",
            min_cross_section=min_cross_section,
        )

        for current_date, ic_value in pearson_ic.items():
            daily_ic_records.append(
                {
                    "factor": factor_name,
                    "date": pd.Timestamp(current_date),
                    "method": "pearson",
                    "ic": float(ic_value),
                }
            )

        for current_date, ic_value in spearman_ic.items():
            daily_ic_records.append(
                {
                    "factor": factor_name,
                    "date": pd.Timestamp(current_date),
                    "method": "spearman",
                    "ic": float(ic_value),
                }
            )

        factor_group_returns_df, group_summary = compute_factor_group_returns(
            data=data,
            factor_column=factor_name,
            target_column=target_column,
            n_groups=n_groups,
            min_cross_section=min_cross_section,
        )
        turnover_summary = compute_factor_turnover_proxies(
            data=data,
            factor_column=factor_name,
            top_fraction=0.20,
            min_cross_section=min_cross_section,
        )
        if not factor_group_returns_df.empty:
            group_return_frames.append(factor_group_returns_df)
            average_group_returns = (
                factor_group_returns_df.groupby("quantile", as_index=False)["average_forward_return"]
                .mean()
                .sort_values("quantile")
            )
            for _, row in average_group_returns.iterrows():
                average_group_return_records.append(
                    {
                        "factor": factor_name,
                        "quantile": int(row["quantile"]),
                        "average_forward_return": float(row["average_forward_return"]),
                    }
                )

        summary_record = {
            "factor": factor_name,
            "feature_family": classify_feature_family(factor_name),
            "coverage_ratio": coverage_ratio,
            "valid_rows": valid_rows,
            "oos_rows": int(len(data)),
            "oos_dates": int(pd.to_datetime(data["date"]).nunique()),
            "selector_score": float(score_map.get(factor_name, np.nan)),
            "model_importance": float(importance_map.get(factor_name, np.nan)),
            **_summarize_ic_series(pearson_ic, prefix="pearson"),
            **_summarize_ic_series(spearman_ic, prefix="spearman"),
            **group_summary,
            **turnover_summary,
        }
        summary_records.append(summary_record)

        single_factor_elapsed = time.perf_counter() - single_factor_start
        total_elapsed = time.perf_counter() - diagnostics_start_time
        average_factor_elapsed = total_elapsed / max(factor_index, 1)
        estimated_remaining = average_factor_elapsed * max(len(factor_columns) - factor_index, 0)
        progress_bar.update(1)
        progress_bar.set_postfix_str(
            (
                f"{factor_name} {format_duration(single_factor_elapsed)} | "
                f"total {format_duration(total_elapsed)} | "
                f"est left {format_duration(estimated_remaining)}"
            )
        )

    progress_bar.close()

    summary_df = pd.DataFrame(summary_records).sort_values(
        ["pearson_ic_mean", "long_short_spread", "selector_score"],
        ascending=[False, False, False],
        na_position="last",
    )
    daily_ic_df = pd.DataFrame(daily_ic_records)
    group_returns_df = (
        pd.concat(group_return_frames, ignore_index=True)
        if group_return_frames
        else pd.DataFrame(columns=["factor", "date", "quantile", "average_forward_return"])
    )
    average_group_returns_df = pd.DataFrame(average_group_return_records)
    return summary_df, daily_ic_df, group_returns_df, average_group_returns_df


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """把小表格转成 Markdown。"""

    if df.empty:
        return "_No data available._"

    headers = " | ".join(df.columns)
    separators = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(str(value) for value in row) for row in df.astype(str).itertuples(index=False, name=None)]
    return "\n".join([f"| {headers} |", f"| {separators} |"] + [f"| {row} |" for row in rows])


def write_factor_diagnostics_report(
    output_path: str | Path,
    dataset_summary: dict[str, Any],
    feature_source_summary: dict[str, Any],
    summary_df: pd.DataFrame,
    average_group_returns_df: pd.DataFrame,
    stage_timing_df: pd.DataFrame | None = None,
) -> None:
    """生成因子诊断 Markdown 报告。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage_timing_df = stage_timing_df if stage_timing_df is not None else pd.DataFrame()

    top_ic_df = summary_df[
        [
            "factor",
            "feature_family",
            "selector_score",
            "model_importance",
            "pearson_ic_mean",
            "spearman_ic_mean",
            "long_short_spread",
            "group_monotonic_spearman",
            "rank_turnover_mean",
            "top_retention_mean",
        ]
    ].head(15)

    top_spread_df = summary_df[
        [
            "factor",
            "feature_family",
            "selector_score",
            "model_importance",
            "long_short_spread",
            "pearson_ic_mean",
            "spearman_ic_mean",
            "group_monotonic_spearman",
            "rank_turnover_mean",
            "top_retention_mean",
        ]
    ].sort_values("long_short_spread", ascending=False).head(15)

    report_text = f"""# Factor Diagnostics Report

## 1. Dataset Summary

- Data path: `{dataset_summary.get("data_path")}`
- Date range: `{dataset_summary.get("min_date")}` to `{dataset_summary.get("max_date")}`
- Sample start date used: `{dataset_summary.get("sample_start_date", "N/A")}`
- OOS start date used: `{dataset_summary.get("oos_start_date_used", "N/A")}`
- Target horizon: `{dataset_summary.get("target_horizon", "N/A")}`
- Target column: `{dataset_summary.get("target_column", "N/A")}`
- OOS rows diagnosed: `{dataset_summary.get("test_rows", "N/A")}`
- OOS dates diagnosed: `{dataset_summary.get("test_date_count", "N/A")}`
- OOS instruments: `{dataset_summary.get("test_instrument_count", "N/A")}`

## 2. Feature Source

```json
{pd.Series(feature_source_summary).to_json(indent=2, force_ascii=False)}
```

## 3. Best Factors By OOS Pearson IC

{dataframe_to_markdown(top_ic_df)}

## 4. Best Factors By OOS Top-Bottom Spread

{dataframe_to_markdown(top_spread_df)}

## 5. Average Quantile Returns For Top IC Factors

{dataframe_to_markdown(average_group_returns_df[average_group_returns_df["factor"].isin(top_ic_df["factor"].head(5))].copy())}

## 6. Runtime Breakdown

{dataframe_to_markdown(stage_timing_df)}

## 7. Interpretation Notes

- `pearson_ic_mean` / `spearman_ic_mean` 是按日期做横截面相关后再取平均，重点看排序能力是否稳定。
- `long_short_spread` 是把单个因子按分组排序后，最高组平均未来收益减去最低组平均未来收益。
- `group_monotonic_spearman` 越接近 `1`，说明分组收益越接近“因子越高，未来收益越高”的单调关系。
- `rank_turnover_mean` 是相邻日共同股票的百分位排名平均绝对变化，越低通常表示信号越稳定。
- `top_retention_mean` 是上一日 Top 20% 股票在下一日仍位于 Top 20% 的比例。
- 上述两项是因子换手代理，不等同于包含持仓重叠、成本和执行约束的组合换手。
- 这份报告默认只看 OOS 样本，目的是确认哪些信号在未来数据上依然成立。
"""

    output_path.write_text(report_text, encoding="utf-8")
