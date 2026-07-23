"""Long-short portfolio backtest layer.

这个模块把模型预测分数 `predicted_y` 转成更接近真实研究流程的多空组合：

- Top-K 股票做多；
- Bottom-K 股票做空；
- 支持全市场排序和行业内排序两种选股方式；
- 用真实 close-to-close 日收益重算组合收益；
- 显式扣除换手成本，并支持简化的年化借券费敏感性；
- 输出 Sharpe、Max Drawdown、Calmar、Turnover Cost 等组合指标。

注意：这仍然是研究型 backtest-style diagnostic，不是实盘交易系统。
它没有处理盘中成交、借券可得性、真实 bid-ask spread、税务和订单撮合。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio import (
    SUPPORTED_HOLDING_CLOCKS,
    load_market_snapshot_frame,
    load_prediction_frame,
    merge_predictions_with_market,
    resolve_holding_window,
)
from src.project_paths import PROJECT_ROOT
from src.provenance import dumps_strict_json, project_relative_path


@dataclass(frozen=True)
class LongShortBacktestConfig:
    """单次多空组合回测配置。

    这个 dataclass 的作用是把一次回测需要的关键假设集中在一起，
    避免函数参数越来越长之后看不清楚当前回测到底用了什么口径。
    """

    run_name: str
    predictions_path: Path
    data_path: Path
    output_dir: Path
    hold_days: int
    step_days: int
    top_k: int
    cost_bps: float
    neutral_mode: str
    signal_delay_days: int = 1
    holding_clock: str = "signal_horizon"
    borrow_cost_bps: float = 0.0
    weight_mode: str = "equal_weight"
    max_abs_weight: float | None = None
    write_outputs: bool = True
    price_adjustment_mode: str = "vendor_adjusted"


def slugify_name(text: str) -> str:
    """把运行名转成适合文件夹使用的短字符串。"""

    cleaned = re.sub(r"[^0-9A-Za-z_\\-]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "long_short_run"


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """安全地把 DataFrame 转成 Markdown。

    pandas 的 `to_markdown` 依赖 `tabulate`。如果本地环境缺这个包，
    这里会退回到普通 CSV 文本，避免报告生成阶段因为展示格式失败。
    """

    if df.empty:
        return "_No data available._"
    preview = df.head(max_rows).copy() if max_rows is not None else df.copy()
    try:
        return preview.to_markdown(index=False)
    except Exception:
        return "```text\n" + preview.to_csv(index=False) + "```"


def compute_nav_max_drawdown(nav_series: pd.Series) -> float:
    """计算净值曲线的最大回撤。

    返回值是负数。例如 `-0.12` 表示从历史高点最多跌了 12%。
    """

    if nav_series.empty:
        return float("nan")
    running_peak = nav_series.cummax()
    drawdown = nav_series / running_peak - 1.0
    return float(drawdown.min())


def annualize_return(total_return: float, daily_count: int) -> float:
    """把回测期总收益年化。

    OOS 样本很短时，年化收益会被明显放大；报告里会单独提示这个限制。
    """

    if daily_count <= 0:
        return float("nan")
    ending_nav = 1.0 + float(total_return)
    if ending_nav <= 0:
        return float("nan")
    return float(ending_nav ** (252.0 / daily_count) - 1.0)


def summarize_long_short_metrics(
    daily_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    sector_df: pd.DataFrame,
) -> dict[str, float | int | str]:
    """汇总多空组合的核心绩效指标。"""

    if daily_df.empty:
        return {}

    daily_returns = daily_df["net_return"].astype(float)
    gross_returns = daily_df["gross_return"].astype(float)
    total_return = float(daily_df["portfolio_nav"].iloc[-1] - 1.0)
    annualized_return = annualize_return(total_return, len(daily_df))
    daily_std = float(daily_returns.std(ddof=1)) if len(daily_df) > 1 else float("nan")
    annualized_vol = daily_std * math.sqrt(252.0) if pd.notna(daily_std) else float("nan")
    # 标准零无风险利率 Sharpe：日均收益 / 日收益样本标准差 × sqrt(252)。
    # 不能使用“复合年化收益 / 年化波动率”，后者在短期强上涨样本中会被放大。
    sharpe = (
        float(daily_returns.mean()) / daily_std * math.sqrt(252.0)
        if pd.notna(daily_std) and daily_std > 1e-12
        else float("nan")
    )
    max_drawdown = compute_nav_max_drawdown(daily_df["portfolio_nav"])
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < -1e-12 else float("nan")

    dated_returns = daily_df[["date", "net_return"]].copy()
    dated_returns["date"] = pd.to_datetime(dated_returns["date"])
    dated_returns = dated_returns.sort_values("date").reset_index(drop=True)
    dated_returns["month"] = dated_returns["date"].dt.to_period("M").astype(str)
    monthly_returns = dated_returns.groupby("month")["net_return"].apply(
        lambda values: float((1.0 + values.astype(float)).prod() - 1.0)
    )
    midpoint = max(1, len(dated_returns) // 2)
    first_half_return = float((1.0 + dated_returns.iloc[:midpoint]["net_return"]).prod() - 1.0)
    second_half_return = float((1.0 + dated_returns.iloc[midpoint:]["net_return"]).prod() - 1.0)

    if not turnover_df.empty:
        avg_gross_turnover = float(turnover_df["gross_turnover"].mean())
        avg_net_turnover = float(turnover_df["net_turnover"].mean())
        avg_turnover_cost_bps = float(turnover_df["turnover_cost_bps"].mean())
        # `turnover_df` stores capital-scaled portfolio turnover. The daily ledger is
        # still the authoritative source for actually deducted costs because it only
        # contains trades that produced a valid return path.
        total_turnover_cost = float(daily_df["transaction_cost"].sum())
    else:
        avg_gross_turnover = avg_net_turnover = avg_turnover_cost_bps = float("nan")
        total_turnover_cost = float("nan")

    # Exposure must be measured from the combined daily portfolio. Averaging the
    # raw +1/-1 books of each overlapping sleeve would report per-sleeve leverage
    # and can overstate portfolio exposure by the number of active sleeves.
    exposure_columns = {
        "long_exposure",
        "short_exposure_abs",
        "gross_exposure",
        "net_exposure",
    }
    if exposure_columns.issubset(daily_df.columns):
        avg_long_weight = float(daily_df["long_exposure"].mean())
        avg_short_weight = float(daily_df["short_exposure_abs"].mean())
        avg_gross_exposure = float(daily_df["gross_exposure"].mean())
        avg_net_exposure = float(daily_df["net_exposure"].mean())
    else:
        avg_long_weight = avg_short_weight = float("nan")
        avg_gross_exposure = avg_net_exposure = float("nan")

    if not sector_df.empty:
        avg_max_abs_sector_net = float(
            sector_df.groupby("signal_date")["net_sector_weight"]
            .apply(lambda series: float(np.abs(series).max()))
            .mean()
        )
        avg_total_abs_sector_net = float(
            sector_df.groupby("signal_date")["net_sector_weight"]
            .apply(lambda series: float(np.abs(series).sum()))
            .mean()
        )
    else:
        avg_max_abs_sector_net = float("nan")
        avg_total_abs_sector_net = float("nan")

    relative_wealth_vs_equal_weight = float(
        daily_df["portfolio_nav"].iloc[-1] / daily_df["benchmark_nav"].iloc[-1] - 1.0
    )
    return {
        "daily_count": int(len(daily_df)),
        "invested_day_count": int((daily_df["gross_exposure"].astype(float) > 1e-12).sum()),
        "cash_day_count": int((daily_df["gross_exposure"].astype(float) <= 1e-12).sum()),
        "daily_ledger_definition": "all market dates from first execution through final liquidation, including cash dates",
        "rebalance_count": int(len(turnover_df)),
        "portfolio_total_return": total_return,
        "portfolio_annualized_return": annualized_return,
        "portfolio_annualized_vol": annualized_vol,
        "portfolio_sharpe": float(sharpe) if not pd.isna(sharpe) else float("nan"),
        "portfolio_max_drawdown": max_drawdown,
        "portfolio_calmar": float(calmar) if not pd.isna(calmar) else float("nan"),
        "sharpe_definition": "mean_daily_net_return / sample_std_daily_net_return * sqrt(252), risk_free_rate=0",
        "hit_ratio": float((daily_returns > 0).mean()),
        "gross_hit_ratio": float((gross_returns > 0).mean()),
        "best_daily_return": float(daily_returns.max()),
        "worst_daily_return": float(daily_returns.min()),
        "average_daily_return": float(daily_returns.mean()),
        "first_half_total_return": first_half_return,
        "second_half_total_return": second_half_return,
        "monthly_period_count": int(len(monthly_returns)),
        "positive_month_ratio": float((monthly_returns > 0.0).mean()) if len(monthly_returns) else float("nan"),
        "best_month_return": float(monthly_returns.max()) if len(monthly_returns) else float("nan"),
        "worst_month_return": float(monthly_returns.min()) if len(monthly_returns) else float("nan"),
        "average_gross_turnover": avg_gross_turnover,
        "average_net_turnover": avg_net_turnover,
        "average_turnover_cost_bps": avg_turnover_cost_bps,
        "total_turnover_cost": total_turnover_cost,
        "average_long_weight": avg_long_weight,
        "average_short_weight_abs": avg_short_weight,
        "average_gross_exposure": avg_gross_exposure,
        "average_net_exposure": avg_net_exposure,
        "average_max_abs_sector_net_weight": avg_max_abs_sector_net,
        "average_total_abs_sector_net_weight": avg_total_abs_sector_net,
        "total_transaction_cost": float(daily_df["transaction_cost"].sum()),
        "total_borrow_cost": float(daily_df["borrow_cost"].sum()),
        "benchmark_total_return": float(daily_df["benchmark_nav"].iloc[-1] - 1.0),
        # The long-short portfolio is dollar neutral with gross exposure near 2,
        # while the equal-weight benchmark is a gross-1 net-long portfolio. Their
        # relative wealth is useful context, but it is not a matched-risk alpha.
        "relative_wealth_vs_equal_weight_long_only": relative_wealth_vs_equal_weight,
        "relative_wealth_definition": "portfolio_nav / equal_weight_long_only_nav - 1; diagnostic only",
        # Compatibility alias for historical analysis scripts. Public reports must
        # use the accurate relative-wealth name above.
        "excess_total_return_vs_benchmark": relative_wealth_vs_equal_weight,
    }


def summarize_return_attribution(
    daily_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """检查回测收益是否被少数日期或少数股票主导。

    ``contribution_df`` 保留每个 sleeve 中每只股票每日的毛收益贡献。
    交易成本和借券费仍在组合日层面扣除，不人为分摊到个股。
    这些指标用于发现异常集中度，不是新的绩效指标。
    """

    if daily_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    daily_audit_columns = [
        column
        for column in [
            "date",
            "long_gross_return",
            "short_gross_return",
            "gross_return",
            "transaction_cost",
            "borrow_cost",
            "net_return",
            "benchmark_return",
            "excess_return",
        ]
        if column in daily_df.columns
    ]
    ordered_daily = daily_df[daily_audit_columns].copy()
    best_days = ordered_daily.nlargest(min(5, len(ordered_daily)), "net_return").assign(audit_bucket="best")
    worst_days = ordered_daily.nsmallest(min(5, len(ordered_daily)), "net_return").assign(audit_bucket="worst")
    extreme_days_df = pd.concat([best_days, worst_days], ignore_index=True).drop_duplicates(
        subset=["date", "audit_bucket"]
    )

    metrics: dict[str, float | int] = {
        "top_5_net_return_days_simple_sum": float(best_days["net_return"].sum()),
        "bottom_5_net_return_days_simple_sum": float(worst_days["net_return"].sum()),
    }
    if contribution_df.empty:
        return metrics, extreme_days_df, pd.DataFrame()

    working = contribution_df.copy()
    working["daily_close_return"] = pd.to_numeric(working["daily_close_return"], errors="coerce")
    working["gross_return_contribution"] = pd.to_numeric(
        working["gross_return_contribution"], errors="coerce"
    )
    working = working.dropna(subset=["instrument_id", "daily_close_return", "gross_return_contribution"])
    if working.empty:
        return metrics, extreme_days_df, pd.DataFrame()

    working["abs_gross_contribution"] = working["gross_return_contribution"].abs()
    working["abs_stock_return"] = working["daily_close_return"].abs()
    instrument_summary_df = (
        working.groupby("instrument_id", as_index=False)
        .agg(
            gross_return_contribution=("gross_return_contribution", "sum"),
            absolute_gross_contribution=("abs_gross_contribution", "sum"),
            position_day_count=("date", "count"),
            max_abs_stock_daily_return=("abs_stock_return", "max"),
        )
        .sort_values("absolute_gross_contribution", ascending=False)
        .reset_index(drop=True)
    )
    total_abs_contribution = float(instrument_summary_df["absolute_gross_contribution"].sum())
    top_5_abs_share = (
        float(instrument_summary_df.head(5)["absolute_gross_contribution"].sum() / total_abs_contribution)
        if total_abs_contribution > 1e-12
        else float("nan")
    )
    metrics.update(
        {
            "selected_position_day_count": int(len(working)),
            "selected_stock_return_abs_gt_20pct_count": int((working["abs_stock_return"] > 0.20).sum()),
            "selected_stock_return_abs_gt_50pct_count": int((working["abs_stock_return"] > 0.50).sum()),
            "max_abs_selected_stock_daily_return": float(working["abs_stock_return"].max()),
            "max_abs_single_position_daily_contribution": float(working["abs_gross_contribution"].max()),
            "top_5_instrument_abs_contribution_share": top_5_abs_share,
        }
    )
    return metrics, extreme_days_df, instrument_summary_df


def allocate_sector_pair_quotas(universe_slice: pd.DataFrame, top_k: int) -> dict[str, int]:
    """给每个行业分配多空配对名额。

    行业内排序的核心是：同一个行业里同时选多头和空头。
    因此某行业如果分到 `q` 个名额，就会选 `q` 个多头和 `q` 个空头。
    """

    sector_counts = universe_slice["sector"].astype(str).value_counts().sort_index()
    total_count = int(sector_counts.sum())
    if total_count <= 0:
        return {}

    quota_rows: list[dict[str, float | int | str]] = []
    for sector_name, sector_count in sector_counts.items():
        capacity = int(sector_count // 2)
        if capacity <= 0:
            continue
        raw_quota = float(top_k) * float(sector_count) / float(total_count)
        base_quota = min(int(math.floor(raw_quota)), capacity)
        quota_rows.append(
            {
                "sector": str(sector_name),
                "quota": base_quota,
                "capacity": capacity,
                "remainder": raw_quota - math.floor(raw_quota),
                "sector_count": int(sector_count),
            }
        )

    quota_df = pd.DataFrame(quota_rows)
    if quota_df.empty:
        return {}

    # 先按比例取 floor，再把剩余名额分给小数部分最大的行业。
    remaining = max(0, int(top_k) - int(quota_df["quota"].sum()))
    while remaining > 0:
        candidates = quota_df[quota_df["quota"] < quota_df["capacity"]].copy()
        if candidates.empty:
            break
        candidates = candidates.sort_values(
            ["remainder", "sector_count", "sector"],
            ascending=[False, False, True],
        )
        chosen_index = candidates.index[0]
        quota_df.loc[chosen_index, "quota"] = int(quota_df.loc[chosen_index, "quota"]) + 1
        remaining -= 1

    return {
        str(row["sector"]): int(row["quota"])
        for _, row in quota_df.iterrows()
        if int(row["quota"]) > 0
    }


def fill_side_from_global_rank(
    ranked_slice: pd.DataFrame,
    selected_ids: set[str],
    target_count: int,
    side: str,
) -> pd.DataFrame:
    """行业内排序无法凑满 Top-K 时，用全市场排序补齐。

    正常 us300 场景下很少触发。保留这个函数是为了让小样本 smoke test
    也能稳定运行，而不是因为某个行业股票太少直接失败。
    """

    if target_count <= 0:
        return pd.DataFrame(columns=ranked_slice.columns)

    available = ranked_slice[~ranked_slice["instrument_id"].astype(str).isin(selected_ids)].copy()
    if side == "long":
        available = available.sort_values("predicted_y", ascending=False)
    elif side == "short":
        available = available.sort_values("predicted_y", ascending=True)
    else:
        raise ValueError(f"Unsupported side: {side}")

    return available.head(target_count).copy()


def select_long_short_books(
    universe_slice: pd.DataFrame,
    *,
    top_k: int,
    neutral_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """根据预测分数选出多头和空头股票。"""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    valid = universe_slice.dropna(subset=["predicted_y", "instrument_id", "sector"]).copy()
    valid["instrument_id"] = valid["instrument_id"].astype(str)
    ranked = valid.sort_values("predicted_y", ascending=False).reset_index(drop=True)

    if len(ranked) < top_k * 2:
        raise ValueError(
            f"Need at least {top_k * 2} valid stocks to build a non-overlapping long-short book; "
            f"got {len(ranked)}."
        )

    if neutral_mode == "unconstrained":
        long_df = ranked.head(top_k).copy()
        short_df = ranked.tail(top_k).sort_values("predicted_y", ascending=True).copy()
        return long_df, short_df

    if neutral_mode != "sector_neutral":
        raise ValueError(f"Unsupported neutral_mode: {neutral_mode}")

    quota_map = allocate_sector_pair_quotas(ranked, top_k=top_k)
    long_parts: list[pd.DataFrame] = []
    short_parts: list[pd.DataFrame] = []
    selected_ids: set[str] = set()

    for sector_name, quota in quota_map.items():
        sector_frame = ranked[ranked["sector"].astype(str) == sector_name].copy()
        if len(sector_frame) < quota * 2:
            continue
        sector_long = sector_frame.head(quota).copy()
        sector_short = sector_frame.tail(quota).sort_values("predicted_y", ascending=True).copy()
        long_parts.append(sector_long)
        short_parts.append(sector_short)
        selected_ids.update(sector_long["instrument_id"].astype(str))
        selected_ids.update(sector_short["instrument_id"].astype(str))

    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame(columns=ranked.columns)
    short_df = pd.concat(short_parts, ignore_index=True) if short_parts else pd.DataFrame(columns=ranked.columns)

    if len(long_df) < top_k:
        fill_long = fill_side_from_global_rank(
            ranked_slice=ranked,
            selected_ids=selected_ids,
            target_count=top_k - len(long_df),
            side="long",
        )
        long_df = pd.concat([long_df, fill_long], ignore_index=True)
        selected_ids.update(fill_long["instrument_id"].astype(str))

    if len(short_df) < top_k:
        fill_short = fill_side_from_global_rank(
            ranked_slice=ranked,
            selected_ids=selected_ids,
            target_count=top_k - len(short_df),
            side="short",
        )
        short_df = pd.concat([short_df, fill_short], ignore_index=True)

    long_df = long_df.sort_values("predicted_y", ascending=False).head(top_k).copy()
    short_df = short_df.sort_values("predicted_y", ascending=True).head(top_k).copy()

    overlap = set(long_df["instrument_id"].astype(str)) & set(short_df["instrument_id"].astype(str))
    if overlap:
        raise ValueError(f"Long and short books overlap: {sorted(overlap)[:5]}")

    return long_df, short_df


def _normalize_positive_weights(raw_scores: pd.Series, *, max_abs_weight: float | None) -> pd.Series:
    """把一组正数分数转成和为 1 的权重。

    `max_abs_weight` 是单票权重上限。它的作用是防止 score-weight 这类
    不等权方法被少数极端预测值控制。
    """

    scores = pd.to_numeric(raw_scores, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores = scores.clip(lower=0.0)
    if scores.sum() <= 1e-12:
        scores = pd.Series(1.0, index=scores.index, dtype=float)

    values = scores.to_numpy(dtype=float)
    if values.sum() <= 1e-12:
        values = np.ones(len(values), dtype=float)
    weights = values / values.sum()
    if max_abs_weight is None:
        return pd.Series(weights, index=scores.index, dtype=float)

    max_abs_weight = float(max_abs_weight)
    if not 0.0 < max_abs_weight <= 1.0:
        raise ValueError("max_abs_weight must fall inside (0, 1].")
    if max_abs_weight * len(values) < 1.0 - 1e-12:
        raise ValueError("max_abs_weight is too small for the selected stock count.")

    if float(weights.max()) <= max_abs_weight + 1e-12:
        return pd.Series(weights, index=scores.index, dtype=float)

    # 单票权重上限的快速投影：
    # 先按原始分数分配权重，超过上限的股票固定在上限；
    # 剩余权重再按剩余股票的原始分数重新分配。
    # 对 Top-K 组合来说，循环次数最多等于股票数，远快于 pandas Series 反复截断。
    adjusted = np.zeros(len(values), dtype=float)
    free_mask = np.ones(len(values), dtype=bool)
    remaining_weight = 1.0
    positive_values = np.clip(values, 0.0, None)

    for _ in range(len(values)):
        free_indices = np.where(free_mask)[0]
        if len(free_indices) == 0:
            break

        free_values = positive_values[free_indices]
        if float(free_values.sum()) <= 1e-12:
            tentative = np.full(len(free_indices), remaining_weight / len(free_indices), dtype=float)
        else:
            tentative = remaining_weight * free_values / float(free_values.sum())

        over_mask = tentative > max_abs_weight + 1e-12
        if not bool(over_mask.any()):
            adjusted[free_indices] = tentative
            break

        capped_indices = free_indices[over_mask]
        adjusted[capped_indices] = max_abs_weight
        free_mask[capped_indices] = False
        remaining_weight -= max_abs_weight * len(capped_indices)
        remaining_weight = max(0.0, remaining_weight)

    if float(adjusted.sum()) <= 1e-12:
        adjusted = np.full(len(values), 1.0 / len(values), dtype=float)
    else:
        adjusted = adjusted / float(adjusted.sum())
    return pd.Series(adjusted, index=scores.index, dtype=float)


def _build_side_abs_weights(side_df: pd.DataFrame, *, side: str, weight_mode: str, max_abs_weight: float | None) -> pd.Series:
    """给多头或空头单边生成绝对权重，单边权重和为 1。

    支持的模式：

    - equal_weight：当前旧口径，每只股票等权；
    - rank_weight：排名越靠前，权重越大；
    - score_weight：预测分数离本侧尾部越远，权重越大；
    - score_vol_weight：score_weight 再除以最近波动率代理，降低高波动股票权重。
    """

    if side_df.empty:
        raise ValueError("side_df must be non-empty.")

    n = len(side_df)
    if weight_mode == "equal_weight":
        raw = pd.Series(1.0, index=side_df.index, dtype=float)
    elif weight_mode == "rank_weight":
        raw = pd.Series(np.arange(n, 0, -1, dtype=float), index=side_df.index)
    elif weight_mode in {"score_weight", "score_vol_weight"}:
        predicted = pd.to_numeric(side_df["predicted_y"], errors="coerce")
        if side == "long":
            raw = predicted - predicted.min()
        elif side == "short":
            raw = predicted.max() - predicted
        else:
            raise ValueError(f"Unsupported side: {side}")
        raw = raw.fillna(0.0).clip(lower=0.0) + 1e-12
        if weight_mode == "score_vol_weight":
            if "realized_vol_20" in side_df.columns:
                risk_proxy = pd.to_numeric(side_df["realized_vol_20"], errors="coerce")
                risk_proxy = risk_proxy.replace([np.inf, -np.inf], np.nan)
                median_vol = float(risk_proxy.median()) if risk_proxy.notna().any() else 0.0
                risk_proxy = risk_proxy.fillna(median_vol).clip(lower=1e-6)
                raw = raw / risk_proxy
            # 没有 realized_vol_20 时保留 score 权重。规模暴露和波动率并非
            # 同一个风险概念，不能为了兼容旧缓存而互相替代。
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    return _normalize_positive_weights(raw, max_abs_weight=max_abs_weight)


def build_signed_weight_frame(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    *,
    weight_mode: str = "equal_weight",
    max_abs_weight: float | None = None,
) -> pd.DataFrame:
    """把多头和空头股票转成 signed weight。

    等权是旧口径；不等权只改变每侧内部权重分配，不改变多空总敞口：

    - 多头总权重仍为 +1；
    - 空头总权重仍为 -1；
    - 净敞口仍接近 0；
    - gross exposure 仍接近 2。
    """

    if long_df.empty or short_df.empty:
        raise ValueError("Both long and short books must be non-empty.")

    long_book = long_df.copy()
    short_book = short_df.copy()
    long_book["side"] = "long"
    short_book["side"] = "short"
    long_abs_weights = _build_side_abs_weights(
        long_book,
        side="long",
        weight_mode=weight_mode,
        max_abs_weight=max_abs_weight,
    )
    short_abs_weights = _build_side_abs_weights(
        short_book,
        side="short",
        weight_mode=weight_mode,
        max_abs_weight=max_abs_weight,
    )
    long_book["weight"] = long_abs_weights
    short_book["weight"] = -short_abs_weights
    selected = pd.concat([long_book, short_book], ignore_index=True)
    selected["abs_weight"] = selected["weight"].abs()
    selected["weight_mode"] = weight_mode
    return selected


def compute_signed_trade_summary(
    previous_weights: dict[str, float],
    target_weights: dict[str, float],
    *,
    cost_bps: float,
    borrow_cost_bps: float,
    borrow_accrual_days: int,
) -> dict[str, float]:
    """计算 signed book 的换手和成本。

    多空组合里权重可以为负，所以不能直接套 long-only turnover。
    这里把多头账本和空头账本分开计算，再合并成 gross turnover。
    """

    all_names = set(previous_weights) | set(target_weights)
    long_turnover = 0.0
    short_turnover = 0.0

    for instrument_id in all_names:
        previous_weight = float(previous_weights.get(instrument_id, 0.0))
        target_weight = float(target_weights.get(instrument_id, 0.0))
        previous_long = max(previous_weight, 0.0)
        target_long = max(target_weight, 0.0)
        previous_short = max(-previous_weight, 0.0)
        target_short = max(-target_weight, 0.0)
        long_turnover += abs(target_long - previous_long)
        short_turnover += abs(target_short - previous_short)

    gross_turnover = long_turnover + short_turnover
    net_turnover = abs(sum(target_weights.values()) - sum(previous_weights.values()))
    long_weight = sum(max(weight, 0.0) for weight in target_weights.values())
    short_weight = sum(max(-weight, 0.0) for weight in target_weights.values())
    gross_exposure = long_weight + short_weight
    net_exposure = sum(target_weights.values())
    turnover_cost = gross_turnover * float(cost_bps) / 10000.0
    borrow_cost_estimate = (
        short_weight
        * float(borrow_cost_bps)
        / 10000.0
        * float(borrow_accrual_days)
        / 252.0
    )

    return {
        "long_turnover": float(long_turnover),
        "short_turnover": float(short_turnover),
        "gross_turnover": float(gross_turnover),
        "net_turnover": float(net_turnover),
        "turnover_cost": float(turnover_cost),
        "turnover_cost_bps": float(turnover_cost * 10000.0),
        "borrow_cost_estimate": float(borrow_cost_estimate),
        "long_weight": float(long_weight),
        "short_weight": float(short_weight),
        "gross_exposure": float(gross_exposure),
        "net_exposure": float(net_exposure),
    }


def compute_sleeve_lifecycle_trade_summary(
    target_weights: dict[str, float],
    *,
    cost_bps: float,
    borrow_cost_bps: float,
    borrow_accrual_days: int,
) -> dict[str, float]:
    """Account for both entry and exit of one independent holding sleeve.

    Each backtest sleeve starts from cash, holds a fixed signed book, and is
    liquidated at its endpoint.  Charging only the entry or only the change from
    an older sleeve understates trading costs, especially when the same names are
    selected repeatedly.  This helper therefore records the full round trip.

    Cross-sleeve trade netting is intentionally not assumed.  That is conservative
    and keeps the accounting reproducible without an order-level execution engine.
    """

    entry = compute_signed_trade_summary(
        previous_weights={},
        target_weights=target_weights,
        cost_bps=cost_bps,
        borrow_cost_bps=borrow_cost_bps,
        borrow_accrual_days=borrow_accrual_days,
    )
    exit_trade = compute_signed_trade_summary(
        previous_weights=target_weights,
        target_weights={},
        cost_bps=cost_bps,
        borrow_cost_bps=0.0,
        borrow_accrual_days=0,
    )

    long_turnover = entry["long_turnover"] + exit_trade["long_turnover"]
    short_turnover = entry["short_turnover"] + exit_trade["short_turnover"]
    gross_turnover = entry["gross_turnover"] + exit_trade["gross_turnover"]
    net_turnover = entry["net_turnover"] + exit_trade["net_turnover"]
    turnover_cost = entry["turnover_cost"] + exit_trade["turnover_cost"]

    return {
        "entry_long_turnover": float(entry["long_turnover"]),
        "entry_short_turnover": float(entry["short_turnover"]),
        "entry_gross_turnover": float(entry["gross_turnover"]),
        "entry_turnover_cost": float(entry["turnover_cost"]),
        "exit_long_turnover": float(exit_trade["long_turnover"]),
        "exit_short_turnover": float(exit_trade["short_turnover"]),
        "exit_gross_turnover": float(exit_trade["gross_turnover"]),
        "exit_turnover_cost": float(exit_trade["turnover_cost"]),
        "long_turnover": float(long_turnover),
        "short_turnover": float(short_turnover),
        "gross_turnover": float(gross_turnover),
        "net_turnover": float(net_turnover),
        "turnover_cost": float(turnover_cost),
        "turnover_cost_bps": float(turnover_cost * 10000.0),
        "borrow_cost_estimate": float(entry["borrow_cost_estimate"]),
        "long_weight": float(entry["long_weight"]),
        "short_weight": float(entry["short_weight"]),
        "gross_exposure": float(entry["gross_exposure"]),
        "net_exposure": float(entry["net_exposure"]),
    }


def scale_trade_summary_to_portfolio(
    sleeve_summary: dict[str, float],
    *,
    sleeve_capital_weight: float,
) -> dict[str, float]:
    """Convert one sleeve's full-notional ledger into portfolio-level values.

    Every sleeve is internally normalized to a +1 long / -1 short book. When
    holding windows overlap, only a fraction of total capital belongs to each
    sleeve. Public turnover, cost, and exposure metrics must therefore use the
    capital-scaled values. The original values remain under `sleeve_*` columns so
    an auditor can reconstruct both layers.
    """

    capital_weight = float(sleeve_capital_weight)
    scaled_keys = {
        "entry_long_turnover",
        "entry_short_turnover",
        "entry_gross_turnover",
        "entry_turnover_cost",
        "exit_long_turnover",
        "exit_short_turnover",
        "exit_gross_turnover",
        "exit_turnover_cost",
        "long_turnover",
        "short_turnover",
        "gross_turnover",
        "net_turnover",
        "turnover_cost",
        "borrow_cost_estimate",
        "long_weight",
        "short_weight",
        "gross_exposure",
        "net_exposure",
    }
    result = {f"sleeve_{key}": float(value) for key, value in sleeve_summary.items()}
    for key in scaled_keys:
        result[key] = float(sleeve_summary[key]) * capital_weight
    result["turnover_cost_bps"] = result["turnover_cost"] * 10000.0
    result["sleeve_capital_weight"] = capital_weight
    return result


def compute_long_short_sector_exposure(
    universe_slice: pd.DataFrame,
    selected_frame: pd.DataFrame,
) -> pd.DataFrame:
    """计算每个行业的多头、空头和净暴露。"""

    universe_count = len(universe_slice)
    universe_sector = (
        universe_slice.groupby("sector")
        .size()
        .div(universe_count)
        .rename("universe_weight")
        .reset_index()
    )

    selected = selected_frame.copy()
    selected["long_weight"] = selected["weight"].clip(lower=0.0)
    selected["short_weight_abs"] = (-selected["weight"]).clip(lower=0.0)
    selected["net_sector_weight"] = selected["weight"]

    sector_exposure = (
        selected.groupby("sector", as_index=False)
        .agg(
            long_weight=("long_weight", "sum"),
            short_weight_abs=("short_weight_abs", "sum"),
            net_sector_weight=("net_sector_weight", "sum"),
        )
        .merge(universe_sector, how="outer", on="sector")
        .fillna(0.0)
    )
    sector_exposure["abs_net_sector_weight"] = sector_exposure["net_sector_weight"].abs()
    return sector_exposure.sort_values("abs_net_sector_weight", ascending=False).reset_index(drop=True)


def add_nav_columns(daily_df: pd.DataFrame) -> pd.DataFrame:
    """给日收益表增加净值和基准净值。"""

    if daily_df.empty:
        return daily_df
    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    daily_df["benchmark_return"] = daily_df["benchmark_return"].fillna(0.0)
    daily_df["active_return_vs_equal_weight_long_only"] = (
        daily_df["net_return"] - daily_df["benchmark_return"]
    )
    # Compatibility alias retained for historical scripts. This is not a
    # risk-matched excess return because the two portfolios have different net
    # and gross exposures.
    daily_df["excess_return"] = daily_df["active_return_vs_equal_weight_long_only"]
    daily_df["portfolio_nav"] = (1.0 + daily_df["net_return"]).cumprod()
    daily_df["benchmark_nav"] = (1.0 + daily_df["benchmark_return"]).cumprod()
    return daily_df


def write_long_short_report(
    output_path: Path,
    *,
    config: LongShortBacktestConfig,
    metrics: dict[str, Any],
    daily_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    extreme_days_df: pd.DataFrame,
    instrument_attribution_df: pd.DataFrame,
) -> None:
    """写出单次多空回测报告。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    caveat = ""
    if metrics.get("rebalance_count", 0) < 10:
        caveat = (
            "\n- Warning: rebalance count is below 10. Sharpe and Calmar are unstable "
            "and should not be presented as robust live-trading evidence.\n"
        )

    report = f"""# Long-Short Backtest Report

## 1. Setup

- Run name: `{config.run_name}`
- Predictions path: `{project_relative_path(config.predictions_path, PROJECT_ROOT)}`
- Market data path: `{project_relative_path(config.data_path, PROJECT_ROOT)}`
- Price adjustment mode: `{config.price_adjustment_mode}`
- Real market-cap coverage: `{metrics.get('market_cap_coverage_ratio', 0.0):.2%}`
- Size exposure available: `{metrics.get('size_exposure_available', False)}`
- Hold days: `{config.hold_days}`
- Holding clock: `{config.holding_clock}`
- Effective executable holding days: `{metrics.get('effective_holding_days')}`
- Rebalance step days: `{config.step_days}`
- Signal delay days: `{config.signal_delay_days}`
- Top-K long / short: `{config.top_k}`
- Neutral mode: `{config.neutral_mode}`
- Weight mode: `{config.weight_mode}`
- Max single-name absolute weight: `{config.max_abs_weight}`
- Transaction cost: `{config.cost_bps}` bps per traded notional
- Borrow fee sensitivity: `{config.borrow_cost_bps}` annualized bps, accrued linearly on short notional
- Turnover accounting: full sleeve round trip (entry + liquidation), capital-scaled, no cross-sleeve netting
- Skipped incomplete return paths: `{metrics.get('skipped_incomplete_return_path_count', 0)}`
- Sharpe definition: `mean(daily net return) / sample std(daily net return) * sqrt(252)`, zero risk-free rate

## 2. Scope Caveat

This is a backtest-style diagnostic built from saved predictions and close-to-close returns.
Under the canonical `signal_horizon` clock, `hold_days` is measured from the signal date.
With a one-day close execution delay, a 10-day target accrues nine executable daily returns
from the execution close through the target endpoint. `execution_horizon` is retained only
as a historical sensitivity definition and must not be mixed with canonical `y_10d` results.
It does not model intraday execution, short borrow availability, bid-ask spread, order-book depth, or tax effects.
{caveat}
## 3. Metrics

```json
{dumps_strict_json(metrics)}
```

## 4. Daily Returns Preview

{dataframe_to_markdown(daily_df.head(10))}

## 5. Turnover / Cost Preview

{dataframe_to_markdown(turnover_df.head(10))}

## 6. Sector Exposure Preview

{dataframe_to_markdown(sector_df.head(15))}

## 7. Return Concentration Audit

Best and worst net-return days:

{dataframe_to_markdown(extreme_days_df, max_rows=10)}

Largest absolute gross-contribution instruments:

{dataframe_to_markdown(instrument_attribution_df, max_rows=10)}

These tables are diagnostics. A result dominated by a few dates, a few stocks, or
extreme adjusted returns requires manual company-action and data-quality review.
"""

    output_path.write_text(report, encoding="utf-8")


def run_long_short_backtest(
    *,
    config: LongShortBacktestConfig,
    market_snapshot_df: pd.DataFrame | None = None,
    prediction_df: pd.DataFrame | None = None,
    merged_prediction_market_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """运行一次 Top-K long / Bottom-K short 回测。"""

    if config.hold_days <= 0 or config.step_days <= 0:
        raise ValueError("hold_days and step_days must be positive.")
    if config.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if config.cost_bps < 0 or config.borrow_cost_bps < 0:
        raise ValueError("cost_bps and borrow_cost_bps must be non-negative.")
    if config.signal_delay_days < 0:
        raise ValueError("signal_delay_days must be non-negative.")
    if config.holding_clock not in SUPPORTED_HOLDING_CLOCKS:
        raise ValueError(
            f"Unsupported holding_clock: {config.holding_clock}. "
            f"Expected one of {sorted(SUPPORTED_HOLDING_CLOCKS)}."
        )
    if config.holding_clock == "signal_horizon" and config.signal_delay_days >= config.hold_days:
        raise ValueError(
            "signal_delay_days must be smaller than hold_days when "
            "holding_clock='signal_horizon'."
        )
    supported_weight_modes = {"equal_weight", "rank_weight", "score_weight", "score_vol_weight"}
    if config.weight_mode not in supported_weight_modes:
        raise ValueError(f"Unsupported weight_mode: {config.weight_mode}")

    if prediction_df is None:
        prediction_df = load_prediction_frame(config.predictions_path)
    else:
        prediction_df = prediction_df.copy()

    market_df = (
        market_snapshot_df
        if market_snapshot_df is not None
        else load_market_snapshot_frame(
            config.data_path,
            price_adjustment_mode=config.price_adjustment_mode,
        )
    )
    for frame_name, frame in (("prediction", prediction_df), ("market", market_df)):
        required_key_columns = {"date", "instrument_id"}
        if not required_key_columns.issubset(frame.columns):
            raise ValueError(
                f"{frame_name} frame is missing key columns: "
                f"{sorted(required_key_columns - set(frame.columns))}"
            )
        if frame.duplicated(subset=["date", "instrument_id"], keep=False).any():
            raise ValueError(
                f"{frame_name} frame contains duplicate date/instrument_id rows."
            )
    if merged_prediction_market_df is None:
        merged_df = merge_predictions_with_market(prediction_df, market_df)
    else:
        # 全量组合网格中，同一个窗口会被不同权重/成本/持有规则重复使用。
        # 允许调用方传入已合并表，避免每个子回测反复做相同 merge。
        merged_df = merged_prediction_market_df.copy()

    market_dates = sorted(pd.to_datetime(market_df["date"]).unique())
    signal_dates = sorted(pd.to_datetime(merged_df["date"]).unique())
    if not market_dates:
        raise ValueError("Market panel contains no valid trading dates.")
    if not signal_dates:
        raise ValueError("Prediction panel contains no valid signal dates after market merge.")
    # `step_days` is a market-trading-day interval. Slicing only the dates that
    # happen to appear in a prediction file would compress time when one complete
    # signal date is missing and silently shift every later rebalance. Require a
    # contiguous prediction-date calendar between the first and last signal, then
    # schedule rebalances on the shared market calendar.
    signal_date_set = {pd.Timestamp(date_value) for date_value in signal_dates}
    signal_calendar = [
        pd.Timestamp(date_value)
        for date_value in market_dates
        if signal_dates[0] <= pd.Timestamp(date_value) <= signal_dates[-1]
    ]
    missing_signal_dates = [
        date_value for date_value in signal_calendar if date_value not in signal_date_set
    ]
    if missing_signal_dates:
        raise ValueError(
            "Prediction panel is missing complete market dates inside its signal range; "
            "step_days cannot be interpreted consistently. Missing dates include: "
            f"{[date.strftime('%Y-%m-%d') for date in missing_signal_dates[:5]]}"
        )
    rebalance_dates = signal_calendar[:: config.step_days]
    effective_holding_days = (
        config.hold_days - config.signal_delay_days
        if config.holding_clock == "signal_horizon"
        else config.hold_days
    )
    max_active_sleeves = max(1, math.ceil(effective_holding_days / config.step_days))
    sleeve_capital = 1.0 / max_active_sleeves

    market_returns = market_df[["date", "instrument_id", "daily_close_return"]].dropna().copy()
    benchmark_daily = (
        market_returns.groupby("date", as_index=False)["daily_close_return"]
        .mean()
        .rename(columns={"daily_close_return": "benchmark_return"})
    )

    weight_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    skipped_trade_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    sleeve_daily_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []

    for rebalance_index, signal_date in enumerate(rebalance_dates):
        signal_date = pd.Timestamp(signal_date)
        sleeve_slot = int(rebalance_index % max_active_sleeves)
        universe_slice = merged_df[merged_df["date"] == signal_date].copy()
        if universe_slice.empty:
            continue

        holding_window = resolve_holding_window(
            market_dates,
            signal_date=signal_date,
            signal_delay_days=config.signal_delay_days,
            hold_days=config.hold_days,
            holding_clock=config.holding_clock,
        )
        if holding_window is None:
            continue
        execution_date, end_date, window_holding_days = holding_window

        long_df, short_df = select_long_short_books(
            universe_slice,
            top_k=config.top_k,
            neutral_mode=config.neutral_mode,
        )
        selected_frame = build_signed_weight_frame(
            long_df,
            short_df,
            weight_mode=config.weight_mode,
            max_abs_weight=config.max_abs_weight,
        )
        target_weights = {
            str(row["instrument_id"]): float(row["weight"])
            for _, row in selected_frame.iterrows()
        }
        sleeve_trade_summary = compute_sleeve_lifecycle_trade_summary(
            target_weights=target_weights,
            cost_bps=config.cost_bps,
            borrow_cost_bps=config.borrow_cost_bps,
            borrow_accrual_days=window_holding_days,
        )
        trade_summary = scale_trade_summary_to_portfolio(
            sleeve_trade_summary,
            sleeve_capital_weight=sleeve_capital,
        )

        # A trade enters the public ledger only when every selected instrument has
        # a complete close-to-close return path for the requested holding window.
        # Silently dropping missing names would change exposure after selection and
        # can create survivorship-like return bias around suspensions or delistings.
        sleeve_returns = market_returns[
            (market_returns["date"] > execution_date) & (market_returns["date"] <= end_date)
        ].copy()
        sleeve_returns["instrument_id"] = sleeve_returns["instrument_id"].astype(str)
        sleeve_returns["target_weight"] = sleeve_returns["instrument_id"].map(target_weights)
        sleeve_returns = sleeve_returns.dropna(subset=["target_weight", "daily_close_return"])
        observed_counts = sleeve_returns.groupby("instrument_id")["date"].nunique()
        incomplete_ids = sorted(
            instrument_id
            for instrument_id in target_weights
            if int(observed_counts.get(instrument_id, 0)) != int(window_holding_days)
        )
        if incomplete_ids:
            skipped_trade_rows.append(
                {
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "execution_date": execution_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "sleeve_slot": sleeve_slot,
                    "execution_status": "skipped_incomplete_return_path",
                    "expected_return_days_per_instrument": int(window_holding_days),
                    "selected_instrument_count": int(len(target_weights)),
                    "incomplete_instrument_count": int(len(incomplete_ids)),
                    "incomplete_instruments": "|".join(incomplete_ids),
                }
            )
            continue

        sector_exposure = compute_long_short_sector_exposure(universe_slice, selected_frame)
        for _, row in sector_exposure.iterrows():
            sector_rows.append(
                {
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "execution_date": execution_date.strftime("%Y-%m-%d"),
                    # sector_exposure.csv 也保留 sleeve 编号，便于将行业暴露
                    # 与同一个并行持仓的权重、换手和收益逐条对齐。
                    "sleeve_slot": sleeve_slot,
                    "sleeve_capital_weight": float(sleeve_capital),
                    "sector": str(row["sector"]),
                    "sleeve_long_weight": float(row["long_weight"]),
                    "sleeve_short_weight_abs": float(row["short_weight_abs"]),
                    "sleeve_net_sector_weight": float(row["net_sector_weight"]),
                    "long_weight": float(row["long_weight"]) * float(sleeve_capital),
                    "short_weight_abs": float(row["short_weight_abs"]) * float(sleeve_capital),
                    "net_sector_weight": float(row["net_sector_weight"]) * float(sleeve_capital),
                    "abs_net_sector_weight": float(row["abs_net_sector_weight"]) * float(sleeve_capital),
                    "universe_weight": float(row["universe_weight"]),
                }
            )

        turnover_rows.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "execution_date": execution_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "sleeve_slot": sleeve_slot,
                "top_k": int(config.top_k),
                "neutral_mode": config.neutral_mode,
                "weight_mode": config.weight_mode,
                "max_abs_weight": config.max_abs_weight,
                "cost_bps": float(config.cost_bps),
                "borrow_cost_bps": float(config.borrow_cost_bps),
                "holding_clock": config.holding_clock,
                "effective_holding_days": int(window_holding_days),
                "execution_status": "executed_complete_return_path",
                **trade_summary,
            }
        )

        for _, row in selected_frame.iterrows():
            weight_rows.append(
                {
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "execution_date": execution_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "holding_clock": config.holding_clock,
                    "effective_holding_days": int(window_holding_days),
                    "sleeve_slot": sleeve_slot,
                    "instrument_id": str(row["instrument_id"]),
                    "side": str(row["side"]),
                    "weight": float(row["weight"]),
                    "abs_weight": float(row["abs_weight"]),
                    "weight_mode": config.weight_mode,
                    "capital_weight": float(sleeve_capital),
                    "effective_weight": float(row["weight"]) * float(sleeve_capital),
                    "predicted_y": float(row["predicted_y"]),
                    "sector": str(row["sector"]),
                    "market_cap": float(row["market_cap"]),
                    "size_exposure_z": float(row["size_exposure_z"]),
                }
            )

        sleeve_returns["long_weight"] = sleeve_returns["target_weight"].clip(lower=0.0)
        sleeve_returns["short_weight"] = sleeve_returns["target_weight"].clip(upper=0.0)
        sleeve_returns["capital_weight"] = float(sleeve_capital)
        sleeve_returns["effective_weight"] = sleeve_returns["target_weight"] * float(sleeve_capital)
        sleeve_returns["gross_return_contribution"] = (
            sleeve_returns["effective_weight"] * sleeve_returns["daily_close_return"]
        )
        contribution_frame = sleeve_returns[
            [
                "date",
                "instrument_id",
                "target_weight",
                "capital_weight",
                "effective_weight",
                "daily_close_return",
                "gross_return_contribution",
            ]
        ].copy()
        contribution_frame["signal_date"] = signal_date.strftime("%Y-%m-%d")
        contribution_frame["execution_date"] = execution_date.strftime("%Y-%m-%d")
        contribution_frame["end_date"] = end_date.strftime("%Y-%m-%d")
        contribution_frame["sleeve_slot"] = sleeve_slot
        contribution_frame["side"] = np.where(contribution_frame["target_weight"] >= 0.0, "long", "short")
        contribution_rows.extend(contribution_frame.to_dict("records"))
        daily_records: list[dict[str, Any]] = []
        # The signal is observed at close t and the position is opened at close
        # t+1.  The execution date therefore has zero close-to-close position
        # return, but the entry trade and its cost still belong to that date.
        # Keeping this explicit row prevents cash/execution days from disappearing
        # from daily Sharpe and annualization calculations.
        daily_records.append(
            {
                "date": execution_date,
                "long_gross_return": 0.0,
                "short_gross_return": 0.0,
                "gross_return": 0.0,
            }
        )
        for current_date, frame in sleeve_returns.groupby("date"):
            # 显式循环比 groupby.apply 更啰嗦，但 pandas 版本兼容性更好，
            # 也更容易看清楚多头贡献、空头贡献和总收益是怎么来的。
            daily_records.append(
                {
                    "date": current_date,
                    "long_gross_return": float((frame["long_weight"] * frame["daily_close_return"]).sum()),
                    "short_gross_return": float((frame["short_weight"] * frame["daily_close_return"]).sum()),
                    "gross_return": float((frame["target_weight"] * frame["daily_close_return"]).sum()),
                }
            )
        daily_sleeve = pd.DataFrame(daily_records)
        daily_sleeve["signal_date"] = signal_date.strftime("%Y-%m-%d")
        daily_sleeve["execution_date"] = execution_date.strftime("%Y-%m-%d")
        daily_sleeve["end_date"] = end_date.strftime("%Y-%m-%d")
        daily_sleeve["sleeve_slot"] = sleeve_slot
        daily_sleeve["capital_weight"] = float(sleeve_capital)
        daily_sleeve["transaction_cost"] = 0.0
        daily_sleeve["borrow_cost"] = 0.0
        holding_return_mask = pd.to_datetime(daily_sleeve["date"]) > execution_date
        daily_sleeve.loc[holding_return_mask, "borrow_cost"] = (
            trade_summary["short_weight"]
            * float(config.borrow_cost_bps)
            / 10000.0
            / 252.0
        )
        if not daily_sleeve.empty:
            # Entry and liquidation are separate real trades. Deduct each at the
            # nearest return-ledger boundary so total cost reconciles exactly with
            # turnover_cost.csv. A one-row window receives both charges.
            daily_sleeve.loc[daily_sleeve.index[0], "transaction_cost"] += trade_summary[
                "entry_turnover_cost"
            ]
            daily_sleeve.loc[daily_sleeve.index[-1], "transaction_cost"] += trade_summary[
                "exit_turnover_cost"
            ]
        daily_sleeve["long_gross_return"] *= float(sleeve_capital)
        daily_sleeve["short_gross_return"] *= float(sleeve_capital)
        daily_sleeve["gross_return"] *= float(sleeve_capital)
        daily_sleeve["long_exposure"] = trade_summary["long_weight"]
        daily_sleeve["short_exposure_abs"] = trade_summary["short_weight"]
        daily_sleeve["gross_exposure"] = trade_summary["gross_exposure"]
        daily_sleeve["net_exposure"] = trade_summary["net_exposure"]
        daily_sleeve["net_return"] = (
            daily_sleeve["gross_return"]
            - daily_sleeve["transaction_cost"]
            - daily_sleeve["borrow_cost"]
        )
        sleeve_daily_rows.extend(daily_sleeve.to_dict("records"))

    weight_df = pd.DataFrame(weight_rows)
    turnover_df = pd.DataFrame(turnover_rows)
    skipped_trade_df = pd.DataFrame(skipped_trade_rows)
    sector_df = pd.DataFrame(sector_rows)
    sleeve_daily_df = pd.DataFrame(sleeve_daily_rows)
    contribution_df = pd.DataFrame(contribution_rows)

    if sleeve_daily_df.empty:
        daily_df = pd.DataFrame(
            columns=[
                "date",
                "long_gross_return",
                "short_gross_return",
                "gross_return",
                "transaction_cost",
                "borrow_cost",
                "net_return",
                "long_exposure",
                "short_exposure_abs",
                "gross_exposure",
                "net_exposure",
                "benchmark_return",
                "excess_return",
                "portfolio_nav",
                "benchmark_nav",
            ]
        )
    else:
        aggregated_daily = (
            sleeve_daily_df.groupby("date", as_index=False)
            .agg(
                long_gross_return=("long_gross_return", "sum"),
                short_gross_return=("short_gross_return", "sum"),
                gross_return=("gross_return", "sum"),
                transaction_cost=("transaction_cost", "sum"),
                borrow_cost=("borrow_cost", "sum"),
                net_return=("net_return", "sum"),
                long_exposure=("long_exposure", "sum"),
                short_exposure_abs=("short_exposure_abs", "sum"),
                gross_exposure=("gross_exposure", "sum"),
                net_exposure=("net_exposure", "sum"),
            )
        )
        # Reindex to every market date from the first executed sleeve through the
        # final liquidation.  This is essential when step_days is larger than the
        # executable holding window: the portfolio is in cash on the omitted dates,
        # so its return/exposure/cost is zero rather than "not observed".
        ledger_start = pd.to_datetime(turnover_df["execution_date"]).min()
        ledger_end = pd.to_datetime(turnover_df["end_date"]).max()
        ledger_calendar = pd.DataFrame(
            {
                "date": [
                    pd.Timestamp(date_value)
                    for date_value in market_dates
                    if ledger_start <= pd.Timestamp(date_value) <= ledger_end
                ]
            }
        )
        daily_df = ledger_calendar.merge(aggregated_daily, how="left", on="date")
        portfolio_daily_columns = [
            "long_gross_return",
            "short_gross_return",
            "gross_return",
            "transaction_cost",
            "borrow_cost",
            "net_return",
            "long_exposure",
            "short_exposure_abs",
            "gross_exposure",
            "net_exposure",
        ]
        daily_df[portfolio_daily_columns] = daily_df[portfolio_daily_columns].fillna(0.0)
        daily_df = daily_df.merge(benchmark_daily, how="left", on="date")
        missing_benchmark_dates = daily_df.loc[
            daily_df["benchmark_return"].isna(), "date"
        ]
        if not missing_benchmark_dates.empty:
            raise ValueError(
                "Equal-weight benchmark is missing market returns inside the portfolio ledger: "
                f"{missing_benchmark_dates.dt.strftime('%Y-%m-%d').head(5).tolist()}"
            )
        daily_df = add_nav_columns(daily_df)
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.strftime("%Y-%m-%d")

    metrics = summarize_long_short_metrics(daily_df, turnover_df, sector_df)
    attribution_metrics, extreme_days_df, instrument_attribution_df = summarize_return_attribution(
        daily_df,
        contribution_df,
    )
    metrics.update(attribution_metrics)
    market_cap = pd.to_numeric(
        market_df["market_cap"] if "market_cap" in market_df.columns else pd.Series(np.nan, index=market_df.index),
        errors="coerce",
    )
    market_cap_coverage_ratio = float(market_cap.notna().mean()) if len(market_df) else 0.0
    metrics.update(
        {
            "run_name": config.run_name,
            "predictions_path": project_relative_path(config.predictions_path, PROJECT_ROOT),
            "data_path": project_relative_path(config.data_path, PROJECT_ROOT),
            "price_adjustment_mode": config.price_adjustment_mode,
            "market_cap_coverage_ratio": market_cap_coverage_ratio,
            "size_exposure_available": bool(market_cap_coverage_ratio >= 0.90),
            "hold_days": int(config.hold_days),
            "holding_clock": config.holding_clock,
            "effective_holding_days": int(effective_holding_days),
            "step_days": int(config.step_days),
            "signal_delay_days": int(config.signal_delay_days),
            "top_k": int(config.top_k),
            "cost_bps": float(config.cost_bps),
            "borrow_cost_bps": float(config.borrow_cost_bps),
            "neutral_mode": config.neutral_mode,
            "weight_mode": config.weight_mode,
            "max_abs_weight": config.max_abs_weight,
            "max_active_sleeves": int(max_active_sleeves),
            "sleeve_capital_weight": float(sleeve_capital),
            "borrow_cost_mode": "annualized_linear_fee_sensitivity_zero_by_default",
            "skipped_incomplete_return_path_count": int(len(skipped_trade_df)),
            "turnover_accounting": "capital_scaled_full_sleeve_round_trip_without_cross_sleeve_netting",
            "is_short_sample_warning": bool(metrics.get("rebalance_count", 0) < 10),
        }
    )

    if config.write_outputs:
        # 全量网格实验会调用上万次回测。此处保留开关：
        # 单次研究时写出详细文件，全量对照实验时只返回内存结果，避免磁盘 IO 成为主要瓶颈。
        config.output_dir.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(config.output_dir / "daily_returns.csv", index=False)
        weight_df.to_csv(config.output_dir / "portfolio_weights.csv", index=False)
        turnover_df.to_csv(config.output_dir / "turnover_cost.csv", index=False)
        skipped_trade_df.to_csv(config.output_dir / "skipped_trades.csv", index=False)
        sector_df.to_csv(config.output_dir / "sector_exposure.csv", index=False)
        sleeve_daily_df.to_csv(config.output_dir / "sleeve_daily_returns.csv", index=False)
        contribution_df.to_csv(config.output_dir / "position_daily_contributions.csv", index=False)
        extreme_days_df.to_csv(config.output_dir / "extreme_return_days.csv", index=False)
        instrument_attribution_df.to_csv(
            config.output_dir / "instrument_return_attribution.csv",
            index=False,
        )
        (config.output_dir / "portfolio_metrics.json").write_text(
            dumps_strict_json(metrics),
            encoding="utf-8",
        )
        write_long_short_report(
            config.output_dir / "portfolio_report.md",
            config=config,
            metrics=metrics,
            daily_df=daily_df,
            turnover_df=turnover_df,
            sector_df=sector_df,
            extreme_days_df=extreme_days_df,
            instrument_attribution_df=instrument_attribution_df,
        )

    return {
        "metrics": metrics,
        "daily_df": daily_df,
        "weight_df": weight_df,
        "turnover_df": turnover_df,
        "skipped_trade_df": skipped_trade_df,
        "sector_df": sector_df,
        "contribution_df": contribution_df,
        "extreme_days_df": extreme_days_df,
        "instrument_attribution_df": instrument_attribution_df,
    }
