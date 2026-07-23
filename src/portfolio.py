"""组合构建与暴露分析模块。

这个模块故意不去碰原有训练主流程，而是专门回答一个更接近实战的问题：

- 已经有了 `predicted_y` 之后，如何把信号转成一个最小可解释组合？
- 这个组合的换手、成本和风格暴露长什么样？
- 成本后结果是否还站得住？

当前版本坚持“先做小而硬”的原则：

1. 只基于已经落盘的预测结果文件做组合分析；
2. 默认使用 `top N` 选股；
3. 支持简单权重方案：等权 / 排名加权；
4. 输出换手、交易成本、行业暴露、市值暴露和净值指标；
5. 不在这里上来就引入复杂优化器，避免把解释成本抬得过高。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import PRICE_ADJUSTMENT_MODES
from src.provenance import dumps_strict_json


REQUIRED_PREDICTION_COLUMNS = {"date", "instrument_id", "y", "predicted_y"}
SUPPORTED_HOLDING_CLOCKS = {"signal_horizon", "execution_horizon"}


def _safe_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std_value = values.std(ddof=0)
    if pd.isna(std_value) or abs(float(std_value)) < 1e-12:
        return values - values.mean()
    return (values - values.mean()) / std_value


def infer_hold_days(
    predictions_path: Path,
    explicit_hold_days: int | None = None,
) -> int:
    """推断持有周期。

    优先级：

    1. 命令行显式传入；
    2. 读取同目录下的 `training_report.md`；
    3. 从路径名里匹配 `5d / 10d` 这样的标记。
    """

    if explicit_hold_days is not None:
        if explicit_hold_days <= 0:
            raise ValueError("hold_days must be a positive integer.")
        return explicit_hold_days

    report_path = predictions_path.with_name("training_report.md")
    if report_path.exists():
        match = re.search(
            r"Active target horizon \(days\): `(\d+)`",
            report_path.read_text(encoding="utf-8"),
        )
        if match:
            return int(match.group(1))

    match = re.search(r"(?<!\d)(\d+)d(?!\d)", str(predictions_path))
    if match:
        return int(match.group(1))

    raise ValueError(
        f"Unable to infer hold_days from {predictions_path}. "
        "Please pass --hold-days explicitly."
    )


def load_prediction_frame(predictions_path: Path) -> pd.DataFrame:
    """读取预测结果文件。"""

    prediction_df = pd.read_csv(predictions_path)
    missing_columns = REQUIRED_PREDICTION_COLUMNS - set(prediction_df.columns)
    if missing_columns:
        raise ValueError(
            f"Prediction file {predictions_path} is missing required columns: {sorted(missing_columns)}"
        )

    prediction_df["date"] = pd.to_datetime(prediction_df["date"])
    prediction_df["instrument_id"] = prediction_df["instrument_id"].astype(str).str.strip()
    prediction_df["y"] = pd.to_numeric(prediction_df["y"], errors="coerce")
    prediction_df["predicted_y"] = pd.to_numeric(prediction_df["predicted_y"], errors="coerce")
    prediction_df = prediction_df.dropna(subset=["date", "instrument_id", "y", "predicted_y"])
    if prediction_df.duplicated(subset=["date", "instrument_id"], keep=False).any():
        raise ValueError(
            f"Prediction file {predictions_path} contains duplicate date/instrument_id rows."
        )
    prediction_df = prediction_df.sort_values(["date", "instrument_id"]).reset_index(drop=True)
    return prediction_df


def load_market_snapshot_frame(
    data_path: Path,
    price_adjustment_mode: str = "vendor_adjusted",
) -> pd.DataFrame:
    """读取用于组合收益和暴露分析的市场快照。

    训练层和组合层必须使用同一套价格口径。否则拆股或现金分红附近可能
    出现虚假的巨大收益，最终同时污染 IC、组合收益、Sharpe 和回撤。

    `vendor_adjusted` 会在存在 `adjustment = Adj Close / Close` 时对 close
    进行复权。公开主线要求原始 OHLC 与非平凡 adjustment 同时存在，以便
    审计公司行为和原始成交额；`raw` 仅供公司行为敏感性检查使用。
    """

    if price_adjustment_mode not in PRICE_ADJUSTMENT_MODES:
        raise ValueError(
            f"Unsupported price_adjustment_mode: {price_adjustment_mode}. "
            f"Supported modes: {list(PRICE_ADJUSTMENT_MODES)}"
        )

    raw = pd.read_csv(data_path)
    required_columns = {"instrument_id", "date", "close", "volume"}
    missing_columns = required_columns - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Market data file {data_path} is missing required columns: {sorted(missing_columns)}")

    raw["date"] = pd.to_datetime(raw["date"])
    raw["instrument_id"] = raw["instrument_id"].astype(str).str.strip()
    raw = raw.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    if raw.duplicated(subset=["date", "instrument_id"], keep=False).any():
        raise ValueError(
            f"Market data file {data_path} contains duplicate date/instrument_id rows."
        )
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")

    # 与 src.data_loader.load_daily_data 保持一致：复权因子只能参与价格清洗，
    # 不能作为预测特征。无 adjustment 列时，约定输入 close 已经是目标口径。
    if price_adjustment_mode == "vendor_adjusted" and "adjustment" in raw.columns:
        adjustment = pd.to_numeric(raw["adjustment"], errors="coerce")
        adjustment = adjustment.where(adjustment > 0.0, np.nan)
        raw["close"] = raw["close"] * adjustment

    if "market_cap" in raw.columns:
        raw["market_cap"] = pd.to_numeric(raw["market_cap"], errors="coerce")
    else:
        raw["market_cap"] = np.nan

    if "sector" not in raw.columns:
        raw["sector"] = "Unknown"

    raw["market_cap"] = raw.groupby("instrument_id")["market_cap"].transform(lambda series: series.ffill())
    raw["sector"] = raw["sector"].fillna("Unknown").astype(str)
    raw["log_market_cap"] = np.log(raw["market_cap"].where(raw["market_cap"] > 0.0))
    raw["size_exposure_z"] = raw.groupby("date")["log_market_cap"].transform(_safe_zscore)

    raw["daily_close_return"] = raw.groupby("instrument_id")["close"].pct_change(fill_method=None)
    raw["realized_vol_20"] = (
        raw.groupby("instrument_id")["daily_close_return"]
        .rolling(window=20, min_periods=5)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    return raw[
        [
            "date",
            "instrument_id",
            "close",
            "daily_close_return",
            "sector",
            "market_cap",
            "size_exposure_z",
            "realized_vol_20",
        ]
    ].copy()


def merge_predictions_with_market(
    prediction_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = prediction_df.merge(
        market_df,
        how="left",
        on=["date", "instrument_id"],
        validate="one_to_one",
    )
    merged["sector"] = merged["sector"].fillna("Unknown")

    # 缺失市值时保留 NaN。组合收益不依赖市值，size exposure 仅在有真实
    # 数据时报告；填成 0 会让“未知暴露”看起来像“恰好中性”。
    merged["size_exposure_z"] = pd.to_numeric(merged["size_exposure_z"], errors="coerce")
    return merged


def _normalize_weight_map(weight_map: dict[str, float]) -> dict[str, float]:
    total_weight = sum(weight_map.values())
    if total_weight <= 0:
        raise ValueError("Weight map must sum to a positive number.")
    return {instrument_id: float(weight / total_weight) for instrument_id, weight in weight_map.items()}


def _apply_max_weight_constraint(
    weight_map: dict[str, float],
    max_weight: float | None,
) -> dict[str, float]:
    normalized = _normalize_weight_map(weight_map)
    if max_weight is None:
        return normalized

    if not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must fall inside (0, 1].")
    if max_weight * len(normalized) < 1.0 - 1e-12:
        raise ValueError("max_weight is too small for the selected stock count.")

    adjusted = normalized.copy()
    for _ in range(50):
        overweight_names = [name for name, weight in adjusted.items() if weight > max_weight + 1e-12]
        if not overweight_names:
            return _normalize_weight_map(adjusted)

        excess_weight = 0.0
        for name in overweight_names:
            excess_weight += adjusted[name] - max_weight
            adjusted[name] = max_weight

        underweight_names = [name for name, weight in adjusted.items() if weight < max_weight - 1e-12]
        if not underweight_names:
            return _normalize_weight_map(adjusted)

        underweight_total = sum(adjusted[name] for name in underweight_names)
        if underweight_total <= 0:
            equal_add = excess_weight / len(underweight_names)
            for name in underweight_names:
                adjusted[name] += equal_add
        else:
            for name in underweight_names:
                adjusted[name] += excess_weight * adjusted[name] / underweight_total

    return _normalize_weight_map(adjusted)


def build_target_weights(
    ranked_slice: pd.DataFrame,
    top_n: int,
    weighting_scheme: str = "equal",
    max_weight: float | None = None,
) -> dict[str, float]:
    """把当日预测分数转成目标权重。"""

    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    selected = ranked_slice.head(top_n).copy()
    if selected.empty:
        raise ValueError("No rows are available to build target weights.")

    if weighting_scheme == "equal":
        raw_weight_map = {instrument_id: 1.0 for instrument_id in selected["instrument_id"]}
    elif weighting_scheme == "rank":
        rank_scores = np.arange(len(selected), 0, -1, dtype=float)
        raw_weight_map = {
            str(instrument_id): float(score)
            for instrument_id, score in zip(selected["instrument_id"], rank_scores)
        }
    else:
        raise ValueError(f"Unsupported weighting_scheme: {weighting_scheme}")

    return _apply_max_weight_constraint(raw_weight_map, max_weight=max_weight)


def build_sector_max_count_map(
    universe_slice: pd.DataFrame,
    top_n: int,
    sector_active_tolerance: float,
) -> dict[str, int]:
    """根据行业权重和容忍度构造每个行业最多可选多少只股票。"""

    if sector_active_tolerance < 0:
        raise ValueError("sector_active_tolerance must be non-negative.")

    unit_weight = 1.0 / top_n
    sector_weights = universe_slice["sector"].value_counts(normalize=True)
    sector_max_count_map: dict[str, int] = {}

    for sector_name, sector_weight in sector_weights.items():
        allowed_weight = min(1.0, float(sector_weight) + sector_active_tolerance)
        max_count = int(math.floor(allowed_weight / unit_weight + 1e-12))
        sector_max_count_map[str(sector_name)] = max(1, max_count)

    return sector_max_count_map


def select_constrained_slice(
    ranked_slice: pd.DataFrame,
    universe_slice: pd.DataFrame,
    top_n: int,
    sector_active_tolerance: float | None = None,
    size_exposure_limit: float | None = None,
) -> pd.DataFrame:
    """在不引入复杂优化器的前提下做轻量约束选股。

    当前约束分两步：

    1. 先限制单一行业最多入选的股票数；
    2. 如果组合 size 暴露仍然过大，再做少量贪心替换。
    """

    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    sector_max_count_map: dict[str, int] | None = None
    if sector_active_tolerance is not None:
        sector_max_count_map = build_sector_max_count_map(
            universe_slice=universe_slice,
            top_n=top_n,
            sector_active_tolerance=sector_active_tolerance,
        )

    selected_rows: list[pd.Series] = []
    sector_count_map: dict[str, int] = {}

    for _, row in ranked_slice.iterrows():
        sector_name = str(row["sector"])
        if sector_max_count_map is not None:
            current_count = sector_count_map.get(sector_name, 0)
            if current_count >= sector_max_count_map.get(sector_name, top_n):
                continue

        selected_rows.append(row)
        sector_count_map[sector_name] = sector_count_map.get(sector_name, 0) + 1
        if len(selected_rows) >= top_n:
            break

    if len(selected_rows) < top_n:
        selected_ids = {str(row["instrument_id"]) for row in selected_rows}
        for _, row in ranked_slice.iterrows():
            instrument_id = str(row["instrument_id"])
            if instrument_id in selected_ids:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= top_n:
                break

    selected_slice = pd.DataFrame(selected_rows).reset_index(drop=True)
    if selected_slice.empty:
        raise ValueError("Constrained selection produced an empty portfolio.")

    if size_exposure_limit is None:
        return selected_slice

    if size_exposure_limit < 0:
        raise ValueError("size_exposure_limit must be non-negative.")

    selected_ids = set(selected_slice["instrument_id"].astype(str))

    def current_size_exposure(frame: pd.DataFrame) -> float:
        return float(frame["size_exposure_z"].mean())

    for _ in range(50):
        current_exposure = current_size_exposure(selected_slice)
        if abs(current_exposure) <= size_exposure_limit:
            break

        drop_candidates = selected_slice.copy()
        if current_exposure > 0:
            drop_candidates = drop_candidates.sort_values(
                ["size_exposure_z", "predicted_y"],
                ascending=[False, True],
            )
            add_candidates = ranked_slice[~ranked_slice["instrument_id"].astype(str).isin(selected_ids)].copy()
            add_candidates = add_candidates.sort_values(
                ["size_exposure_z", "predicted_y"],
                ascending=[True, False],
            )
        else:
            drop_candidates = drop_candidates.sort_values(
                ["size_exposure_z", "predicted_y"],
                ascending=[True, True],
            )
            add_candidates = ranked_slice[~ranked_slice["instrument_id"].astype(str).isin(selected_ids)].copy()
            add_candidates = add_candidates.sort_values(
                ["size_exposure_z", "predicted_y"],
                ascending=[False, False],
            )

        best_swap: tuple[int, pd.Series] | None = None
        best_objective = None

        for drop_index, drop_row in drop_candidates.iterrows():
            drop_instrument = str(drop_row["instrument_id"])
            drop_sector = str(drop_row["sector"])
            current_sector_counts = selected_slice["sector"].astype(str).value_counts().to_dict()

            for _, add_row in add_candidates.head(80).iterrows():
                add_instrument = str(add_row["instrument_id"])
                add_sector = str(add_row["sector"])
                if add_instrument in selected_ids:
                    continue

                if sector_max_count_map is not None:
                    next_sector_counts = current_sector_counts.copy()
                    next_sector_counts[drop_sector] = next_sector_counts.get(drop_sector, 0) - 1
                    next_sector_counts[add_sector] = next_sector_counts.get(add_sector, 0) + 1
                    if next_sector_counts.get(add_sector, 0) > sector_max_count_map.get(add_sector, top_n):
                        continue

                next_exposure = current_exposure + (
                    float(add_row["size_exposure_z"]) - float(drop_row["size_exposure_z"])
                ) / len(selected_slice)
                if abs(next_exposure) >= abs(current_exposure) - 1e-12:
                    continue

                alpha_loss = float(drop_row["predicted_y"]) - float(add_row["predicted_y"])
                objective = abs(next_exposure) + max(alpha_loss, 0.0) * 5.0

                if best_objective is None or objective < best_objective:
                    best_objective = objective
                    best_swap = (drop_index, add_row)

            if best_swap is not None and best_objective is not None and best_objective < abs(current_exposure):
                break

        if best_swap is None:
            break

        drop_index, add_row = best_swap
        drop_instrument = str(selected_slice.loc[drop_index, "instrument_id"])
        selected_ids.remove(drop_instrument)
        selected_ids.add(str(add_row["instrument_id"]))
        selected_slice.loc[drop_index] = add_row

    selected_slice = selected_slice.sort_values("predicted_y", ascending=False).reset_index(drop=True)
    return selected_slice


def compute_trade_summary(
    previous_weights: dict[str, float],
    target_weights: dict[str, float],
    buy_cost_bps: float,
    sell_cost_bps: float,
) -> dict[str, float]:
    """计算换手与交易成本。"""

    instrument_union = set(previous_weights) | set(target_weights)
    buy_turnover = 0.0
    sell_turnover = 0.0

    for instrument_id in instrument_union:
        previous_weight = float(previous_weights.get(instrument_id, 0.0))
        target_weight = float(target_weights.get(instrument_id, 0.0))
        delta_weight = target_weight - previous_weight
        if delta_weight > 0:
            buy_turnover += delta_weight
        elif delta_weight < 0:
            sell_turnover += -delta_weight

    transaction_cost = (
        buy_turnover * (buy_cost_bps / 10000.0)
        + sell_turnover * (sell_cost_bps / 10000.0)
    )
    return {
        "buy_turnover": float(buy_turnover),
        "sell_turnover": float(sell_turnover),
        "turnover": float(0.5 * (buy_turnover + sell_turnover)),
        "transaction_cost": float(transaction_cost),
        "transaction_cost_bps": float(transaction_cost * 10000.0),
    }


def compute_portfolio_exposures(
    universe_slice: pd.DataFrame,
    selected_slice: pd.DataFrame,
) -> tuple[dict[str, float | str | bool], pd.DataFrame]:
    """计算组合和基准的 size / sector 暴露。"""

    universe_slice = universe_slice.copy()
    selected_slice = selected_slice.copy()
    universe_count = len(universe_slice)

    if universe_count <= 0:
        raise ValueError("universe_slice must contain at least one row.")

    valid_universe_size = pd.to_numeric(universe_slice["size_exposure_z"], errors="coerce")
    valid_selected_size = pd.to_numeric(selected_slice["size_exposure_z"], errors="coerce")
    size_exposure_available = bool(
        valid_universe_size.notna().mean() >= 0.90
        and valid_selected_size.notna().all()
    )
    if size_exposure_available:
        portfolio_size_exposure = float((selected_slice["weight"] * valid_selected_size).sum())
        universe_size_exposure = float(valid_universe_size.mean())
        active_size_exposure = portfolio_size_exposure - universe_size_exposure
        weighted_market_cap = float(
            (selected_slice["weight"] * pd.to_numeric(selected_slice["market_cap"], errors="coerce")).sum()
        )
        universe_market_cap = float(pd.to_numeric(universe_slice["market_cap"], errors="coerce").mean())
    else:
        portfolio_size_exposure = float("nan")
        universe_size_exposure = float("nan")
        active_size_exposure = float("nan")
        weighted_market_cap = float("nan")
        universe_market_cap = float("nan")

    portfolio_sector = selected_slice.groupby("sector", as_index=False)["weight"].sum()
    portfolio_sector = portfolio_sector.rename(columns={"weight": "portfolio_weight"})

    universe_sector = (
        universe_slice.groupby("sector")
        .size()
        .div(universe_count)
        .rename("universe_weight")
        .reset_index()
    )

    sector_exposure_df = universe_sector.merge(portfolio_sector, how="outer", on="sector").fillna(0.0)
    sector_exposure_df["active_weight"] = (
        sector_exposure_df["portfolio_weight"] - sector_exposure_df["universe_weight"]
    )
    sector_exposure_df = sector_exposure_df.sort_values("active_weight", ascending=False).reset_index(drop=True)

    top_overweight = sector_exposure_df.iloc[0]
    top_underweight = sector_exposure_df.iloc[-1]
    exposure_summary = {
        "size_exposure_available": size_exposure_available,
        "portfolio_size_exposure": portfolio_size_exposure,
        "universe_size_exposure": universe_size_exposure,
        "active_size_exposure": active_size_exposure,
        "portfolio_weighted_market_cap": weighted_market_cap,
        "universe_average_market_cap": universe_market_cap,
        "top_overweight_sector": str(top_overweight["sector"]),
        "top_overweight_active_weight": float(top_overweight["active_weight"]),
        "top_underweight_sector": str(top_underweight["sector"]),
        "top_underweight_active_weight": float(top_underweight["active_weight"]),
    }
    return exposure_summary, sector_exposure_df


def shift_market_date(
    market_dates: list[pd.Timestamp],
    date_value: pd.Timestamp,
    offset: int,
) -> pd.Timestamp | pd.NaT:
    """把某个市场日期向后平移指定交易日数。"""

    market_index = {current_date: idx for idx, current_date in enumerate(market_dates)}
    if date_value not in market_index:
        return pd.NaT
    target_index = market_index[date_value] + offset
    if target_index < 0 or target_index >= len(market_dates):
        return pd.NaT
    return market_dates[target_index]


def resolve_holding_window(
    market_dates: list[pd.Timestamp] | list[np.datetime64],
    *,
    signal_date: pd.Timestamp,
    signal_delay_days: int,
    hold_days: int,
    holding_clock: str,
) -> tuple[pd.Timestamp, pd.Timestamp, int] | None:
    """Resolve one signal's execution date, endpoint, and accrual length.

    The canonical ``signal_horizon`` clock measures ``hold_days`` from the
    signal date.  A ``y_10d`` signal at ``t`` with a one-day close execution
    proxy therefore accrues nine daily returns through ``t+10``.  The legacy
    ``execution_horizon`` clock starts the full holding period after execution
    and is retained only as a separately labelled sensitivity definition.
    """

    if signal_delay_days < 0:
        raise ValueError("signal_delay_days must be non-negative.")
    if hold_days <= 0:
        raise ValueError("hold_days must be positive.")
    if holding_clock not in SUPPORTED_HOLDING_CLOCKS:
        raise ValueError(
            f"Unsupported holding_clock: {holding_clock}. "
            f"Expected one of {sorted(SUPPORTED_HOLDING_CLOCKS)}."
        )
    if holding_clock == "signal_horizon" and signal_delay_days >= hold_days:
        raise ValueError(
            "signal_delay_days must be smaller than hold_days when "
            "holding_clock='signal_horizon'."
        )

    execution_date = shift_market_date(market_dates, signal_date, signal_delay_days)
    if pd.isna(execution_date):
        return None
    execution_date = pd.Timestamp(execution_date)

    if holding_clock == "signal_horizon":
        end_date = shift_market_date(market_dates, signal_date, hold_days)
        effective_holding_days = hold_days - signal_delay_days
    else:
        end_date = shift_market_date(market_dates, execution_date, hold_days)
        effective_holding_days = hold_days

    if pd.isna(end_date):
        return None
    return execution_date, pd.Timestamp(end_date), int(effective_holding_days)


def compute_nav_max_drawdown(nav_series: pd.Series) -> float:
    running_peak = nav_series.cummax()
    drawdown = nav_series / running_peak - 1.0
    return float(drawdown.min()) if not drawdown.empty else float("nan")


def summarize_period_metrics(
    period_df: pd.DataFrame,
    annualization_days: int,
) -> dict[str, float | int | str]:
    """汇总组合期度表现。"""

    if period_df.empty:
        return {}

    annual_factor = 252.0 / annualization_days
    portfolio_nav = float(period_df["portfolio_nav"].iloc[-1])
    benchmark_nav = float(period_df["benchmark_nav"].iloc[-1])
    period_returns = period_df["net_return"].astype(float)
    period_count = len(period_df)

    annualized_return = portfolio_nav ** (annual_factor / period_count) - 1.0 if period_count > 0 else float("nan")
    annualized_vol = (
        float(period_returns.std(ddof=1) * math.sqrt(annual_factor))
        if period_count > 1
        else float("nan")
    )
    daily_sample_std = float(period_returns.std(ddof=1)) if period_count > 1 else float("nan")
    sharpe = (
        float(period_returns.mean() / daily_sample_std * math.sqrt(annual_factor))
        if np.isfinite(daily_sample_std) and daily_sample_std > 0
        else float("nan")
    )

    benchmark_returns = period_df["benchmark_return"].astype(float)
    excess_returns = period_df["excess_return"].astype(float)
    benchmark_annualized_return = (
        benchmark_nav ** (annual_factor / period_count) - 1.0 if period_count > 0 else float("nan")
    )

    return {
        "period_count": int(period_count),
        "annualization_days": int(annualization_days),
        "portfolio_total_return": float(portfolio_nav - 1.0),
        "portfolio_annualized_return": float(annualized_return),
        "portfolio_annualized_vol": annualized_vol,
        "portfolio_sharpe": float(sharpe) if not pd.isna(sharpe) else float("nan"),
        "sharpe_definition": (
            "mean daily net return / sample std daily net return * sqrt(252), "
            "zero risk-free rate"
        ),
        "portfolio_max_drawdown": compute_nav_max_drawdown(period_df["portfolio_nav"]),
        "benchmark_total_return": float(benchmark_nav - 1.0),
        "benchmark_annualized_return": float(benchmark_annualized_return),
        "benchmark_max_drawdown": compute_nav_max_drawdown(period_df["benchmark_nav"]),
        "excess_total_return_vs_benchmark": float(portfolio_nav / benchmark_nav - 1.0),
        "average_turnover": float(period_df["turnover"].mean()),
        "average_transaction_cost_bps": float(period_df["transaction_cost_bps"].mean()),
        "total_transaction_cost": float(
            period_df.get("buy_cost", pd.Series(0.0, index=period_df.index)).sum()
            + period_df.get("sell_cost", pd.Series(0.0, index=period_df.index)).sum()
        ),
        "positive_period_ratio": float((period_returns > 0).mean()),
        "average_excess_return": float(excess_returns.mean()),
        "best_period_return": float(period_returns.max()),
        "worst_period_return": float(period_returns.min()),
        "average_abs_active_size_exposure": float(period_df["active_size_exposure"].abs().mean()),
        "average_buy_turnover": float(period_df["buy_turnover"].mean()),
        "average_sell_turnover": float(period_df["sell_turnover"].mean()),
        "average_benchmark_return": float(benchmark_returns.mean()),
    }


def build_schedule_summary_frame(schedule_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """把调仓计划整理成表，便于单独落盘。"""

    if not schedule_rows:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "execution_date",
                "end_date",
                "selected_count",
                "turnover",
                "buy_turnover",
                "sell_turnover",
                "transaction_cost_bps",
                "active_size_exposure",
                "top_overweight_sector",
                "top_underweight_sector",
            ]
        )
    return pd.DataFrame(schedule_rows)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def write_portfolio_report(
    output_path: Path,
    *,
    run_name: str,
    predictions_path: Path,
    data_path: Path,
    hold_days: int,
    step_days: int,
    signal_delay_days: int,
    holding_clock: str,
    top_n: int,
    weighting_scheme: str,
    max_weight: float | None,
    constraint_mode: str,
    sector_active_tolerance: float | None,
    size_exposure_limit: float | None,
    buy_cost_bps: float,
    sell_cost_bps: float,
    metrics: dict[str, Any],
    period_df: pd.DataFrame,
    sector_summary_df: pd.DataFrame,
) -> None:
    """写出组合分析 Markdown 报告。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    period_preview = period_df.head(10).copy()
    sector_preview = sector_summary_df.head(10).copy()

    report = f"""# Portfolio Diagnostic Report

## 1. Run Setup

- Run name: `{run_name}`
- Predictions path: `{predictions_path}`
- Market data path: `{data_path}`
- Hold days: `{hold_days}`
- Holding clock: `{holding_clock}`
- Effective executable holding days: `{metrics.get('effective_holding_days')}`
- Rebalance step days: `{step_days}`
- Signal delay days: `{signal_delay_days}`
- Top N holdings: `{top_n}`
- Weighting scheme: `{weighting_scheme}`
- Max weight constraint: `{max_weight if max_weight is not None else "disabled"}`
- Constraint mode: `{constraint_mode}`
- Sector active tolerance: `{sector_active_tolerance if sector_active_tolerance is not None else "disabled"}`
- Size exposure limit: `{size_exposure_limit if size_exposure_limit is not None else "disabled"}`
- Buy cost (bps): `{buy_cost_bps}`
- Sell cost (bps): `{sell_cost_bps}`
- Sharpe definition: `mean(daily net return) / sample std(daily net return) * sqrt(252)`, zero risk-free rate

## 2. Important Scope Note

这份结果是 **independent backtest-style portfolio diagnostic**。

- 选股信号来自预测文件中的 `predicted_y`
- 实际组合收益使用市场价格表里的 `close-to-close` 日收益重算
- 支持 `signal_delay_days`
- canonical `signal_horizon` 将持有终点对齐到信号日 forward-return 标签终点
- 支持 `effective_holding_days > step_days` 时的重叠 sleeve
- 当前重点是看：权重、换手、成本、size/sector 暴露、以及成本后净值指标

## 3. Aggregate Metrics

```json
{dumps_strict_json(metrics)}
```

## 4. Period Summary Preview

{_dataframe_to_markdown(period_preview)}

## 5. Average Sector Active Exposure

{_dataframe_to_markdown(sector_preview)}
"""

    output_path.write_text(report, encoding="utf-8")


def run_portfolio_diagnostic(
    *,
    predictions_path: Path,
    market_snapshot_df: pd.DataFrame,
    data_path: Path,
    output_dir: Path,
    hold_days: int,
    step_days: int,
    top_n: int,
    weighting_scheme: str,
    max_weight: float | None,
    constraint_mode: str,
    sector_active_tolerance: float | None,
    size_exposure_limit: float | None,
    buy_cost_bps: float,
    sell_cost_bps: float,
    run_name: str,
    signal_delay_days: int = 1,
    holding_clock: str = "signal_horizon",
) -> dict[str, Any]:
    """执行单个预测结果文件的组合诊断。

    当前版本比最早的“直接吃 `y` 做 period spread”更真实一些：

    - 选股仍然来自预测文件里的 `predicted_y`
    - 真实收益改为从市场价格表中重算
    - 支持 `signal_delay_days`
    - 支持 `hold_days > step_days` 时的重叠持仓
    - 成本按每个 sleeve 的开平仓收取
    """

    if signal_delay_days < 0:
        raise ValueError("signal_delay_days must be non-negative.")
    if hold_days <= 0 or step_days <= 0:
        raise ValueError("hold_days and step_days must be positive integers.")
    if holding_clock not in SUPPORTED_HOLDING_CLOCKS:
        raise ValueError(
            f"Unsupported holding_clock: {holding_clock}. "
            f"Expected one of {sorted(SUPPORTED_HOLDING_CLOCKS)}."
        )
    if holding_clock == "signal_horizon" and signal_delay_days >= hold_days:
        raise ValueError(
            "signal_delay_days must be smaller than hold_days when "
            "holding_clock='signal_horizon'."
        )

    prediction_df = load_prediction_frame(predictions_path)
    merged_df = merge_predictions_with_market(prediction_df, market_snapshot_df)

    signal_dates = sorted(pd.to_datetime(merged_df["date"]).unique())
    market_dates = sorted(pd.to_datetime(market_snapshot_df["date"]).unique())
    if not signal_dates or not market_dates:
        raise ValueError("Signal dates and market dates must both be non-empty.")

    market_returns = market_snapshot_df[["date", "instrument_id", "daily_close_return"]].copy()
    market_returns = market_returns.dropna(subset=["daily_close_return"])
    benchmark_daily_return = (
        market_returns.groupby("date", as_index=False)["daily_close_return"]
        .mean()
        .rename(columns={"daily_close_return": "benchmark_return"})
    )

    weight_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    sleeve_daily_rows: list[dict[str, Any]] = []

    effective_holding_days = (
        hold_days - signal_delay_days
        if holding_clock == "signal_horizon"
        else hold_days
    )
    max_active_sleeves = max(1, math.ceil(effective_holding_days / step_days))
    sleeve_capital = 1.0 / max_active_sleeves
    previous_weights_by_sleeve: dict[int, dict[str, float]] = {}

    rebalance_dates = signal_dates[::step_days]
    for rebalance_index, signal_date in enumerate(rebalance_dates):
        signal_date = pd.Timestamp(signal_date)
        sleeve_slot = int(rebalance_index % max_active_sleeves)
        previous_weights = previous_weights_by_sleeve.get(sleeve_slot, {})
        universe_slice = merged_df[merged_df["date"] == signal_date].copy()
        if universe_slice.empty:
            continue
        ranked_slice = universe_slice.sort_values("predicted_y", ascending=False).reset_index(drop=True)
        if constraint_mode == "unconstrained":
            selected_input_slice = ranked_slice
        elif constraint_mode == "sector_size_constrained":
            selected_input_slice = select_constrained_slice(
                ranked_slice=ranked_slice,
                universe_slice=universe_slice,
                top_n=top_n,
                sector_active_tolerance=sector_active_tolerance,
                size_exposure_limit=size_exposure_limit,
            )
        else:
            raise ValueError(f"Unsupported constraint_mode: {constraint_mode}")

        target_weights = build_target_weights(
            ranked_slice=selected_input_slice,
            top_n=top_n,
            weighting_scheme=weighting_scheme,
            max_weight=max_weight,
        )

        selected_slice = selected_input_slice[selected_input_slice["instrument_id"].isin(target_weights)].copy()
        selected_slice["weight"] = selected_slice["instrument_id"].map(target_weights).astype(float)
        selected_slice = selected_slice.sort_values("weight", ascending=False).reset_index(drop=True)

        holding_window = resolve_holding_window(
            market_dates,
            signal_date=signal_date,
            signal_delay_days=signal_delay_days,
            hold_days=hold_days,
            holding_clock=holding_clock,
        )
        if holding_window is None:
            continue
        execution_date, end_date, window_holding_days = holding_window

        trade_summary = compute_trade_summary(
            previous_weights=previous_weights,
            target_weights=target_weights,
            buy_cost_bps=buy_cost_bps,
            sell_cost_bps=sell_cost_bps,
        )
        exposure_summary, sector_exposure_df = compute_portfolio_exposures(
            universe_slice=universe_slice,
            selected_slice=selected_slice,
        )
        schedule_rows.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "execution_date": execution_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "holding_clock": holding_clock,
                "effective_holding_days": int(window_holding_days),
                "sleeve_slot": sleeve_slot,
                "selected_count": int(len(selected_slice)),
                "turnover": float(sleeve_capital) * trade_summary["turnover"],
                "buy_turnover": float(sleeve_capital) * trade_summary["buy_turnover"],
                "sell_turnover": float(sleeve_capital) * trade_summary["sell_turnover"],
                "transaction_cost_bps": float(sleeve_capital) * trade_summary["transaction_cost_bps"],
                "active_size_exposure": exposure_summary["active_size_exposure"],
                "top_overweight_sector": exposure_summary["top_overweight_sector"],
                "top_underweight_sector": exposure_summary["top_underweight_sector"],
            }
        )

        for _, row in selected_slice.iterrows():
            weight_rows.append(
                {
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "execution_date": execution_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "holding_clock": holding_clock,
                    "effective_holding_days": int(window_holding_days),
                    "sleeve_slot": sleeve_slot,
                    "instrument_id": str(row["instrument_id"]),
                    "weight": float(row["weight"]),
                    "capital_weight": float(sleeve_capital),
                    "effective_weight": float(row["weight"]) * float(sleeve_capital),
                    "predicted_y": float(row["predicted_y"]),
                    "signal_y": float(row["y"]),
                    "sector": str(row["sector"]),
                    "market_cap": float(row["market_cap"]),
                    "size_exposure_z": float(row["size_exposure_z"]),
                }
            )

        for _, row in sector_exposure_df.iterrows():
            sector_rows.append(
                {
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "execution_date": execution_date.strftime("%Y-%m-%d"),
                    "sleeve_slot": sleeve_slot,
                    "sector": str(row["sector"]),
                    "portfolio_weight": float(row["portfolio_weight"]),
                    "universe_weight": float(row["universe_weight"]),
                    "active_weight": float(row["active_weight"]),
                }
            )

        sleeve_return_frame = market_returns[
            (market_returns["date"] > execution_date) & (market_returns["date"] <= end_date)
        ].copy()
        if sleeve_return_frame.empty:
            previous_weights_by_sleeve[sleeve_slot] = target_weights
            continue

        sleeve_return_frame["target_weight"] = sleeve_return_frame["instrument_id"].map(target_weights)
        sleeve_return_frame = sleeve_return_frame.dropna(subset=["target_weight", "daily_close_return"])
        if sleeve_return_frame.empty:
            previous_weights_by_sleeve[sleeve_slot] = target_weights
            continue

        daily_sleeve_return = (
            sleeve_return_frame.groupby("date")
            .apply(lambda frame: float((frame["target_weight"] * frame["daily_close_return"]).sum()))
            .rename("gross_return")
            .reset_index()
        )
        daily_sleeve_return["signal_date"] = signal_date.strftime("%Y-%m-%d")
        daily_sleeve_return["execution_date"] = execution_date.strftime("%Y-%m-%d")
        daily_sleeve_return["end_date"] = end_date.strftime("%Y-%m-%d")
        daily_sleeve_return["holding_clock"] = holding_clock
        daily_sleeve_return["effective_holding_days"] = int(window_holding_days)
        daily_sleeve_return["sleeve_slot"] = sleeve_slot
        daily_sleeve_return["capital_weight"] = float(sleeve_capital)
        daily_sleeve_return["buy_cost"] = 0.0
        daily_sleeve_return["sell_cost"] = 0.0
        if not daily_sleeve_return.empty:
            daily_sleeve_return.loc[daily_sleeve_return.index[0], "buy_cost"] = (
                float(sleeve_capital)
                * trade_summary["buy_turnover"]
                * (buy_cost_bps / 10000.0)
            )
            daily_sleeve_return.loc[daily_sleeve_return.index[0], "sell_cost"] = (
                float(sleeve_capital)
                * trade_summary["sell_turnover"]
                * (sell_cost_bps / 10000.0)
            )
        daily_sleeve_return["net_return"] = (
            daily_sleeve_return["gross_return"] * float(sleeve_capital)
            - daily_sleeve_return["buy_cost"]
            - daily_sleeve_return["sell_cost"]
        )
        sleeve_daily_rows.extend(daily_sleeve_return.to_dict("records"))
        previous_weights_by_sleeve[sleeve_slot] = target_weights

    schedule_df = build_schedule_summary_frame(schedule_rows)
    weight_df = pd.DataFrame(weight_rows)
    sector_df = pd.DataFrame(sector_rows)
    sleeve_daily_df = pd.DataFrame(sleeve_daily_rows)

    if sleeve_daily_df.empty:
        period_df = pd.DataFrame(
            columns=[
                "date",
                "gross_return",
                "buy_cost",
                "sell_cost",
                "net_return",
                "benchmark_return",
                "excess_return",
                "portfolio_nav",
                "benchmark_nav",
                "turnover",
                "buy_turnover",
                "sell_turnover",
                "transaction_cost_bps",
                "active_size_exposure",
            ]
        )
    else:
        daily_portfolio = (
            sleeve_daily_df.groupby("date", as_index=False)
            .agg(
                gross_return=("gross_return", "sum"),
                buy_cost=("buy_cost", "sum"),
                sell_cost=("sell_cost", "sum"),
                net_return=("net_return", "sum"),
            )
        )
        daily_portfolio = daily_portfolio.merge(benchmark_daily_return, how="left", on="date")
        schedule_event_df = schedule_df.copy()
        if not schedule_event_df.empty:
            schedule_event_df["date"] = pd.to_datetime(schedule_event_df["execution_date"])
            daily_events = schedule_event_df.groupby("date", as_index=False).agg(
                turnover=("turnover", "sum"),
                buy_turnover=("buy_turnover", "sum"),
                sell_turnover=("sell_turnover", "sum"),
                transaction_cost_bps=("transaction_cost_bps", "sum"),
                active_size_exposure=("active_size_exposure", "mean"),
            )
        else:
            daily_events = pd.DataFrame(
                columns=["date", "turnover", "buy_turnover", "sell_turnover", "transaction_cost_bps", "active_size_exposure"]
            )
        period_df = daily_portfolio.merge(daily_events, how="left", on="date")
        period_df["benchmark_return"] = period_df["benchmark_return"].fillna(0.0)
        period_df["turnover"] = period_df["turnover"].fillna(0.0)
        period_df["buy_turnover"] = period_df["buy_turnover"].fillna(0.0)
        period_df["sell_turnover"] = period_df["sell_turnover"].fillna(0.0)
        period_df["transaction_cost_bps"] = period_df["transaction_cost_bps"].fillna(0.0)
        period_df["active_size_exposure"] = period_df["active_size_exposure"].ffill().fillna(0.0)
        period_df["excess_return"] = period_df["net_return"] - period_df["benchmark_return"]
        period_df["portfolio_nav"] = (1.0 + period_df["net_return"]).cumprod()
        period_df["benchmark_nav"] = (1.0 + period_df["benchmark_return"]).cumprod()
        period_df["date"] = pd.to_datetime(period_df["date"]).dt.strftime("%Y-%m-%d")

    if not period_df.empty:
        sector_summary_df = (
            sector_df.groupby("sector", as_index=False)
            .agg(
                average_portfolio_weight=("portfolio_weight", "mean"),
                average_universe_weight=("universe_weight", "mean"),
                average_active_weight=("active_weight", "mean"),
                average_abs_active_weight=("active_weight", lambda series: float(np.mean(np.abs(series)))),
            )
            .sort_values("average_abs_active_weight", ascending=False)
            .reset_index(drop=True)
        )
        average_max_abs_sector_active_weight = float(
            sector_df.groupby("signal_date")["active_weight"]
            .apply(lambda series: float(np.abs(series).max()))
            .mean()
        )
    else:
        sector_summary_df = pd.DataFrame()
        average_max_abs_sector_active_weight = float("nan")

    metrics = summarize_period_metrics(period_df=period_df, annualization_days=1)
    metrics.update(
        {
            "run_name": run_name,
            "hold_days": int(hold_days),
            "holding_clock": holding_clock,
            "effective_holding_days": int(effective_holding_days),
            "rebalance_step_days": int(step_days),
            "signal_delay_days": int(signal_delay_days),
            "max_active_sleeves": int(max_active_sleeves),
            "sleeve_capital_weight": float(sleeve_capital),
            "top_n": int(top_n),
            "weighting_scheme": weighting_scheme,
            "max_weight": float(max_weight) if max_weight is not None else None,
            "constraint_mode": constraint_mode,
            "sector_active_tolerance": (
                float(sector_active_tolerance) if sector_active_tolerance is not None else None
            ),
            "size_exposure_limit": float(size_exposure_limit) if size_exposure_limit is not None else None,
            "buy_cost_bps": float(buy_cost_bps),
            "sell_cost_bps": float(sell_cost_bps),
            "average_max_abs_sector_active_weight": average_max_abs_sector_active_weight,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    period_df.to_csv(output_dir / "period_summary.csv", index=False)
    schedule_df.to_csv(output_dir / "rebalance_schedule.csv", index=False)
    sleeve_daily_df.to_csv(output_dir / "sleeve_daily_returns.csv", index=False)
    weight_df.to_csv(output_dir / "portfolio_weights.csv", index=False)
    sector_df.to_csv(output_dir / "sector_active_exposure.csv", index=False)
    sector_summary_df.to_csv(output_dir / "sector_exposure_summary.csv", index=False)
    (output_dir / "portfolio_metrics.json").write_text(
        dumps_strict_json(metrics),
        encoding="utf-8",
    )
    write_portfolio_report(
        output_path=output_dir / "portfolio_report.md",
        run_name=run_name,
        predictions_path=predictions_path,
        data_path=data_path,
        hold_days=hold_days,
        step_days=step_days,
        signal_delay_days=signal_delay_days,
        holding_clock=holding_clock,
        top_n=top_n,
        weighting_scheme=weighting_scheme,
        max_weight=max_weight,
        constraint_mode=constraint_mode,
        sector_active_tolerance=sector_active_tolerance,
        size_exposure_limit=size_exposure_limit,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        metrics=metrics,
        period_df=period_df,
        sector_summary_df=sector_summary_df,
    )

    return {
        "metrics": metrics,
        "period_df": period_df,
        "weight_df": weight_df,
        "sector_df": sector_df,
        "sector_summary_df": sector_summary_df,
    }
