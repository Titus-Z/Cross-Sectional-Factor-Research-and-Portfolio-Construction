"""Market-data quality and corporate-action audit helpers.

The project relies on vendor-adjusted daily history, but an adjusted price flag is
not proof that every split, distribution, merger, or ticker change is correct.
This module creates a review table around every adjustment-factor change so a
release run can show whether a large raw-price jump was removed and whether a
large residual adjusted return remains.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_corporate_action_audit(
    data: pd.DataFrame,
    price_adjustment_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return event-level and summary evidence for vendor adjustment changes.

    Parameters
    ----------
    data:
        Loaded market panel after the project's price-policy transformation.
        The original vendor ``adjustment`` ratio must still be present.
    price_adjustment_mode:
        ``vendor_adjusted`` means OHLC in ``data`` has already been multiplied by
        ``adjustment``. ``raw`` means OHLC remains raw and adjusted close is
        reconstructed only for this diagnostic.

    Notes
    -----
    An adjustment change is a review candidate, not a definitive corporate-action
    classification. Yahoo history can be revised, and a licensed action master is
    still required for production-grade reconciliation.
    """

    output_columns = [
        "instrument_id",
        "date",
        "previous_adjustment",
        "adjustment",
        "adjustment_change_pct",
        "raw_close_reconstructed",
        "adjusted_close_reconstructed",
        "raw_daily_return",
        "adjusted_daily_return",
        "raw_jump_removed_by_adjustment",
        "large_adjustment_change_gt_20pct",
        "residual_adjusted_return_gt_20pct",
        "residual_adjusted_return_gt_50pct",
        "review_priority",
    ]
    required_columns = {"instrument_id", "date", "close", "adjustment"}
    if data.empty or not required_columns.issubset(data.columns):
        summary = {
            "audit_available": False,
            "reason": "required columns are missing or the market panel is empty",
            "adjustment_event_count": 0,
        }
        return pd.DataFrame(columns=output_columns), summary
    if price_adjustment_mode not in {"vendor_adjusted", "raw"}:
        raise ValueError(f"Unsupported price_adjustment_mode: {price_adjustment_mode}")

    frame = data[["instrument_id", "date", "close", "adjustment"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["adjustment"] = pd.to_numeric(frame["adjustment"], errors="coerce")
    frame["adjustment"] = frame["adjustment"].where(frame["adjustment"] > 0.0, np.nan)
    frame = frame.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    if price_adjustment_mode == "vendor_adjusted":
        frame["adjusted_close_reconstructed"] = frame["close"]
        frame["raw_close_reconstructed"] = frame["close"] / frame["adjustment"]
    else:
        frame["raw_close_reconstructed"] = frame["close"]
        frame["adjusted_close_reconstructed"] = frame["close"] * frame["adjustment"]

    grouped = frame.groupby("instrument_id", sort=False)
    frame["previous_adjustment"] = grouped["adjustment"].shift(1)
    frame["adjustment_change_pct"] = grouped["adjustment"].pct_change(fill_method=None)
    frame["raw_daily_return"] = grouped["raw_close_reconstructed"].pct_change(fill_method=None)
    frame["adjusted_daily_return"] = grouped["adjusted_close_reconstructed"].pct_change(fill_method=None)

    # Yahoo ratios can contain harmless floating-point noise around one.  Use
    # the same tolerance as the run manifest so the event file contains actual
    # factor moves rather than thousands of numerical artifacts.
    event_mask = frame["adjustment_change_pct"].abs().gt(1e-6)
    events = frame.loc[event_mask].copy()
    events["raw_jump_removed_by_adjustment"] = (
        events["raw_daily_return"].abs().gt(0.20)
        & events["adjusted_daily_return"].abs().le(0.20)
    )
    events["large_adjustment_change_gt_20pct"] = events["adjustment_change_pct"].abs().gt(0.20)
    events["residual_adjusted_return_gt_20pct"] = events["adjusted_daily_return"].abs().gt(0.20)
    events["residual_adjusted_return_gt_50pct"] = events["adjusted_daily_return"].abs().gt(0.50)
    events["review_priority"] = np.select(
        [
            events["residual_adjusted_return_gt_50pct"],
            events["residual_adjusted_return_gt_20pct"],
            events["large_adjustment_change_gt_20pct"],
        ],
        ["critical", "high", "medium"],
        default="low",
    )
    events = events[output_columns].sort_values(
        ["residual_adjusted_return_gt_50pct", "residual_adjusted_return_gt_20pct", "date"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    summary = {
        "audit_available": True,
        "price_adjustment_mode": price_adjustment_mode,
        "adjustment_event_count": int(len(events)),
        "instrument_count_with_adjustment_events": int(events["instrument_id"].nunique()),
        "large_adjustment_change_gt_20pct_count": int(events["large_adjustment_change_gt_20pct"].sum()),
        "raw_jump_removed_by_adjustment_count": int(events["raw_jump_removed_by_adjustment"].sum()),
        "residual_adjusted_return_gt_20pct_count": int(events["residual_adjusted_return_gt_20pct"].sum()),
        "residual_adjusted_return_gt_50pct_count": int(events["residual_adjusted_return_gt_50pct"].sum()),
        "max_abs_adjusted_return_on_event": (
            float(events["adjusted_daily_return"].abs().max())
            if events["adjusted_daily_return"].notna().any()
            else None
        ),
        "manual_review_required": bool(events["residual_adjusted_return_gt_20pct"].any()),
        "limitation": (
            "Adjustment-factor changes are vendor review candidates, not independently verified "
            "corporate-action labels. No-event data may already be adjusted and cannot prove completeness."
        ),
    }
    return events, summary


def build_universe_coverage_audit(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize each instrument's observable history and flag incomplete coverage.

    The flags identify review candidates only. A late first date may be an IPO or
    missing vendor history; an early last date may be a delisting, merger, ticker
    change, or download failure. Daily OHLCV alone cannot distinguish these cases.
    """

    output_columns = [
        "instrument_id",
        "first_date",
        "last_date",
        "observation_count",
        "global_trading_date_count",
        "coverage_ratio",
        "trading_dates_before_first_observation",
        "trading_dates_after_last_observation",
        "late_start_flag",
        "early_end_flag",
        "fewer_than_252_observations_flag",
        "coverage_below_80pct_flag",
        "review_reason",
    ]
    required_columns = {"instrument_id", "date"}
    if data.empty or not required_columns.issubset(data.columns):
        return pd.DataFrame(columns=output_columns), {
            "audit_available": False,
            "reason": "instrument_id/date are missing or the market panel is empty",
        }

    frame = data[["instrument_id", "date"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["instrument_id", "date"]).drop_duplicates()
    global_dates = pd.Index(sorted(frame["date"].unique()))
    date_position = {pd.Timestamp(date): index for index, date in enumerate(global_dates)}
    global_date_count = int(len(global_dates))

    rows: list[dict[str, Any]] = []
    for instrument_id, instrument_frame in frame.groupby("instrument_id", sort=True):
        first_date = pd.Timestamp(instrument_frame["date"].min())
        last_date = pd.Timestamp(instrument_frame["date"].max())
        observation_count = int(instrument_frame["date"].nunique())
        before_count = int(date_position[first_date])
        after_count = int(global_date_count - date_position[last_date] - 1)
        coverage_ratio = observation_count / global_date_count if global_date_count else float("nan")
        late_start = before_count >= 20
        early_end = after_count >= 20
        short_history = observation_count < 252
        low_coverage = bool(pd.notna(coverage_ratio) and coverage_ratio < 0.80)
        reasons = [
            label
            for flag, label in [
                (late_start, "late_start_possible_ipo_or_missing_history"),
                (early_end, "early_end_possible_delisting_merger_or_data_gap"),
                (short_history, "fewer_than_252_observations"),
                (low_coverage, "coverage_below_80pct"),
            ]
            if flag
        ]
        rows.append(
            {
                "instrument_id": instrument_id,
                "first_date": str(first_date.date()),
                "last_date": str(last_date.date()),
                "observation_count": observation_count,
                "global_trading_date_count": global_date_count,
                "coverage_ratio": coverage_ratio,
                "trading_dates_before_first_observation": before_count,
                "trading_dates_after_last_observation": after_count,
                "late_start_flag": late_start,
                "early_end_flag": early_end,
                "fewer_than_252_observations_flag": short_history,
                "coverage_below_80pct_flag": low_coverage,
                "review_reason": ";".join(reasons),
            }
        )

    audit_df = pd.DataFrame(rows, columns=output_columns)
    summary = {
        "audit_available": True,
        "instrument_count": int(len(audit_df)),
        "global_trading_date_count": global_date_count,
        "late_start_instrument_count": int(audit_df["late_start_flag"].sum()),
        "early_end_instrument_count": int(audit_df["early_end_flag"].sum()),
        "fewer_than_252_observations_count": int(audit_df["fewer_than_252_observations_flag"].sum()),
        "coverage_below_80pct_count": int(audit_df["coverage_below_80pct_flag"].sum()),
        "median_coverage_ratio": float(audit_df["coverage_ratio"].median()),
        "limitation": (
            "Coverage flags cannot identify authoritative IPO, delisting, merger, or ticker-change events "
            "without a point-in-time security master."
        ),
    }
    return audit_df, summary
