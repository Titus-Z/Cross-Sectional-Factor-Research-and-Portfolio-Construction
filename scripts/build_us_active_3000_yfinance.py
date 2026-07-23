"""Build a recoverable yfinance daily dataset for active US equities.

This script is intentionally independent from the training pipeline.  It solves
one data-engineering problem only:

1. fetch the official NASDAQ Trader symbol directories;
2. screen common-stock candidates by recent dollar volume;
3. download long daily OHLCV history in small Parquet batches.

The output can later be converted to CSV or wired into model training, but this
script does not start any model run.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_TARGET_COLUMN, add_forward_return_targets

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    raise ImportError("yfinance is required. Install requirements.txt first.") from exc

try:
    import pyarrow  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    raise ImportError("pyarrow is required for Parquet output. Install requirements.txt first.") from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

STANDARD_COLUMNS = [
    "instrument_id",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "adjustment",
    "next_open",
    "market_cap",
    "turnover",
    "sector",
    "y_1d",
    "y_5d",
    "y_10d",
    "y",
]

SECURITY_NAME_EXCLUDE_PATTERN = re.compile(
    r"\b(?:"
    r"warrant|warrants|right|rights|unit|units|preferred|preference|"
    r"note|notes|bond|debenture|etf|etn|fund|closed end|closed-end|"
    r"spac|acquisition corp\.?\s+unit|contingent value"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Estimate:
    candidate_count: int
    selected_count: int
    batch_count: int
    estimated_trading_days: int
    estimated_rows: int
    estimated_csv_gb: float
    estimated_parquet_gb_low: float
    estimated_parquet_gb_high: float
    disk_free_gb: float


def iter_with_progress(items: Iterable, description: str):
    """Return a tqdm iterator when available; otherwise return plain items."""

    if tqdm is None:
        return items
    return tqdm(items, desc=description)


def resolve_path(path: str | Path) -> Path:
    """Resolve project-relative paths without forcing users to cd perfectly."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def yahoo_symbol(symbol: str) -> str:
    """Convert NASDAQ directory symbols into Yahoo Finance ticker style."""

    return str(symbol).strip().replace(".", "-")


def fetch_symbol_directory(url: str) -> pd.DataFrame:
    """Download and parse one NASDAQ Trader pipe-delimited symbol file."""

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = [
        line
        for line in response.text.splitlines()
        if line.strip() and not line.startswith("File Creation Time")
    ]
    return pd.read_csv(StringIO("\n".join(lines)), sep="|", dtype=str)


def build_candidate_universe() -> pd.DataFrame:
    """Build a filtered common-stock candidate universe from NASDAQ Trader."""

    nasdaq = fetch_symbol_directory(NASDAQ_LISTED_URL)
    nasdaq = nasdaq.rename(
        columns={
            "Symbol": "raw_symbol",
            "Security Name": "security_name",
            "ETF": "etf",
            "Test Issue": "test_issue",
            "Market Category": "exchange",
        }
    )
    nasdaq["source"] = "nasdaqlisted"

    other = fetch_symbol_directory(OTHER_LISTED_URL)
    other = other.rename(
        columns={
            "ACT Symbol": "raw_symbol",
            "Security Name": "security_name",
            "ETF": "etf",
            "Test Issue": "test_issue",
            "Exchange": "exchange",
        }
    )
    other["source"] = "otherlisted"

    combined = pd.concat(
        [
            nasdaq[["raw_symbol", "security_name", "etf", "test_issue", "exchange", "source"]],
            other[["raw_symbol", "security_name", "etf", "test_issue", "exchange", "source"]],
        ],
        ignore_index=True,
    )
    combined = combined.dropna(subset=["raw_symbol"]).copy()
    combined["raw_symbol"] = combined["raw_symbol"].astype(str).str.strip()
    combined["yahoo_symbol"] = combined["raw_symbol"].map(yahoo_symbol)
    combined["security_name"] = combined["security_name"].fillna("")
    combined["etf"] = combined["etf"].fillna("N")
    combined["test_issue"] = combined["test_issue"].fillna("N")

    is_valid = (
        combined["raw_symbol"].ne("")
        & combined["yahoo_symbol"].str.match(r"^[A-Z0-9.-]+$", na=False)
        & combined["etf"].str.upper().eq("N")
        & combined["test_issue"].str.upper().eq("N")
        & ~combined["security_name"].str.contains(SECURITY_NAME_EXCLUDE_PATTERN, na=False)
    )
    filtered = combined.loc[is_valid].drop_duplicates("yahoo_symbol").copy()
    return filtered.sort_values("yahoo_symbol").reset_index(drop=True)


def chunked(items: list[str], size: int) -> list[list[str]]:
    """Split a list into deterministic chunks."""

    return [items[index : index + size] for index in range(0, len(items), size)]


def download_raw_history(
    symbols: list[str],
    start_date: str,
    end_date: str | None,
    auto_adjust: bool,
) -> pd.DataFrame:
    """Call yfinance once for a batch of tickers."""

    return yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        interval="1d",
        group_by="ticker",
        threads=True,
    )


def extract_symbol_history(raw_history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract one symbol's OHLCV frame from a yfinance batch result."""

    if raw_history is None or raw_history.empty:
        return pd.DataFrame()

    symbol_frame: pd.DataFrame
    if isinstance(raw_history.columns, pd.MultiIndex):
        level_zero = raw_history.columns.get_level_values(0)
        level_one = raw_history.columns.get_level_values(1)
        if symbol in level_zero:
            symbol_frame = raw_history[symbol].copy()
        elif symbol in level_one:
            symbol_frame = raw_history.xs(symbol, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        symbol_frame = raw_history.copy()

    if symbol_frame.empty:
        return pd.DataFrame()

    symbol_frame = symbol_frame.dropna(how="all").reset_index()
    symbol_frame.columns = [str(column).strip() for column in symbol_frame.columns]
    return symbol_frame


def normalize_symbol_history(symbol_frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize one ticker to the project's standard daily schema."""

    if symbol_frame.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    rename_map = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    normalized = symbol_frame.rename(columns=rename_map).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(normalized.columns):
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    normalized["instrument_id"] = symbol
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)

    numeric_columns = ["open", "high", "low", "close", "volume"]
    if "adj_close" in normalized.columns:
        numeric_columns.append("adj_close")
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"])
    normalized = normalized.sort_values("date").reset_index(drop=True)
    if normalized.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    normalized["vwap"] = (
        normalized["open"] + normalized["high"] + normalized["low"] + normalized["close"]
    ) / 4.0
    if "adj_close" in normalized.columns:
        raw_close = pd.to_numeric(normalized["close"], errors="coerce")
        normalized["adjustment"] = pd.to_numeric(
            normalized["adj_close"], errors="coerce"
        ) / raw_close.where(raw_close.ne(0.0), np.nan)
    else:
        normalized["adjustment"] = 1.0
    normalized["next_open"] = normalized["open"].shift(-1)
    normalized["market_cap"] = np.nan
    normalized["turnover"] = normalized["close"] * normalized["volume"]
    normalized["sector"] = "Unknown"

    # Forward labels must not be calculated on one ticker's private row clock.
    # A suspended or partially downloaded ticker could otherwise turn its fifth
    # available observation into a return longer than five market days. Labels
    # are added after all symbols in the batch are concatenated, using the batch's
    # shared market-date union. The canonical loader recomputes them again from
    # the complete merged file before model training.
    target_columns = ["y_1d", "y_5d", "y_10d", "y"]
    for target_column in target_columns:
        normalized[target_column] = np.nan
    return normalized[STANDARD_COLUMNS].copy()


def normalize_batch_history(raw_history: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Normalize all downloaded symbols in one batch."""

    frames = []
    for symbol in symbols:
        symbol_frame = extract_symbol_history(raw_history, symbol)
        normalized = normalize_symbol_history(symbol_frame, symbol)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    merged = add_forward_return_targets(merged, price_column="close")
    merged["y"] = merged[DEFAULT_TARGET_COLUMN]
    return merged[STANDARD_COLUMNS].copy()


def retry_download(
    symbols: list[str],
    start_date: str,
    end_date: str | None,
    auto_adjust: bool,
    retry: int,
    retry_sleep_sec: float,
) -> pd.DataFrame:
    """Download a batch with simple retry handling."""

    last_error: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            raw = download_raw_history(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                auto_adjust=auto_adjust,
            )
            return normalize_batch_history(raw, symbols)
        except Exception as exc:  # yfinance/network failures are intentionally isolated.
            last_error = exc
            if attempt < retry:
                time.sleep(retry_sleep_sec * attempt)
    raise RuntimeError(f"Failed to download batch after {retry} attempts: {last_error}")


def screen_liquidity(
    candidates: pd.DataFrame,
    lookback_days: int,
    min_valid_days: int,
    min_last_close: float,
    top_n: int,
    batch_size: int,
    retry: int,
    retry_sleep_sec: float,
    auto_adjust: bool,
    failed_records: list[dict],
) -> pd.DataFrame:
    """Rank candidates by recent average dollar volume."""

    symbols = candidates["yahoo_symbol"].tolist()
    start = (date.today() - timedelta(days=max(lookback_days * 2, lookback_days + 30))).isoformat()
    all_stats = []

    for batch_index, batch_symbols in enumerate(
        iter_with_progress(chunked(symbols, batch_size), "Screening liquidity"),
        start=1,
    ):
        try:
            history = retry_download(
                symbols=batch_symbols,
                start_date=start,
                end_date=None,
                auto_adjust=auto_adjust,
                retry=retry,
                retry_sleep_sec=retry_sleep_sec,
            )
        except Exception as exc:
            for symbol in batch_symbols:
                failed_records.append(
                    {
                        "stage": "liquidity",
                        "symbol": symbol,
                        "batch_index": batch_index,
                        "reason": str(exc),
                    }
                )
            continue

        downloaded_symbols = set(history["instrument_id"].dropna().unique()) if not history.empty else set()
        for missing_symbol in sorted(set(batch_symbols) - downloaded_symbols):
            failed_records.append(
                {
                    "stage": "liquidity",
                    "symbol": missing_symbol,
                    "batch_index": batch_index,
                    "reason": "missing_from_yfinance_response",
                }
            )

        if history.empty:
            continue

        for symbol, symbol_df in history.groupby("instrument_id", sort=False):
            valid = symbol_df.dropna(subset=["close", "volume"]).sort_values("date").tail(60).copy()
            if valid.empty:
                continue
            valid["dollar_volume"] = valid["close"] * valid["volume"]
            all_stats.append(
                {
                    "instrument_id": symbol,
                    "valid_days": int(len(valid)),
                    "last_date": valid["date"].max(),
                    "last_close": float(valid["close"].iloc[-1]),
                    "avg_dollar_volume_60d": float(valid["dollar_volume"].mean()),
                    "median_dollar_volume_60d": float(valid["dollar_volume"].median()),
                }
            )

    if not all_stats:
        raise ValueError("Liquidity screening downloaded no usable price data.")

    stats = pd.DataFrame(all_stats)
    enriched = stats.merge(candidates, left_on="instrument_id", right_on="yahoo_symbol", how="left")
    eligible = enriched[
        (enriched["valid_days"] >= min_valid_days)
        & (enriched["last_close"] >= min_last_close)
        & np.isfinite(enriched["avg_dollar_volume_60d"])
    ].copy()
    eligible = eligible.sort_values("avg_dollar_volume_60d", ascending=False).reset_index(drop=True)
    eligible["liquidity_rank"] = np.arange(1, len(eligible) + 1)
    return eligible.head(top_n).copy()


def estimate_download(
    candidate_count: int,
    selected_count: int,
    start_date: str,
    batch_size: int,
    output_root: Path,
) -> Estimate:
    """Estimate rows, size, and disk pressure before the full history run."""

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(date.today())
    estimated_trading_days = len(pd.bdate_range(start=start_ts, end=end_ts))
    estimated_rows = int(selected_count * estimated_trading_days)

    # Calibrated from the current MyQuant us_large_cap_300 CSV: about 265 bytes/row.
    estimated_csv_gb = estimated_rows * 265.0 / 1024**3
    estimated_parquet_gb_low = estimated_csv_gb * 0.30
    estimated_parquet_gb_high = estimated_csv_gb * 0.65
    disk_free_gb = shutil.disk_usage(output_root.parent if output_root.parent.exists() else PROJECT_ROOT).free / 1024**3
    batch_count = math.ceil(selected_count / batch_size)
    return Estimate(
        candidate_count=candidate_count,
        selected_count=selected_count,
        batch_count=batch_count,
        estimated_trading_days=estimated_trading_days,
        estimated_rows=estimated_rows,
        estimated_csv_gb=estimated_csv_gb,
        estimated_parquet_gb_low=estimated_parquet_gb_low,
        estimated_parquet_gb_high=estimated_parquet_gb_high,
        disk_free_gb=disk_free_gb,
    )


def print_estimate(estimate: Estimate) -> None:
    """Print a compact pre-run estimate."""

    print("[Estimate] Candidate symbols:", estimate.candidate_count)
    print("[Estimate] Selected symbols:", estimate.selected_count)
    print("[Estimate] Batches:", estimate.batch_count)
    print("[Estimate] Trading days:", estimate.estimated_trading_days)
    print("[Estimate] Rows:", f"{estimate.estimated_rows:,}")
    print("[Estimate] CSV size:", f"{estimate.estimated_csv_gb:.2f} GB")
    print(
        "[Estimate] Parquet size:",
        f"{estimate.estimated_parquet_gb_low:.2f}-{estimate.estimated_parquet_gb_high:.2f} GB",
    )
    print("[Estimate] Disk free:", f"{estimate.disk_free_gb:.2f} GB")


def save_failed_records(records: list[dict], failed_output: Path) -> None:
    """Write failed symbols even when the list is empty."""

    failed_output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["stage", "symbol", "batch_index", "reason"]
    pd.DataFrame(records, columns=columns).to_csv(failed_output, index=False)


def download_history_batches(
    symbols: list[str],
    output_root: Path,
    start_date: str,
    end_date: str | None,
    batch_size: int,
    retry: int,
    retry_sleep_sec: float,
    auto_adjust: bool,
    force: bool,
    failed_records: list[dict],
) -> pd.DataFrame:
    """Download selected symbols to one Parquet file per batch."""

    output_root.mkdir(parents=True, exist_ok=True)
    summary_records = []
    batches = chunked(symbols, batch_size)

    for batch_index, batch_symbols in enumerate(
        iter_with_progress(batches, "Downloading full history"),
        start=1,
    ):
        batch_path = output_root / f"batch_{batch_index:04d}.parquet"
        manifest_path = output_root / f"batch_{batch_index:04d}_symbols.txt"

        if batch_path.exists() and not force:
            summary_records.append(
                {
                    "batch_index": batch_index,
                    "path": str(batch_path),
                    "requested_symbols": len(batch_symbols),
                    "downloaded_symbols": np.nan,
                    "rows": np.nan,
                    "date_min": None,
                    "date_max": None,
                    "status": "skipped_existing",
                    "file_size_mb": batch_path.stat().st_size / 1024**2,
                }
            )
            continue

        try:
            batch_df = retry_download(
                symbols=batch_symbols,
                start_date=start_date,
                end_date=end_date,
                auto_adjust=auto_adjust,
                retry=retry,
                retry_sleep_sec=retry_sleep_sec,
            )
        except Exception as exc:
            for symbol in batch_symbols:
                failed_records.append(
                    {
                        "stage": "history",
                        "symbol": symbol,
                        "batch_index": batch_index,
                        "reason": str(exc),
                    }
                )
            summary_records.append(
                {
                    "batch_index": batch_index,
                    "path": str(batch_path),
                    "requested_symbols": len(batch_symbols),
                    "downloaded_symbols": 0,
                    "rows": 0,
                    "date_min": None,
                    "date_max": None,
                    "status": "failed",
                    "file_size_mb": 0.0,
                }
            )
            continue

        downloaded_symbols = set(batch_df["instrument_id"].dropna().unique()) if not batch_df.empty else set()
        for missing_symbol in sorted(set(batch_symbols) - downloaded_symbols):
            failed_records.append(
                {
                    "stage": "history",
                    "symbol": missing_symbol,
                    "batch_index": batch_index,
                    "reason": "missing_from_yfinance_response",
                }
            )

        if not batch_df.empty:
            batch_df.to_parquet(batch_path, index=False)
            manifest_path.write_text("\n".join(batch_symbols) + "\n", encoding="utf-8")

        summary_records.append(
            {
                "batch_index": batch_index,
                "path": str(batch_path),
                "requested_symbols": len(batch_symbols),
                "downloaded_symbols": len(downloaded_symbols),
                "rows": int(len(batch_df)),
                "date_min": batch_df["date"].min() if not batch_df.empty else None,
                "date_max": batch_df["date"].max() if not batch_df.empty else None,
                "status": "finished" if not batch_df.empty else "empty",
                "file_size_mb": batch_path.stat().st_size / 1024**2 if batch_path.exists() else 0.0,
            }
        )

    return pd.DataFrame(summary_records)


def merge_parquet_to_csv(output_root: Path, merged_csv: Path) -> None:
    """Merge Parquet batches into one CSV without holding all batches in memory."""

    parquet_files = sorted(output_root.glob("batch_*.parquet"))
    parquet_files = [path for path in parquet_files if not path.name.endswith("_symbols.parquet")]
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet batch files found under {output_root}")

    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    if merged_csv.exists():
        merged_csv.unlink()

    wrote_header = False
    for path in iter_with_progress(parquet_files, "Merging Parquet to CSV"):
        frame = pd.read_parquet(path)
        frame.to_csv(merged_csv, mode="a", index=False, header=not wrote_header)
        wrote_header = True
    print(f"[Info] Merged CSV written to: {merged_csv}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build active US equity yfinance Parquet dataset.")
    parser.add_argument("--start-date", type=str, default="2010-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=3000)
    parser.add_argument("--candidate-limit", type=int, default=None, help="Limit candidates for smoke tests only.")
    parser.add_argument("--liquidity-lookback-days", type=int, default=120)
    parser.add_argument("--liquidity-min-valid-days", type=int, default=40)
    parser.add_argument("--min-last-close", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=75)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--retry-sleep-sec", type=float, default=5.0)
    parser.add_argument("--output-root", type=str, default="data/yfinance/us_active_3000_daily_parquet")
    parser.add_argument("--universe-output", type=str, default="data/universe/us_active_3000_symbols.csv")
    parser.add_argument("--summary-output", type=str, default="data/yfinance/us_active_3000_download_summary.csv")
    parser.add_argument("--failed-output", type=str, default="data/yfinance/us_active_3000_failed_symbols.csv")
    parser.add_argument("--liquidity-output", type=str, default="data/universe/us_active_3000_liquidity_candidates.csv")
    parser.add_argument("--merged-csv", type=str, default="data/us_active_3000_daily.csv")
    parser.add_argument(
        "--auto-adjust",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Preserve raw OHLC and the Adj Close / Close audit factor by default; "
            "use --auto-adjust only for a separately labelled compatibility download."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing Parquet batches.")
    parser.add_argument("--estimate-only", action="store_true", help="Build candidate universe and print estimates only.")
    parser.add_argument("--merge-only", action="store_true", help="Merge existing Parquet batches into one CSV.")
    return parser.parse_args()


def main() -> None:
    """Run the active universe builder."""

    args = parse_args()
    output_root = resolve_path(args.output_root)
    universe_output = resolve_path(args.universe_output)
    summary_output = resolve_path(args.summary_output)
    failed_output = resolve_path(args.failed_output)
    liquidity_output = resolve_path(args.liquidity_output)
    merged_csv = resolve_path(args.merged_csv)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive.")
    if args.retry <= 0:
        raise ValueError("--retry must be positive.")

    if args.merge_only:
        merge_parquet_to_csv(output_root=output_root, merged_csv=merged_csv)
        return

    print("[Info] Fetching NASDAQ Trader symbol directories...")
    candidates = build_candidate_universe()
    if args.candidate_limit is not None:
        if args.candidate_limit <= 0:
            raise ValueError("--candidate-limit must be positive when provided.")
        candidates = candidates.head(args.candidate_limit).copy()
        print(f"[Info] Candidate universe limited for smoke test: {len(candidates)} symbols")
    selected_count_for_estimate = min(args.top_n, len(candidates))
    estimate = estimate_download(
        candidate_count=len(candidates),
        selected_count=selected_count_for_estimate,
        start_date=args.start_date,
        batch_size=args.batch_size,
        output_root=output_root,
    )
    print_estimate(estimate)

    if args.estimate_only:
        return

    failed_records: list[dict] = []
    print("[Info] Screening liquidity with recent yfinance data...")
    selected = screen_liquidity(
        candidates=candidates,
        lookback_days=args.liquidity_lookback_days,
        min_valid_days=args.liquidity_min_valid_days,
        min_last_close=args.min_last_close,
        top_n=args.top_n,
        batch_size=args.batch_size,
        retry=args.retry,
        retry_sleep_sec=args.retry_sleep_sec,
        auto_adjust=args.auto_adjust,
        failed_records=failed_records,
    )

    universe_output.parent.mkdir(parents=True, exist_ok=True)
    liquidity_output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(universe_output, index=False)
    selected.to_csv(liquidity_output, index=False)
    print(f"[Info] Selected active universe: {len(selected)} symbols -> {universe_output}")

    selected_symbols = selected["instrument_id"].tolist()
    print("[Info] Downloading full history to Parquet batches...")
    summary_df = download_history_batches(
        symbols=selected_symbols,
        output_root=output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        retry=args.retry,
        retry_sleep_sec=args.retry_sleep_sec,
        auto_adjust=args.auto_adjust,
        force=args.force,
        failed_records=failed_records,
    )

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output, index=False)
    save_failed_records(failed_records, failed_output)
    print(f"[Info] Download summary written to: {summary_output}")
    print(f"[Info] Failed symbols written to: {failed_output}")
    print("[Info] Done.")


if __name__ == "__main__":
    main()
