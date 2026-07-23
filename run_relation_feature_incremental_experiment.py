"""Run relation-feature incremental experiment for MyQuant.

This script answers one narrow question:

    Does baseline + relation-derived features improve MyQuant baseline OOS ranking?

It intentionally stays separate from the main training pipeline. The relation
edge table is an upstream artifact from the stock relation project; this script
turns it into model-ready features, trains the same lightweight linear models
with and without those features, and writes paired OOS evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.mined_factor_model_ablation import (  # noqa: E402
    evaluate_prediction_frame,
    get_numeric_feature_columns,
    train_and_predict,
)
from factor_mining_workspace.single_factor_case_study import (  # noqa: E402
    dataframe_to_markdown,
    load_or_build_preprocessed_train_test,
)
from src.project_paths import resolve_project_path  # noqa: E402
from src.reporting import safe_corr  # noqa: E402
from src.runtime_config import (  # noqa: E402
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)


DEFAULT_OUTPUT_DIR = "outputs/relation_feature_incremental_experiment"
DEFAULT_OOS_START_DATE = "2025-06-01"
DEFAULT_SECONDARY_START_DATE = "2026-01-01"
DEFAULT_UNIVERSE_PATH = "data/universe/us_active_3000_liquidity_candidates.csv"

RELATION_FEATURE_COLUMNS = [
    "weighted_peer_return_1d",
    "weighted_peer_return_5d",
    "weighted_peer_return_20d",
    "lead_lag_peer_return_1d",
    "relation_dispersion",
    "source_influence_centrality",
    "target_neighbor_centrality",
    "same_sector_peer_strength",
    "top1_peer_return",
    "topk_peer_return_std",
]

TYPED_STABLE_RELATION_FEATURE_COLUMNS = [
    "stable_weighted_peer_return_1d",
    "stable_weighted_peer_return_5d",
    "stable_weighted_peer_return_20d",
    "same_sector_weighted_peer_return_1d",
    "same_sector_weighted_peer_return_5d",
    "stable_edge_strength",
    "same_sector_edge_strength",
    "stable_relation_dispersion",
]

QUALITY_RELATION_FEATURE_COLUMNS = [
    "quality_weighted_peer_return_1d",
    "quality_weighted_peer_return_5d",
    "quality_weighted_peer_return_20d",
    "quality_edge_strength",
    "quality_relation_dispersion",
    "mean_edge_quality_score",
    "directed_lag_weighted_peer_return_1d",
    "directed_lag_weighted_peer_return_5d",
    "directed_lag_edge_strength",
    "mean_directed_lag_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether stock relation graph features improve MyQuant baseline "
            "OOS Rank IC and long-short diagnostics."
        )
    )
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="MyQuant us300 daily CSV.")
    parser.add_argument(
        "--edge-path",
        required=True,
        help="Daily relation edge table CSV produced by the external stock-relation project.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory under MyQuant.")
    parser.add_argument("--cache-dir", default=".cache", help="MyQuant preprocessing cache directory.")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="Sample start date.")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="Primary OOS start date.")
    parser.add_argument(
        "--secondary-start-date",
        default=DEFAULT_SECONDARY_START_DATE,
        help="Secondary OOS subperiod start date, evaluated on the same trained model.",
    )
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--models", nargs="+", default=["ridge", "elastic_net"])
    parser.add_argument("--top-frac", type=float, default=0.10, help="Top/bottom fraction for rank turnover proxy.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--universe-path", default="", help="Optional liquidity universe CSV for filtering a larger data panel.")
    parser.add_argument("--universe-top-n", type=int, default=0, help="Use top N instruments from universe-path before training.")
    parser.add_argument("--dynamic-lag-selection-path", default="", help="Pair lag selection CSV from stock relation project.")
    parser.add_argument("--max-dynamic-relation-factors", type=int, default=30)
    parser.add_argument("--stable-edge-threshold", type=float, default=0.50)
    parser.add_argument("--quality-edge-threshold", type=float, default=0.50)
    parser.add_argument(
        "--no-alpha191",
        action="store_true",
        help="Use lightweight technical baseline only. Default keeps full Alpha191 baseline.",
    )
    parser.add_argument(
        "--force-rebuild-relation-features",
        action="store_true",
        help="Rebuild relation feature panel even if a cached CSV exists in the output directory.",
    )
    parser.add_argument(
        "--disable-preprocessing-cache",
        action="store_true",
        help="Forwarded to MyQuant cache loader. Not recommended for normal runs.",
    )
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_top_universe_symbols(universe_path: Path, top_n: int) -> list[str]:
    universe = pd.read_csv(universe_path)
    if "instrument_id" not in universe.columns:
        raise ValueError(f"Universe file is missing instrument_id: {universe_path}")
    if "liquidity_rank" in universe.columns:
        universe = universe.sort_values("liquidity_rank", ascending=True)
    symbols = universe["instrument_id"].dropna().astype(str).head(top_n).tolist()
    if not symbols:
        raise ValueError("Universe filter produced no symbols.")
    return symbols


def prepare_filtered_data_path(args: argparse.Namespace, output_dir: Path) -> Path:
    data_path = resolve_path(args.data_path)
    if int(args.universe_top_n) <= 0:
        return data_path
    universe_path = resolve_path(args.universe_path or DEFAULT_UNIVERSE_PATH)
    symbols = read_top_universe_symbols(universe_path, int(args.universe_top_n))
    output_path = output_dir / f"us{int(args.universe_top_n)}_daily_panel.csv"
    if output_path.exists():
        return output_path

    usecols = [
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
    allowed = set(symbols)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=500_000):
        chunk = chunk[chunk["instrument_id"].astype(str).isin(allowed)].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"])
        if args.sample_start_date:
            chunk = chunk[chunk["date"] >= pd.Timestamp(args.sample_start_date)].copy()
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No rows remain after universe filtering.")
    filtered = pd.concat(chunks, ignore_index=True).sort_values(["instrument_id", "date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    return output_path


def load_market_returns(data_path: Path) -> pd.DataFrame:
    usecols = ["instrument_id", "date", "close"]
    market_df = pd.read_csv(data_path, usecols=usecols)
    market_df["date"] = pd.to_datetime(market_df["date"])
    market_df = market_df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    close = pd.to_numeric(market_df["close"], errors="coerce")
    by_symbol = market_df.groupby("instrument_id", group_keys=False)
    market_df["ret_1d"] = by_symbol["close"].pct_change(1, fill_method=None)
    market_df["ret_5d"] = by_symbol["close"].pct_change(5, fill_method=None)
    market_df["ret_20d"] = by_symbol["close"].pct_change(20, fill_method=None)
    market_df["ret_1d_lag1"] = by_symbol["ret_1d"].shift(1)
    return market_df[["date", "instrument_id", "ret_1d", "ret_5d", "ret_20d", "ret_1d_lag1"]].copy()


def weighted_std_from_sums(sum_w: pd.Series, sum_wx: pd.Series, sum_wx2: pd.Series) -> pd.Series:
    mean = sum_wx / sum_w.replace(0.0, np.nan)
    variance = (sum_wx2 / sum_w.replace(0.0, np.nan)) - np.square(mean)
    return np.sqrt(variance.clip(lower=0.0))


def build_relation_feature_panel(
    edge_path: Path,
    data_path: Path,
    output_path: Path,
    stable_edge_threshold: float = 0.50,
    quality_edge_threshold: float = 0.50,
) -> pd.DataFrame:
    start = time.perf_counter()
    if not edge_path.exists():
        raise FileNotFoundError(f"Relation edge table not found: {edge_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Market data not found: {data_path}")

    returns = load_market_returns(data_path)
    source_returns = returns.rename(columns={"instrument_id": "source"})
    edge_columns = [
        "date",
        "source",
        "target",
        "weight",
        "rank",
        "same_sector",
        "source_influence_centrality",
        "target_neighbor_centrality",
    ]
    available_edge_columns = set(pd.read_csv(edge_path, nrows=0).columns)
    optional_edge_columns = [
        column
        for column in [
            "edge_stability_score",
            "typed_edge",
            "directed_lag_score",
            "directed_lag_factor_count",
            "edge_quality_score",
            "edge_quality_bucket",
        ]
        if column in available_edge_columns
    ]
    edges = pd.read_csv(edge_path, usecols=edge_columns + optional_edge_columns)
    edges["date"] = pd.to_datetime(edges["date"])
    for column in ["weight", "rank", "source_influence_centrality", "target_neighbor_centrality"]:
        edges[column] = pd.to_numeric(edges[column], errors="coerce")
    if edges["same_sector"].dtype != bool:
        edges["same_sector"] = edges["same_sector"].astype(str).str.lower().isin({"true", "1", "yes"})
    if "edge_stability_score" not in edges.columns:
        edges["edge_stability_score"] = 0.0
    edges["edge_stability_score"] = pd.to_numeric(edges["edge_stability_score"], errors="coerce").fillna(0.0)
    if "typed_edge" not in edges.columns:
        edges["typed_edge"] = np.where(edges["same_sector"], "same_sector_corr_edge", "pure_corr_edge")
    for column in ["directed_lag_score", "directed_lag_factor_count", "edge_quality_score"]:
        if column not in edges.columns:
            edges[column] = 0.0
        edges[column] = pd.to_numeric(edges[column], errors="coerce").fillna(0.0)
    if "edge_quality_bucket" not in edges.columns:
        edges["edge_quality_bucket"] = "missing_quality_bucket"
    edges["is_stable_edge"] = (
        edges["typed_edge"].astype(str).eq("stable_corr_edge")
        | (edges["edge_stability_score"] >= float(stable_edge_threshold))
    )
    edges["is_directed_lag_edge"] = (
        edges["typed_edge"].astype(str).eq("directed_lag_edge")
        | (edges["directed_lag_factor_count"] > 0)
    )
    edges["is_quality_edge"] = (
        edges["is_directed_lag_edge"]
        | edges["is_stable_edge"]
        | edges["same_sector"]
        | (edges["edge_quality_score"] >= float(quality_edge_threshold))
    )

    merged = edges.merge(source_returns, on=["date", "source"], how="left")
    merged["valid_weight_1d"] = np.where(merged["ret_1d"].notna(), merged["weight"], 0.0)
    merged["valid_weight_5d"] = np.where(merged["ret_5d"].notna(), merged["weight"], 0.0)
    merged["valid_weight_20d"] = np.where(merged["ret_20d"].notna(), merged["weight"], 0.0)
    merged["valid_weight_lag1"] = np.where(merged["ret_1d_lag1"].notna(), merged["weight"], 0.0)
    merged["w_ret_1d"] = merged["weight"] * merged["ret_1d"].fillna(0.0)
    merged["w_ret_5d"] = merged["weight"] * merged["ret_5d"].fillna(0.0)
    merged["w_ret_20d"] = merged["weight"] * merged["ret_20d"].fillna(0.0)
    merged["w_ret_1d_lag1"] = merged["weight"] * merged["ret_1d_lag1"].fillna(0.0)
    merged["w_ret_1d_sq"] = merged["weight"] * np.square(merged["ret_1d"].fillna(0.0))
    merged["w_source_centrality"] = merged["weight"] * merged["source_influence_centrality"].fillna(0.0)
    merged["same_sector_weight"] = np.where(merged["same_sector"], merged["weight"], 0.0)
    merged["stable_weight"] = np.where(merged["is_stable_edge"], merged["weight"], 0.0)
    merged["quality_weight"] = np.where(merged["is_quality_edge"], merged["weight"], 0.0)
    merged["directed_lag_weight"] = np.where(merged["is_directed_lag_edge"], merged["weight"], 0.0)
    merged["stable_w_ret_1d"] = merged["stable_weight"] * merged["ret_1d"].fillna(0.0)
    merged["stable_w_ret_5d"] = merged["stable_weight"] * merged["ret_5d"].fillna(0.0)
    merged["stable_w_ret_20d"] = merged["stable_weight"] * merged["ret_20d"].fillna(0.0)
    merged["stable_w_ret_1d_sq"] = merged["stable_weight"] * np.square(merged["ret_1d"].fillna(0.0))
    merged["same_sector_w_ret_1d"] = merged["same_sector_weight"] * merged["ret_1d"].fillna(0.0)
    merged["same_sector_w_ret_5d"] = merged["same_sector_weight"] * merged["ret_5d"].fillna(0.0)
    merged["quality_w_ret_1d"] = merged["quality_weight"] * merged["ret_1d"].fillna(0.0)
    merged["quality_w_ret_5d"] = merged["quality_weight"] * merged["ret_5d"].fillna(0.0)
    merged["quality_w_ret_20d"] = merged["quality_weight"] * merged["ret_20d"].fillna(0.0)
    merged["quality_w_ret_1d_sq"] = merged["quality_weight"] * np.square(merged["ret_1d"].fillna(0.0))
    merged["directed_lag_w_ret_1d"] = merged["directed_lag_weight"] * merged["ret_1d"].fillna(0.0)
    merged["directed_lag_w_ret_5d"] = merged["directed_lag_weight"] * merged["ret_5d"].fillna(0.0)
    merged["w_edge_quality_score"] = merged["weight"] * merged["edge_quality_score"].fillna(0.0)
    merged["w_directed_lag_score"] = merged["directed_lag_weight"] * merged["directed_lag_score"].fillna(0.0)

    group_keys = ["date", "target"]
    grouped = merged.groupby(group_keys, sort=True)
    sums = grouped[
        [
            "valid_weight_1d",
            "valid_weight_5d",
            "valid_weight_20d",
            "valid_weight_lag1",
            "w_ret_1d",
            "w_ret_5d",
            "w_ret_20d",
            "w_ret_1d_lag1",
            "w_ret_1d_sq",
            "w_source_centrality",
            "weight",
            "same_sector_weight",
            "stable_weight",
            "quality_weight",
            "directed_lag_weight",
            "stable_w_ret_1d",
            "stable_w_ret_5d",
            "stable_w_ret_20d",
            "stable_w_ret_1d_sq",
            "same_sector_w_ret_1d",
            "same_sector_w_ret_5d",
            "quality_w_ret_1d",
            "quality_w_ret_5d",
            "quality_w_ret_20d",
            "quality_w_ret_1d_sq",
            "directed_lag_w_ret_1d",
            "directed_lag_w_ret_5d",
            "w_edge_quality_score",
            "w_directed_lag_score",
            "target_neighbor_centrality",
        ]
    ].agg(
        {
            "valid_weight_1d": "sum",
            "valid_weight_5d": "sum",
            "valid_weight_20d": "sum",
            "valid_weight_lag1": "sum",
            "w_ret_1d": "sum",
            "w_ret_5d": "sum",
            "w_ret_20d": "sum",
            "w_ret_1d_lag1": "sum",
            "w_ret_1d_sq": "sum",
            "w_source_centrality": "sum",
            "weight": "sum",
            "same_sector_weight": "sum",
            "stable_weight": "sum",
            "quality_weight": "sum",
            "directed_lag_weight": "sum",
            "stable_w_ret_1d": "sum",
            "stable_w_ret_5d": "sum",
            "stable_w_ret_20d": "sum",
            "stable_w_ret_1d_sq": "sum",
            "same_sector_w_ret_1d": "sum",
            "same_sector_w_ret_5d": "sum",
            "quality_w_ret_1d": "sum",
            "quality_w_ret_5d": "sum",
            "quality_w_ret_20d": "sum",
            "quality_w_ret_1d_sq": "sum",
            "directed_lag_w_ret_1d": "sum",
            "directed_lag_w_ret_5d": "sum",
            "w_edge_quality_score": "sum",
            "w_directed_lag_score": "sum",
            "target_neighbor_centrality": "mean",
        }
    )
    panel = sums.reset_index()
    panel["weighted_peer_return_1d"] = panel["w_ret_1d"] / panel["valid_weight_1d"].replace(0.0, np.nan)
    panel["weighted_peer_return_5d"] = panel["w_ret_5d"] / panel["valid_weight_5d"].replace(0.0, np.nan)
    panel["weighted_peer_return_20d"] = panel["w_ret_20d"] / panel["valid_weight_20d"].replace(0.0, np.nan)
    panel["lead_lag_peer_return_1d"] = panel["w_ret_1d_lag1"] / panel["valid_weight_lag1"].replace(0.0, np.nan)
    panel["relation_dispersion"] = weighted_std_from_sums(
        panel["valid_weight_1d"],
        panel["w_ret_1d"],
        panel["w_ret_1d_sq"],
    )
    panel["source_influence_centrality"] = panel["w_source_centrality"] / panel["weight"].replace(0.0, np.nan)
    panel["same_sector_peer_strength"] = panel["same_sector_weight"]
    panel["stable_weighted_peer_return_1d"] = panel["stable_w_ret_1d"] / panel["stable_weight"].replace(0.0, np.nan)
    panel["stable_weighted_peer_return_5d"] = panel["stable_w_ret_5d"] / panel["stable_weight"].replace(0.0, np.nan)
    panel["stable_weighted_peer_return_20d"] = panel["stable_w_ret_20d"] / panel["stable_weight"].replace(0.0, np.nan)
    panel["same_sector_weighted_peer_return_1d"] = panel["same_sector_w_ret_1d"] / panel["same_sector_weight"].replace(0.0, np.nan)
    panel["same_sector_weighted_peer_return_5d"] = panel["same_sector_w_ret_5d"] / panel["same_sector_weight"].replace(0.0, np.nan)
    panel["stable_edge_strength"] = panel["stable_weight"]
    panel["same_sector_edge_strength"] = panel["same_sector_weight"]
    panel["stable_relation_dispersion"] = weighted_std_from_sums(
        panel["stable_weight"],
        panel["stable_w_ret_1d"],
        panel["stable_w_ret_1d_sq"],
    )
    panel["quality_weighted_peer_return_1d"] = panel["quality_w_ret_1d"] / panel["quality_weight"].replace(0.0, np.nan)
    panel["quality_weighted_peer_return_5d"] = panel["quality_w_ret_5d"] / panel["quality_weight"].replace(0.0, np.nan)
    panel["quality_weighted_peer_return_20d"] = panel["quality_w_ret_20d"] / panel["quality_weight"].replace(0.0, np.nan)
    panel["quality_edge_strength"] = panel["quality_weight"]
    panel["quality_relation_dispersion"] = weighted_std_from_sums(
        panel["quality_weight"],
        panel["quality_w_ret_1d"],
        panel["quality_w_ret_1d_sq"],
    )
    panel["mean_edge_quality_score"] = panel["w_edge_quality_score"] / panel["weight"].replace(0.0, np.nan)
    panel["directed_lag_weighted_peer_return_1d"] = panel["directed_lag_w_ret_1d"] / panel["directed_lag_weight"].replace(0.0, np.nan)
    panel["directed_lag_weighted_peer_return_5d"] = panel["directed_lag_w_ret_5d"] / panel["directed_lag_weight"].replace(0.0, np.nan)
    panel["directed_lag_edge_strength"] = panel["directed_lag_weight"]
    panel["mean_directed_lag_score"] = panel["w_directed_lag_score"] / panel["directed_lag_weight"].replace(0.0, np.nan)

    top1 = (
        merged.loc[merged["rank"] == 1, ["date", "target", "ret_1d"]]
        .rename(columns={"ret_1d": "top1_peer_return"})
        .drop_duplicates(subset=["date", "target"], keep="first")
    )
    topk_std = (
        merged.groupby(group_keys)["ret_1d"]
        .std()
        .reset_index()
        .rename(columns={"ret_1d": "topk_peer_return_std"})
    )
    panel = panel.merge(top1, on=group_keys, how="left").merge(topk_std, on=group_keys, how="left")
    panel = panel.rename(columns={"target": "instrument_id"})
    panel = panel[
        ["date", "instrument_id", *RELATION_FEATURE_COLUMNS, *TYPED_STABLE_RELATION_FEATURE_COLUMNS, *QUALITY_RELATION_FEATURE_COLUMNS]
    ].copy()
    panel = panel.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    metadata = {
        "edge_path": str(edge_path),
        "data_path": str(data_path),
        "rows": int(len(panel)),
        "dates": int(panel["date"].nunique()),
        "instruments": int(panel["instrument_id"].nunique()),
        "min_date": str(panel["date"].min().date()) if not panel.empty else "",
        "max_date": str(panel["date"].max().date()) if not panel.empty else "",
        "feature_columns": RELATION_FEATURE_COLUMNS,
        "typed_stable_feature_columns": TYPED_STABLE_RELATION_FEATURE_COLUMNS,
        "quality_feature_columns": QUALITY_RELATION_FEATURE_COLUMNS,
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "leakage_boundary": (
            "Edge weights are rolling correlations estimated before the edge date; "
            "peer returns use same-date close-to-close returns except lead_lag_peer_return_1d, "
            "which uses previous-day peer return."
        ),
    }
    (output_path.parent / "relation_feature_panel_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return panel


def load_or_build_relation_feature_panel(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    feature_path = output_dir / "relation_features_panel.csv"
    if feature_path.exists() and not args.force_rebuild_relation_features:
        panel = pd.read_csv(feature_path)
        panel["date"] = pd.to_datetime(panel["date"])
        return panel
    return build_relation_feature_panel(
        edge_path=resolve_path(args.edge_path),
        data_path=resolve_path(args.data_path),
        output_path=feature_path,
        stable_edge_threshold=float(getattr(args, "stable_edge_threshold", 0.50)),
        quality_edge_threshold=float(getattr(args, "quality_edge_threshold", 0.50)),
    )


def cross_sectional_winsor_zscore(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = data.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

        def transform(group: pd.Series) -> pd.Series:
            valid = group.dropna()
            if len(valid) < 5:
                return group
            lower = valid.quantile(0.01)
            upper = valid.quantile(0.99)
            clipped = group.clip(lower=lower, upper=upper)
            std = clipped.std(ddof=0)
            if not np.isfinite(std) or std <= 0:
                return clipped - clipped.mean()
            return (clipped - clipped.mean()) / std

        result[column] = values.groupby(result["date"]).transform(transform)
    return result


def attach_relation_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    relation_panel: pd.DataFrame,
    relation_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    relation = relation_panel.copy()
    relation["date"] = pd.to_datetime(relation["date"])
    available_columns = [column for column in relation_columns if column in relation.columns]
    relation = cross_sectional_winsor_zscore(relation, available_columns)
    relation = relation[["date", "instrument_id", *available_columns]].copy()

    train_enriched = train_df.merge(relation, on=["date", "instrument_id"], how="left")
    test_enriched = test_df.merge(relation, on=["date", "instrument_id"], how="left")
    coverage = {}
    for name, frame in [("train", train_enriched), ("test", test_enriched)]:
        non_null = frame[available_columns].notna().any(axis=1) if available_columns else pd.Series(False, index=frame.index)
        coverage[f"{name}_rows"] = int(len(frame))
        coverage[f"{name}_relation_any_non_null_rows"] = int(non_null.sum())
        coverage[f"{name}_relation_any_non_null_ratio"] = float(non_null.mean()) if len(frame) else float("nan")
        coverage[f"{name}_min_date"] = str(pd.to_datetime(frame["date"]).min().date()) if not frame.empty else ""
        coverage[f"{name}_max_date"] = str(pd.to_datetime(frame["date"]).max().date()) if not frame.empty else ""
        coverage[f"{name}_relation_column_count"] = int(len(available_columns))
    return train_enriched, test_enriched, coverage


def load_dynamic_lag_selection(
    selection_path: Path,
    available_feature_columns: list[str],
    max_factors: int,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if not selection_path.exists():
        return pd.DataFrame(), [], {"dynamic_lag_selection_status": "missing"}
    selection = pd.read_csv(selection_path)
    required = {"source", "target", "factor", "best_lag", "selection_status"}
    missing = required - set(selection.columns)
    if missing:
        raise ValueError(f"Dynamic lag selection file missing columns: {sorted(missing)}")
    selected = selection[selection["selection_status"].astype(str).eq("selected_directed_lag_edge")].copy()
    selected["factor"] = selected["factor"].astype(str)
    available = set(available_feature_columns)
    selected = selected[selected["factor"].isin(available)].copy()
    if selected.empty:
        return selected, [], {
            "dynamic_lag_selection_status": "no_selected_rows_with_available_features",
            "selection_path": str(selection_path),
        }
    rank_col = "primary_oos_rank_ic" if "primary_oos_rank_ic" in selected.columns else "train_rank_ic"
    factor_rank = (
        selected.assign(abs_rank=selected[rank_col].abs())
        .groupby("factor", as_index=False)
        .agg(selected_pair_count=("factor", "size"), mean_abs_rank=("abs_rank", "mean"))
        .sort_values(["selected_pair_count", "mean_abs_rank"], ascending=False)
    )
    kept_factors = factor_rank["factor"].head(max_factors).astype(str).tolist()
    selected = selected[selected["factor"].isin(kept_factors)].copy()
    selected["best_lag"] = pd.to_numeric(selected["best_lag"], errors="coerce").astype("Int64")
    selected = selected.dropna(subset=["best_lag"]).copy()
    selected["best_lag"] = selected["best_lag"].astype(int)
    return selected, kept_factors, {
        "dynamic_lag_selection_status": "loaded",
        "selection_path": str(selection_path),
        "selected_rows": int(len(selected)),
        "selected_factor_count": int(len(kept_factors)),
        "selected_factors": kept_factors,
    }


def load_selected_edges(edge_path: Path, selected_pairs: pd.DataFrame) -> pd.DataFrame:
    pair_keys = selected_pairs[["source", "target"]].drop_duplicates().copy()
    if pair_keys.empty:
        return pd.DataFrame()
    chunks: list[pd.DataFrame] = []
    usecols = [column for column in ["date", "source", "target", "weight"] if column in pd.read_csv(edge_path, nrows=0).columns]
    for chunk in pd.read_csv(edge_path, usecols=usecols, chunksize=500_000):
        merged = chunk.merge(pair_keys, on=["source", "target"], how="inner")
        if not merged.empty:
            chunks.append(merged)
    if not chunks:
        return pd.DataFrame(columns=["date", "source", "target", "weight"])
    edges = pd.concat(chunks, ignore_index=True)
    edges["date"] = pd.to_datetime(edges["date"])
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    return edges


def build_dynamic_lag_relation_panel(
    edge_path: Path,
    feature_frame: pd.DataFrame,
    lag_selection: pd.DataFrame,
    selected_factors: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if lag_selection.empty or not selected_factors:
        return pd.DataFrame(columns=["date", "instrument_id"]), [], {"dynamic_lag_feature_status": "empty_selection"}
    selected_edges = load_selected_edges(edge_path, lag_selection[["source", "target"]])
    if selected_edges.empty:
        return pd.DataFrame(columns=["date", "instrument_id"]), [], {"dynamic_lag_feature_status": "no_matching_edges"}

    feature_frame = feature_frame[["date", "instrument_id", *selected_factors]].copy()
    feature_frame["date"] = pd.to_datetime(feature_frame["date"])
    feature_frame = feature_frame.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    relation_panels: list[pd.DataFrame] = []
    dynamic_columns: list[str] = []
    selected_edges = selected_edges.merge(
        lag_selection[["source", "target", "factor", "best_lag"]],
        on=["source", "target"],
        how="inner",
    )
    for (factor, lag), group_edges in selected_edges.groupby(["factor", "best_lag"], sort=True):
        factor = str(factor)
        lag = int(lag)
        if factor not in selected_factors:
            continue
        source_values = feature_frame[["instrument_id", "date", factor]].copy()
        source_values[factor] = source_values.groupby("instrument_id")[factor].shift(lag)
        source_values = source_values.rename(columns={"instrument_id": "source", factor: "source_factor_value"})
        merged = group_edges[["date", "source", "target", "weight"]].merge(
            source_values,
            on=["date", "source"],
            how="left",
        )
        merged["weighted_value"] = merged["weight"] * pd.to_numeric(merged["source_factor_value"], errors="coerce")
        merged["valid_weight"] = np.where(merged["source_factor_value"].notna(), merged["weight"], 0.0)
        grouped = merged.groupby(["date", "target"], sort=True)[["weighted_value", "valid_weight"]].sum().reset_index()
        column = f"dynamic_lag_{factor}_lag{lag}"
        grouped[column] = grouped["weighted_value"] / grouped["valid_weight"].replace(0.0, np.nan)
        relation_panels.append(grouped.rename(columns={"target": "instrument_id"})[["date", "instrument_id", column]])
        dynamic_columns.append(column)

    if not relation_panels:
        return pd.DataFrame(columns=["date", "instrument_id"]), [], {"dynamic_lag_feature_status": "no_columns_built"}
    panel = relation_panels[0]
    for extra in relation_panels[1:]:
        panel = panel.merge(extra, on=["date", "instrument_id"], how="outer")
    panel = panel.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return panel, dynamic_columns, {
        "dynamic_lag_feature_status": "built",
        "dynamic_lag_feature_count": int(len(dynamic_columns)),
        "dynamic_lag_feature_rows": int(len(panel)),
    }


def build_preprocessing_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        data_path=str(args.data_path),
        model_dir="models",
        output_dir=str(args.output_dir),
        cache_dir=str(args.cache_dir),
        sample_start_date=str(args.sample_start_date),
        oos_start_date=str(args.oos_start_date),
        target_horizon=int(args.target_horizon),
        test_size=float(args.test_size),
        disable_preprocessing_cache=bool(args.disable_preprocessing_cache),
        include_alpha_seeds=not bool(args.no_alpha191),
        factor="",
    )


def filter_prediction_period(
    prediction_df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    result = prediction_df[pd.to_datetime(prediction_df["date"]) >= start].copy()
    if end_date is not None:
        result = result[pd.to_datetime(result["date"]) <= pd.Timestamp(end_date)].copy()
    return result


def long_short_return_by_date(prediction_df: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_value, frame in prediction_df.dropna(subset=["predicted_y", "y"]).groupby("date", sort=True):
        n = len(frame)
        if n < 20:
            continue
        k = max(1, int(math.floor(n * top_frac)))
        ranked = frame.sort_values("predicted_y")
        short_ret = float(pd.to_numeric(ranked.head(k)["y"], errors="coerce").mean())
        long_ret = float(pd.to_numeric(ranked.tail(k)["y"], errors="coerce").mean())
        rows.append(
            {
                "date": pd.Timestamp(date_value),
                "long_return": long_ret,
                "short_return": short_ret,
                "long_short_return": long_ret - short_ret,
                "cross_section_size": int(n),
                "leg_size": int(k),
            }
        )
    return pd.DataFrame(rows)


def daily_rank_ic(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_value, frame in prediction_df.dropna(subset=["predicted_y", "y"]).groupby("date", sort=True):
        if len(frame) < 20:
            continue
        rows.append(
            {
                "date": pd.Timestamp(date_value),
                "rank_ic": safe_corr(frame["predicted_y"], frame["y"], method="spearman"),
                "pearson_ic": safe_corr(frame["predicted_y"], frame["y"], method="pearson"),
                "cross_section_size": int(len(frame)),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["rank_ic"])


def summarize_period_metrics(
    prediction_df: pd.DataFrame,
    experiment: str,
    period_name: str,
    start_date: str | pd.Timestamp,
    top_frac: float,
) -> dict[str, Any]:
    period_df = filter_prediction_period(prediction_df, start_date)
    metrics = evaluate_prediction_frame(period_df, experiment)
    ic_df = daily_rank_ic(period_df)
    ls_df = long_short_return_by_date(period_df, top_frac=top_frac)
    metrics.update(
        {
            "period": period_name,
            "period_start": str(pd.Timestamp(start_date).date()),
            "rows": int(len(period_df)),
            "dates": int(pd.to_datetime(period_df["date"]).nunique()) if not period_df.empty else 0,
            "instruments": int(period_df["instrument_id"].nunique()) if not period_df.empty else 0,
            "daily_rank_ic_std": float(ic_df["rank_ic"].std()) if len(ic_df) > 1 else float("nan"),
            "daily_rank_ic_positive_share": float((ic_df["rank_ic"] > 0).mean()) if not ic_df.empty else float("nan"),
            "daily_long_short_mean": float(ls_df["long_short_return"].mean()) if not ls_df.empty else float("nan"),
            "daily_long_short_positive_share": float((ls_df["long_short_return"] > 0).mean()) if not ls_df.empty else float("nan"),
        }
    )
    return metrics


def build_period_delta_table(period_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if period_metrics_df.empty:
        return pd.DataFrame()
    baseline = period_metrics_df[period_metrics_df["experiment"] == "baseline_linear"].copy()
    relation = period_metrics_df[period_metrics_df["experiment"] != "baseline_linear"].copy()
    for _, relation_row in relation.iterrows():
        period = relation_row["period"]
        base_match = baseline[baseline["period"] == period]
        if base_match.empty:
            continue
        base_row = base_match.iloc[0]
        row: dict[str, Any] = {
            "period": period,
            "baseline_experiment": base_row["experiment"],
            "relation_experiment": relation_row["experiment"],
        }
        for column in [
            "pearson_corr",
            "spearman_corr",
            "rmse",
            "mae",
            "pearson_ic_mean",
            "spearman_ic_mean",
            "long_short_return",
            "daily_long_short_mean",
            "daily_rank_ic_positive_share",
            "daily_long_short_positive_share",
        ]:
            if column in relation_row.index and column in base_row.index:
                row[f"delta_{column}"] = float(relation_row[column] - base_row[column])
        rows.append(row)
    return pd.DataFrame(rows)


def build_monthly_delta(
    baseline_predictions: pd.DataFrame,
    relation_predictions: pd.DataFrame,
    top_frac: float,
    relation_experiment: str = "baseline_plus_relation_linear",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    merged_dates = sorted(
        set(pd.to_datetime(baseline_predictions["date"]).dt.to_period("M").astype(str))
        & set(pd.to_datetime(relation_predictions["date"]).dt.to_period("M").astype(str))
    )
    for month in merged_dates:
        start = pd.Period(month, freq="M").start_time
        end = pd.Period(month, freq="M").end_time
        base_window = filter_prediction_period(baseline_predictions, start, end)
        relation_window = filter_prediction_period(relation_predictions, start, end)
        if base_window.empty or relation_window.empty:
            continue
        base_ic = daily_rank_ic(base_window)
        relation_ic = daily_rank_ic(relation_window)
        base_ls = long_short_return_by_date(base_window, top_frac=top_frac)
        relation_ls = long_short_return_by_date(relation_window, top_frac=top_frac)
        rows.append(
            {
                "relation_experiment": relation_experiment,
                "window_id": month,
                "window_start": str(start.date()),
                "window_end": str(end.date()),
                "baseline_rank_ic": float(base_ic["rank_ic"].mean()) if not base_ic.empty else float("nan"),
                "relation_rank_ic": float(relation_ic["rank_ic"].mean()) if not relation_ic.empty else float("nan"),
                "delta_rank_ic": (
                    float(relation_ic["rank_ic"].mean() - base_ic["rank_ic"].mean())
                    if not base_ic.empty and not relation_ic.empty
                    else float("nan")
                ),
                "baseline_long_short_return": float(base_ls["long_short_return"].mean()) if not base_ls.empty else float("nan"),
                "relation_long_short_return": float(relation_ls["long_short_return"].mean()) if not relation_ls.empty else float("nan"),
                "delta_long_short_return": (
                    float(relation_ls["long_short_return"].mean() - base_ls["long_short_return"].mean())
                    if not base_ls.empty and not relation_ls.empty
                    else float("nan")
                ),
                "date_count": int(pd.to_datetime(base_window["date"]).nunique()),
            }
        )
    return pd.DataFrame(rows)


def rank_turnover_proxy(prediction_df: pd.DataFrame, top_frac: float) -> dict[str, Any]:
    previous_weights: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for date_value, frame in prediction_df.dropna(subset=["predicted_y"]).groupby("date", sort=True):
        n = len(frame)
        if n < 20:
            continue
        k = max(1, int(math.floor(n * top_frac)))
        ranked = frame.sort_values("predicted_y")
        weights: dict[str, float] = {}
        for symbol in ranked.tail(k)["instrument_id"].astype(str):
            weights[symbol] = 1.0 / k
        for symbol in ranked.head(k)["instrument_id"].astype(str):
            weights[symbol] = weights.get(symbol, 0.0) - 1.0 / k
        universe = set(weights) | set(previous_weights)
        gross_turnover = sum(abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)) for symbol in universe)
        rows.append({"date": pd.Timestamp(date_value), "gross_turnover": float(gross_turnover), "leg_size": int(k)})
        previous_weights = weights
    turnover_df = pd.DataFrame(rows)
    return {
        "average_gross_turnover": float(turnover_df["gross_turnover"].iloc[1:].mean()) if len(turnover_df) > 1 else float("nan"),
        "median_gross_turnover": float(turnover_df["gross_turnover"].iloc[1:].median()) if len(turnover_df) > 1 else float("nan"),
        "rebalance_count": int(len(turnover_df)),
    }


def build_turnover_table(prediction_map: dict[str, pd.DataFrame], top_frac: float) -> pd.DataFrame:
    rows = []
    for experiment, predictions in prediction_map.items():
        rows.append({"experiment": experiment, **rank_turnover_proxy(predictions, top_frac=top_frac)})
    result = pd.DataFrame(rows)
    if not result.empty and "baseline_linear" in set(result["experiment"]):
        base_turnover = float(result.loc[result["experiment"] == "baseline_linear", "average_gross_turnover"].iloc[0])
        result["delta_vs_baseline_average_gross_turnover"] = result["average_gross_turnover"] - base_turnover
    return result


def build_acceptance_summary(
    period_delta_df: pd.DataFrame,
    monthly_delta_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    experiments = [exp for exp in period_delta_df.get("relation_experiment", pd.Series(dtype=str)).dropna().unique()]
    priority = [
        "baseline_plus_dynamic_lag_relation_linear",
        "baseline_plus_quality_relation_linear",
        "baseline_plus_typed_stable_relation_linear",
        "baseline_plus_static_relation_linear",
        "baseline_plus_relation_linear",
    ]
    for experiment in experiments:
        primary = period_delta_df[(period_delta_df["period"] == "primary") & (period_delta_df["relation_experiment"] == experiment)]
        secondary = period_delta_df[(period_delta_df["period"] == "secondary") & (period_delta_df["relation_experiment"] == experiment)]
        primary_rank_delta = float(primary["delta_spearman_ic_mean"].iloc[0]) if not primary.empty else float("nan")
        secondary_rank_delta = float(secondary["delta_spearman_ic_mean"].iloc[0]) if not secondary.empty else float("nan")
        monthly = monthly_delta_df[monthly_delta_df["relation_experiment"] == experiment] if "relation_experiment" in monthly_delta_df.columns else monthly_delta_df
        monthly_rank_delta_mean = float(monthly["delta_rank_ic"].mean()) if not monthly.empty else float("nan")
        monthly_positive_share = float((monthly["delta_rank_ic"] > 0).mean()) if not monthly.empty else float("nan")
        ls_delta_abs = monthly["delta_long_short_return"].abs().dropna() if not monthly.empty else pd.Series(dtype=float)
        top_month_abs_delta_share = float(ls_delta_abs.max() / ls_delta_abs.sum()) if not ls_delta_abs.empty and ls_delta_abs.sum() > 0 else float("nan")
        turnover_delta = float("nan")
        if not turnover_df.empty:
            turnover_row = turnover_df[turnover_df["experiment"] == experiment]
            if not turnover_row.empty:
                turnover_delta = float(turnover_row["delta_vs_baseline_average_gross_turnover"].iloc[0])

        check_values = [
            ("primary_rank_ic_delta_at_least_0.003", primary_rank_delta, 0.003, np.isfinite(primary_rank_delta) and primary_rank_delta >= 0.003),
            ("primary_rank_ic_delta_at_least_0.005_stretch", primary_rank_delta, 0.005, np.isfinite(primary_rank_delta) and primary_rank_delta >= 0.005),
            (
                "primary_secondary_delta_same_positive_direction",
                secondary_rank_delta,
                0.0,
                np.isfinite(primary_rank_delta)
                and np.isfinite(secondary_rank_delta)
                and primary_rank_delta > 0
                and secondary_rank_delta > 0,
            ),
            ("monthly_paired_rank_delta_positive", monthly_rank_delta_mean, 0.0, np.isfinite(monthly_rank_delta_mean) and monthly_rank_delta_mean > 0),
            ("monthly_positive_delta_share_at_least_0.55", monthly_positive_share, 0.55, np.isfinite(monthly_positive_share) and monthly_positive_share >= 0.55),
            ("turnover_delta_not_materially_higher", turnover_delta, 0.10, np.isfinite(turnover_delta) and turnover_delta <= 0.10),
            ("top_month_abs_delta_share_below_0.35", top_month_abs_delta_share, 0.35, np.isfinite(top_month_abs_delta_share) and top_month_abs_delta_share < 0.35),
        ]
        for check, value, threshold, passed in check_values:
            rows.append({"experiment": experiment, "check": check, "value": value, "threshold": threshold, "passed": bool(passed)})
        summaries[experiment] = {
            "primary_rank_ic_delta": primary_rank_delta,
            "secondary_rank_ic_delta": secondary_rank_delta,
            "monthly_rank_delta_mean": monthly_rank_delta_mean,
            "monthly_positive_delta_share": monthly_positive_share,
            "turnover_delta": turnover_delta,
            "top_month_abs_delta_share": top_month_abs_delta_share,
        }

    check_df = pd.DataFrame(rows)
    hard_checks = {
        "primary_rank_ic_delta_at_least_0.003",
        "primary_secondary_delta_same_positive_direction",
        "monthly_paired_rank_delta_positive",
        "monthly_positive_delta_share_at_least_0.55",
        "turnover_delta_not_materially_higher",
        "top_month_abs_delta_share_below_0.35",
    }
    priority_rank = {experiment: rank for rank, experiment in enumerate(priority)}

    def rank_experiment(experiment: str) -> tuple[Any, ...]:
        experiment_checks = check_df[check_df["experiment"] == experiment]
        hard_check_rows = experiment_checks[experiment_checks["check"].isin(hard_checks)]
        hard_pass_count = int(hard_check_rows["passed"].sum()) if not hard_check_rows.empty else 0
        summary = summaries.get(experiment, {})
        primary_delta = float(summary.get("primary_rank_ic_delta", float("nan")))
        secondary_delta = float(summary.get("secondary_rank_ic_delta", float("nan")))
        turnover_delta = float(summary.get("turnover_delta", float("nan")))
        predictive_flag = bool(np.isfinite(primary_delta) and primary_delta >= 0.003 and np.isfinite(secondary_delta) and secondary_delta > 0)
        turnover_ok = bool(np.isfinite(turnover_delta) and turnover_delta <= 0.10)
        return (
            hard_pass_count,
            int(predictive_flag),
            primary_delta if np.isfinite(primary_delta) else -999.0,
            secondary_delta if np.isfinite(secondary_delta) else -999.0,
            int(turnover_ok),
            -priority_rank.get(experiment, 999),
        )

    chosen = max(summaries, key=rank_experiment) if summaries else (experiments[0] if experiments else "")
    chosen_checks = check_df[check_df["experiment"] == chosen]
    hard_pass = bool(not chosen_checks.empty and chosen_checks[chosen_checks["check"].isin(hard_checks)]["passed"].all())
    chosen_summary = summaries.get(chosen, {})
    predictive = bool(
        np.isfinite(chosen_summary.get("primary_rank_ic_delta", float("nan")))
        and chosen_summary.get("primary_rank_ic_delta", float("nan")) >= 0.003
        and np.isfinite(chosen_summary.get("secondary_rank_ic_delta", float("nan")))
        and chosen_summary.get("secondary_rank_ic_delta", float("nan")) > 0
    )
    high_turnover = bool(
        np.isfinite(chosen_summary.get("turnover_delta", float("nan")))
        and chosen_summary.get("turnover_delta", float("nan")) > 0.10
    )
    if hard_pass:
        status = "relation_module_survives_incremental_gate"
    elif predictive and high_turnover:
        status = "predictive_but_high_turnover"
    elif predictive:
        status = "research_only_relation_effect"
    else:
        status = "failed_scaleup"
    summary = {
        "status": status,
        "chosen_experiment": chosen,
        "experiment_summaries": summaries,
        **chosen_summary,
    }
    return check_df, summary


def write_report(
    output_dir: Path,
    dataset_summary: dict[str, Any],
    relation_coverage: dict[str, Any],
    model_metrics_df: pd.DataFrame,
    period_delta_df: pd.DataFrame,
    monthly_delta_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    acceptance_df: pd.DataFrame,
    acceptance_summary: dict[str, Any],
    relation_feature_importance_df: pd.DataFrame,
) -> None:
    chosen_experiment = str(acceptance_summary.get("chosen_experiment", ""))
    primary_delta = period_delta_df[
        (period_delta_df["period"] == "primary")
        & (period_delta_df["relation_experiment"].astype(str) == chosen_experiment)
    ]
    secondary_delta = period_delta_df[
        (period_delta_df["period"] == "secondary")
        & (period_delta_df["relation_experiment"].astype(str) == chosen_experiment)
    ]
    primary_rank_delta = (
        float(primary_delta["delta_spearman_ic_mean"].iloc[0]) if not primary_delta.empty else float("nan")
    )
    secondary_rank_delta = (
        float(secondary_delta["delta_spearman_ic_mean"].iloc[0]) if not secondary_delta.empty else float("nan")
    )
    report = f"""# Relation Feature Incremental Experiment

## 1. Question

This experiment tests one claim only:

```text
MacQuant baseline + relation graph features 是否比 MacQuant baseline 更好？
```

It does not claim that the relation graph is a standalone alpha.

## 2. Status

- P0 status: `{acceptance_summary.get("status")}`
- Chosen experiment: `{chosen_experiment}`
- Primary OOS Rank IC delta: `{primary_rank_delta:.6f}`
- Secondary OOS Rank IC delta: `{secondary_rank_delta:.6f}`
- Monthly paired Rank IC delta mean: `{acceptance_summary.get("monthly_rank_delta_mean"):.6f}`
- Turnover delta proxy: `{acceptance_summary.get("turnover_delta"):.6f}`
- Top month abs long-short delta share: `{acceptance_summary.get("top_month_abs_delta_share"):.6f}`

## 3. Dataset And Boundary

```json
{json.dumps(dataset_summary, ensure_ascii=False, indent=2)}
```

Relation feature coverage:

```json
{json.dumps(relation_coverage, ensure_ascii=False, indent=2)}
```

Important boundary:

- Daily edge weights come from the upstream relation edge table.
- For edge date `t`, the exported rolling correlations use history before `t`.
- `weighted_peer_return_1d/5d/20d` use peer close-to-close returns known after close on date `t`.
- `lead_lag_peer_return_1d` uses previous-day peer return.
- This run uses a linear model comparison; it is not NN/GNN evidence.

## 4. Relation Features

| Feature | Meaning |
| --- | --- |
| weighted_peer_return_1d | Weighted average same-date 1d return of source peers. |
| weighted_peer_return_5d | Weighted average 5d return of source peers. |
| weighted_peer_return_20d | Weighted average 20d return of source peers. |
| lead_lag_peer_return_1d | Weighted average previous-day 1d return of source peers. |
| relation_dispersion | Weighted cross-peer dispersion of 1d peer returns. |
| source_influence_centrality | Weighted average influence centrality of incoming source peers. |
| target_neighbor_centrality | Target exposure to influential neighbors from the edge table. |
| same_sector_peer_strength | Sum of retained peer weights from the same sector. |
| top1_peer_return | 1d return of the strongest relation peer. |
| topk_peer_return_std | Unweighted standard deviation of top-k peer 1d returns. |
| stable_* | Same peer-return features restricted to stable rolling-correlation edges. |
| same_sector_* | Peer-return features restricted to same-sector edges. |
| quality_* | Peer-return features restricted to edges with stability, same-sector, directed-lag, or quality-score evidence. |
| directed_lag_* | Peer-return features restricted to train-selected directed-lag edge pairs. |
| dynamic_lag_* | Source factor values aligned by train-selected pair-specific lag. |

## 5. Model Metrics

{dataframe_to_markdown(model_metrics_df.round(6))}

## 6. Period Delta

Positive `delta_spearman_ic_mean` means relation features improved Rank IC versus the baseline on the same OOS rows.

{dataframe_to_markdown(period_delta_df.round(6))}

## 7. Monthly Paired Delta

This is a paired OOS month-window check using the same trained model. It is not a full retrained expanding-window study.

{dataframe_to_markdown(monthly_delta_df.round(6))}

## 8. Turnover Proxy

Turnover is computed from daily top/bottom 10% equal-weight rank selections. This is a proxy, not a full costed portfolio backtest.

{dataframe_to_markdown(turnover_df.round(6))}

## 9. Acceptance Checks

{dataframe_to_markdown(acceptance_df)}

## 10. Relation Feature Coefficients

This table is only from linear model coefficients. Treat it as diagnostic, not causal attribution.

{dataframe_to_markdown(relation_feature_importance_df.round(8))}

## 11. Interpretation

If P0 passes, the defensible project claim is:

```text
Relation graph can be used as a MacQuant feature-engineering layer with positive incremental OOS evidence.
```

If P0 fails, the defensible claim is narrower:

```text
The project produced reusable relation edge tables and a MacQuant integration test, but current relation features did not survive the incremental alpha gate.
```
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def relation_feature_coefficients(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    relation_columns: list[str],
    model_name: str,
    random_seed: int,
) -> pd.DataFrame:
    from factor_mining_workspace.mined_factor_model_ablation import build_feature_matrices
    from src.model import build_model

    if model_name != "ridge":
        return pd.DataFrame(columns=["feature", "abs_coefficient"])
    train_x, _ = build_feature_matrices(train_df, test_df, feature_columns)
    y_train = pd.to_numeric(train_df["y"], errors="coerce")
    valid = y_train.notna()
    model = build_model(model_name="ridge", random_state=random_seed)
    model.fit(train_x.loc[valid], y_train.loc[valid])
    coefs = np.asarray(model.model.named_steps["model"].coef_, dtype=float)
    coef_df = pd.DataFrame({"feature": feature_columns, "coefficient": coefs})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    return (
        coef_df[coef_df["feature"].isin(relation_columns)]
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_data_path = resolve_path(args.data_path)
    filtered_data_path = prepare_filtered_data_path(args, output_dir=output_dir)
    args.data_path = str(filtered_data_path)

    run_config = vars(args).copy()
    run_config["original_data_path"] = str(original_data_path)
    run_config["resolved_data_path"] = str(resolve_path(args.data_path))
    run_config["resolved_edge_path"] = str(resolve_path(args.edge_path))
    (output_dir / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    relation_panel = load_or_build_relation_feature_panel(args, output_dir=output_dir)
    preprocessing_args = build_preprocessing_args(args)
    train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(preprocessing_args)

    static_relation_columns = [column for column in RELATION_FEATURE_COLUMNS if column in relation_panel.columns]
    typed_stable_relation_columns = [column for column in TYPED_STABLE_RELATION_FEATURE_COLUMNS if column in relation_panel.columns]
    quality_relation_columns = [column for column in QUALITY_RELATION_FEATURE_COLUMNS if column in relation_panel.columns]

    train_static_df, test_static_df, static_coverage = attach_relation_features(
        train_df=train_df,
        test_df=test_df,
        relation_panel=relation_panel,
        relation_columns=static_relation_columns,
    )
    baseline_feature_columns = get_numeric_feature_columns(train_df)
    static_relation_columns = [column for column in static_relation_columns if column in train_static_df.columns and column in test_static_df.columns]
    static_model_columns = baseline_feature_columns + static_relation_columns

    train_typed_df, test_typed_df, typed_coverage = attach_relation_features(
        train_df=train_df,
        test_df=test_df,
        relation_panel=relation_panel,
        relation_columns=static_relation_columns + typed_stable_relation_columns,
    )
    typed_stable_relation_columns = [
        column for column in typed_stable_relation_columns if column in train_typed_df.columns and column in test_typed_df.columns
    ]
    typed_model_columns = baseline_feature_columns + static_relation_columns + typed_stable_relation_columns

    train_quality_df, test_quality_df, quality_coverage = attach_relation_features(
        train_df=train_df,
        test_df=test_df,
        relation_panel=relation_panel,
        relation_columns=static_relation_columns + quality_relation_columns,
    )
    quality_relation_columns = [
        column for column in quality_relation_columns if column in train_quality_df.columns and column in test_quality_df.columns
    ]
    quality_model_columns = baseline_feature_columns + static_relation_columns + quality_relation_columns

    dynamic_metadata: dict[str, Any] = {"dynamic_lag_selection_status": "not_requested"}
    dynamic_relation_columns: list[str] = []
    train_dynamic_df = pd.DataFrame()
    test_dynamic_df = pd.DataFrame()
    dynamic_model_columns: list[str] = []
    dynamic_path_raw = str(args.dynamic_lag_selection_path).strip()
    dynamic_path = resolve_path(dynamic_path_raw) if dynamic_path_raw else None
    if dynamic_path is not None and dynamic_path.exists():
        combined_features = pd.concat([train_df, test_df], ignore_index=True, sort=False)
        lag_selection, selected_dynamic_factors, dynamic_selection_metadata = load_dynamic_lag_selection(
            selection_path=dynamic_path,
            available_feature_columns=baseline_feature_columns,
            max_factors=int(args.max_dynamic_relation_factors),
        )
        dynamic_panel, dynamic_relation_columns, dynamic_feature_metadata = build_dynamic_lag_relation_panel(
            edge_path=resolve_path(args.edge_path),
            feature_frame=combined_features,
            lag_selection=lag_selection,
            selected_factors=selected_dynamic_factors,
        )
        dynamic_metadata = {**dynamic_selection_metadata, **dynamic_feature_metadata}
        if dynamic_relation_columns:
            train_dynamic_df, test_dynamic_df, dynamic_coverage = attach_relation_features(
                train_df=train_static_df,
                test_df=test_static_df,
                relation_panel=dynamic_panel,
                relation_columns=dynamic_relation_columns,
            )
            dynamic_metadata["coverage"] = dynamic_coverage
            dynamic_relation_columns = [
                column for column in dynamic_relation_columns if column in train_dynamic_df.columns and column in test_dynamic_df.columns
            ]
            dynamic_model_columns = baseline_feature_columns + static_relation_columns + dynamic_relation_columns
            if quality_relation_columns:
                train_dynamic_df, test_dynamic_df, quality_dynamic_coverage = attach_relation_features(
                    train_df=train_dynamic_df,
                    test_df=test_dynamic_df,
                    relation_panel=relation_panel,
                    relation_columns=quality_relation_columns,
                )
                dynamic_metadata["quality_edge_coverage"] = quality_dynamic_coverage
                dynamic_model_columns = baseline_feature_columns + static_relation_columns + quality_relation_columns + dynamic_relation_columns
    elif dynamic_path is not None:
        dynamic_metadata = {"dynamic_lag_selection_status": "missing", "selection_path": str(dynamic_path)}

    experiment_specs: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]] = {
        "baseline_linear": (train_df, test_df, baseline_feature_columns, []),
        "baseline_plus_static_relation_linear": (train_static_df, test_static_df, static_model_columns, static_relation_columns),
    }
    if typed_stable_relation_columns:
        experiment_specs["baseline_plus_typed_stable_relation_linear"] = (
            train_typed_df,
            test_typed_df,
            typed_model_columns,
            static_relation_columns + typed_stable_relation_columns,
        )
    if quality_relation_columns:
        experiment_specs["baseline_plus_quality_relation_linear"] = (
            train_quality_df,
            test_quality_df,
            quality_model_columns,
            static_relation_columns + quality_relation_columns,
        )
    if dynamic_relation_columns:
        experiment_specs["baseline_plus_dynamic_lag_relation_linear"] = (
            train_dynamic_df,
            test_dynamic_df,
            dynamic_model_columns,
            static_relation_columns + quality_relation_columns + dynamic_relation_columns,
        )

    model_metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_map: dict[str, pd.DataFrame] = {}

    for experiment, (exp_train, exp_test, feature_columns, _relation_cols) in experiment_specs.items():
        start = time.perf_counter()
        predictions, runtime = train_and_predict(
            train_df=exp_train,
            test_df=exp_test,
            feature_columns=feature_columns,
            model_names=list(args.models),
            random_seed=int(args.random_seed),
        )
        prediction_map[experiment] = predictions
        predictions.to_csv(prediction_dir / f"{experiment}_predictions.csv", index=False)
        runtime_rows.extend([{"experiment": experiment, **row.to_dict()} for _, row in runtime.iterrows()])
        runtime_rows.append(
            {
                "experiment": experiment,
                "model": "group_total",
                "feature_count": len(feature_columns),
                "train_rows": int(len(exp_train)),
                "oos_rows": int(len(exp_test)),
                "runtime_seconds": time.perf_counter() - start,
            }
        )
        metrics = summarize_period_metrics(
            predictions,
            experiment=experiment,
            period_name="primary",
            start_date=args.oos_start_date,
            top_frac=float(args.top_frac),
        )
        metrics.update({"feature_count": len(feature_columns)})
        model_metric_rows.append(metrics)
        secondary = summarize_period_metrics(
            predictions,
            experiment=experiment,
            period_name="secondary",
            start_date=args.secondary_start_date,
            top_frac=float(args.top_frac),
        )
        secondary.update({"feature_count": len(feature_columns)})
        model_metric_rows.append(secondary)

    model_metrics_df = pd.DataFrame(model_metric_rows)
    period_delta_df = build_period_delta_table(model_metrics_df)
    monthly_frames: list[pd.DataFrame] = []
    baseline_predictions = filter_prediction_period(prediction_map["baseline_linear"], args.oos_start_date)
    for experiment, predictions in prediction_map.items():
        if experiment == "baseline_linear":
            continue
        monthly_frames.append(
            build_monthly_delta(
                baseline_predictions=baseline_predictions,
                relation_predictions=filter_prediction_period(predictions, args.oos_start_date),
                top_frac=float(args.top_frac),
                relation_experiment=experiment,
            )
        )
    monthly_delta_df = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    turnover_df = build_turnover_table(
        {experiment: filter_prediction_period(predictions, args.oos_start_date) for experiment, predictions in prediction_map.items()},
        top_frac=float(args.top_frac),
    )
    acceptance_df, acceptance_summary = build_acceptance_summary(
        period_delta_df=period_delta_df,
        monthly_delta_df=monthly_delta_df,
        turnover_df=turnover_df,
    )
    coefficient_experiment = acceptance_summary.get("chosen_experiment") or next(
        (
            name
            for name in [
                "baseline_plus_dynamic_lag_relation_linear",
                "baseline_plus_quality_relation_linear",
                "baseline_plus_typed_stable_relation_linear",
                "baseline_plus_static_relation_linear",
            ]
            if name in experiment_specs
        ),
        "baseline_plus_static_relation_linear",
    )
    coef_train, coef_test, coef_feature_columns, coef_relation_columns = experiment_specs[coefficient_experiment]
    relation_feature_importance_df = relation_feature_coefficients(
        train_df=coef_train,
        test_df=coef_test,
        feature_columns=coef_feature_columns,
        relation_columns=coef_relation_columns,
        model_name="ridge",
        random_seed=int(args.random_seed),
    )

    dataset_summary.update(
        {
            "original_data_path": str(original_data_path),
            "target_column": target_column,
            "baseline_feature_count": int(len(baseline_feature_columns)),
            "static_relation_feature_count": int(len(static_relation_columns)),
            "typed_stable_relation_feature_count": int(len(typed_stable_relation_columns)),
            "quality_relation_feature_count": int(len(quality_relation_columns)),
            "dynamic_relation_feature_count": int(len(dynamic_relation_columns)),
            "experiment_feature_counts": {name: int(len(spec[2])) for name, spec in experiment_specs.items()},
            "models": list(args.models),
            "edge_path": str(resolve_path(args.edge_path)),
            "relation_feature_panel_path": str(output_dir / "relation_features_panel.csv"),
            "dynamic_lag_metadata": dynamic_metadata,
            "primary_oos_start_date": args.oos_start_date,
            "secondary_oos_start_date": args.secondary_start_date,
            "alpha191_included": not bool(args.no_alpha191),
            "universe_top_n": int(args.universe_top_n),
        }
    )
    relation_coverage = {
        "static": static_coverage,
        "typed_stable": typed_coverage,
        "quality": quality_coverage,
        "dynamic_lag": dynamic_metadata.get("coverage", {}),
    }

    model_metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    period_delta_df.to_csv(output_dir / "period_delta.csv", index=False)
    monthly_delta_df.to_csv(output_dir / "walkforward_paired_delta.csv", index=False)
    monthly_delta_df.to_csv(output_dir / "month_concentration.csv", index=False)
    turnover_df.to_csv(output_dir / "turnover_proxy.csv", index=False)
    acceptance_df.to_csv(output_dir / "acceptance_summary.csv", index=False)
    relation_feature_importance_df.to_csv(output_dir / "relation_feature_coefficients.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output_dir / "runtime.csv", index=False)

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "relation_coverage.json").write_text(
        json.dumps(relation_coverage, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "final_status.json").write_text(
        json.dumps(acceptance_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        output_dir=output_dir,
        dataset_summary=dataset_summary,
        relation_coverage=relation_coverage,
        model_metrics_df=model_metrics_df,
        period_delta_df=period_delta_df,
        monthly_delta_df=monthly_delta_df,
        turnover_df=turnover_df,
        acceptance_df=acceptance_df,
        acceptance_summary=acceptance_summary,
        relation_feature_importance_df=relation_feature_importance_df,
    )
    print(f"[Done] output_dir={output_dir}")
    print(f"[Done] status={acceptance_summary['status']}")
    print(f"[Done] primary_rank_ic_delta={acceptance_summary['primary_rank_ic_delta']:.6f}")
    print(f"[Done] report={output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
