"""Export one canonical run into a compact, auditable public evidence package.

The training and portfolio directories contain large or private artifacts such as
predictions, models, and caches. This exporter keeps only the tables and manifests
needed to verify public claims. It never copies raw data or binary model files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Direct execution sets sys.path to scripts/. Add the repository root so the
# exporter works from a fresh clone without installing MyQuant as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import resolve_project_path
from src.provenance import dumps_strict_json, sha256_file
from src.alpha191 import CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS


REQUIRED_TRAINING_FILES = (
    "run_manifest.json",
    "data_quality_summary.json",
    "corporate_action_audit.csv",
    "universe_coverage_audit.csv",
    "walk_forward_fold_metrics.csv",
    "walk_forward_model_summary.csv",
    "stage_timing.csv",
    "final_model_timing.csv",
)

PUBLIC_PORTFOLIO_DETAIL_FILES = (
    "daily_returns.csv",
    "portfolio_weights.csv",
    "turnover_cost.csv",
    "skipped_trades.csv",
    "sector_exposure.csv",
    "extreme_return_days.csv",
    "position_daily_contributions.csv",
    "instrument_return_attribution.csv",
    "portfolio_metrics.json",
    "portfolio_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export canonical run evidence for GitHub review.")
    parser.add_argument("--training-output", default="outputs/public_us300_release_v1")
    parser.add_argument("--model-dir", default="models/public_us300_release_v1")
    parser.add_argument(
        "--portfolio-summary",
        default="outputs/public_us300_release_v1_backtest/portfolio_grid_summary.csv",
    )
    parser.add_argument(
        "--backtest-manifest",
        default="outputs/public_us300_release_v1_backtest/backtest_run_manifest.json",
    )
    parser.add_argument("--public-dir", default="results/public/us300_release_v1")
    parser.add_argument("--experiment-id", default="us300_release_v1")
    parser.add_argument(
        "--allow-pre-release",
        action="store_true",
        help="Allow a dirty/missing-commit source for local review. Never use this flag for a public release.",
    )
    parser.add_argument(
        "--allow-missing-backtest",
        action="store_true",
        help="Export prediction evidence without portfolio evidence for local review only.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_public_staging_directory(public_dir: Path) -> Path:
    """Create a clean sibling directory for one all-or-nothing evidence export.

    Overwriting files in place can leave obsolete tables or figures from an older
    experiment beside a new manifest.  The exporter therefore builds a complete
    package in a temporary sibling directory and replaces the public directory
    only after every contract check and write has succeeded.
    """

    allowed_root = (PROJECT_ROOT / "results" / "public").resolve()
    resolved_public_dir = public_dir.resolve()
    try:
        relative_public_dir = resolved_public_dir.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"Public evidence output must stay under {allowed_root}: {resolved_public_dir}"
        ) from exc
    if not relative_public_dir.parts:
        raise ValueError("Public evidence output must be a run-specific subdirectory.")

    staging_dir = resolved_public_dir.with_name(f".{resolved_public_dir.name}.staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir


def replace_public_directory(public_dir: Path, staging_dir: Path) -> None:
    """Atomically replace stale evidence while preserving the old package on failure."""

    resolved_public_dir = public_dir.resolve()
    backup_dir = resolved_public_dir.with_name(f".{resolved_public_dir.name}.previous")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if resolved_public_dir.exists():
        resolved_public_dir.rename(backup_dir)
    try:
        staging_dir.rename(resolved_public_dir)
    except Exception:
        if backup_dir.exists() and not resolved_public_dir.exists():
            backup_dir.rename(resolved_public_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def require_matching_fingerprint(path: Path, fingerprint: dict[str, Any], label: str) -> None:
    """Fail when a saved artifact has changed after its manifest was written."""

    if not path.is_file():
        raise FileNotFoundError(f"Required {label} artifact is missing: {path}")
    expected_sha256 = str((fingerprint or {}).get("sha256") or "")
    if not expected_sha256:
        raise ValueError(f"Manifest has no SHA256 for {label}: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} artifact no longer matches its manifest: {path} "
            f"({actual_sha256} != {expected_sha256})"
        )


def validate_training_contract(
    training_output: Path,
    model_dir: Path,
    run_manifest: dict[str, Any],
    *,
    allow_pre_release: bool,
) -> None:
    """Validate the fixed public US300 experiment and all exported source files."""

    if run_manifest.get("status") != "completed":
        raise ValueError(f"Training run is not completed: {run_manifest.get('status')!r}")
    if int(run_manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported training manifest schema: {run_manifest.get('schema_version')!r}")

    git_state = run_manifest.get("environment", {}).get("git", {}) or {}
    if not allow_pre_release:
        if not git_state.get("commit"):
            raise ValueError("Public evidence requires a recorded source commit.")
        if git_state.get("dirty_tracked_worktree") is not False:
            raise ValueError("Public evidence requires a clean source worktree.")

    configuration = run_manifest.get("effective_configuration", {}) or {}
    arguments = run_manifest.get("arguments", {}) or {}
    data = run_manifest.get("data", {}) or {}
    if configuration.get("target_column") != "y_10d":
        raise ValueError(f"Canonical public target must be y_10d: {configuration.get('target_column')!r}")
    if pd.Timestamp(configuration.get("oos_start_date")) != pd.Timestamp("2025-06-01"):
        raise ValueError(
            "Canonical public OOS start must be 2025-06-01: "
            f"{configuration.get('oos_start_date')!r}"
        )
    if configuration.get("price_adjustment_mode") != "vendor_adjusted":
        raise ValueError("Canonical public training must use vendor_adjusted prices.")
    if int(data.get("instrument_count", -1)) != 300:
        raise ValueError(f"Canonical public run must contain 300 instruments: {data.get('instrument_count')!r}")
    expected_arguments = {
        "data_path": "data/us_large_cap_300_daily.csv",
        "sample_start_date": "2022-01-01",
        "oos_start_date": "2025-06-01",
        "target_horizon": 10,
        "n_splits": 3,
        "top_n": 50,
        "missing_threshold": 0.6,
        "variance_threshold": 0.001,
        "correlation_threshold": 0.95,
        "feature_score_method": "correlation",
        "validation_score_metric": "pearson_ic_mean",
        "max_alpha": 0,
        "alpha_factors": list(CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS),
        "universe_label": "us_large_cap_300_static_snapshot",
        "refresh_caches": True,
        "random_state": 42,
    }
    for key, expected_value in expected_arguments.items():
        if arguments.get(key) != expected_value:
            raise ValueError(
                f"Canonical public argument {key!r} must be {expected_value!r}: "
                f"found {arguments.get(key)!r}"
            )
    if list(configuration.get("models", [])) != ["ridge", "lasso"]:
        raise ValueError(f"Canonical public models must be ['ridge', 'lasso']: {configuration.get('models')!r}")

    data_quality = data.get("data_quality", {}) or {}
    if float(data_quality.get("missing_or_invalid_adjustment_ratio", 1.0)) > 0.001:
        raise ValueError("Canonical data has too many missing or invalid adjustment factors.")
    if float(data_quality.get("non_unit_adjustment_ratio", 0.0)) <= 0.01:
        raise ValueError(
            "Canonical data does not preserve auditable vendor adjustment factors; "
            "an auto-adjusted all-ones file is not accepted."
        )
    if float(data_quality.get("turnover_raw_close_evaluable_ratio", 0.0)) < 0.95:
        raise ValueError("Canonical raw-dollar-turnover consistency is evaluable on less than 95% of rows.")
    if float(data_quality.get("turnover_raw_close_consistency_ratio", 0.0)) < 0.99:
        raise ValueError("Canonical turnover is not consistent with raw close times volume on at least 99% of evaluable rows.")

    features = run_manifest.get("features", {}) or {}
    if features.get("selection_score_basis") != (
        "absolute_mean_daily_cross_sectional_pearson_ic_train_only"
    ):
        raise ValueError(
            "Canonical public feature selection must use train-only mean daily cross-sectional IC."
        )
    feature_metadata = features.get("metadata", {}) or {}
    raw_feature_columns = set(feature_metadata.get("raw_feature_columns", []))
    forbidden_absolute_raw_features = {"open", "high", "low", "close", "volume", "vwap"}
    if raw_feature_columns & forbidden_absolute_raw_features:
        raise ValueError(
            "Canonical public candidates contain direct absolute OHLCV/VWAP levels: "
            f"{sorted(raw_feature_columns & forbidden_absolute_raw_features)}"
        )
    fundamental_columns = set(feature_metadata.get("fundamental_raw_columns", [])) | set(
        feature_metadata.get("fundamental_rank_columns", [])
    )
    if fundamental_columns:
        raise ValueError(
            "Canonical public US300 evidence must not contain optional fundamental "
            f"features: {sorted(fundamental_columns)}"
        )
    generated_alpha_columns = list(feature_metadata.get("alpha_feature_columns", []))
    if generated_alpha_columns != list(CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS):
        raise ValueError(
            "Canonical feature metadata does not contain the exact documented "
            f"scale-invariant Alpha list: {generated_alpha_columns}"
        )
    if float(data_quality.get("market_cap_coverage_ratio", 1.0)) > 1e-12:
        raise ValueError(
            "Canonical public US300 data contract currently requires zero market-cap "
            "coverage; point-in-time size data belongs in a separate ablation."
        )
    preprocessing = features.get("preprocessing", {}) or {}
    if preprocessing.get("sector_neutralization_used") is not True:
        raise ValueError("Canonical public features must record actual sector neutralization.")
    if preprocessing.get("size_neutralization_used") is not False:
        raise ValueError(
            "Canonical public US300 must not claim size neutralization without point-in-time market cap."
        )
    if float(preprocessing.get("market_cap_coverage_ratio", 1.0)) > 1e-12:
        raise ValueError("Canonical preprocessing observed unexpected market-cap coverage.")
    validation = run_manifest.get("validation", {}) or {}
    if validation.get("model_weight_policy") != (
        "50pct_validation_positive_score_plus_50pct_equal_weight"
    ):
        raise ValueError("Canonical public model weights must use the documented shrinkage policy.")

    artifact_paths = {
        "predictions": training_output / "predictions.csv",
        "test_predictions_with_actual": training_output / "test_predictions_with_actual.csv",
        "walk_forward_fold_metrics": training_output / "walk_forward_fold_metrics.csv",
        "walk_forward_model_summary": training_output / "walk_forward_model_summary.csv",
        "stage_timing": training_output / "stage_timing.csv",
        "final_model_timing": training_output / "final_model_timing.csv",
        "data_quality_summary": training_output / "data_quality_summary.json",
        "corporate_action_audit": training_output / "corporate_action_audit.csv",
        "universe_coverage_audit": training_output / "universe_coverage_audit.csv",
        "selected_features": model_dir / "selected_features.csv",
        "selected_feature_scores": model_dir / "selected_feature_scores.csv",
        "model_weights": model_dir / "model_weights.csv",
    }
    artifact_fingerprints = run_manifest.get("artifacts", {}) or {}
    for artifact_name, artifact_path in artifact_paths.items():
        require_matching_fingerprint(
            artifact_path,
            artifact_fingerprints.get(artifact_name, {}),
            f"training/{artifact_name}",
        )


def validate_backtest_contract(
    *,
    run_manifest: dict[str, Any],
    backtest_manifest: dict[str, Any],
    portfolio_summary_path: Path,
    portfolio_df: pd.DataFrame,
    allow_pre_release: bool,
) -> None:
    """Prove that portfolio evidence uses the same data, commit, and predictions."""

    if backtest_manifest.get("status") != "completed":
        raise ValueError(f"Backtest run is not completed: {backtest_manifest.get('status')!r}")
    if int(backtest_manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported backtest manifest schema: {backtest_manifest.get('schema_version')!r}")
    if backtest_manifest.get("holding_clock") != "signal_horizon":
        raise ValueError("Public backtest manifest must use holding_clock=signal_horizon.")
    if backtest_manifest.get("price_adjustment_mode") != "vendor_adjusted":
        raise ValueError("Public backtest must use vendor_adjusted prices.")

    training_git = run_manifest.get("environment", {}).get("git", {}) or {}
    backtest_git = backtest_manifest.get("environment", {}).get("git", {}) or {}
    if training_git.get("commit") != backtest_git.get("commit"):
        raise ValueError("Training and backtest manifests were produced from different commits.")
    if not allow_pre_release and backtest_git.get("dirty_tracked_worktree") is not False:
        raise ValueError("Public backtest evidence requires a clean source worktree.")

    training_data_sha = str(run_manifest.get("data", {}).get("fingerprint", {}).get("sha256") or "")
    backtest_data_sha = str(backtest_manifest.get("market_data", {}).get("sha256") or "")
    if not training_data_sha or training_data_sha != backtest_data_sha:
        raise ValueError("Training and backtest manifests use different market-data files.")

    prediction_sha = str(
        run_manifest.get("artifacts", {}).get("test_predictions_with_actual", {}).get("sha256") or ""
    )
    backtest_prediction_shas = {
        str(item.get("sha256") or "")
        for item in backtest_manifest.get("prediction_inputs", [])
        if isinstance(item, dict)
    }
    if not prediction_sha or prediction_sha not in backtest_prediction_shas:
        raise ValueError("Backtest input is not the canonical training prediction artifact.")

    require_matching_fingerprint(
        portfolio_summary_path,
        backtest_manifest.get("artifacts", {}).get("portfolio_grid_summary", {}),
        "backtest/portfolio_grid_summary",
    )
    if portfolio_df.empty:
        raise ValueError("Canonical public portfolio grid is empty.")
    # This call enforces required columns, signal-horizon clocks, and a non-empty
    # 20-bps public comparison slice before anything is copied to results/public.
    public_slice = format_public_portfolio_summary(portfolio_df)
    if public_slice.empty:
        raise ValueError("Canonical portfolio grid has no required 10d/20bps public rows.")
    skipped_paths = pd.to_numeric(
        portfolio_df.get(
            "skipped_incomplete_return_path_count",
            pd.Series(np.nan, index=portfolio_df.index),
        ),
        errors="coerce",
    )
    if skipped_paths.isna().any():
        raise ValueError(
            "Canonical portfolio grid must report skipped_incomplete_return_path_count "
            "for every configuration."
        )
    if skipped_paths.gt(0).any():
        affected_runs = portfolio_df.loc[
            skipped_paths.gt(0), "run_name"
        ].astype(str).head(5).tolist()
        raise ValueError(
            "Canonical public portfolio evidence cannot skip sleeves after observing "
            "an incomplete future return path. Resolve the market-data/security-master "
            f"gap before release. Affected runs include: {affected_runs}"
        )

    expected_grid_values = {
        "hold_days": {10, 20},
        "top_k": {10, 20, 30, 50},
        "cost_bps": {5.0, 10.0, 20.0, 50.0},
        "neutral_mode": {"unconstrained", "sector_neutral"},
    }
    for column, expected_values in expected_grid_values.items():
        if column == "neutral_mode":
            actual_values = set(portfolio_df[column].dropna().astype(str))
        else:
            actual_values = set(pd.to_numeric(portfolio_df[column], errors="coerce").dropna())
        if actual_values != expected_values:
            raise ValueError(
                f"Canonical portfolio grid has unexpected {column} values: "
                f"{sorted(actual_values)} != {sorted(expected_values)}"
            )
    expected_grid_count = 64
    if len(portfolio_df) != expected_grid_count or int(backtest_manifest.get("grid_run_count", -1)) != expected_grid_count:
        raise ValueError(
            "Canonical portfolio grid must contain exactly 64 completed configurations: "
            f"rows={len(portfolio_df)}, manifest={backtest_manifest.get('grid_run_count')}"
        )
    if "run_name" not in portfolio_df.columns or portfolio_df["run_name"].duplicated().any():
        raise ValueError("Canonical portfolio grid requires one unique run_name per configuration.")

    selected_internal_rows = select_public_portfolio_rows(portfolio_df)
    detailed_run_fingerprints = (
        (backtest_manifest.get("artifacts", {}) or {}).get("detailed_runs", {}) or {}
    )
    for run_name in selected_internal_rows["run_name"].astype(str):
        run_fingerprints = detailed_run_fingerprints.get(run_name, {}) or {}
        for filename in PUBLIC_PORTFOLIO_DETAIL_FILES:
            require_matching_fingerprint(
                portfolio_summary_path.parent / run_name / filename,
                run_fingerprints.get(filename, {}),
                f"backtest/{run_name}/{filename}",
            )

    grid_key_columns = ["hold_days", "top_k", "cost_bps", "neutral_mode"]
    if bool(portfolio_df.duplicated(subset=grid_key_columns, keep=False).any()):
        raise ValueError("Canonical portfolio grid contains duplicate parameter configurations.")
    if int(portfolio_df[grid_key_columns].drop_duplicates().shape[0]) != expected_grid_count:
        raise ValueError("Canonical portfolio grid does not contain the complete 64-cell Cartesian grid.")

    hold_days = pd.to_numeric(portfolio_df["hold_days"], errors="coerce")
    step_days = pd.to_numeric(portfolio_df["step_days"], errors="coerce")
    effective_days = pd.to_numeric(portfolio_df["effective_holding_days"], errors="coerce")
    signal_delay = pd.to_numeric(portfolio_df["signal_delay_days"], errors="coerce")
    borrow_cost = pd.to_numeric(portfolio_df["borrow_cost_bps"], errors="coerce")
    if bool((step_days != hold_days).fillna(True).any()):
        raise ValueError("Canonical portfolio grid requires step_days=hold_days.")
    if bool((signal_delay != 1).fillna(True).any()):
        raise ValueError("Canonical portfolio grid requires a one-trading-day signal delay.")
    if bool((borrow_cost != 0.0).fillna(True).any()):
        raise ValueError("Canonical portfolio grid requires zero borrow fee; stress tests must be exported separately.")
    if set(portfolio_df["holding_clock"].dropna().astype(str)) != {"signal_horizon"}:
        raise ValueError("Every canonical portfolio row must use holding_clock=signal_horizon.")
    if set(portfolio_df["price_adjustment_mode"].dropna().astype(str)) != {"vendor_adjusted"}:
        raise ValueError("Every canonical portfolio row must use vendor_adjusted prices.")
    expected_turnover_accounting = (
        "capital_scaled_full_sleeve_round_trip_without_cross_sleeve_netting"
    )
    if "turnover_accounting" not in portfolio_df.columns or set(
        portfolio_df["turnover_accounting"].dropna().astype(str)
    ) != {expected_turnover_accounting}:
        raise ValueError(
            "Canonical portfolio rows must use full sleeve entry/liquidation turnover "
            "without cross-sleeve netting."
        )
    gross_turnover = pd.to_numeric(portfolio_df["average_gross_turnover"], errors="coerce")
    expected_gross_turnover = pd.Series(4.0, index=portfolio_df.index)
    if bool((gross_turnover - expected_gross_turnover).abs().gt(1e-9).fillna(True).any()):
        raise ValueError(
            "Canonical step_days=hold_days long-short rows must report four units of "
            "round-trip gross turnover per executed sleeve (+1/-1 entry and liquidation)."
        )
    observed_cost_bps = pd.to_numeric(
        portfolio_df["average_turnover_cost_bps"], errors="coerce"
    )
    expected_cost_bps = pd.to_numeric(portfolio_df["cost_bps"], errors="coerce") * 4.0
    if bool((observed_cost_bps - expected_cost_bps).abs().gt(1e-9).fillna(True).any()):
        raise ValueError(
            "Canonical portfolio turnover cost does not reconcile to full round-trip traded notional."
        )
    mismatch = effective_days != (hold_days - signal_delay)
    if bool(mismatch.fillna(True).any()):
        raise ValueError("Portfolio effective holding days do not match the signal-horizon contract.")


def build_feature_family_summary(run_manifest: dict[str, Any]) -> pd.DataFrame:
    metadata = run_manifest.get("features", {}).get("metadata", {})
    counts = metadata.get("feature_counts", {})
    rows = [
        ("raw_feature", counts.get("raw_feature_count", 0)),
        ("fundamental_raw", counts.get("fundamental_raw_count", 0)),
        ("base_feature", counts.get("base_feature_count", 0)),
        ("advanced_feature", counts.get("advanced_feature_count", 0)),
        ("context_feature", counts.get("context_feature_count", 0)),
        ("alpha_feature", counts.get("alpha_feature_count", 0)),
        ("candidate_feature", counts.get("candidate_feature_count", 0)),
    ]
    return pd.DataFrame(rows, columns=["family", "feature_count"])


def build_selection_funnel(run_manifest: dict[str, Any]) -> pd.DataFrame:
    stage_counts = run_manifest.get("features", {}).get("selection_stage_counts", {})
    return pd.DataFrame(
        [{"stage": stage, "feature_count": int(count)} for stage, count in stage_counts.items()]
    )


def build_oos_metrics(run_manifest: dict[str, Any]) -> dict[str, Any]:
    data = run_manifest.get("data", {})
    metrics = dict(run_manifest.get("oos_metrics", {}))
    return {
        "oos_start": data.get("test_min_date"),
        "oos_end": data.get("test_max_date"),
        "oos_dates": data.get("test_date_count"),
        "oos_rows": data.get("test_rows"),
        **metrics,
        "interpretation": (
            "Final OOS audit only. These values are not model-selection inputs and "
            "must be interpreted together with docs/LIMITATIONS.md."
        ),
    }


def build_runtime_summary(run_manifest: dict[str, Any]) -> dict[str, Any]:
    """Extract human-readable timing evidence from the canonical manifest."""

    runtime = run_manifest.get("runtime", {}) or {}
    return {
        "total_runtime_seconds": run_manifest.get("total_runtime_seconds"),
        "stage_timing": runtime.get("stage_timing", []),
        "final_model_timing": runtime.get("final_model_timing", []),
        "interpretation": (
            "Cache-hit and cache-miss timings are not directly comparable. Inspect each stage status "
            "before using runtime values in public claims."
        ),
    }


def build_portfolio_anomaly_summary(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Keep concentration and extreme-return diagnostics for every grid row."""

    preferred_columns = [
        "run_name",
        "hold_days",
        "holding_clock",
        "effective_holding_days",
        "step_days",
        "top_k",
        "cost_bps",
        "neutral_mode",
        "turnover_accounting",
        "skipped_incomplete_return_path_count",
        "portfolio_total_return",
        "portfolio_sharpe",
        "first_half_total_return",
        "second_half_total_return",
        "top_5_net_return_days_simple_sum",
        "bottom_5_net_return_days_simple_sum",
        "top_5_instrument_abs_contribution_share",
        "max_abs_selected_stock_daily_return",
        "selected_stock_return_abs_gt_20pct_count",
        "selected_stock_return_abs_gt_50pct_count",
    ]
    available_columns = [column for column in preferred_columns if column in portfolio_df.columns]
    return portfolio_df[available_columns].copy()


def select_public_portfolio_rows(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Keep a compact 20-bps slice while retaining the full grid separately."""

    if portfolio_df.empty:
        return portfolio_df.copy()
    required = {
        "hold_days",
        "holding_clock",
        "effective_holding_days",
        "top_k",
        "cost_bps",
        "neutral_mode",
    }
    if not required.issubset(portfolio_df.columns):
        raise ValueError(f"Portfolio summary is missing columns: {sorted(required - set(portfolio_df.columns))}")
    invalid_clocks = sorted(
        set(portfolio_df["holding_clock"].dropna().astype(str)) - {"signal_horizon"}
    )
    if invalid_clocks:
        raise ValueError(
            "Public evidence only accepts holding_clock=signal_horizon; "
            f"found {invalid_clocks}. Export legacy execution-horizon runs separately."
        )
    mask = (
        (pd.to_numeric(portfolio_df["hold_days"], errors="coerce") == 10)
        & (pd.to_numeric(portfolio_df["top_k"], errors="coerce").isin([20, 50]))
        & (pd.to_numeric(portfolio_df["cost_bps"], errors="coerce") == 20.0)
    )
    return portfolio_df.loc[mask].copy().reset_index(drop=True)


def format_public_portfolio_summary(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize internal grid columns into the stable public evidence schema."""

    selected = select_public_portfolio_rows(portfolio_df)
    if selected.empty:
        return selected
    selected = selected.copy()
    selected["portfolio"] = selected.apply(
        lambda row: (
            f"top{int(row['top_k'])}_{row['neutral_mode']}_{float(row['cost_bps']):g}bps"
        ),
        axis=1,
    )
    rename_map = {
        "step_days": "rebalance_days",
        "cost_bps": "transaction_cost_bps",
        "borrow_cost_bps": "borrow_cost_bps_annual",
        "portfolio_total_return": "cumulative_return",
        "benchmark_total_return": "benchmark_return",
        "relative_wealth_vs_equal_weight_long_only": (
            "relative_wealth_vs_equal_weight_long_only"
        ),
        "portfolio_sharpe": "sharpe",
        "portfolio_max_drawdown": "max_drawdown",
    }
    selected = selected.rename(columns=rename_map)
    public_columns = [
        "portfolio",
        "top_k",
        "neutral_mode",
        "hold_days",
        "holding_clock",
        "effective_holding_days",
        "rebalance_days",
        "signal_delay_days",
        "transaction_cost_bps",
        "borrow_cost_bps_annual",
        "price_adjustment_mode",
        "daily_count",
        "invested_day_count",
        "cash_day_count",
        "rebalance_count",
        "cumulative_return",
        "benchmark_return",
        "relative_wealth_vs_equal_weight_long_only",
        "sharpe",
        "max_drawdown",
        "average_gross_turnover",
        "average_turnover_cost_bps",
        "total_turnover_cost",
        "skipped_incomplete_return_path_count",
        "turnover_accounting",
    ]
    missing = [column for column in public_columns if column not in selected.columns]
    if missing:
        raise ValueError(f"Portfolio summary cannot be exported; missing columns: {missing}")
    return selected[public_columns].copy()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a compact table without making report generation depend on tabulate."""

    if df.empty:
        return "_No rows available._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```csv\n" + df.to_csv(index=False) + "```"


def write_public_package_readme(
    output_path: Path,
    *,
    experiment_id: str,
    public_status: str,
    run_manifest: dict[str, Any],
    model_summary_df: pd.DataFrame,
    public_portfolio_df: pd.DataFrame,
) -> None:
    """Replace any legacy package README with one derived from verified artifacts."""

    data = run_manifest.get("data", {}) or {}
    configuration = run_manifest.get("effective_configuration", {}) or {}
    source_commit = (
        (run_manifest.get("environment", {}).get("git", {}) or {}).get("commit")
    )
    oos_metrics = build_oos_metrics(run_manifest)
    model_columns = [
        column
        for column in ["model", "pearson_ic_mean", "spearman_ic_mean", "rmse", "mae"]
        if column in model_summary_df.columns
    ]
    oos_rows = pd.DataFrame(
        [
            {"metric": key, "value": oos_metrics.get(key)}
            for key in [
                "pearson_corr",
                "spearman_corr",
                "pearson_ic_mean",
                "spearman_ic_mean",
                "rmse",
                "mae",
                "long_short_spread",
                "prediction_coverage_ratio",
            ]
            if key in oos_metrics
        ]
    )
    report = f"""# Canonical US300 Evidence Package

> Public status: `{public_status}`. A release candidate still requires CI and manual
> review of the data-quality, anomaly, and limitations files.

## Contract

- Experiment: `{experiment_id}`
- Universe: `{data.get('universe_label', data.get('universe'))}`
- Instruments: `{data.get('instrument_count')}`
- In-sample dates: `{data.get('train_min_date')}` to `{data.get('train_max_date')}`
- Final OOS dates: `{data.get('test_min_date')}` to `{data.get('test_max_date')}`
- Target: `{configuration.get('target_column')}`
- Models: `{', '.join(configuration.get('models', []))}`
- Price mode: `{configuration.get('price_adjustment_mode')}`
- Source commit: `{source_commit}`

## Walk-Forward Model Summary

{dataframe_to_markdown(model_summary_df[model_columns].copy() if model_columns else pd.DataFrame())}

## Final OOS Audit

{dataframe_to_markdown(oos_rows)}

The label spread above is a same-date diagnostic and is not cumulative portfolio return.

## Cost-Aware Portfolio Slice

{dataframe_to_markdown(public_portfolio_df)}

Each sleeve pays both entry and liquidation turnover. Costs are capital-scaled,
cross-sleeve netting is not assumed, and incomplete selected-stock return paths are
reported instead of silently dropped.

## Evidence Files

The directory includes complete fold metrics, final OOS metrics, selected features,
model weights, runtime, data-quality and corporate-action audits, the complete
portfolio grid, anomaly diagnostics, source/backtest manifests, and SHA256-backed
provenance. `portfolio_runs/` contains the daily ledger, positions, turnover, costs,
sector exposure, and attribution for every displayed 20 bps row. Raw market data,
predictions, model binaries, and caches are excluded.

Read [`../../../docs/LIMITATIONS.md`](../../../docs/LIMITATIONS.md) before citing any
number. A release-candidate status does not establish stable tradability.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    training_output = resolve_project_path(args.training_output)
    model_dir = resolve_project_path(args.model_dir)
    portfolio_summary = resolve_project_path(args.portfolio_summary)
    backtest_manifest_path = resolve_project_path(args.backtest_manifest)
    public_dir = resolve_project_path(args.public_dir)

    for filename in REQUIRED_TRAINING_FILES:
        if not (training_output / filename).is_file():
            raise FileNotFoundError(f"Required training artifact is missing: {training_output / filename}")

    run_manifest = read_json(training_output / "run_manifest.json")
    validate_training_contract(
        training_output,
        model_dir,
        run_manifest,
        allow_pre_release=bool(args.allow_pre_release),
    )

    portfolio_df = pd.DataFrame()
    backtest_manifest: dict[str, Any] | None = None
    if portfolio_summary.is_file():
        if not backtest_manifest_path.is_file():
            raise FileNotFoundError(
                f"Portfolio summary exists without its backtest manifest: {backtest_manifest_path}"
            )
        portfolio_df = pd.read_csv(portfolio_summary)
        backtest_manifest = read_json(backtest_manifest_path)
        validate_backtest_contract(
            run_manifest=run_manifest,
            backtest_manifest=backtest_manifest,
            portfolio_summary_path=portfolio_summary,
            portfolio_df=portfolio_df,
            allow_pre_release=bool(args.allow_pre_release),
        )
    elif not args.allow_missing_backtest:
        raise FileNotFoundError(
            "Canonical public evidence requires the portfolio grid. Run the canonical backtest first, "
            "or use --allow-missing-backtest only for a local pre-release review."
        )

    git_state = run_manifest.get("environment", {}).get("git", {})
    data_quality = run_manifest.get("data", {}).get("data_quality", {}) or {}
    corporate_action_review_required = bool(
        (data_quality.get("corporate_action_audit", {}) or {}).get("manual_review_required", False)
    )
    if backtest_manifest is None:
        public_status = "pre_release_backtest_missing"
    elif git_state.get("dirty_tracked_worktree") is not False:
        public_status = "pre_release_source_worktree_not_clean"
    elif not git_state.get("commit"):
        public_status = "pre_release_source_commit_missing"
    elif corporate_action_review_required:
        public_status = "pre_release_corporate_action_review_required"
    else:
        public_status = "release_candidate_requires_review"

    # No existing public file is touched until all source contracts above pass.
    # All outputs are written to a clean staging directory so legacy evidence
    # cannot survive beside the new run by accident.
    staging_dir = prepare_public_staging_directory(public_dir)
    for filename in REQUIRED_TRAINING_FILES:
        copy_required(training_output / filename, staging_dir / filename)

    # Preserve the complete sanitized run manifest as the primary provenance record.
    copy_required(training_output / "run_manifest.json", staging_dir / "source_run_manifest.json")
    build_feature_family_summary(run_manifest).to_csv(
        staging_dir / "feature_family_summary.csv", index=False
    )
    build_selection_funnel(run_manifest).to_csv(
        staging_dir / "feature_selection_funnel.csv", index=False
    )
    (staging_dir / "oos_metrics.json").write_text(
        dumps_strict_json(build_oos_metrics(run_manifest)),
        encoding="utf-8",
    )
    (staging_dir / "data_summary.json").write_text(
        dumps_strict_json(run_manifest.get("data", {})),
        encoding="utf-8",
    )
    (staging_dir / "runtime_summary.json").write_text(
        dumps_strict_json(build_runtime_summary(run_manifest)),
        encoding="utf-8",
    )

    for filename in ("selected_features.csv", "selected_feature_scores.csv", "model_weights.csv"):
        copy_required(model_dir / filename, staging_dir / filename)

    backtest_exported = backtest_manifest is not None
    public_portfolio_run_names: list[str] = []
    if backtest_exported:
        portfolio_df.to_csv(staging_dir / "portfolio_grid_summary.csv", index=False)
        public_portfolio_df = format_public_portfolio_summary(portfolio_df)
        public_portfolio_df.to_csv(
            staging_dir / "portfolio_cost_summary.csv", index=False
        )
        build_portfolio_anomaly_summary(portfolio_df).to_csv(
            staging_dir / "portfolio_anomaly_summary.csv", index=False
        )
        copy_required(backtest_manifest_path, staging_dir / "backtest_run_manifest.json")
        selected_internal_rows = select_public_portfolio_rows(portfolio_df)
        public_portfolio_run_names = selected_internal_rows["run_name"].astype(str).tolist()
        for run_name in public_portfolio_run_names:
            for filename in PUBLIC_PORTFOLIO_DETAIL_FILES:
                copy_required(
                    portfolio_summary.parent / run_name / filename,
                    staging_dir / "portfolio_runs" / run_name / filename,
                )
    else:
        public_portfolio_df = pd.DataFrame()

    compact_manifest = {
        "experiment_id": args.experiment_id,
        "public_status": public_status,
        "source_git": git_state,
        "reproduction_command": "scripts/run_canonical_us300.sh",
        "backtest_command": "scripts/run_canonical_us300_backtest.sh",
        "evidence_export_command": "python scripts/export_public_evidence.py",
        "data": run_manifest.get("data", {}),
        "effective_configuration": run_manifest.get("effective_configuration", {}),
        "features": {
            "candidate_count": run_manifest.get("features", {}).get("candidate_count"),
            "selected_count": run_manifest.get("features", {}).get("selected_count"),
        },
        "validation": run_manifest.get("validation", {}),
        "runtime_seconds": run_manifest.get("total_runtime_seconds"),
        "backtest_exported": backtest_exported,
        "public_portfolio_run_names": public_portfolio_run_names,
        "release_note": (
            "A release candidate still requires CI, README metric reconciliation, "
            "and manual limitations review."
        ),
    }
    (staging_dir / "experiment_manifest.json").write_text(
        dumps_strict_json(compact_manifest),
        encoding="utf-8",
    )
    model_summary_df = pd.read_csv(training_output / "walk_forward_model_summary.csv")
    write_public_package_readme(
        staging_dir / "README.md",
        experiment_id=args.experiment_id,
        public_status=public_status,
        run_manifest=run_manifest,
        model_summary_df=model_summary_df,
        public_portfolio_df=public_portfolio_df,
    )
    replace_public_directory(public_dir, staging_dir)
    print(f"[Evidence] Exported compact package to: {public_dir}")
    print(f"[Evidence] Public status: {public_status}")


if __name__ == "__main__":
    main()
