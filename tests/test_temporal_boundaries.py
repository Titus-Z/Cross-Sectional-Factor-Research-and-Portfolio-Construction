"""Regression tests for the leakage and data-provenance boundaries fixed before release."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from factor_mining_workspace.mined_factor_model_ablation import load_factor_zoo
from src.alpha191 import (
    CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS,
    generate_alpha191_features,
)
from src.data_loader import load_daily_data
from src.data_quality import build_corporate_action_audit, build_universe_coverage_audit
from src.feature_cache import build_feature_cache_key
from src.feature_generator import generate_feature_matrix
from src.feature_selector import FeatureSelector
from src.fmp_fundamentals import _build_quarterly_fundamental_panel, _normalize_statement_frame
from src.long_short_backtest import (
    LongShortBacktestConfig,
    compute_sleeve_lifecycle_trade_summary,
    run_long_short_backtest,
    scale_trade_summary_to_portfolio,
)
from src.portfolio import load_market_snapshot_frame, resolve_holding_window
from src.preprocessing import RAW_MARKET_CAP_EXPOSURE_COLUMN, apply_cross_sectional_preprocessing
from src.provenance import dumps_strict_json
from src.time_series_pipeline import purge_training_label_overlap
from src.utils import decay_linear
from src.validation import calculate_model_weights, generate_walk_forward_folds
from src.validation_cache import build_validation_cache_key


class TemporalBoundaryTests(unittest.TestCase):
    def test_fundamental_panel_uses_latest_source_availability_date(self) -> None:
        """Merged statement fields cannot appear before the last source filing is public."""

        income = _normalize_statement_frame(
            [
                {
                    "date": "2023-12-31",
                    "acceptedDate": "2024-02-01",
                    "revenue": 100.0,
                    "netIncome": 10.0,
                    "eps": 1.0,
                }
            ],
            symbol="AAA",
            source_name="income_statement",
        )
        balance = _normalize_statement_frame(
            [
                {
                    "date": "2023-12-31",
                    "acceptedDate": "2024-02-15",
                    "totalAssets": 500.0,
                    "totalStockholdersEquity": 200.0,
                }
            ],
            symbol="AAA",
            source_name="balance_sheet",
        )

        panel = _build_quarterly_fundamental_panel("AAA", income, balance)

        self.assertEqual(pd.Timestamp(panel.loc[0, "effective_date"]), pd.Timestamp("2024-02-15"))

    def test_loader_preserves_missing_market_observations(self) -> None:
        """Neither an early nor an interior OHLCV gap may be manufactured by filling."""

        dates = pd.date_range("2024-01-02", periods=15, freq="B")
        frame = pd.DataFrame(
            {
                "instrument_id": "AAA",
                "date": dates,
                "open": [
                    np.nan if index in {0, 5} else 100.0 + index
                    for index in range(len(dates))
                ],
                "high": [102.0 + index for index in range(len(dates))],
                "low": [98.0 + index for index in range(len(dates))],
                "close": [100.0 + index for index in range(len(dates))],
                "volume": [np.nan if index == 8 else 1_000_000.0 for index in range(len(dates))],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "daily.csv"
            frame.to_csv(csv_path, index=False)
            loaded = load_daily_data(csv_path)

        self.assertTrue(pd.isna(loaded.loc[0, "open"]))
        self.assertEqual(float(loaded.loc[1, "open"]), 101.0)
        self.assertTrue(pd.isna(loaded.loc[5, "open"]))
        self.assertTrue(pd.isna(loaded.loc[8, "volume"]))

    def test_loader_preserves_fundamental_availability_metadata(self) -> None:
        """Filing timestamps must remain auditable after loading the merged CSV."""

        dates = pd.date_range("2024-02-15", periods=12, freq="B")
        frame = pd.DataFrame(
            {
                "instrument_id": "AAA",
                "date": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000.0,
                "effective_date": "2024-02-15",
                "filing_date": "2024-02-14",
                "accepted_date": "2024-02-15",
                "fiscal_period": "Q4",
                "eps": 1.25,
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "fundamental_daily.csv"
            frame.to_csv(csv_path, index=False)
            loaded = load_daily_data(csv_path)

        self.assertEqual(pd.Timestamp(loaded.loc[0, "effective_date"]), pd.Timestamp("2024-02-15"))
        self.assertEqual(str(loaded.loc[0, "fiscal_period"]), "Q4")

    def test_market_cap_source_is_preserved_per_row(self) -> None:
        """One observed market cap must not relabel every missing row as provided."""

        frame = pd.DataFrame(
            {
                "instrument_id": ["AAA", "BBB"],
                "date": ["2024-01-02", "2024-01-02"],
                "open": [10.0, 20.0],
                "high": [11.0, 21.0],
                "low": [9.0, 19.0],
                "close": [10.5, 20.5],
                "volume": [1000.0, 2000.0],
                "market_cap": [1_000_000.0, np.nan],
                "market_cap_source": ["fmp", "fmp"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "market_cap_source.csv"
            frame.to_csv(csv_path, index=False)
            loaded = load_daily_data(csv_path)

        sources = loaded.set_index("instrument_id")["market_cap_source"].to_dict()
        self.assertEqual(sources["AAA"], "fmp")
        self.assertEqual(sources["BBB"], "missing")

    def test_returns_are_built_in_feature_layer_after_raw_loading(self) -> None:
        """The CSV loader must not create a model feature before the raw split."""

        dates = pd.date_range("2024-01-02", periods=15, freq="B")
        frame = pd.DataFrame(
            {
                "instrument_id": "AAA",
                "date": dates,
                "open": np.arange(100.0, 115.0),
                "high": np.arange(101.0, 116.0),
                "low": np.arange(99.0, 114.0),
                "close": np.arange(100.0, 115.0),
                "volume": 1_000_000.0,
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "daily.csv"
            frame.to_csv(csv_path, index=False)
            loaded = load_daily_data(csv_path)

        self.assertNotIn("log_return", loaded.columns)
        featured, feature_columns, _ = generate_feature_matrix(loaded, alpha_factor_names=[])
        self.assertIn("log_return", featured.columns)
        self.assertIn("log_return", feature_columns)
        for scale_dependent_column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "close_ma_5",
            "ema_close_12",
            "boll_upper_20",
        ]:
            self.assertNotIn(scale_dependent_column, feature_columns)

    def test_canonical_alpha_subset_is_invariant_to_per_stock_price_rescaling(self) -> None:
        """Public Alpha formulas cannot inherit a later vendor price-level scale."""

        dates = pd.date_range("2024-01-02", periods=45, freq="B")
        rows = []
        multipliers = {"AAA": 0.1, "BBB": 7.0, "CCC": 125.0}
        for instrument_index, instrument_id in enumerate(multipliers):
            for date_index, date_value in enumerate(dates):
                close = 80.0 + 10.0 * instrument_index + 0.3 * date_index
                rows.append(
                    {
                        "instrument_id": instrument_id,
                        "date": date_value,
                        "open": close * (0.995 + 0.0005 * ((date_index + instrument_index) % 5)),
                        "high": close * 1.015,
                        "low": close * 0.985,
                        "close": close,
                        "vwap": close * 1.001,
                        "volume": 1_000_000.0 + 10_000.0 * date_index + 50_000.0 * instrument_index,
                        "turnover": close * (1_000_000.0 + 10_000.0 * date_index + 50_000.0 * instrument_index),
                    }
                )
        original = pd.DataFrame(rows)
        rescaled = original.copy()
        scale = rescaled["instrument_id"].map(multipliers).astype(float)
        for column in ["open", "high", "low", "close", "vwap"]:
            rescaled[column] = rescaled[column] * scale

        original_factors = generate_alpha191_features(
            original,
            factor_names=list(CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS),
        )
        rescaled_factors = generate_alpha191_features(
            rescaled,
            factor_names=list(CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS),
        )

        np.testing.assert_allclose(
            original_factors.to_numpy(dtype=float),
            rescaled_factors.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )

    def test_missing_market_cap_never_creates_synthetic_size_features(self) -> None:
        """OHLCV alone cannot support market-cap or size-neutralization claims."""

        dates = pd.date_range("2024-01-02", periods=15, freq="B")
        frame = pd.DataFrame(
            {
                "instrument_id": "AAA",
                "date": dates,
                "open": np.arange(100.0, 115.0),
                "high": np.arange(101.0, 116.0),
                "low": np.arange(99.0, 114.0),
                "close": np.arange(100.0, 115.0),
                "volume": 1_000_000.0,
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "daily.csv"
            frame.to_csv(csv_path, index=False)
            loaded = load_daily_data(csv_path)

        self.assertTrue(loaded["market_cap"].isna().all())
        featured, feature_columns, _ = generate_feature_matrix(loaded, alpha_factor_names=[])
        self.assertNotIn("market_cap", feature_columns)
        self.assertNotIn("shares_outstanding_proxy", featured.columns)
        self.assertNotIn("turnover_rate_proxy", featured.columns)

    def test_repeated_preprocessing_reuses_raw_market_cap_exposure(self) -> None:
        """A mined-factor pass must not neutralize against standardized market cap."""

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"] * 4),
                "instrument_id": ["AAA", "BBB", "CCC", "DDD"],
                "sector": ["Tech", "Tech", "Health", "Health"],
                "market_cap": [100.0, 200.0, 400.0, 800.0],
                "baseline_signal": [1.0, 2.0, 3.0, 5.0],
            }
        )
        original_market_cap = frame["market_cap"].copy()

        first_pass, _ = apply_cross_sectional_preprocessing(
            frame,
            feature_columns=["market_cap", "baseline_signal"],
        )
        first_pass["mined_signal"] = [2.0, 1.0, 4.0, 3.0]
        second_pass, second_summary = apply_cross_sectional_preprocessing(
            first_pass,
            feature_columns=["mined_signal"],
        )

        pd.testing.assert_series_equal(
            pd.to_numeric(second_pass[RAW_MARKET_CAP_EXPOSURE_COLUMN]),
            original_market_cap,
            check_names=False,
        )
        self.assertTrue(bool(second_summary["size_neutralization_used"]))
        self.assertEqual(float(second_summary["market_cap_coverage_ratio"]), 1.0)

    def test_outer_split_purges_one_target_horizon(self) -> None:
        """The final target-horizon observations per stock cannot be training rows."""

        dates = pd.date_range("2024-01-02", periods=20, freq="B")
        frame = pd.DataFrame(
            [(symbol, date) for symbol in ["AAA", "BBB"] for date in dates],
            columns=["instrument_id", "date"],
        )
        purged, summary = purge_training_label_overlap(frame, target_horizon=3)

        self.assertEqual(summary["purged_date_count"], 3)
        self.assertEqual(summary["purged_row_count"], 6)
        self.assertEqual(pd.to_datetime(purged["date"]).max(), dates[-4])

    def test_purge_handles_irregular_instrument_calendars(self) -> None:
        """A stock with missing dates still loses its final N own observations."""

        dates = pd.date_range("2024-01-02", periods=12, freq="B")
        aaa = pd.DataFrame({"instrument_id": "AAA", "date": dates})
        bbb_dates = dates.delete([7, 9])
        bbb = pd.DataFrame({"instrument_id": "BBB", "date": bbb_dates})
        frame = pd.concat([aaa, bbb], ignore_index=True)

        purged, summary = purge_training_label_overlap(frame, target_horizon=3)

        self.assertEqual(summary["purged_row_count"], 6)
        self.assertEqual(len(purged[purged["instrument_id"] == "AAA"]), len(aaa) - 3)
        self.assertEqual(len(purged[purged["instrument_id"] == "BBB"]), len(bbb) - 3)
        self.assertEqual(
            pd.to_datetime(purged[purged["instrument_id"] == "BBB"]["date"]).max(),
            bbb_dates[-4],
        )

    def test_forward_target_does_not_jump_over_missing_market_date(self) -> None:
        """A missing t+N close must produce NaN rather than a later-horizon label."""

        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        frame = pd.DataFrame(
            {
                "instrument_id": ["AAA"] * 5 + ["BBB"] * 4,
                "date": list(dates) + [dates[0], dates[1], dates[3], dates[4]],
                "close": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 23.0, 24.0],
            }
        )
        from src.data_loader import add_forward_return_targets

        labelled = add_forward_return_targets(frame, horizons=(2,))
        bbb_first = labelled[
            (labelled["instrument_id"] == "BBB") & (labelled["date"] == dates[0])
        ].iloc[0]
        aaa_first = labelled[
            (labelled["instrument_id"] == "AAA") & (labelled["date"] == dates[0])
        ].iloc[0]

        self.assertTrue(pd.isna(bbb_first["y_2d"]))
        self.assertAlmostEqual(float(aaa_first["y_2d"]), 0.2)

    def test_walk_forward_fold_purges_label_overlap(self) -> None:
        """Every validation fold must have a target-horizon gap after training."""

        dates = pd.date_range("2024-01-02", periods=36, freq="B")
        frame = pd.DataFrame({"instrument_id": "AAA", "date": dates, "y": 0.0})
        folds = generate_walk_forward_folds(frame, n_splits=3, purge_days=5)

        for train_fold, valid_fold, summary in folds:
            self.assertEqual(summary["purged_train_date_count"], 5)
            train_max = pd.to_datetime(train_fold["date"]).max()
            valid_min = pd.to_datetime(valid_fold["date"]).min()
            self.assertGreater(valid_min, train_max)

    def test_adjusted_price_is_consistent_between_labels_and_portfolio(self) -> None:
        """A two-for-one split must not create a false -50% model/backtest return."""

        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        frame = pd.DataFrame(
            {
                "instrument_id": "AAA",
                "date": dates,
                "open": [100.0, 102.0, 51.0, 52.0, 53.0],
                "high": [101.0, 103.0, 52.0, 53.0, 54.0],
                "low": [99.0, 101.0, 50.0, 51.0, 52.0],
                "close": [100.0, 102.0, 51.0, 52.0, 53.0],
                "volume": 1_000_000.0,
                "adjustment": [0.5, 0.5, 1.0, 1.0, 1.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "split_daily.csv"
            frame.to_csv(csv_path, index=False)
            loaded = load_daily_data(csv_path, price_adjustment_mode="vendor_adjusted")
            market = load_market_snapshot_frame(csv_path, price_adjustment_mode="vendor_adjusted")

        self.assertAlmostEqual(float(loaded.loc[1, "close"]), 51.0)
        self.assertAlmostEqual(float(loaded.loc[2, "close"]), 51.0)
        self.assertAlmostEqual(float(loaded.loc[0, "turnover"]), 100_000_000.0)
        self.assertAlmostEqual(float(loaded.loc[1, "y_1d"]), 0.0)
        self.assertAlmostEqual(float(market.loc[2, "daily_close_return"]), 0.0)

    def test_corporate_action_audit_distinguishes_raw_and_adjusted_split_returns(self) -> None:
        """A split-like raw jump should be documented as removed by the adjustment."""

        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        adjusted_panel = pd.DataFrame(
            {
                "instrument_id": "AAA",
                "date": dates,
                "close": [50.0, 51.0, 51.0],
                "adjustment": [0.5, 0.5, 1.0],
            }
        )
        events, summary = build_corporate_action_audit(
            adjusted_panel,
            price_adjustment_mode="vendor_adjusted",
        )

        self.assertEqual(len(events), 1)
        self.assertTrue(bool(events.loc[0, "raw_jump_removed_by_adjustment"]))
        self.assertFalse(bool(events.loc[0, "residual_adjusted_return_gt_20pct"]))
        self.assertEqual(summary["raw_jump_removed_by_adjustment_count"], 1)

    def test_universe_coverage_audit_flags_early_ending_history(self) -> None:
        """An instrument ending far before the panel must remain visible for review."""

        dates = pd.date_range("2024-01-02", periods=60, freq="B")
        full = pd.DataFrame({"instrument_id": "AAA", "date": dates})
        early_end = pd.DataFrame({"instrument_id": "BBB", "date": dates[:20]})
        audit, summary = build_universe_coverage_audit(pd.concat([full, early_end], ignore_index=True))
        bbb = audit[audit["instrument_id"] == "BBB"].iloc[0]

        self.assertTrue(bool(bbb["early_end_flag"]))
        self.assertIn("possible_delisting", str(bbb["review_reason"]))
        self.assertEqual(summary["early_end_instrument_count"], 1)

    def test_cache_key_separates_raw_and_adjusted_prices(self) -> None:
        """Raw and adjusted price experiments cannot reuse one feature cache."""

        common = {
            "data_path": Path("data/example.csv"),
            "sample_start_date": "2024-01-01",
            "oos_start_date": "2026-01-01",
            "test_size": 0.2,
            "target_horizon": 10,
            "history_window": 260,
            "alpha_factor_names": ["alpha001"],
        }
        adjusted_key = build_feature_cache_key(
            **common,
            price_adjustment_mode="vendor_adjusted",
        )
        raw_key = build_feature_cache_key(
            **common,
            price_adjustment_mode="raw",
        )
        self.assertNotEqual(adjusted_key, raw_key)

    def test_factor_zoo_loader_rejects_label_formula(self) -> None:
        """A hand-edited zoo cannot bypass the searcher's forbidden-field gate."""

        factor_zoo = pd.DataFrame(
            {
                "candidate_id": ["unsafe_candidate"],
                "formula": ["rank(y_10d)"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            factor_zoo_path = Path(tmp_dir) / "factor_zoo.csv"
            factor_zoo.to_csv(factor_zoo_path, index=False)
            with self.assertRaisesRegex(ValueError, "forbidden fields"):
                load_factor_zoo(factor_zoo_path)

    def test_factor_zoo_loader_enforces_canonical_feature_allowlist(self) -> None:
        """Strict model ablation rejects scale-dependent intermediate fields."""

        factor_zoo = pd.DataFrame(
            {
                "candidate_id": ["raw_price_candidate"],
                "formula": ["rank(close)"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            factor_zoo_path = Path(tmp_dir) / "factor_zoo.csv"
            factor_zoo.to_csv(factor_zoo_path, index=False)
            with self.assertRaisesRegex(ValueError, "non-canonical feature fields"):
                load_factor_zoo(
                    factor_zoo_path,
                    allowed_formula_fields={"return_std_20"},
                )

    def test_validation_cache_key_separates_price_modes(self) -> None:
        """Strict factor ablation cannot reuse validation from another price convention."""

        common = {
            "data_path": Path("data/example.csv"),
            "sample_start_date": "2024-01-01",
            "oos_start_date": "2026-01-01",
            "test_size": 0.2,
            "target_horizon": 10,
            "history_window": 260,
            "feature_columns": ["momentum_10"],
            "model_names": ["ridge"],
            "selector_config": {"top_n": 1},
            "n_splits": 3,
            "random_state": 42,
            "score_metric": "pearson_ic_mean",
            "apply_preprocessing": True,
            "apply_neutralization": True,
            "winsorize_quantile": 0.01,
        }
        adjusted_key = build_validation_cache_key(
            **common,
            price_adjustment_mode="vendor_adjusted",
        )
        raw_key = build_validation_cache_key(
            **common,
            price_adjustment_mode="raw",
        )
        self.assertNotEqual(adjusted_key, raw_key)

    def test_feature_selector_handles_large_finite_values_without_overflow(self) -> None:
        """Variance and correlation filters must remain defined for extreme formulas."""

        selector = FeatureSelector(
            missing_threshold=0.5,
            variance_threshold=0.001,
            correlation_threshold=0.95,
            top_n=2,
        )
        features = pd.DataFrame(
            {
                "huge_signal": [1e200, 2e200, 3e200, 4e200],
                "ordinary_signal": [0.1, -0.2, 0.3, -0.4],
                "invalid_signal": [np.inf, np.inf, np.inf, np.inf],
                "constant_signal": [1.0, 1.0, 1.0, 1.0],
            }
        )
        target = pd.Series([0.01, -0.02, 0.03, -0.04])

        transformed = selector.fit_transform(features, target)

        self.assertIn("huge_signal", selector.columns_after_variance_)
        self.assertNotIn("invalid_signal", selector.columns_after_missing_)
        self.assertNotIn("constant_signal", selector.columns_after_variance_)
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())

    def test_validation_cache_key_separates_random_seeds(self) -> None:
        """Stochastic model validation cannot reuse results from another seed."""

        common = {
            "data_path": Path("data/example.csv"),
            "sample_start_date": "2024-01-01",
            "oos_start_date": "2026-01-01",
            "test_size": 0.2,
            "target_horizon": 10,
            "history_window": 260,
            "feature_columns": ["momentum_10"],
            "model_names": ["random_forest"],
            "selector_config": {"top_n": 1},
            "n_splits": 3,
            "score_metric": "pearson_ic_mean",
            "apply_preprocessing": True,
            "apply_neutralization": True,
            "winsorize_quantile": 0.01,
        }
        seed_7_key = build_validation_cache_key(**common, random_state=7)
        seed_42_key = build_validation_cache_key(**common, random_state=42)

        self.assertNotEqual(seed_7_key, seed_42_key)

    def test_all_nan_validation_scores_fall_back_to_equal_weights(self) -> None:
        """Undefined validation IC must not create NaN ensemble predictions."""

        summary = pd.DataFrame(
            {
                "model": ["ridge", "lasso"],
                "pearson_ic_mean": [np.nan, np.nan],
            }
        )
        weights = calculate_model_weights(summary, score_metric="pearson_ic_mean")

        self.assertEqual(weights, {"ridge": 0.5, "lasso": 0.5})

    def test_public_json_encodes_nonfinite_metrics_as_null(self) -> None:
        """Public evidence JSON must be readable by strict non-Python parsers."""

        encoded = dumps_strict_json(
            {"valid": 1.5, "nan_metric": float("nan"), "inf_metric": float("inf")}
        )
        decoded = json.loads(encoded)

        self.assertEqual(decoded["valid"], 1.5)
        self.assertIsNone(decoded["nan_metric"])
        self.assertIsNone(decoded["inf_metric"])

    def test_validation_weights_are_shrunk_toward_equal(self) -> None:
        """Tiny validation differences must not create an all-or-nothing ensemble."""

        summary = pd.DataFrame(
            {
                "model": ["ridge", "lasso"],
                "pearson_ic_mean": [0.03, 0.02],
            }
        )
        weights = calculate_model_weights(summary, score_metric="pearson_ic_mean")

        self.assertAlmostEqual(weights["ridge"], 0.55)
        self.assertAlmostEqual(weights["lasso"], 0.45)

    def test_decay_linear_renormalizes_observed_values_without_filling_gaps(self) -> None:
        """A missing historical observation must not be copied from either neighbour."""

        values = pd.Series([1.0, np.nan, 3.0])
        result = decay_linear(values, window=3)

        # Original positions have weights 1, 2, 3. Only values 1 and 3 exist,
        # so the valid weighted mean is (1*1 + 3*3) / (1+3) = 2.5.
        self.assertAlmostEqual(float(result.iloc[-1]), 2.5)

    def test_signal_horizon_ends_at_the_forward_label_endpoint(self) -> None:
        """A y_10d signal with one-day close execution must end at t+10."""

        dates = list(pd.date_range("2024-01-02", periods=15, freq="B"))
        window = resolve_holding_window(
            dates,
            signal_date=dates[0],
            signal_delay_days=1,
            hold_days=10,
            holding_clock="signal_horizon",
        )

        self.assertIsNotNone(window)
        execution_date, end_date, effective_days = window
        self.assertEqual(execution_date, dates[1])
        self.assertEqual(end_date, dates[10])
        self.assertEqual(effective_days, 9)
        accrued_return_dates = [date for date in dates if execution_date < date <= end_date]
        self.assertEqual(accrued_return_dates, dates[2:11])

    def test_execution_horizon_is_kept_as_an_explicit_sensitivity_mode(self) -> None:
        """The historical clock may be reproduced without contaminating canonical runs."""

        dates = list(pd.date_range("2024-01-02", periods=15, freq="B"))
        window = resolve_holding_window(
            dates,
            signal_date=dates[0],
            signal_delay_days=1,
            hold_days=10,
            holding_clock="execution_horizon",
        )

        self.assertIsNotNone(window)
        execution_date, end_date, effective_days = window
        self.assertEqual(execution_date, dates[1])
        self.assertEqual(end_date, dates[11])
        self.assertEqual(effective_days, 10)

    def test_overlapping_sleeve_turnover_is_scaled_to_portfolio_capital(self) -> None:
        """Two half-capital sleeves cannot each report full-portfolio turnover."""

        sleeve_summary = compute_sleeve_lifecycle_trade_summary(
            target_weights={"AAA": 1.0, "BBB": -1.0},
            cost_bps=20.0,
            borrow_cost_bps=0.0,
            borrow_accrual_days=9,
        )
        portfolio_summary = scale_trade_summary_to_portfolio(
            sleeve_summary,
            sleeve_capital_weight=0.5,
        )

        self.assertAlmostEqual(sleeve_summary["gross_turnover"], 4.0)
        self.assertAlmostEqual(portfolio_summary["sleeve_gross_turnover"], 4.0)
        self.assertAlmostEqual(portfolio_summary["gross_turnover"], 2.0)
        self.assertAlmostEqual(portfolio_summary["turnover_cost"], 0.004)
        self.assertAlmostEqual(portfolio_summary["turnover_cost_bps"], 40.0)

    def test_sleeve_lifecycle_charges_entry_and_exit(self) -> None:
        """A +1/-1 sleeve must trade four units over its complete round trip."""

        lifecycle = compute_sleeve_lifecycle_trade_summary(
            target_weights={"AAA": 1.0, "BBB": -1.0},
            cost_bps=20.0,
            borrow_cost_bps=0.0,
            borrow_accrual_days=9,
        )

        self.assertAlmostEqual(lifecycle["entry_gross_turnover"], 2.0)
        self.assertAlmostEqual(lifecycle["exit_gross_turnover"], 2.0)
        self.assertAlmostEqual(lifecycle["gross_turnover"], 4.0)
        self.assertAlmostEqual(lifecycle["turnover_cost"], 0.008)
        self.assertAlmostEqual(lifecycle["turnover_cost_bps"], 80.0)

    def test_backtest_daily_ledger_keeps_execution_and_cash_dates(self) -> None:
        """Sharpe must include zero-return cash dates between non-overlapping sleeves."""

        dates = pd.date_range("2024-01-02", periods=24, freq="B")
        instruments = ["AAA", "BBB", "CCC", "DDD"]
        market_rows = []
        prediction_rows = []
        for instrument_index, instrument_id in enumerate(instruments):
            for date_index, date_value in enumerate(dates):
                market_rows.append(
                    {
                        "date": date_value,
                        "instrument_id": instrument_id,
                        "daily_close_return": 0.001 * (instrument_index - 1),
                        "sector": "Shared",
                        "market_cap": 1_000_000.0,
                        "size_exposure_z": 0.0,
                        "realized_vol_20": 0.02,
                    }
                )
                # Canonical prediction artifacts contain one cross-section for
                # every market date in their signal range. The backtest then
                # schedules the requested 10-market-day rebalance interval.
                if date_index <= 20:
                    prediction_rows.append(
                        {
                            "date": date_value,
                            "instrument_id": instrument_id,
                            "y": 0.0,
                            "predicted_y": float(instrument_index),
                        }
                    )

        result = run_long_short_backtest(
            config=LongShortBacktestConfig(
                run_name="ledger_cash_days",
                predictions_path=Path("unused_predictions.csv"),
                data_path=Path("unused_market.csv"),
                output_dir=Path("unused_output"),
                hold_days=10,
                step_days=10,
                top_k=1,
                cost_bps=0.0,
                neutral_mode="unconstrained",
                signal_delay_days=1,
                write_outputs=False,
            ),
            market_snapshot_df=pd.DataFrame(market_rows),
            prediction_df=pd.DataFrame(prediction_rows),
        )

        daily = result["daily_df"].copy()
        self.assertEqual(daily["date"].tolist(), dates[1:21].strftime("%Y-%m-%d").tolist())
        self.assertEqual(float(daily.loc[daily["date"] == dates[1].strftime("%Y-%m-%d"), "gross_return"].iloc[0]), 0.0)
        self.assertEqual(float(daily.loc[daily["date"] == dates[11].strftime("%Y-%m-%d"), "gross_return"].iloc[0]), 0.0)
        self.assertEqual(int(result["metrics"]["daily_count"]), 20)

    def test_backtest_rejects_missing_complete_signal_date(self) -> None:
        """One absent cross-section cannot shift every later rebalance date forward."""

        dates = pd.date_range("2024-01-02", periods=12, freq="B")
        instruments = ["AAA", "BBB", "CCC", "DDD"]
        market_rows = []
        prediction_rows = []
        for instrument_index, instrument_id in enumerate(instruments):
            for date_index, date_value in enumerate(dates):
                market_rows.append(
                    {
                        "date": date_value,
                        "instrument_id": instrument_id,
                        "daily_close_return": 0.001 * (instrument_index - 1),
                        "sector": "Shared",
                        "market_cap": 1_000_000.0,
                        "size_exposure_z": 0.0,
                        "realized_vol_20": 0.02,
                    }
                )
                # Remove the entire fifth-date cross-section. If the simulator
                # sliced only the dates present in this file, later step_days
                # rebalances would silently move to different market dates.
                if date_index != 5:
                    prediction_rows.append(
                        {
                            "date": date_value,
                            "instrument_id": instrument_id,
                            "y": 0.0,
                            "predicted_y": float(instrument_index),
                        }
                    )

        with self.assertRaisesRegex(ValueError, "missing complete market dates"):
            run_long_short_backtest(
                config=LongShortBacktestConfig(
                    run_name="missing_signal_date",
                    predictions_path=Path("unused_predictions.csv"),
                    data_path=Path("unused_market.csv"),
                    output_dir=Path("unused_output"),
                    hold_days=3,
                    step_days=3,
                    top_k=1,
                    cost_bps=0.0,
                    neutral_mode="unconstrained",
                    signal_delay_days=1,
                    write_outputs=False,
                ),
                market_snapshot_df=pd.DataFrame(market_rows),
                prediction_df=pd.DataFrame(prediction_rows),
            )


if __name__ == "__main__":
    unittest.main()
