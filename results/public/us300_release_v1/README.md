# Canonical US300 Evidence Package

> Public status: `release_candidate_requires_review`. The canonical run, OOS
> audit, portfolio grid, provenance files, and threshold-event review are
> published for research inspection. This status does not establish stable or
> executable live performance.

## Contract

- Experiment: `us300_release_v1`
- Universe: `us_large_cap_300_static_snapshot`
- Instruments: `300`
- In-sample dates: `2022-01-03` to `2025-05-15`
- Final OOS dates: `2025-06-02` to `2026-05-20`
- Target: `y_10d`
- Models: `ridge, lasso`
- Price mode: `vendor_adjusted`
- Source commit: `f814c4d78796cfb101b5dfa52d6fddc0610a70ae`

## Walk-Forward Model Summary

| model   |   pearson_ic_mean |   spearman_ic_mean |      rmse |       mae |
|:--------|------------------:|-------------------:|----------:|----------:|
| ridge   |         0.0261287 |          0.0235119 | 0.0652329 | 0.0461502 |
| lasso   |         0.0221299 |          0.01632   | 0.065081  | 0.0459547 |

## Final OOS Audit

| metric                    |      value |
|:--------------------------|-----------:|
| pearson_corr              | 0.0689692  |
| spearman_corr             | 0.00928286 |
| pearson_ic_mean           | 0.112551   |
| spearman_ic_mean          | 0.0610037  |
| rmse                      | 0.0720927  |
| mae                       | 0.0483574  |
| long_short_spread         | 0.028369   |
| prediction_coverage_ratio | 1          |

The label spread above is a same-date diagnostic and is not cumulative portfolio return.

## Cost-Aware Portfolio Slice

| portfolio                  |   top_k | neutral_mode   |   hold_days | holding_clock   |   effective_holding_days |   rebalance_days |   signal_delay_days |   transaction_cost_bps |   borrow_cost_bps_annual | price_adjustment_mode   |   daily_count |   invested_day_count |   cash_day_count |   rebalance_count |   cumulative_return |   benchmark_return |   relative_wealth_vs_equal_weight_long_only |   sharpe |   max_drawdown |   average_gross_turnover |   average_turnover_cost_bps |   total_turnover_cost |   skipped_incomplete_return_path_count | turnover_accounting                                                |
|:---------------------------|--------:|:---------------|------------:|:----------------|-------------------------:|-----------------:|--------------------:|-----------------------:|-------------------------:|:------------------------|--------------:|---------------------:|-----------------:|------------------:|--------------------:|-------------------:|--------------------------------------------:|---------:|---------------:|-------------------------:|----------------------------:|----------------------:|---------------------------------------:|:-------------------------------------------------------------------|
| top20_unconstrained_20bps  |      20 | unconstrained  |          10 | signal_horizon  |                        9 |               10 |                   1 |                     20 |                        0 | vendor_adjusted         |           250 |                  250 |                0 |                25 |            0.995038 |           0.327191 |                                   0.503204  |  3.16226 |     -0.0971034 |                        4 |                          80 |                   0.2 |                                      0 | capital_scaled_full_sleeve_round_trip_without_cross_sleeve_netting |
| top20_sector_neutral_20bps |      20 | sector_neutral |          10 | signal_horizon  |                        9 |               10 |                   1 |                     20 |                        0 | vendor_adjusted         |           250 |                  250 |                0 |                25 |            0.382865 |           0.327191 |                                   0.0419488 |  1.96554 |     -0.0809019 |                        4 |                          80 |                   0.2 |                                      0 | capital_scaled_full_sleeve_round_trip_without_cross_sleeve_netting |
| top50_unconstrained_20bps  |      50 | unconstrained  |          10 | signal_horizon  |                        9 |               10 |                   1 |                     20 |                        0 | vendor_adjusted         |           250 |                  250 |                0 |                25 |            0.431634 |           0.327191 |                                   0.0786954 |  2.57767 |     -0.0577558 |                        4 |                          80 |                   0.2 |                                      0 | capital_scaled_full_sleeve_round_trip_without_cross_sleeve_netting |
| top50_sector_neutral_20bps |      50 | sector_neutral |          10 | signal_horizon  |                        9 |               10 |                   1 |                     20 |                        0 | vendor_adjusted         |           250 |                  250 |                0 |                25 |            0.222793 |           0.327191 |                                  -0.0786602 |  1.68049 |     -0.0604015 |                        4 |                          80 |                   0.2 |                                      0 | capital_scaled_full_sleeve_round_trip_without_cross_sleeve_netting |

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

The [manual threshold-event review](MANUAL_DATA_REVIEW.md) classifies TGT, CVNA,
and SATS using same-day company or SEC disclosures. It does not prove complete
vendor corporate-action coverage.

Read [`../../../docs/LIMITATIONS.md`](../../../docs/LIMITATIONS.md) before citing any
number. A release-candidate status does not establish stable tradability.
