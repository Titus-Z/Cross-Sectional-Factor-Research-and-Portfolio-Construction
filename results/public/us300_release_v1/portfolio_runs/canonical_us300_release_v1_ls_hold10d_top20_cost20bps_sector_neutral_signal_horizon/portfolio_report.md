# Long-Short Backtest Report

## 1. Setup

- Run name: `canonical_us300_release_v1_ls_hold10d_top20_cost20bps_sector_neutral_signal_horizon`
- Predictions path: `outputs/public_us300_release_v1/test_predictions_with_actual.csv`
- Market data path: `data/us_large_cap_300_daily.csv`
- Price adjustment mode: `vendor_adjusted`
- Real market-cap coverage: `0.00%`
- Size exposure available: `False`
- Hold days: `10`
- Holding clock: `signal_horizon`
- Effective executable holding days: `9`
- Rebalance step days: `10`
- Signal delay days: `1`
- Top-K long / short: `20`
- Neutral mode: `sector_neutral`
- Weight mode: `equal_weight`
- Max single-name absolute weight: `None`
- Transaction cost: `20.0` bps per traded notional
- Borrow fee sensitivity: `0.0` annualized bps, accrued linearly on short notional
- Turnover accounting: full sleeve round trip (entry + liquidation), capital-scaled, no cross-sleeve netting
- Skipped incomplete return paths: `0`
- Sharpe definition: `mean(daily net return) / sample std(daily net return) * sqrt(252)`, zero risk-free rate

## 2. Scope Caveat

This is a backtest-style diagnostic built from saved predictions and close-to-close returns.
Under the canonical `signal_horizon` clock, `hold_days` is measured from the signal date.
With a one-day close execution delay, a 10-day target accrues nine executable daily returns
from the execution close through the target endpoint. `execution_horizon` is retained only
as a historical sensitivity definition and must not be mixed with canonical `y_10d` results.
It does not model intraday execution, short borrow availability, bid-ask spread, order-book depth, or tax effects.

## 3. Metrics

```json
{
  "daily_count": 250,
  "invested_day_count": 250,
  "cash_day_count": 0,
  "daily_ledger_definition": "all market dates from first execution through final liquidation, including cash dates",
  "rebalance_count": 25,
  "portfolio_total_return": 0.3828647069403699,
  "portfolio_annualized_return": 0.38645548549092323,
  "portfolio_annualized_vol": 0.17401110224649177,
  "portfolio_sharpe": 1.9655421202359142,
  "portfolio_max_drawdown": -0.08090188026479561,
  "portfolio_calmar": 4.77684182649447,
  "sharpe_definition": "mean_daily_net_return / sample_std_daily_net_return * sqrt(252), risk_free_rate=0",
  "hit_ratio": 0.512,
  "gross_hit_ratio": 0.536,
  "best_daily_return": 0.04413806740062946,
  "worst_daily_return": -0.03788625497201974,
  "average_daily_return": 0.0013572466303736423,
  "first_half_total_return": 0.2694519248728604,
  "second_half_total_return": 0.08933995832797348,
  "monthly_period_count": 13,
  "positive_month_ratio": 0.7692307692307693,
  "best_month_return": 0.06288670453692413,
  "worst_month_return": -0.03512718092302347,
  "average_gross_turnover": 4.000000000000001,
  "average_net_turnover": 0.0,
  "average_turnover_cost_bps": 80.00000000000001,
  "total_turnover_cost": 0.20000000000000007,
  "average_long_weight": 1.0,
  "average_short_weight_abs": 1.0,
  "average_gross_exposure": 2.0,
  "average_net_exposure": 0.0,
  "average_max_abs_sector_net_weight": 0.0,
  "average_total_abs_sector_net_weight": 0.0,
  "total_transaction_cost": 0.20000000000000007,
  "total_borrow_cost": 0.0,
  "benchmark_total_return": 0.327190618891692,
  "relative_wealth_vs_equal_weight_long_only": 0.041948825779954735,
  "relative_wealth_definition": "portfolio_nav / equal_weight_long_only_nav - 1; diagnostic only",
  "excess_total_return_vs_benchmark": 0.041948825779954735,
  "top_5_net_return_days_simple_sum": 0.1565106862499408,
  "bottom_5_net_return_days_simple_sum": -0.1611016089915333,
  "selected_position_day_count": 9000,
  "selected_stock_return_abs_gt_20pct_count": 3,
  "selected_stock_return_abs_gt_50pct_count": 0,
  "max_abs_selected_stock_daily_return": 0.3583731788330714,
  "max_abs_single_position_daily_contribution": 0.01791865894165357,
  "top_5_instrument_abs_contribution_share": 0.14954302588562945,
  "run_name": "canonical_us300_release_v1_ls_hold10d_top20_cost20bps_sector_neutral_signal_horizon",
  "predictions_path": "outputs/public_us300_release_v1/test_predictions_with_actual.csv",
  "data_path": "data/us_large_cap_300_daily.csv",
  "price_adjustment_mode": "vendor_adjusted",
  "market_cap_coverage_ratio": 0.0,
  "size_exposure_available": false,
  "hold_days": 10,
  "holding_clock": "signal_horizon",
  "effective_holding_days": 9,
  "step_days": 10,
  "signal_delay_days": 1,
  "top_k": 20,
  "cost_bps": 20.0,
  "borrow_cost_bps": 0.0,
  "neutral_mode": "sector_neutral",
  "weight_mode": "equal_weight",
  "max_abs_weight": null,
  "max_active_sleeves": 1,
  "sleeve_capital_weight": 1.0,
  "borrow_cost_mode": "annualized_linear_fee_sensitivity_zero_by_default",
  "skipped_incomplete_return_path_count": 0,
  "turnover_accounting": "capital_scaled_full_sleeve_round_trip_without_cross_sleeve_netting",
  "is_short_sample_warning": false
}
```

## 4. Daily Returns Preview

| date       |   long_gross_return |   short_gross_return |   gross_return |   transaction_cost |   borrow_cost |   net_return |   long_exposure |   short_exposure_abs |   gross_exposure |   net_exposure |   benchmark_return |   active_return_vs_equal_weight_long_only |   excess_return |   portfolio_nav |   benchmark_nav |
|:-----------|--------------------:|---------------------:|---------------:|-------------------:|--------------:|-------------:|----------------:|---------------------:|-----------------:|---------------:|-------------------:|------------------------------------------:|----------------:|----------------:|----------------:|
| 2025-06-03 |         0           |          0           |    0           |              0.004 |             0 | -0.004       |               1 |                    1 |                2 |              0 |         0.00700721 |                              -0.0110072   |    -0.0110072   |        0.996    |         1.00701 |
| 2025-06-04 |         0.000813423 |          0.00375865  |    0.00457207  |              0     |             0 |  0.00457207  |               1 |                    1 |                2 |              0 |        -0.00212558 |                               0.00669765  |     0.00669765  |        1.00055  |         1.00487 |
| 2025-06-05 |        -0.0109331   |          0.00371504  |   -0.00721803  |              0     |             0 | -0.00721803  |               1 |                    1 |                2 |              0 |        -0.00172149 |                              -0.00549654  |    -0.00549654  |        0.993332 |         1.00314 |
| 2025-06-06 |         0.0140676   |         -0.00539097  |    0.00867662  |              0     |             0 |  0.00867662  |               1 |                    1 |                2 |              0 |         0.0094382  |                              -0.000761574 |    -0.000761574 |        1.00195  |         1.0126  |
| 2025-06-09 |         0.0137219   |          0.00646592  |    0.0201879   |              0     |             0 |  0.0201879   |               1 |                    1 |                2 |              0 |        -0.00130468 |                               0.0214925   |     0.0214925   |        1.02218  |         1.01128 |
| 2025-06-10 |         0.00460647  |          0.00277352  |    0.00737999  |              0     |             0 |  0.00737999  |               1 |                    1 |                2 |              0 |         0.00434525 |                               0.00303475  |     0.00303475  |        1.02972  |         1.01568 |
| 2025-06-11 |        -0.00510422  |          0.00282337  |   -0.00228085  |              0     |             0 | -0.00228085  |               1 |                    1 |                2 |              0 |        -0.00124608 |                              -0.00103477  |    -0.00103477  |        1.02737  |         1.01441 |
| 2025-06-12 |         0.00248184  |         -0.00629544  |   -0.0038136   |              0     |             0 | -0.0038136   |               1 |                    1 |                2 |              0 |         0.00412834 |                              -0.00794194  |    -0.00794194  |        1.02345  |         1.0186  |
| 2025-06-13 |        -0.00876617  |          0.00815104  |   -0.000615134 |              0     |             0 | -0.000615134 |               1 |                    1 |                2 |              0 |        -0.0111391  |                               0.0105239   |     0.0105239   |        1.02283  |         1.00725 |
| 2025-06-16 |         0.0290357   |         -0.000154641 |    0.028881    |              0.004 |             0 |  0.024881    |               1 |                    1 |                2 |              0 |         0.0102416  |                               0.0146394   |     0.0146394   |        1.04827  |         1.01757 |

## 5. Turnover / Cost Preview

| signal_date   | execution_date   | end_date   |   sleeve_slot |   top_k | neutral_mode   | weight_mode   | max_abs_weight   |   cost_bps |   borrow_cost_bps | holding_clock   |   effective_holding_days | execution_status              |   sleeve_entry_long_turnover |   sleeve_entry_short_turnover |   sleeve_entry_gross_turnover |   sleeve_entry_turnover_cost |   sleeve_exit_long_turnover |   sleeve_exit_short_turnover |   sleeve_exit_gross_turnover |   sleeve_exit_turnover_cost |   sleeve_long_turnover |   sleeve_short_turnover |   sleeve_gross_turnover |   sleeve_net_turnover |   sleeve_turnover_cost |   sleeve_turnover_cost_bps |   sleeve_borrow_cost_estimate |   sleeve_long_weight |   sleeve_short_weight |   sleeve_gross_exposure |   sleeve_net_exposure |   exit_gross_turnover |   entry_turnover_cost |   exit_long_turnover |   borrow_cost_estimate |   entry_gross_turnover |   net_turnover |   short_weight |   long_turnover |   net_exposure |   exit_turnover_cost |   gross_turnover |   entry_long_turnover |   exit_short_turnover |   turnover_cost |   entry_short_turnover |   gross_exposure |   short_turnover |   long_weight |   turnover_cost_bps |   sleeve_capital_weight |
|:--------------|:-----------------|:-----------|--------------:|--------:|:---------------|:--------------|:-----------------|-----------:|------------------:|:----------------|-------------------------:|:------------------------------|-----------------------------:|------------------------------:|------------------------------:|-----------------------------:|----------------------------:|-----------------------------:|-----------------------------:|----------------------------:|-----------------------:|------------------------:|------------------------:|----------------------:|-----------------------:|---------------------------:|------------------------------:|---------------------:|----------------------:|------------------------:|----------------------:|----------------------:|----------------------:|---------------------:|-----------------------:|-----------------------:|---------------:|---------------:|----------------:|---------------:|---------------------:|-----------------:|----------------------:|----------------------:|----------------:|-----------------------:|-----------------:|-----------------:|--------------:|--------------------:|------------------------:|
| 2025-06-02    | 2025-06-03       | 2025-06-16 |             0 |      20 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-06-16    | 2025-06-17       | 2025-07-01 |             0 |      20 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-01    | 2025-07-02       | 2025-07-16 |             0 |      20 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-16    | 2025-07-17       | 2025-07-30 |             0 |      20 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-30    | 2025-07-31       | 2025-08-13 |             0 |      20 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-08-13    | 2025-08-14       | 2025-08-27 |             0 |      20 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-08-27    | 2025-08-28       | 2025-09-11 |             0 |      20 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-09-11    | 2025-09-12       | 2025-09-25 |             0 |      20 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-09-25    | 2025-09-26       | 2025-10-09 |             0 |      20 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-10-09    | 2025-10-10       | 2025-10-23 |             0 |      20 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |

## 6. Sector Exposure Preview

| signal_date   | execution_date   |   sleeve_slot |   sleeve_capital_weight | sector                 |   sleeve_long_weight |   sleeve_short_weight_abs |   sleeve_net_sector_weight |   long_weight |   short_weight_abs |   net_sector_weight |   abs_net_sector_weight |   universe_weight |
|:--------------|:-----------------|--------------:|------------------------:|:-----------------------|---------------------:|--------------------------:|---------------------------:|--------------:|-------------------:|--------------------:|------------------------:|------------------:|
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Communication Services |                 0    |                      0    |                          0 |          0    |               0    |                   0 |                       0 |         0.0433333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Consumer Discretionary |                 0.05 |                      0.05 |                          0 |          0.05 |               0.05 |                   0 |                       0 |         0.09      |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Consumer Staples       |                 0.05 |                      0.05 |                          0 |          0.05 |               0.05 |                   0 |                       0 |         0.0566667 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Energy                 |                 0.05 |                      0.05 |                          0 |          0.05 |               0.05 |                   0 |                       0 |         0.0633333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Financials             |                 0.1  |                      0.1  |                          0 |          0.1  |               0.1  |                   0 |                       0 |         0.143333  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Health Care            |                 0.05 |                      0.05 |                          0 |          0.05 |               0.05 |                   0 |                       0 |         0.0833333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Industrials            |                 0.4  |                      0.4  |                          0 |          0.4  |               0.4  |                   0 |                       0 |         0.146667  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Information Technology |                 0.15 |                      0.15 |                          0 |          0.15 |               0.15 |                   0 |                       0 |         0.153333  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Materials              |                 0    |                      0    |                          0 |          0    |               0    |                   0 |                       0 |         0.0333333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Real Estate            |                 0    |                      0    |                          0 |          0    |               0    |                   0 |                       0 |         0.0333333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Unknown                |                 0.1  |                      0.1  |                          0 |          0.1  |               0.1  |                   0 |                       0 |         0.1       |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Utilities              |                 0.05 |                      0.05 |                          0 |          0.05 |               0.05 |                   0 |                       0 |         0.0533333 |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Communication Services |                 0    |                      0    |                          0 |          0    |               0    |                   0 |                       0 |         0.0433333 |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Consumer Discretionary |                 0.05 |                      0.05 |                          0 |          0.05 |               0.05 |                   0 |                       0 |         0.09      |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Consumer Staples       |                 0.05 |                      0.05 |                          0 |          0.05 |               0.05 |                   0 |                       0 |         0.0566667 |

## 7. Return Concentration Audit

Best and worst net-return days:

| date       |   long_gross_return |   short_gross_return |   gross_return |   transaction_cost |   borrow_cost |   net_return |   benchmark_return |   excess_return | audit_bucket   |
|:-----------|--------------------:|---------------------:|---------------:|-------------------:|--------------:|-------------:|-------------------:|----------------:|:---------------|
| 2025-11-24 |           0.0407485 |           0.00338952 |      0.0441381 |              0     |             0 |    0.0441381 |        0.00730622  |      0.0368318  | best           |
| 2026-02-06 |           0.0529498 |          -0.0218564  |      0.0310934 |              0     |             0 |    0.0310934 |        0.0222329   |      0.00886047 | best           |
| 2026-02-25 |           0.0242383 |           0.00478564 |      0.0290239 |              0     |             0 |    0.0290239 |        0.0056865   |      0.0233374  | best           |
| 2026-03-09 |           0.0284471 |          -0.00183544 |      0.0266117 |              0     |             0 |    0.0266117 |        0.00608482  |      0.0205269  | best           |
| 2026-01-09 |           0.0268586 |          -0.00121495 |      0.0256437 |              0     |             0 |    0.0256437 |        0.00738005  |      0.0182636  | best           |
| 2026-02-04 |          -0.0287845 |          -0.00510175 |     -0.0338863 |              0.004 |             0 |   -0.0378863 |        0.000918344 |     -0.0388046  | worst          |
| 2026-01-08 |          -0.0208714 |          -0.0122314  |     -0.0331027 |              0     |             0 |   -0.0331027 |        0.00375458  |     -0.0368573  | worst          |
| 2026-01-30 |          -0.0303236 |          -0.00130665 |     -0.0316303 |              0     |             0 |   -0.0316303 |       -0.00451585  |     -0.0271144  | worst          |
| 2026-04-28 |          -0.0237104 |          -0.00646764 |     -0.0301781 |              0     |             0 |   -0.0301781 |       -0.00420625  |     -0.0259718  | worst          |
| 2025-12-12 |          -0.0269011 |          -0.00140314 |     -0.0283043 |              0     |             0 |   -0.0283043 |       -0.00860871  |     -0.0196956  | worst          |

Largest absolute gross-contribution instruments:

| instrument_id   |   gross_return_contribution |   absolute_gross_contribution |   position_day_count |   max_abs_stock_daily_return |
|:----------------|----------------------------:|------------------------------:|---------------------:|-----------------------------:|
| HOOD            |                  0.010949   |                      0.245175 |                  153 |                    0.139516  |
| AXON            |                 -0.0250733  |                      0.230994 |                  180 |                    0.175521  |
| SNDK            |                  0.0929356  |                      0.214051 |                   90 |                    0.179023  |
| VRT             |                  0.044423   |                      0.205306 |                  144 |                    0.244915  |
| COIN            |                  0.00387508 |                      0.198278 |                  126 |                    0.166958  |
| LITE            |                  0.0465946  |                      0.164682 |                   72 |                    0.147285  |
| FIX             |                  0.0550591  |                      0.158969 |                  144 |                    0.223709  |
| DAL             |                  0.0035526  |                      0.130984 |                  135 |                    0.119921  |
| GEV             |                  0.0110458  |                      0.127422 |                  108 |                    0.156245  |
| NRG             |                  0.00129573 |                      0.113138 |                  117 |                    0.0830167 |

These tables are diagnostics. A result dominated by a few dates, a few stocks, or
extreme adjusted returns requires manual company-action and data-quality review.
