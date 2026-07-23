# Long-Short Backtest Report

## 1. Setup

- Run name: `canonical_us300_release_v1_ls_hold10d_top20_cost20bps_unconstrained_signal_horizon`
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
- Neutral mode: `unconstrained`
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
  "portfolio_total_return": 0.9950376171318256,
  "portfolio_annualized_return": 1.0060913143486303,
  "portfolio_annualized_vol": 0.22867413328588856,
  "portfolio_sharpe": 3.1622632156088737,
  "portfolio_max_drawdown": -0.09710340130416961,
  "portfolio_calmar": 10.361030621338584,
  "sharpe_definition": "mean_daily_net_return / sample_std_daily_net_return * sqrt(252), risk_free_rate=0",
  "hit_ratio": 0.56,
  "gross_hit_ratio": 0.56,
  "best_daily_return": 0.05011000229176997,
  "worst_daily_return": -0.05647948065408061,
  "average_daily_return": 0.0028695547621079604,
  "first_half_total_return": 0.4239639484456277,
  "second_half_total_return": 0.40104503299368743,
  "monthly_period_count": 13,
  "positive_month_ratio": 0.7692307692307693,
  "best_month_return": 0.16811817615643454,
  "worst_month_return": -0.02146153965251474,
  "average_gross_turnover": 4.000000000000001,
  "average_net_turnover": 0.0,
  "average_turnover_cost_bps": 80.00000000000001,
  "total_turnover_cost": 0.20000000000000007,
  "average_long_weight": 1.0,
  "average_short_weight_abs": 1.0,
  "average_gross_exposure": 2.0,
  "average_net_exposure": 0.0,
  "average_max_abs_sector_net_weight": 0.138,
  "average_total_abs_sector_net_weight": 0.528,
  "total_transaction_cost": 0.20000000000000007,
  "total_borrow_cost": 0.0,
  "benchmark_total_return": 0.327190618891692,
  "relative_wealth_vs_equal_weight_long_only": 0.5032035253517977,
  "relative_wealth_definition": "portfolio_nav / equal_weight_long_only_nav - 1; diagnostic only",
  "excess_total_return_vs_benchmark": 0.5032035253517977,
  "top_5_net_return_days_simple_sum": 0.2207909927823644,
  "bottom_5_net_return_days_simple_sum": -0.18417668452779581,
  "selected_position_day_count": 9000,
  "selected_stock_return_abs_gt_20pct_count": 6,
  "selected_stock_return_abs_gt_50pct_count": 0,
  "max_abs_selected_stock_daily_return": 0.49109265960450155,
  "max_abs_single_position_daily_contribution": 0.024554632980225078,
  "top_5_instrument_abs_contribution_share": 0.15688657845409457,
  "run_name": "canonical_us300_release_v1_ls_hold10d_top20_cost20bps_unconstrained_signal_horizon",
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
  "neutral_mode": "unconstrained",
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
| 2025-06-03 |         0           |           0          |     0          |              0.004 |             0 |  -0.004      |               1 |                    1 |                2 |              0 |         0.00700721 |                              -0.0110072   |    -0.0110072   |         0.996   |         1.00701 |
| 2025-06-04 |         0.00524796  |           0.0020629  |     0.00731086 |              0     |             0 |   0.00731086 |               1 |                    1 |                2 |              0 |        -0.00212558 |                               0.00943643  |     0.00943643  |         1.00328 |         1.00487 |
| 2025-06-05 |        -0.00310743  |           0.00072847 |    -0.00237896 |              0     |             0 |  -0.00237896 |               1 |                    1 |                2 |              0 |        -0.00172149 |                              -0.000657472 |    -0.000657472 |         1.00089 |         1.00314 |
| 2025-06-06 |         0.00814633  |          -0.00556475 |     0.00258158 |              0     |             0 |   0.00258158 |               1 |                    1 |                2 |              0 |         0.0094382  |                              -0.00685662  |    -0.00685662  |         1.00348 |         1.0126  |
| 2025-06-09 |         0.00835736  |           0.00767755 |     0.0160349  |              0     |             0 |   0.0160349  |               1 |                    1 |                2 |              0 |        -0.00130468 |                               0.0173396   |     0.0173396   |         1.01957 |         1.01128 |
| 2025-06-10 |         0.00767938  |           0.00321107 |     0.0108904  |              0     |             0 |   0.0108904  |               1 |                    1 |                2 |              0 |         0.00434525 |                               0.0065452   |     0.0065452   |         1.03067 |         1.01568 |
| 2025-06-11 |        -0.00193768  |           0.00742147 |     0.00548379 |              0     |             0 |   0.00548379 |               1 |                    1 |                2 |              0 |        -0.00124608 |                               0.00672987  |     0.00672987  |         1.03632 |         1.01441 |
| 2025-06-12 |        -0.000500437 |          -0.00456806 |    -0.0050685  |              0     |             0 |  -0.0050685  |               1 |                    1 |                2 |              0 |         0.00412834 |                              -0.00919684  |    -0.00919684  |         1.03107 |         1.0186  |
| 2025-06-13 |        -0.0196088   |           0.012488   |    -0.00712073 |              0     |             0 |  -0.00712073 |               1 |                    1 |                2 |              0 |        -0.0111391  |                               0.00401835  |     0.00401835  |         1.02373 |         1.00725 |
| 2025-06-16 |         0.0575584   |          -0.00415368 |     0.0534048  |              0.004 |             0 |   0.0494048  |               1 |                    1 |                2 |              0 |         0.0102416  |                               0.0391631   |     0.0391631   |         1.07431 |         1.01757 |

## 5. Turnover / Cost Preview

| signal_date   | execution_date   | end_date   |   sleeve_slot |   top_k | neutral_mode   | weight_mode   | max_abs_weight   |   cost_bps |   borrow_cost_bps | holding_clock   |   effective_holding_days | execution_status              |   sleeve_entry_long_turnover |   sleeve_entry_short_turnover |   sleeve_entry_gross_turnover |   sleeve_entry_turnover_cost |   sleeve_exit_long_turnover |   sleeve_exit_short_turnover |   sleeve_exit_gross_turnover |   sleeve_exit_turnover_cost |   sleeve_long_turnover |   sleeve_short_turnover |   sleeve_gross_turnover |   sleeve_net_turnover |   sleeve_turnover_cost |   sleeve_turnover_cost_bps |   sleeve_borrow_cost_estimate |   sleeve_long_weight |   sleeve_short_weight |   sleeve_gross_exposure |   sleeve_net_exposure |   exit_gross_turnover |   entry_turnover_cost |   exit_long_turnover |   borrow_cost_estimate |   entry_gross_turnover |   net_turnover |   short_weight |   long_turnover |   net_exposure |   exit_turnover_cost |   gross_turnover |   entry_long_turnover |   exit_short_turnover |   turnover_cost |   entry_short_turnover |   gross_exposure |   short_turnover |   long_weight |   turnover_cost_bps |   sleeve_capital_weight |
|:--------------|:-----------------|:-----------|--------------:|--------:|:---------------|:--------------|:-----------------|-----------:|------------------:|:----------------|-------------------------:|:------------------------------|-----------------------------:|------------------------------:|------------------------------:|-----------------------------:|----------------------------:|-----------------------------:|-----------------------------:|----------------------------:|-----------------------:|------------------------:|------------------------:|----------------------:|-----------------------:|---------------------------:|------------------------------:|---------------------:|----------------------:|------------------------:|----------------------:|----------------------:|----------------------:|---------------------:|-----------------------:|-----------------------:|---------------:|---------------:|----------------:|---------------:|---------------------:|-----------------:|----------------------:|----------------------:|----------------:|-----------------------:|-----------------:|-----------------:|--------------:|--------------------:|------------------------:|
| 2025-06-02    | 2025-06-03       | 2025-06-16 |             0 |      20 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-06-16    | 2025-06-17       | 2025-07-01 |             0 |      20 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-01    | 2025-07-02       | 2025-07-16 |             0 |      20 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-16    | 2025-07-17       | 2025-07-30 |             0 |      20 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-30    | 2025-07-31       | 2025-08-13 |             0 |      20 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-08-13    | 2025-08-14       | 2025-08-27 |             0 |      20 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-08-27    | 2025-08-28       | 2025-09-11 |             0 |      20 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-09-11    | 2025-09-12       | 2025-09-25 |             0 |      20 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-09-25    | 2025-09-26       | 2025-10-09 |             0 |      20 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-10-09    | 2025-10-10       | 2025-10-23 |             0 |      20 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |

## 6. Sector Exposure Preview

| signal_date   | execution_date   |   sleeve_slot |   sleeve_capital_weight | sector                 |   sleeve_long_weight |   sleeve_short_weight_abs |   sleeve_net_sector_weight |   long_weight |   short_weight_abs |   net_sector_weight |   abs_net_sector_weight |   universe_weight |
|:--------------|:-----------------|--------------:|------------------------:|:-----------------------|---------------------:|--------------------------:|---------------------------:|--------------:|-------------------:|--------------------:|------------------------:|------------------:|
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Consumer Discretionary |                 0.2  |                      0.05 |                       0.15 |          0.2  |               0.05 |                0.15 |                    0.15 |         0.09      |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Information Technology |                 0.3  |                      0.45 |                      -0.15 |          0.3  |               0.45 |               -0.15 |                    0.15 |         0.153333  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Financials             |                 0.15 |                      0.05 |                       0.1  |          0.15 |               0.05 |                0.1  |                    0.1  |         0.143333  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Materials              |                 0    |                      0.1  |                      -0.1  |          0    |               0.1  |               -0.1  |                    0.1  |         0.0333333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Unknown                |                 0.1  |                      0    |                       0.1  |          0.1  |               0    |                0.1  |                    0.1  |         0.1       |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Communication Services |                 0.05 |                      0.1  |                      -0.05 |          0.05 |               0.1  |               -0.05 |                    0.05 |         0.0433333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Consumer Staples       |                 0.05 |                      0.1  |                      -0.05 |          0.05 |               0.1  |               -0.05 |                    0.05 |         0.0566667 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Energy                 |                 0    |                      0.05 |                      -0.05 |          0    |               0.05 |               -0.05 |                    0.05 |         0.0633333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Health Care            |                 0.05 |                      0    |                       0.05 |          0.05 |               0    |                0.05 |                    0.05 |         0.0833333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Industrials            |                 0.05 |                      0.05 |                       0    |          0.05 |               0.05 |                0    |                    0    |         0.146667  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Real Estate            |                 0    |                      0    |                       0    |          0    |               0    |                0    |                    0    |         0.0333333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Utilities              |                 0.05 |                      0.05 |                       0    |          0.05 |               0.05 |                0    |                    0    |         0.0533333 |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Consumer Discretionary |                 0.2  |                      0.05 |                       0.15 |          0.2  |               0.05 |                0.15 |                    0.15 |         0.09      |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Information Technology |                 0.35 |                      0.5  |                      -0.15 |          0.35 |               0.5  |               -0.15 |                    0.15 |         0.153333  |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Financials             |                 0.1  |                      0    |                       0.1  |          0.1  |               0    |                0.1  |                    0.1  |         0.143333  |

## 7. Return Concentration Audit

Best and worst net-return days:

| date       |   long_gross_return |   short_gross_return |   gross_return |   transaction_cost |   borrow_cost |   net_return |   benchmark_return |   excess_return | audit_bucket   |
|:-----------|--------------------:|---------------------:|---------------:|-------------------:|--------------:|-------------:|-------------------:|----------------:|:---------------|
| 2025-11-24 |           0.0527884 |          -0.00267836 |      0.05011   |              0     |             0 |    0.05011   |        0.00730622  |       0.0428038 | best           |
| 2025-06-16 |           0.0575584 |          -0.00415368 |      0.0534048 |              0.004 |             0 |    0.0494048 |        0.0102416   |       0.0391631 | best           |
| 2026-04-08 |           0.0636891 |          -0.018162   |      0.045527  |              0     |             0 |    0.045527  |        0.0249213   |       0.0206057 | best           |
| 2025-11-05 |           0.0369611 |           0.00129406 |      0.0382552 |              0     |             0 |    0.0382552 |        0.00641125  |       0.031844  | best           |
| 2026-02-06 |           0.0580561 |          -0.0205621  |      0.037494  |              0     |             0 |    0.037494  |        0.0222329   |       0.0152611 | best           |
| 2026-02-04 |          -0.0424452 |          -0.0100342  |     -0.0524795 |              0.004 |             0 |   -0.0564795 |        0.000918344 |      -0.0573978 | worst          |
| 2026-01-08 |          -0.0315461 |          -0.00303235 |     -0.0345785 |              0     |             0 |   -0.0345785 |        0.00375458  |      -0.038333  | worst          |
| 2026-04-28 |          -0.0311957 |          -0.00313433 |     -0.0343301 |              0     |             0 |   -0.0343301 |       -0.00420625  |      -0.0301238 | worst          |
| 2025-12-12 |          -0.0260182 |          -0.00344865 |     -0.0294668 |              0     |             0 |   -0.0294668 |       -0.00860871  |      -0.0208581 | worst          |
| 2025-11-20 |          -0.0418943 |           0.0165724  |     -0.0253219 |              0.004 |             0 |   -0.0293219 |       -0.0153045   |      -0.0140174 | worst          |

Largest absolute gross-contribution instruments:

| instrument_id   |   gross_return_contribution |   absolute_gross_contribution |   position_day_count |   max_abs_stock_daily_return |
|:----------------|----------------------------:|------------------------------:|---------------------:|-----------------------------:|
| SNDK            |                  0.143747   |                      0.326636 |                  135 |                     0.179023 |
| HOOD            |                  0.0134373  |                      0.270827 |                  171 |                     0.139516 |
| LITE            |                  0.0750103  |                      0.245628 |                  126 |                     0.171251 |
| SATS            |                  0.0692047  |                      0.229363 |                  135 |                     0.491093 |
| MU              |                  0.0847891  |                      0.215194 |                  126 |                     0.192916 |
| COIN            |                  0.00387508 |                      0.198278 |                  126 |                     0.166958 |
| APP             |                  0.0212197  |                      0.190496 |                  108 |                     0.1968   |
| INTC            |                  0.033965   |                      0.177892 |                  108 |                     0.235999 |
| AXON            |                 -0.00384443 |                      0.150783 |                  108 |                     0.175521 |
| WDC             |                  0.0422193  |                      0.147612 |                   81 |                     0.107021 |

These tables are diagnostics. A result dominated by a few dates, a few stocks, or
extreme adjusted returns requires manual company-action and data-quality review.
