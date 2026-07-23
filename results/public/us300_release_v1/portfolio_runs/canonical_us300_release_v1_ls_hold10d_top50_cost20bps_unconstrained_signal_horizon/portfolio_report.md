# Long-Short Backtest Report

## 1. Setup

- Run name: `canonical_us300_release_v1_ls_hold10d_top50_cost20bps_unconstrained_signal_horizon`
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
- Top-K long / short: `50`
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
  "portfolio_total_return": 0.4316343629081103,
  "portfolio_annualized_return": 0.43574982143658825,
  "portfolio_annualized_vol": 0.1444440789802248,
  "portfolio_sharpe": 2.5776668417336075,
  "portfolio_max_drawdown": -0.05775575365326613,
  "portfolio_calmar": 7.544699772296128,
  "sharpe_definition": "mean_daily_net_return / sample_std_daily_net_return * sqrt(252), risk_free_rate=0",
  "hit_ratio": 0.544,
  "gross_hit_ratio": 0.568,
  "best_daily_return": 0.030057692534046256,
  "worst_daily_return": -0.03433616316233426,
  "average_daily_return": 0.001477494892349507,
  "first_half_total_return": 0.20173508903379478,
  "second_half_total_return": 0.19130611727344826,
  "monthly_period_count": 13,
  "positive_month_ratio": 0.7692307692307693,
  "best_month_return": 0.0909575982553732,
  "worst_month_return": -0.023554344693738183,
  "average_gross_turnover": 4.000000000000002,
  "average_net_turnover": 0.0,
  "average_turnover_cost_bps": 80.00000000000004,
  "total_turnover_cost": 0.2000000000000001,
  "average_long_weight": 1.0,
  "average_short_weight_abs": 1.0,
  "average_gross_exposure": 2.0,
  "average_net_exposure": 0.0,
  "average_max_abs_sector_net_weight": 0.0808,
  "average_total_abs_sector_net_weight": 0.29120000000000007,
  "total_transaction_cost": 0.2000000000000001,
  "total_borrow_cost": 0.0,
  "benchmark_total_return": 0.327190618891692,
  "relative_wealth_vs_equal_weight_long_only": 0.07869536035723113,
  "relative_wealth_definition": "portfolio_nav / equal_weight_long_only_nav - 1; diagnostic only",
  "excess_total_return_vs_benchmark": 0.07869536035723113,
  "top_5_net_return_days_simple_sum": 0.13509067083883736,
  "bottom_5_net_return_days_simple_sum": -0.13120527360416537,
  "selected_position_day_count": 22500,
  "selected_stock_return_abs_gt_20pct_count": 15,
  "selected_stock_return_abs_gt_50pct_count": 0,
  "max_abs_selected_stock_daily_return": 0.49109265960450155,
  "max_abs_single_position_daily_contribution": 0.009821853192090032,
  "top_5_instrument_abs_contribution_share": 0.09550052233125071,
  "run_name": "canonical_us300_release_v1_ls_hold10d_top50_cost20bps_unconstrained_signal_horizon",
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
  "top_k": 50,
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
| 2025-06-03 |         0           |          0           |     0          |              0.004 |             0 |  -0.004      |               1 |                    1 |                2 |              0 |         0.00700721 |                               -0.0110072  |     -0.0110072  |        0.996    |         1.00701 |
| 2025-06-04 |        -0.00243882  |          0.00393915  |     0.00150033 |              0     |             0 |   0.00150033 |               1 |                    1 |                2 |              0 |        -0.00212558 |                                0.00362591 |      0.00362591 |        0.997494 |         1.00487 |
| 2025-06-05 |        -0.00374906  |          0.00215819  |    -0.00159087 |              0     |             0 |  -0.00159087 |               1 |                    1 |                2 |              0 |        -0.00172149 |                                0.00013062 |      0.00013062 |        0.995907 |         1.00314 |
| 2025-06-06 |         0.0116709   |         -0.00653501  |     0.00513586 |              0     |             0 |   0.00513586 |               1 |                    1 |                2 |              0 |         0.0094382  |                               -0.00430233 |     -0.00430233 |        1.00102  |         1.0126  |
| 2025-06-09 |         0.0055392   |          0.00853874  |     0.0140779  |              0     |             0 |   0.0140779  |               1 |                    1 |                2 |              0 |        -0.00130468 |                                0.0153826  |      0.0153826  |        1.01511  |         1.01128 |
| 2025-06-10 |         0.00620106  |          0.0010297   |     0.00723076 |              0     |             0 |   0.00723076 |               1 |                    1 |                2 |              0 |         0.00434525 |                                0.00288551 |      0.00288551 |        1.02245  |         1.01568 |
| 2025-06-11 |         0.000316853 |          0.00494814  |     0.005265   |              0     |             0 |   0.005265   |               1 |                    1 |                2 |              0 |        -0.00124608 |                                0.00651108 |      0.00651108 |        1.02784  |         1.01441 |
| 2025-06-12 |         0.0015214   |         -0.00661135  |    -0.00508995 |              0     |             0 |  -0.00508995 |               1 |                    1 |                2 |              0 |         0.00412834 |                               -0.00921829 |     -0.00921829 |        1.02261  |         1.0186  |
| 2025-06-13 |        -0.0168837   |          0.00906109  |    -0.0078226  |              0     |             0 |  -0.0078226  |               1 |                    1 |                2 |              0 |        -0.0111391  |                                0.00331648 |      0.00331648 |        1.01461  |         1.00725 |
| 2025-06-16 |         0.0345081   |         -0.000450377 |     0.0340577  |              0.004 |             0 |   0.0300577  |               1 |                    1 |                2 |              0 |         0.0102416  |                                0.0198161  |      0.0198161  |        1.0451   |         1.01757 |

## 5. Turnover / Cost Preview

| signal_date   | execution_date   | end_date   |   sleeve_slot |   top_k | neutral_mode   | weight_mode   | max_abs_weight   |   cost_bps |   borrow_cost_bps | holding_clock   |   effective_holding_days | execution_status              |   sleeve_entry_long_turnover |   sleeve_entry_short_turnover |   sleeve_entry_gross_turnover |   sleeve_entry_turnover_cost |   sleeve_exit_long_turnover |   sleeve_exit_short_turnover |   sleeve_exit_gross_turnover |   sleeve_exit_turnover_cost |   sleeve_long_turnover |   sleeve_short_turnover |   sleeve_gross_turnover |   sleeve_net_turnover |   sleeve_turnover_cost |   sleeve_turnover_cost_bps |   sleeve_borrow_cost_estimate |   sleeve_long_weight |   sleeve_short_weight |   sleeve_gross_exposure |   sleeve_net_exposure |   exit_gross_turnover |   entry_turnover_cost |   exit_long_turnover |   borrow_cost_estimate |   entry_gross_turnover |   net_turnover |   short_weight |   long_turnover |   net_exposure |   exit_turnover_cost |   gross_turnover |   entry_long_turnover |   exit_short_turnover |   turnover_cost |   entry_short_turnover |   gross_exposure |   short_turnover |   long_weight |   turnover_cost_bps |   sleeve_capital_weight |
|:--------------|:-----------------|:-----------|--------------:|--------:|:---------------|:--------------|:-----------------|-----------:|------------------:|:----------------|-------------------------:|:------------------------------|-----------------------------:|------------------------------:|------------------------------:|-----------------------------:|----------------------------:|-----------------------------:|-----------------------------:|----------------------------:|-----------------------:|------------------------:|------------------------:|----------------------:|-----------------------:|---------------------------:|------------------------------:|---------------------:|----------------------:|------------------------:|----------------------:|----------------------:|----------------------:|---------------------:|-----------------------:|-----------------------:|---------------:|---------------:|----------------:|---------------:|---------------------:|-----------------:|----------------------:|----------------------:|----------------:|-----------------------:|-----------------:|-----------------:|--------------:|--------------------:|------------------------:|
| 2025-06-02    | 2025-06-03       | 2025-06-16 |             0 |      50 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-06-16    | 2025-06-17       | 2025-07-01 |             0 |      50 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-01    | 2025-07-02       | 2025-07-16 |             0 |      50 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-16    | 2025-07-17       | 2025-07-30 |             0 |      50 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-30    | 2025-07-31       | 2025-08-13 |             0 |      50 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-08-13    | 2025-08-14       | 2025-08-27 |             0 |      50 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-08-27    | 2025-08-28       | 2025-09-11 |             0 |      50 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-09-11    | 2025-09-12       | 2025-09-25 |             0 |      50 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-09-25    | 2025-09-26       | 2025-10-09 |             0 |      50 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-10-09    | 2025-10-10       | 2025-10-23 |             0 |      50 | unconstrained  | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |

## 6. Sector Exposure Preview

| signal_date   | execution_date   |   sleeve_slot |   sleeve_capital_weight | sector                 |   sleeve_long_weight |   sleeve_short_weight_abs |   sleeve_net_sector_weight |   long_weight |   short_weight_abs |   net_sector_weight |   abs_net_sector_weight |   universe_weight |
|:--------------|:-----------------|--------------:|------------------------:|:-----------------------|---------------------:|--------------------------:|---------------------------:|--------------:|-------------------:|--------------------:|------------------------:|------------------:|
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Industrials            |                 0.18 |                      0.1  |                       0.08 |          0.18 |               0.1  |                0.08 |                    0.08 |         0.146667  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Consumer Discretionary |                 0.1  |                      0.16 |                      -0.06 |          0.1  |               0.16 |               -0.06 |                    0.06 |         0.09      |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Financials             |                 0.14 |                      0.2  |                      -0.06 |          0.14 |               0.2  |               -0.06 |                    0.06 |         0.143333  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Consumer Staples       |                 0.08 |                      0.04 |                       0.04 |          0.08 |               0.04 |                0.04 |                    0.04 |         0.0566667 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Unknown                |                 0.08 |                      0.04 |                       0.04 |          0.08 |               0.04 |                0.04 |                    0.04 |         0.1       |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Communication Services |                 0.04 |                      0.06 |                      -0.02 |          0.04 |               0.06 |               -0.02 |                    0.02 |         0.0433333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Energy                 |                 0.06 |                      0.04 |                       0.02 |          0.06 |               0.04 |                0.02 |                    0.02 |         0.0633333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Health Care            |                 0.04 |                      0.06 |                      -0.02 |          0.04 |               0.06 |               -0.02 |                    0.02 |         0.0833333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Information Technology |                 0.2  |                      0.22 |                      -0.02 |          0.2  |               0.22 |               -0.02 |                    0.02 |         0.153333  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Materials              |                 0.04 |                      0.04 |                       0    |          0.04 |               0.04 |                0    |                    0    |         0.0333333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Real Estate            |                 0    |                      0    |                       0    |          0    |               0    |                0    |                    0    |         0.0333333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Utilities              |                 0.04 |                      0.04 |                       0    |          0.04 |               0.04 |                0    |                    0    |         0.0533333 |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Financials             |                 0.12 |                      0.16 |                      -0.04 |          0.12 |               0.16 |               -0.04 |                    0.04 |         0.143333  |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Unknown                |                 0.1  |                      0.06 |                       0.04 |          0.1  |               0.06 |                0.04 |                    0.04 |         0.1       |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Communication Services |                 0.04 |                      0.02 |                       0.02 |          0.04 |               0.02 |                0.02 |                    0.02 |         0.0433333 |

## 7. Return Concentration Audit

Best and worst net-return days:

| date       |   long_gross_return |   short_gross_return |   gross_return |   transaction_cost |   borrow_cost |   net_return |   benchmark_return |   excess_return | audit_bucket   |
|:-----------|--------------------:|---------------------:|---------------:|-------------------:|--------------:|-------------:|-------------------:|----------------:|:---------------|
| 2025-06-16 |           0.0345081 |         -0.000450377 |      0.0340577 |              0.004 |             0 |    0.0300577 |        0.0102416   |      0.0198161  | best           |
| 2026-04-08 |           0.0427072 |         -0.0130061   |      0.0297011 |              0     |             0 |    0.0297011 |        0.0249213   |      0.00477983 | best           |
| 2025-10-29 |           0.0126524 |          0.0134044   |      0.0260568 |              0     |             0 |    0.0260568 |       -0.00408722  |      0.0301441  | best           |
| 2025-11-24 |           0.0273694 |         -0.00245165  |      0.0249178 |              0     |             0 |    0.0249178 |        0.00730622  |      0.0176116  | best           |
| 2025-11-05 |           0.0268776 |         -0.00252037  |      0.0243572 |              0     |             0 |    0.0243572 |        0.00641125  |      0.017946   | best           |
| 2026-02-04 |          -0.0243256 |         -0.00601058  |     -0.0303362 |              0.004 |             0 |   -0.0343362 |        0.000918344 |     -0.0352545  | worst          |
| 2026-04-28 |          -0.0227228 |         -0.00467227  |     -0.027395  |              0     |             0 |   -0.027395  |       -0.00420625  |     -0.0231888  | worst          |
| 2025-10-30 |          -0.0212095 |         -0.00345299  |     -0.0246625 |              0     |             0 |   -0.0246625 |       -0.00502998  |     -0.0196325  | worst          |
| 2025-12-12 |          -0.0242108 |         -0.000317805 |     -0.0245286 |              0     |             0 |   -0.0245286 |       -0.00860871  |     -0.0159199  | worst          |
| 2025-12-17 |          -0.0204383 |          0.000155287 |     -0.020283  |              0     |             0 |   -0.020283  |       -0.0062761   |     -0.0140069  | worst          |

Largest absolute gross-contribution instruments:

| instrument_id   |   gross_return_contribution |   absolute_gross_contribution |   position_day_count |   max_abs_stock_daily_return |
|:----------------|----------------------------:|------------------------------:|---------------------:|-----------------------------:|
| SNDK            |                  0.0798135  |                     0.204763  |                  216 |                     0.27565  |
| HOOD            |                  0.0112854  |                     0.144773  |                  225 |                     0.158321 |
| LITE            |                  0.0426048  |                     0.12455   |                  162 |                     0.235666 |
| MU              |                  0.0464303  |                     0.121308  |                  189 |                     0.192916 |
| SATS            |                  0.0314367  |                     0.112704  |                  171 |                     0.491093 |
| COIN            |                  0.00229641 |                     0.106104  |                  162 |                     0.166958 |
| COHR            |                  0.0277683  |                     0.10139   |                  162 |                     0.183243 |
| WDC             |                  0.0297225  |                     0.0960759 |                  153 |                     0.131764 |
| INTC            |                  0.014587   |                     0.0879888 |                  144 |                     0.235999 |
| AXON            |                 -0.0056972  |                     0.0868004 |                  171 |                     0.175521 |

These tables are diagnostics. A result dominated by a few dates, a few stocks, or
extreme adjusted returns requires manual company-action and data-quality review.
