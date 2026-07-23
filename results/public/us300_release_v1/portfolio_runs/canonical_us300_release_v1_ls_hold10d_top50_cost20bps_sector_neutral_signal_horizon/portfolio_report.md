# Long-Short Backtest Report

## 1. Setup

- Run name: `canonical_us300_release_v1_ls_hold10d_top50_cost20bps_sector_neutral_signal_horizon`
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
  "portfolio_total_return": 0.22279348120867026,
  "portfolio_annualized_return": 0.22476266678179924,
  "portfolio_annualized_vol": 0.12535251208882686,
  "portfolio_sharpe": 1.6804900440757966,
  "portfolio_max_drawdown": -0.06040154216015048,
  "portfolio_calmar": 3.7211411951346656,
  "sharpe_definition": "mean_daily_net_return / sample_std_daily_net_return * sqrt(252), risk_free_rate=0",
  "hit_ratio": 0.508,
  "gross_hit_ratio": 0.532,
  "best_daily_return": 0.02683518258738659,
  "worst_daily_return": -0.030111429633786144,
  "average_daily_return": 0.0008359271768458907,
  "first_half_total_return": 0.13240568520225793,
  "second_half_total_return": 0.07981927076802786,
  "monthly_period_count": 13,
  "positive_month_ratio": 0.8461538461538461,
  "best_month_return": 0.05263252859403589,
  "worst_month_return": -0.02032830245496775,
  "average_gross_turnover": 4.000000000000002,
  "average_net_turnover": 0.0,
  "average_turnover_cost_bps": 80.00000000000004,
  "total_turnover_cost": 0.2000000000000001,
  "average_long_weight": 1.0,
  "average_short_weight_abs": 1.0,
  "average_gross_exposure": 2.0,
  "average_net_exposure": 0.0,
  "average_max_abs_sector_net_weight": 0.0,
  "average_total_abs_sector_net_weight": 0.0,
  "total_transaction_cost": 0.2000000000000001,
  "total_borrow_cost": 0.0,
  "benchmark_total_return": 0.327190618891692,
  "relative_wealth_vs_equal_weight_long_only": -0.07866024382405712,
  "relative_wealth_definition": "portfolio_nav / equal_weight_long_only_nav - 1; diagnostic only",
  "excess_total_return_vs_benchmark": -0.07866024382405712,
  "top_5_net_return_days_simple_sum": 0.11040517296465785,
  "bottom_5_net_return_days_simple_sum": -0.1137963325041158,
  "selected_position_day_count": 22500,
  "selected_stock_return_abs_gt_20pct_count": 10,
  "selected_stock_return_abs_gt_50pct_count": 1,
  "max_abs_selected_stock_daily_return": 0.7024765852638799,
  "max_abs_single_position_daily_contribution": 0.014049531705277599,
  "top_5_instrument_abs_contribution_share": 0.090116656512324,
  "run_name": "canonical_us300_release_v1_ls_hold10d_top50_cost20bps_sector_neutral_signal_horizon",
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
| 2025-06-03 |         0           |          0           |     0          |              0.004 |             0 |  -0.004      |               1 |                    1 |                2 |              0 |         0.00700721 |                               -0.0110072  |     -0.0110072  |        0.996    |         1.00701 |
| 2025-06-04 |        -0.00177103  |          0.00386686  |     0.00209583 |              0     |             0 |   0.00209583 |               1 |                    1 |                2 |              0 |        -0.00212558 |                                0.00422141 |      0.00422141 |        0.998087 |         1.00487 |
| 2025-06-05 |        -0.00150262  |          0.00285967  |     0.00135705 |              0     |             0 |   0.00135705 |               1 |                    1 |                2 |              0 |        -0.00172149 |                                0.00307854 |      0.00307854 |        0.999442 |         1.00314 |
| 2025-06-06 |         0.0113082   |         -0.00576642  |     0.0055418  |              0     |             0 |   0.0055418  |               1 |                    1 |                2 |              0 |         0.0094382  |                               -0.00389639 |     -0.00389639 |        1.00498  |         1.0126  |
| 2025-06-09 |         0.00363131  |          0.00792137  |     0.0115527  |              0     |             0 |   0.0115527  |               1 |                    1 |                2 |              0 |        -0.00130468 |                                0.0128574  |      0.0128574  |        1.01659  |         1.01128 |
| 2025-06-10 |         0.00614745  |          0.000870613 |     0.00701806 |              0     |             0 |   0.00701806 |               1 |                    1 |                2 |              0 |         0.00434525 |                                0.00267282 |      0.00267282 |        1.02373  |         1.01568 |
| 2025-06-11 |        -0.000973593 |          0.00265647  |     0.00168288 |              0     |             0 |   0.00168288 |               1 |                    1 |                2 |              0 |        -0.00124608 |                                0.00292896 |      0.00292896 |        1.02545  |         1.01441 |
| 2025-06-12 |         0.00317573  |         -0.00542121  |    -0.00224549 |              0     |             0 |  -0.00224549 |               1 |                    1 |                2 |              0 |         0.00412834 |                               -0.00637383 |     -0.00637383 |        1.02315  |         1.0186  |
| 2025-06-13 |        -0.0164757   |          0.00801364  |    -0.00846202 |              0     |             0 |  -0.00846202 |               1 |                    1 |                2 |              0 |        -0.0111391  |                                0.00267706 |      0.00267706 |        1.01449  |         1.00725 |
| 2025-06-16 |         0.0320468   |         -0.00121165  |     0.0308352  |              0.004 |             0 |   0.0268352  |               1 |                    1 |                2 |              0 |         0.0102416  |                                0.0165936  |      0.0165936  |        1.04171  |         1.01757 |

## 5. Turnover / Cost Preview

| signal_date   | execution_date   | end_date   |   sleeve_slot |   top_k | neutral_mode   | weight_mode   | max_abs_weight   |   cost_bps |   borrow_cost_bps | holding_clock   |   effective_holding_days | execution_status              |   sleeve_entry_long_turnover |   sleeve_entry_short_turnover |   sleeve_entry_gross_turnover |   sleeve_entry_turnover_cost |   sleeve_exit_long_turnover |   sleeve_exit_short_turnover |   sleeve_exit_gross_turnover |   sleeve_exit_turnover_cost |   sleeve_long_turnover |   sleeve_short_turnover |   sleeve_gross_turnover |   sleeve_net_turnover |   sleeve_turnover_cost |   sleeve_turnover_cost_bps |   sleeve_borrow_cost_estimate |   sleeve_long_weight |   sleeve_short_weight |   sleeve_gross_exposure |   sleeve_net_exposure |   exit_gross_turnover |   entry_turnover_cost |   exit_long_turnover |   borrow_cost_estimate |   entry_gross_turnover |   net_turnover |   short_weight |   long_turnover |   net_exposure |   exit_turnover_cost |   gross_turnover |   entry_long_turnover |   exit_short_turnover |   turnover_cost |   entry_short_turnover |   gross_exposure |   short_turnover |   long_weight |   turnover_cost_bps |   sleeve_capital_weight |
|:--------------|:-----------------|:-----------|--------------:|--------:|:---------------|:--------------|:-----------------|-----------:|------------------:|:----------------|-------------------------:|:------------------------------|-----------------------------:|------------------------------:|------------------------------:|-----------------------------:|----------------------------:|-----------------------------:|-----------------------------:|----------------------------:|-----------------------:|------------------------:|------------------------:|----------------------:|-----------------------:|---------------------------:|------------------------------:|---------------------:|----------------------:|------------------------:|----------------------:|----------------------:|----------------------:|---------------------:|-----------------------:|-----------------------:|---------------:|---------------:|----------------:|---------------:|---------------------:|-----------------:|----------------------:|----------------------:|----------------:|-----------------------:|-----------------:|-----------------:|--------------:|--------------------:|------------------------:|
| 2025-06-02    | 2025-06-03       | 2025-06-16 |             0 |      50 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-06-16    | 2025-06-17       | 2025-07-01 |             0 |      50 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-01    | 2025-07-02       | 2025-07-16 |             0 |      50 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-16    | 2025-07-17       | 2025-07-30 |             0 |      50 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-07-30    | 2025-07-31       | 2025-08-13 |             0 |      50 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-08-13    | 2025-08-14       | 2025-08-27 |             0 |      50 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-08-27    | 2025-08-28       | 2025-09-11 |             0 |      50 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-09-11    | 2025-09-12       | 2025-09-25 |             0 |      50 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-09-25    | 2025-09-26       | 2025-10-09 |             0 |      50 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |
| 2025-10-09    | 2025-10-10       | 2025-10-23 |             0 |      50 | sector_neutral | equal_weight  |                  |         20 |                 0 | signal_horizon  |                        9 | executed_complete_return_path |                            1 |                             1 |                             2 |                        0.004 |                           1 |                            1 |                            2 |                       0.004 |                      2 |                       2 |                       4 |                     0 |                  0.008 |                         80 |                             0 |                    1 |                     1 |                       2 |                     0 |                     2 |                 0.004 |                    1 |                      0 |                      2 |              0 |              1 |               2 |              0 |                0.004 |                4 |                     1 |                     1 |           0.008 |                      1 |                2 |                2 |             1 |                  80 |                       1 |

## 6. Sector Exposure Preview

| signal_date   | execution_date   |   sleeve_slot |   sleeve_capital_weight | sector                 |   sleeve_long_weight |   sleeve_short_weight_abs |   sleeve_net_sector_weight |   long_weight |   short_weight_abs |   net_sector_weight |   abs_net_sector_weight |   universe_weight |
|:--------------|:-----------------|--------------:|------------------------:|:-----------------------|---------------------:|--------------------------:|---------------------------:|--------------:|-------------------:|--------------------:|------------------------:|------------------:|
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Communication Services |                 0.04 |                      0.04 |                          0 |          0.04 |               0.04 |                   0 |                       0 |         0.0433333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Consumer Discretionary |                 0.08 |                      0.08 |                          0 |          0.08 |               0.08 |                   0 |                       0 |         0.09      |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Consumer Staples       |                 0.14 |                      0.14 |                          0 |          0.14 |               0.14 |                   0 |                       0 |         0.0566667 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Energy                 |                 0.06 |                      0.06 |                          0 |          0.06 |               0.06 |                   0 |                       0 |         0.0633333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Financials             |                 0.14 |                      0.14 |                          0 |          0.14 |               0.14 |                   0 |                       0 |         0.143333  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Health Care            |                 0.08 |                      0.08 |                          0 |          0.08 |               0.08 |                   0 |                       0 |         0.0833333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Industrials            |                 0.14 |                      0.14 |                          0 |          0.14 |               0.14 |                   0 |                       0 |         0.146667  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Information Technology |                 0.14 |                      0.14 |                          0 |          0.14 |               0.14 |                   0 |                       0 |         0.153333  |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Materials              |                 0.02 |                      0.02 |                          0 |          0.02 |               0.02 |                   0 |                       0 |         0.0333333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Real Estate            |                 0.02 |                      0.02 |                          0 |          0.02 |               0.02 |                   0 |                       0 |         0.0333333 |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Unknown                |                 0.1  |                      0.1  |                          0 |          0.1  |               0.1  |                   0 |                       0 |         0.1       |
| 2025-06-02    | 2025-06-03       |             0 |                       1 | Utilities              |                 0.04 |                      0.04 |                          0 |          0.04 |               0.04 |                   0 |                       0 |         0.0533333 |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Communication Services |                 0.04 |                      0.04 |                          0 |          0.04 |               0.04 |                   0 |                       0 |         0.0433333 |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Consumer Discretionary |                 0.08 |                      0.08 |                          0 |          0.08 |               0.08 |                   0 |                       0 |         0.09      |
| 2025-06-16    | 2025-06-17       |             0 |                       1 | Consumer Staples       |                 0.14 |                      0.14 |                          0 |          0.14 |               0.14 |                   0 |                       0 |         0.0566667 |

## 7. Return Concentration Audit

Best and worst net-return days:

| date       |   long_gross_return |   short_gross_return |   gross_return |   transaction_cost |   borrow_cost |   net_return |   benchmark_return |   excess_return | audit_bucket   |
|:-----------|--------------------:|---------------------:|---------------:|-------------------:|--------------:|-------------:|-------------------:|----------------:|:---------------|
| 2025-06-16 |          0.0320468  |          -0.00121165 |      0.0308352 |              0.004 |             0 |    0.0268352 |        0.0102416   |      0.0165936  | best           |
| 2025-11-24 |          0.0220585  |           0.00194687 |      0.0240054 |              0     |             0 |    0.0240054 |        0.00730622  |      0.0166992  | best           |
| 2026-04-08 |          0.038711   |          -0.0176445  |      0.0210665 |              0     |             0 |    0.0210665 |        0.0249213   |     -0.00385481 | best           |
| 2026-03-09 |          0.0210888  |          -0.00119298 |      0.0198958 |              0     |             0 |    0.0198958 |        0.00608482  |      0.013811   | best           |
| 2025-11-05 |          0.0183817  |           0.00022062 |      0.0186023 |              0     |             0 |    0.0186023 |        0.00641125  |      0.012191   | best           |
| 2026-02-04 |         -0.0203144  |          -0.00579698 |     -0.0261114 |              0.004 |             0 |   -0.0301114 |        0.000918344 |     -0.0310298  | worst          |
| 2025-10-30 |         -0.0179527  |          -0.00585186 |     -0.0238045 |              0     |             0 |   -0.0238045 |       -0.00502998  |     -0.0187745  | worst          |
| 2026-04-28 |         -0.0155385  |          -0.00648657 |     -0.022025  |              0     |             0 |   -0.022025  |       -0.00420625  |     -0.0178188  | worst          |
| 2025-12-12 |         -0.0176522  |          -0.00191647 |     -0.0195687 |              0     |             0 |   -0.0195687 |       -0.00860871  |     -0.01096    | worst          |
| 2026-01-08 |         -0.00940993 |          -0.00887668 |     -0.0182866 |              0     |             0 |   -0.0182866 |        0.00375458  |     -0.0220412  | worst          |

Largest absolute gross-contribution instruments:

| instrument_id   |   gross_return_contribution |   absolute_gross_contribution |   position_day_count |   max_abs_stock_daily_return |
|:----------------|----------------------------:|------------------------------:|---------------------:|-----------------------------:|
| SNDK            |                  0.0573358  |                     0.14649   |                  153 |                     0.179023 |
| HOOD            |                  0.0112854  |                     0.144773  |                  225 |                     0.158321 |
| SATS            |                  0.0460628  |                     0.123234  |                  171 |                     0.702477 |
| COIN            |                  0.00229641 |                     0.106104  |                  162 |                     0.166958 |
| LITE            |                  0.0300041  |                     0.0982511 |                  126 |                     0.171251 |
| AXON            |                 -0.0100293  |                     0.0923974 |                  180 |                     0.175521 |
| MU              |                  0.0288368  |                     0.0842894 |                  126 |                     0.192916 |
| VRT             |                  0.0177692  |                     0.0821225 |                  144 |                     0.244915 |
| CVNA            |                  0.00908917 |                     0.0804315 |                  144 |                     0.141673 |
| APP             |                  0.0084879  |                     0.0761984 |                  108 |                     0.1968   |

These tables are diagnostics. A result dominated by a few dates, a few stocks, or
extreme adjusted returns requires manual company-action and data-quality review.
