# Methodology

## 1. Research Objective

MyQuant studies a cross-sectional ranking problem. On each date, the model assigns a score to every stock using information available by that date. The score is evaluated against the stock's subsequent return.

The canonical target is:

```text
y_10d[t] = close[t + 10] / close[t] - 1
```

This label measures a forward close-to-close return. It is never allowed into the feature matrix.

## 2. Canonical Data Boundary

The public canonical experiment uses a controlled static snapshot of 300 U.S. large-cap stocks. It is not a point-in-time constituent history:

- raw sampled date range: 2022-01-03 to 2026-06-04;
- pre-OOS sample: 2022-01-03 to 2025-05-30, followed by the documented 10-observation boundary purge;
- final OOS dates with potentially known 10-day labels: 2025-06-02 to 2026-05-20;
- 300 instruments and 244 shared-market OOS dates before row-level missing labels. Exact usable rows come from the clean run manifest.

US3000 support is a scalability extension. Its data builder and compatible model pipeline are retained, while formal US3000 model and portfolio metrics remain outside the public evidence claim.

### 2.1 Price Convention

The canonical experiment uses `vendor_adjusted` OHLC for labels and return paths. The input must preserve raw OHLC and `adjustment = adjusted_close / raw_close`; the loader applies that factor to OHLC and VWAP before constructing returns and labels. An all-ones file produced by `yfinance(auto_adjust=True)` cannot support the required corporate-action or raw-dollar-turnover audit and is rejected from public evidence.

The same rule is applied when portfolio close-to-close returns are reconstructed. `adjustment` is a data-cleaning field and is never part of the feature matrix. A `raw` mode exists only for corporate-action sensitivity analysis. Cache keys include the price convention so adjusted and raw experiments cannot reuse each other's feature matrices or validation results.

## 3. Leakage Controls

### 3.1 Split Raw Data First

[`src/time_series_pipeline.py`](../src/time_series_pipeline.py) performs the train/OOS split on raw observations before creating the two feature matrices. Even `log_return` is constructed inside the post-split feature generator rather than in the CSV loader.

Training features are generated using training observations only. OOS features are generated using:

- the trailing history from the training period;
- the current OOS observation;
- earlier OOS observations.

Future OOS observations are not needed by trailing rolling operators.

Missing observations inside trailing windows are not forward- or backward-filled. Weighted rolling operators keep the original time positions of valid observations and renormalize only across those observed values. This avoids manufacturing a flat or interpolated market path while retaining the information that was actually available.

### 3.2 Historical Context

Some formulas use windows as long as 230-250 trading days. The OOS matrix therefore receives up to 260 trailing training observations per stock. Those rows provide history for rolling calculations and are removed before OOS evaluation.

### 3.3 Forward-Label Purge

A 10-day label near a split boundary can use a price from the next block. The current pipeline removes the final 10 training observations separately for every instrument before the outer OOS boundary and before every walk-forward validation block. This is stricter than deleting 10 global dates because suspended, newly listed, or incomplete instruments can have irregular calendars. Purged rows remain available as observable history for rolling OOS features, but their cross-boundary labels never enter feature selection or model fitting.

### 3.4 Train-Only Decisions

The following decisions are made inside the training or walk-forward validation boundary:

- missingness filtering;
- low-variance filtering;
- correlation pruning;
- Top-N feature selection by absolute mean daily cross-sectional Pearson IC;
- model hyperparameter comparison;
- validation-score model weighting with 50% shrinkage toward equal weights.

The final OOS period is reserved for one audit pass. It must not be used to choose factors, hyperparameters, model weights, Top-K, or transaction-cost assumptions.

### 3.5 Forbidden Fields

Targets and future-derived fields such as `y`, `y_*`, `future`, `target`, `next_open`, and `predicted_y` are excluded from the feature interface. `adjustment` is not used as a model feature in the canonical experiment.

## 4. Feature Families

The clean manifest is the only source of exact feature counts. Candidate families
include scale-free technical structure, context variables, available fundamental
fields, and a bounded Alpha191-style subset. All-empty market-cap, valuation, or
market-cap-derived columns are excluded.

Absolute OHLCV/VWAP levels, moving-price levels, raw Bollinger bands, and similar
scale-dependent intermediates remain available for formula construction but do not
enter the canonical candidate list directly. This prevents later vendor split
rescaling from acting as an arbitrary historical price-level feature. The retained
technical candidates emphasize returns, momentum, realized volatility, range and
channel position, volume/turnover ratios, ranks, and normalized oscillators. The
canonical experiment uses only Alpha001, Alpha002, Alpha004, Alpha005, Alpha006,
Alpha015, Alpha018, Alpha019, Alpha020, Alpha022, and Alpha023. These formulas are
unchanged when one stock's OHLC/VWAP history is multiplied by an arbitrary positive
constant. Scale-dependent Alpha191 formulas remain available for explicitly labelled
raw-price or point-in-time-adjustment research, but they cannot enter public baseline
evidence. Dollar turnover is derived from raw close times volume when absent and its
consistency is recorded in `data_quality_summary.json`.

The canonical experiment contains zero fundamental features. Fundamental download and point-in-time-style merge utilities are optional research extensions and cannot be cited as part of the public US300 result.

## 5. Cross-Sectional Preprocessing

For each date, cross-sectionally varying features pass through:

1. 1% winsorization to reduce the influence of extreme values;
2. cross-sectional z-score normalization;
3. linear neutralization against sector dummy variables;
4. additional log-market-cap neutralization only when real positive market capitalization covers at least 90% of that date's cross-section.

Date-level market context fields are not cross-sectionally standardized because every stock has the same value on a given date. When real market capitalization is available, preprocessing retains a non-feature copy of the observable raw capitalization so later mined-factor blocks use the same exposure even if a model feature named `market_cap` is standardized. The canonical CSV currently lacks real market-cap coverage, so the expected clean-run policy is sector neutralization without size neutralization. No price-volume proxy is substituted. The later sector-neutral portfolio constraint remains useful because model predictions and holdings can reintroduce aggregate sector exposure.

## 6. Feature Selection

Each training fold applies the following sequence:

1. remove features whose missing rate exceeds the configured threshold;
2. impute remaining missing values using training-derived statistics;
3. remove features below the variance threshold;
4. prune one feature from highly correlated pairs;
5. rank surviving variables by absolute mean daily cross-sectional Pearson IC with `y_10d` inside that training fold;
6. retain at most 50 features.

No statistic is fitted on the final OOS matrix.

## 7. Walk-Forward Validation

The release protocol uses three expanding-window folds with a 10-observation
per-instrument label purge. Later folds receive more historical training data,
while every validation block occurs strictly after the purged training block. The
clean fold table records the actual pre-purge and effective boundaries; documentation
does not hard-code dates from an obsolete run. The primary selection metric is mean
daily cross-sectional Pearson IC. Reports also retain IC median, standard deviation,
non-annualized mean/std ICIR, positive-date ratio, Rank IC, RMSE, MAE, prediction
coverage, group monotonicity, and a same-date long-short spread.

## 8. Models

The public experiment compares Ridge and Lasso. Linear models make the contribution of selected features easier to inspect and provide a disciplined baseline before flexible tree models are introduced. Positive validation scores are normalized and blended 50/50 with equal weights so a small, noisy fold-level difference cannot create a near-all-or-nothing ensemble.

The codebase also supports random forest, XGBoost, and LightGBM. Their presence is an implementation capability; models are not mixed into the canonical result unless they use the same data, folds, features, and OOS boundary.

## 9. Formulaic Factor Mining

The factor-mining workspace contains four search families:

- warm-start GP-style mutation and recombination;
- contextual-bandit action selection;
- probabilistic grammar sampling;
- PPO-based multi-step formula generation.

All searchers share a constrained formula language, forbidden-field checks, candidate caching, and factor diagnostics. A valid mining claim requires a strict experiment in which candidate selection uses training/validation data only and the selected factors are retrained inside the same model protocol as the baseline.

The strict model-layer audit is progressive. It first compares a technical/context baseline with the same baseline plus the canonical 11-formula scale-invariant Alpha191 subset, then compares that bounded Alpha191 baseline with a validation-selected mined-factor augmentation. The price-adjustment mode is recorded and checked across PPO mining, validation-only factor selection, strict model fitting, and portfolio evaluation. Materialized formulas receive the same daily winsorization, z-score, and available sector/real-size neutralization as canonical candidates before Top-N selection. Ridge, Lasso, and XGBoost are evaluated under identical folds by default. Because a mined formula has already been selected using pre-OOS validation data, its historical fold table is an in-sample calibration diagnostic rather than a nested, independent estimate of mined-factor performance. A fully independent fold estimate would require rerunning formula mining inside every outer fold. The current claim therefore comes only from the untouched final OOS comparison. Fixed 3/6/12-month OOS windows are diagnostic views of that one audit and are never used to reselect formulas or tune the models.

The canonical public package does not currently claim mined-factor improvement.

## 10. Portfolio Diagnostic

Predictions are converted into a dollar-neutral long-short portfolio:

- rank stocks by `predicted_y` on each rebalance date;
- buy the Top-K and short the Bottom-K;
- allocate +1 total long weight and -1 total short weight;
- optionally select within sectors;
- delay the signal by one trading day;
- interpret `step_days` on the shared market trading calendar, not on the compressed list of dates present in a prediction file. A missing complete signal date inside the prediction range fails closed instead of shifting every later rebalance;
- use `holding_clock=signal_horizon` for the canonical experiment: a 10-day signal observed at `t`, executed with a close proxy at `t+1`, stops at the label endpoint `t+10` and therefore accrues nine executable close-to-close daily returns;
- retain `holding_clock=execution_horizon` only as a historical sensitivity mode, where execution at `t+1` is followed by ten complete returns through `t+11`;
- when positions overlap, maintain up to `ceil(effective_holding_days / step_days)` rotating sleeves and allocate equal capital to each active sleeve;
- treat every sleeve as a complete cash-to-position-to-cash lifecycle: charge entry turnover when the sleeve opens and liquidation turnover when it closes. Cross-sleeve order netting is not assumed, so this accounting is conservative and does not give repeated names a free exit or re-entry;
- retain every market date between the first execution and final liquidation in the daily ledger. Execution and cash dates carry zero gross position return but still participate in Sharpe, annualization, benchmark comparison, and any applicable trade cost;
- reject a sleeve from the performance ledger when any selected stock lacks the complete requested return path. Rejected attempts are written to `skipped_trades.csv` instead of silently dropping stocks and changing the selected portfolio after the fact;
- deduct proportional costs from traded notional. When holdings overlap, every sleeve keeps a full-notional audit ledger but its turnover, exposure, and cost are scaled by that sleeve's share of total portfolio capital before entering public metrics. The 5/10/20/50 bps grid is an all-in friction sensitivity range, not a venue- or order-level slippage estimate.

The benchmark is the daily equal-weight close return of stocks available in the market snapshot. The dollar-neutral long-short book has gross exposure near 2, while this benchmark is a gross-1 net-long portfolio. Their risk budgets differ, so `portfolio_nav / benchmark_nav - 1` is reported only as `relative_wealth_vs_equal_weight_long_only`; it is market context, not matched-risk alpha or strict excess return. Sharpe uses `mean(daily net return) / sample_std(daily net return) * sqrt(252)` with a zero risk-free rate. Borrow fee is configurable as an annualized bps assumption, while borrow availability, recalls, and hard-to-borrow exclusions remain unmodeled. Portfolio returns, turnover, cost drag, Sharpe, drawdown, relative wealth, first/second-half return, and monthly stability are reported separately from model IC. Each detailed run also saves position-day gross contributions, the five best and worst net-return days, instrument contribution concentration, and selected-stock returns above 20% and 50% absolute thresholds. These are anomaly diagnostics used to challenge unusually strong results. Any future public portfolio row remains a research diagnostic unless calibrated slippage, nonlinear impact, and short-locate constraints are added.

Single-factor diagnostics report two separate turnover proxies. `rank_turnover_mean` is the mean absolute change in percentile rank among stocks observed on adjacent dates. `top_retention_mean` is the fraction of the prior date's Top 20% set that remains in the next date's Top 20%. They describe signal stability and must not be interpreted as realized portfolio turnover.
