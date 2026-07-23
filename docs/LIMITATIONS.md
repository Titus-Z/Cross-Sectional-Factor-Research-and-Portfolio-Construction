# Research Limitations

This document defines the boundary of every public performance claim.

## 1. Universe Bias

The US300 dataset is a controlled large-cap research universe. It is not a fully point-in-time historical constituent database. Selecting stocks using a recent membership or liquidity snapshot can create survivorship and look-ahead bias because future survivors may be represented in earlier periods while delisted or failed companies are absent.

The US3000 builder ranks currently listed candidates by recent dollar volume. It does not reconstruct historical monthly eligibility, delistings, ticker changes, or delisting returns. Formal US3000 benchmark results are therefore pending.

## 2. Price Adjustments and Corporate Actions

Yahoo Finance adjusted history helps normalize splits and distributions, but it is not an institutional corporate-action master. The project does not independently reconcile:

- splits and reverse splits;
- cash and special dividends;
- spin-offs and mergers;
- ticker and share-class changes;
- delisting returns.

Any production-grade study requires a documented total-return convention and independent corporate-action validation.

The release pipeline writes an event-level `corporate_action_audit.csv`. It can
show whether a vendor adjustment removed a split-like raw jump and can flag a
large residual adjusted return. It cannot infer authoritative event types or
prove that the vendor history contains every action.

The current canonical code uses `vendor_adjusted` prices consistently in labels, features, and portfolio close returns. Superseded local reference artifacts predate this unified policy, so they remain outside the public result tree. Vendor-adjusted history can also be revised retrospectively, which limits exact bit-for-bit reproduction over time.

Back-adjusted vendors may apply a later split factor to earlier absolute prices.
That makes absolute historical price levels unsafe even when the model never sees
the `adjustment` column directly. The public canonical feature set therefore
excludes direct OHLC/VWAP levels and limits Alpha191 to an explicit 11-formula
price-scale-invariant subset. Formula-zoo loaders also reject fields outside the
canonical candidate list. The full Alpha implementation remains exploratory and
must not be mixed into public results without a separate invariance audit.

The canonical input must retain raw OHLC and `Adj Close / Close`. An all-ones
auto-adjusted file hides the adjustment path and prevents raw-dollar-turnover
verification, so it cannot produce public evidence. This requirement reduces one
class of historical rescaling leakage; it does not turn Yahoo data into a complete
point-in-time corporate-action database.

## 3. Point-in-Time Fundamentals

Fundamental utilities are optional and do not contribute to the canonical US300 result. A valid merge must use the public filing or availability date plus a conservative reporting delay. Fiscal period-end dates alone would leak information that the market had not yet received. When fields from multiple statements are combined, the implementation uses the latest source availability date so a later-filed statement is not exposed early.

The current free/API-limited fundamental sample has incomplete coverage and should only support controlled ablation, not a broad claim. The downloaded endpoint may expose subsequently restated historical values rather than a complete as-reported vintage archive. Availability-date merging cannot remove that revision bias; serious point-in-time research requires vendor data with filing vintages.

## 4. OOS Window Length

The final audit is configured from 2025-06-01 and has 244 shared-market dates
with potentially known 10-day labels from 2025-06-02 to 2026-05-20. This is a
material improvement over the superseded 96-date audit, but it still represents
roughly one year and cannot cover a full recession, crisis, low-volatility,
rate-cutting, and rate-hiking cycle. High IC or Sharpe can still be dominated by
one regime. Clean reports therefore retain 3/6/12-month, first/second-half, and
monthly stability views without treating overlapping windows as independent
proofs.

## 5. Overlapping Labels

Ten-day forward returns overlap on adjacent dates. Daily IC has many observations, but those observations are serially dependent. Effective sample size is lower than the row count suggests. The clean canonical rerun must report its exact rebalance count; no minimum sample-size claim is inferred from row count alone.

## 6. Multiple Testing

The project contains many technical features, Alpha191 formulas, model choices, portfolio settings, and automated factor-search candidates. Repeatedly observing the same OOS period can turn it into an implicit validation set. Search procedures need train-only rewards, purged or embargoed validation, and an untouched final audit.

The public package does not claim mined-factor improvement because a strict same-protocol incremental result is not yet included.

## 7. Cross-Sectional Dependence

Stock observations on the same date share market, industry, and macro shocks. Row count therefore overstates independent information. Daily cross-sectional IC is more appropriate than a pooled correlation, but inference should still account for serial dependence in the daily IC series.

## 8. Execution Model

The portfolio simulator applies proportional transaction cost to capital-scaled traded notional. Each independent sleeve pays both entry and final liquidation turnover; cross-sleeve netting is not assumed. Overlapping sleeves retain both raw sleeve turnover and portfolio-level turnover so reported cost matches the amount deducted from the daily return ledger. Every market date between first execution and final liquidation remains in the daily ledger, including zero-return execution or cash dates. A sleeve with an incomplete selected-stock return path is excluded and recorded in `skipped_trades.csv`; the simulator does not silently convert missing selected positions into cash. Because that exclusion is known only after the signal date, any such sleeve creates selection risk. The public evidence exporter therefore requires zero skipped sleeves; otherwise the market-data or security-master gap must be resolved before release. The 5/10/20/50 bps grid should be read as an all-in friction sensitivity range, not as a calibrated execution model. The canonical grid includes 20 bps and zero borrow fee as sensitivity assumptions; annualized borrow fee is configurable. The simulation does not calibrate:

- commissions and exchange fees separately;
- realized bid-ask spread;
- order-size-dependent slippage;
- nonlinear market impact;
- participation-rate or capacity limits;
- borrow availability and recalls;
- hard-to-borrow exclusions;
- margin, financing, and locate fees.

The saved pre-release portfolio rows used an execution-date holding clock and ended one trading day after the `y_10d` label endpoint. They are invalid as release evidence. Canonical reruns use `holding_clock=signal_horizon`: signal at `t`, close execution proxy at `t+1`, endpoint at `t+10`, and nine executable daily returns. The old `execution_horizon` definition remains available only for a separately labelled sensitivity comparison.

The portfolio result is a research diagnostic, not a live performance record.

## 9. Benchmark Interpretation

The reported benchmark is an equal-weight long-only reference over the available universe. The long-short book is approximately dollar neutral with gross exposure near 2, so its relative wealth versus that gross-1 benchmark is not a risk-matched excess return or asset-pricing alpha. Sector neutralization reduces one selected exposure; size neutralization is available only when real market capitalization has sufficient coverage. Neither control guarantees neutrality to beta, momentum, volatility, quality, or other omitted styles.

## 10. Data Vendor Reliability

Yahoo Finance is convenient for reproducible educational research. Downloads may change, fail, or contain vendor corrections. The repository does not distribute raw vendor data, and a rerun may not reproduce every historical row bit-for-bit.

## 11. Historical Synthetic Size Proxy

A superseded local artifact was produced when the pipeline could substitute a price-volume proxy for missing market capitalization. That quantity measures trading activity, not company size. It could contaminate market-cap ranks, turnover-rate proxies, valuation fields, and the reported size-neutralization flag. The current code leaves missing market capitalization as missing, excludes all-empty derived fields, and records actual coverage. This is an additional reason old feature counts and metrics are excluded from public evidence.

### Return concentration

A high cumulative return or Sharpe can be driven by a few dates or a few names. The current backtest writes `extreme_return_days.csv`, `position_daily_contributions.csv`, and `instrument_return_attribution.csv`, but those files only become evidence after the clean canonical rerun. Any configuration with large selected-stock returns, concentrated absolute contribution, or sharply different first- and second-half returns requires manual review before publication.

### Static sector classification

The bundled sector map is a current static classification. Applying it to older dates can misclassify firms that changed business mix or classification. Sector-neutral outputs therefore control the project's static taxonomy; they do not prove point-in-time industry neutrality.

## 12. Provenance Gap

The superseded local evidence package was assembled from saved artifacts and did not store its exact Git commit. The loader used at that time allowed backward filling of observable fields, the saved folds did not record a target-horizon purge, training/backtest price adjustment was not governed by the current unified policy, and Sharpe used compound annualized return divided by annualized volatility. The current loader preserves missing OHLCV without forward/backward filling, the protocol purges the last 10 observations of every instrument before each `y_10d` boundary, price mode is recorded, and Sharpe now uses the standard daily mean/std formula. A fresh run must be completed before a tagged public release can claim current-code performance or full code-to-result provenance.

## 13. Practical Interpretation

After the clean canonical rerun, the strongest conclusion this repository may test is:

```text
In a controlled US300 sample, the documented feature and linear-model pipeline
can be evaluated for cross-sectional ranking quality across three purged
walk-forward folds and one held-out 2026 audit.
```

Even a positive clean result would not establish a persistent, scalable,
executable trading strategy.
