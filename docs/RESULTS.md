# Results and Evidence Status

> **Status: research release candidate.** The canonical US300 experiment was
> regenerated from a clean source commit under the current leakage, data-quality,
> price, metric, holding-clock, and transaction-cost rules.

## Canonical Contract

| Item | Value |
|---|---|
| Universe | Static US300 large-cap snapshot |
| Target | Forward 10-trading-day close-to-close return |
| Models | Ridge and Lasso |
| Validation | 3 purged expanding walk-forward folds |
| Final OOS | 2025-06-02 to 2026-05-20 |
| Final OOS dates | 244 |
| Candidate / selected features | 71 / 50 |
| Canonical Alpha191 scope | 11 price-scale-invariant formulas |
| Price policy | Vendor-adjusted |

The static universe is not point-in-time membership. Read
[`LIMITATIONS.md`](LIMITATIONS.md) before interpreting any metric.

## Walk-Forward Summary

| Model | Mean Pearson IC | Mean Rank IC | RMSE | MAE |
|---|---:|---:|---:|---:|
| Ridge | 0.0261 | 0.0235 | 0.0652 | 0.0462 |
| Lasso | 0.0221 | 0.0163 | 0.0651 | 0.0460 |

These are internal walk-forward summaries used for model weighting. The complete
fold table is published in
[`walk_forward_fold_metrics.csv`](../results/public/us300_release_v1/walk_forward_fold_metrics.csv).

## Final Untouched OOS Audit

| Metric | Result |
|---|---:|
| Aggregate Pearson correlation | 0.0690 |
| Aggregate Spearman correlation | 0.0093 |
| Mean daily cross-sectional Pearson IC | 0.1126 |
| Mean daily cross-sectional Rank IC | 0.0610 |
| Pearson IC positive-date ratio | 81.56% |
| Rank IC positive-date ratio | 72.95% |
| RMSE | 0.0721 |
| MAE | 0.0484 |
| Same-date Top-Bottom label spread | 2.84% |

Aggregate correlations and mean daily IC answer different questions. The
Top-Bottom label spread is a same-date prediction diagnostic and is not cumulative
portfolio return.

## Portfolio Evidence

The public package contains all 64 combinations:

```text
hold_days = {10, 20}
top_k = {10, 20, 30, 50}
cost_bps = {5, 10, 20, 50}
neutral_mode = {unconstrained, sector_neutral}
```

The main conservative slice is fixed for presentation rather than selected as the
highest-Sharpe grid cell:

| Setting | Value |
|---|---:|
| Portfolio | Top20 long / Bottom20 short |
| Neutralization | Sector-neutral using the project's static sector map |
| Signal / rebalance horizon | 10 trading days |
| Execution delay | 1 trading day |
| Effective return days per sleeve | 9 |
| Cost | 20 bps per traded notional |
| Rebalances | 25 |
| Cumulative net return | 38.29% |
| Sharpe | 1.97 |
| Maximum drawdown | -8.09% |
| Equal-weight long-only benchmark return | 32.72% |
| Relative wealth versus benchmark | 4.19% |

This comparison is not risk-matched alpha. The long-short book has gross exposure
near 2 while the benchmark is gross-1 long-only. The simulator charges separate
entry and liquidation turnover, producing an average 80 bps capital cost per
rebalance at a 20 bps rate.

Borrow cost is zero in this displayed slice. Bid-ask spread, slippage, nonlinear
impact, capacity, borrow availability, recalls, margin, financing, and taxes are
not calibrated. The result is a research backtest, not a live track record.

## Data-Quality Review

The canonical data has:

- no duplicate instrument-date rows;
- no nonpositive close rows;
- no invalid OHLC ordering;
- two adjusted absolute daily returns above 50%;
- one adjustment-change date with residual adjusted return above 20%;
- no adjusted absolute daily return above 100%.

The threshold events were reviewed against same-day primary sources:

- TGT on 2024-11-20: Target third-quarter results and revised guidance;
- CVNA on 2023-06-08: Carvana Form 8-K and improved outlook;
- SATS on 2025-08-26: EchoStar spectrum transaction announcement.

See [`MANUAL_DATA_REVIEW.md`](../results/public/us300_release_v1/MANUAL_DATA_REVIEW.md).
This narrow event review does not prove complete corporate-action or delisting
coverage.

## Evidence Package

The complete compact package is stored in
[`results/public/us300_release_v1/`](../results/public/us300_release_v1/README.md).
It contains:

1. every walk-forward fold and model summary;
2. final OOS metrics;
3. selected features, feature-family counts, and model weights;
4. the complete portfolio grid and cost sensitivity;
5. daily ledgers and position-level attribution for the four displayed 20 bps rows;
6. data-quality, corporate-action, universe-coverage, and manual review artifacts;
7. source commit, environment, data SHA256, and artifact provenance.

Raw vendor data, predictions, binary models, and caches remain excluded from Git.

## Claim Boundary

The evidence supports a reproducible research claim:

```text
In a controlled static US300 sample, this feature and linear-model pipeline
produced positive mean daily cross-sectional IC over a one-year final OOS window,
and one pre-specified sector-neutral portfolio diagnostic remained positive after
20 bps proportional traded-notional costs.
```

It does not establish persistent alpha, live executability, point-in-time universe
validity, capacity, or production readiness. Formula-mining results remain a
separate experiment and have no public stable-increment claim.
