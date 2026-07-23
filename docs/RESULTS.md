# Results and Evidence Status

> **Status: pre-release.** MyQuant currently makes no public predictive-performance
> or portfolio-performance claim. The canonical US300 experiment must be rerun from
> a clean commit with the current leakage, data-quality, price, metric, holding-clock,
> and transaction-cost rules.

## Why Current Numbers Are Withheld

The saved historical artifact set predates several release-blocking corrections:

- missing OHLCV observations are now preserved instead of filled;
- each training boundary purges the final target-horizon observations per stock;
- training labels and portfolio returns now share one adjusted-price policy;
- missing market capitalization is no longer replaced by a price-volume proxy;
- Sharpe now uses daily mean net return divided by daily sample standard deviation;
- the canonical 10-day portfolio ends at the same `t+10` endpoint as `y_10d`;
- every independent sleeve now pays both entry and liquidation turnover;
- a sleeve with an incomplete selected-stock return path is rejected and audited.

These changes can materially alter IC, turnover, cost drag, return, Sharpe, and
drawdown. Reusing historical values would misrepresent the current pipeline.

## Required Release Evidence

The clean public result package must contain:

1. Every expanding walk-forward fold for Ridge and Lasso, including date and purge boundaries.
2. Final untouched OOS aggregate correlation, daily cross-sectional Pearson IC, Rank IC, RMSE, MAE, coverage, and Top-Bottom label spread.
3. Feature-selection funnel, selected features, feature-family counts, and model weights.
4. The complete 64-cell long-short grid for `hold_days={10,20}`, `top_k={10,20,30,50}`, `cost_bps={5,10,20,50}`, and both neutralization modes.
5. Daily returns, full sleeve entry/liquidation turnover, skipped return paths, benchmark comparison, Sharpe, drawdown, and anomaly attribution.
6. Source commit, clean-worktree status, command, environment, data SHA256, prediction SHA256, and artifact hashes.

The exporter and release gate reject an incomplete grid, a legacy holding clock,
dirty or missing source provenance, a mismatched data/prediction fingerprint, or a
package still marked pre-release.

## Interpretation Rules

- Aggregate Pearson/Spearman correlation and daily cross-sectional IC answer different questions and are reported separately.
- `long_short_spread` is a same-date Top-Bottom realized-label diagnostic. It is not cumulative portfolio return.
- Portfolio return, benchmark-relative wealth, turnover, cost drag, Sharpe, and drawdown come only from the portfolio simulator.
- Benchmark-relative wealth compares a gross-2 dollar-neutral book with a gross-1 long-only universe. It is market context, not a risk-matched excess-return or alpha estimate.
- Canonical validation folds are used for model and feature decisions. In the separate mined-factor experiment, formulas have already seen a pre-OOS validation segment, so downstream fold tables are calibration diagnostics unless formula mining is nested inside every outer fold. Final OOS is opened only after all such decisions are fixed.
- A short OOS window or a small rebalance count cannot establish stable tradability, even if its point estimate is high.
- Formula-mining results remain a separate incremental experiment and are not part of the canonical baseline claim.

## Legacy Audit Boundary

Superseded numeric artifacts remain local and are excluded from the public result
tree. They predate the current research contract and must not be quoted in a
resume, interview, README, or current research conclusion. Only a package exported
to [`results/public/us300_release_v1/`](../results/public/us300_release_v1/README.md)
by the current provenance checks can become public evidence.

See [`DATA.md`](DATA.md), [`METHODOLOGY.md`](METHODOLOGY.md),
[`LIMITATIONS.md`](LIMITATIONS.md), and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
for the contracts that govern the clean rerun.
