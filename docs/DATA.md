# Data Card

## Canonical Research Dataset

The public research contract uses a static list of 300 U.S. large-cap stocks and
Yahoo Finance daily OHLCV. Each row represents one instrument on one trading date.

Required fields:

```text
instrument_id,date,open,high,low,close,volume
```

Optional fields include `vwap`, `adjustment`, `sector`, `market_cap`, and
`turnover`. The public canonical dataset must retain raw OHLC plus
`adjustment = adjusted_close / raw_close`; an all-ones file whose OHLC was
silently auto-adjusted is insufficient for the required audit. Forward-return
columns are reconstructed from the selected adjusted
close convention before training. Their endpoints use one shared market trading
calendar. If a stock has no close on the exact `t+N` market date, its label is
missing; the target builder does not jump to that stock's next observed row.

The current canonical US300 CSV contains the `market_cap` column but has no
usable market-cap observations. The loader preserves those values as missing.
It does not substitute `close * volume` or another liquidity proxy. Therefore,
the clean canonical run excludes market-cap-derived features and does not claim
size neutralization. Real size controls require point-in-time shares outstanding
or market capitalization with availability timestamps.

## Universe Construction

The built-in US300 list is a recent large-cap snapshot assembled from high-weight
S&P 500 names. The original retrieval timestamp and licensed constituent source
were not stored by the historical project. It therefore provides a controlled
cross-sectional sample, not point-in-time membership.

Consequences:

- companies that later failed or were delisted may be absent from early dates;
- recent entrants can appear throughout their available downloaded history;
- results can contain survivorship and constituent look-ahead bias;
- the universe cannot support a claim of historical index replication.

The run manifest records the explicit label
`us_large_cap_300_static_snapshot` so this limitation cannot be hidden behind a
generic `US300` name.

The built-in sector labels are also a static current/project mapping rather than
a point-in-time classification history. Sector-neutral preprocessing and
industry-within ranking can reduce exposure to this taxonomy, but they do not
reconstruct past sector changes. Unknown labels remain `Unknown`, and their
coverage is written to the run manifest.

## Price And Corporate-Action Policy

Canonical runs use `vendor_adjusted` OHLC. When a raw Yahoo file contains
`adjustment = Adj Close / Close`, the loader applies that factor before computing
returns, labels, and price-derived features. The same adjusted close convention is
used in the portfolio simulator. If an adjustment-aware file contains a missing or
nonpositive factor, that row's adjusted prices remain missing. The loader does not
replace the factor with `1.0` and silently mix raw and adjusted observations.

Yahoo downloads therefore use `auto_adjust=False`. A file whose OHLC was already
auto-adjusted and whose adjustment is all ones may be used only in a separately
labelled compatibility experiment; the public evidence exporter rejects it. The
manifest also verifies that supplied dollar turnover is consistent with raw close
times volume. If turnover is absent, the loader constructs it from the preserved
raw close before applying any back-adjustment.

Back-adjustment can rescale historical absolute prices using a later corporate
action. For that reason, public model candidates exclude direct OHLC/VWAP levels,
and the canonical Alpha191 block is restricted to the following formulas whose
values are invariant to multiplying one stock's entire price history by an
arbitrary positive constant:

```text
alpha001, alpha002, alpha004, alpha005, alpha006,
alpha015, alpha018, alpha019, alpha020, alpha022, alpha023
```

The complete Alpha191 implementation remains available for explicitly labelled
diagnostic research, but it is outside the canonical public training contract.

This policy reduces false jumps around splits and distributions. It does not
independently validate reverse splits, special dividends, spin-offs, mergers,
ticker changes, or delisting returns. Yahoo may revise adjusted history.

## Missing Data

Missing OHLCV observations remain missing. Neither forward fill nor backward fill
is used to manufacture market prices or volume. Rolling features enforce their
own minimum-history rules, and any final model-feature imputation is fitted on
training folds only. Point-in-time fundamentals are carried forward only by the
dedicated availability-date merge, not by the generic market-data loader.
If one quarterly row combines multiple statement endpoints, its effective date
is the latest availability date among those sources. FMP history may still
contain later restatements, so the optional fundamental sample is described as
point-in-time-style rather than a complete as-reported vintage database.

The run manifest reports:

- duplicate instrument/date rows;
- missing ratios for core market columns;
- nonpositive closes;
- zero-volume and stale-price observations;
- adjusted-price daily returns above 50% and 100%;
- adjustment-factor changes above 20%;
- unknown-sector coverage;
- non-unit adjustment coverage;
- raw-close turnover evaluable and consistency ratios;
- minimum, median, and maximum observations per instrument;
- instruments with fewer than 252 observations and the range of first/last dates.

Each canonical run also writes `corporate_action_audit.csv` and
`data_quality_summary.json`. For every adjustment-factor change, the audit
reconstructs raw and adjusted close returns, identifies large raw jumps removed by
the vendor adjustment, and flags residual adjusted returns above 20% or 50% for
manual review. These rows are review candidates; they do not independently prove
the type or completeness of a corporate action.

`universe_coverage_audit.csv` retains one row per instrument with its first and
last observation, history length, coverage ratio, and early-end/late-start flags.
An early end is labelled only as a possible delisting, merger, ticker change, or
data gap. The project does not infer an event type without a point-in-time
security master.

## US3000 Extension

The US3000 builder starts from currently listed U.S. symbols and ranks them by
recent average dollar volume. It supports resumable Parquet downloads, but does
not reconstruct historical eligibility, delistings, ticker changes, or delisting
returns. US3000 model and portfolio results therefore remain outside the public
canonical evidence until a separate, same-protocol benchmark is completed.

## Production Upgrade Path

A production-grade dataset would require:

1. point-in-time universe membership;
2. delisted securities and delisting returns;
3. a licensed corporate-action master;
4. filing-availability timestamps for fundamentals;
5. documented exchange calendar and stale-price rules;
6. immutable vendor snapshots or versioned data hashes.
