# Cross-Sectional Factor Research and Portfolio Construction

MyQuant is a leakage-aware research framework for cross-sectional U.S. equity return prediction, formulaic alpha research, walk-forward validation, and cost-aware portfolio diagnostics.

> **Pre-release evidence boundary:** the intended public baseline is **US300 with a forward 10-trading-day return target**. Public metrics are withheld until a clean rerun verifies missing-OHLCV preservation, the 10-day label purge, one adjusted-price policy, removal of a historical synthetic market-cap proxy, the standard Sharpe formula, signal-horizon endpoint alignment, and full entry/liquidation transaction-cost accounting. The repository also contains a scalable US3000 data pipeline, but no formal US3000 benchmark result is claimed here.

## Research Question

For each trading date, can information available at or before that date rank U.S. equities by their subsequent 10-trading-day return?

The canonical target is:

```text
y_10d[t] = close[t + 10] / close[t] - 1
```

The model output, `predicted_y`, is a cross-sectional ranking signal. It is not a forecast of the future stock price and does not directly constitute a trading order.

## Research Pipeline

```mermaid
flowchart LR
    A[Yahoo Finance OHLCV] --> B[Time split on raw data]
    B --> C[Technical and Alpha191 features]
    C --> D[Daily winsorization and z-score]
    D --> E[Sector neutralization and conditional real-size neutralization]
    E --> F[Train-only feature selection]
    F --> G[Expanding walk-forward validation]
    G --> H[Ridge and Lasso ensemble]
    H --> I[Final untouched OOS audit]
    I --> J[Long-short portfolio diagnostics]
```

The key leakage controls are implemented in [`src/data_loader.py`](src/data_loader.py), [`src/time_series_pipeline.py`](src/time_series_pipeline.py), and [`src/validation.py`](src/validation.py): missing OHLCV observations remain missing, forward labels use one shared market calendar, raw observations are split before feature matrices are built, and the final 10 training observations of every instrument are purged before each validation/OOS boundary. A missing stock close on the exact `t+10` endpoint produces a missing label instead of a later-horizon return. Test-period rolling features may use trailing training history, current observations, and earlier test observations only.

The canonical price convention is `vendor_adjusted`: labels and portfolio close returns use the vendor adjustment factor, while the source file must retain raw OHLC and `Adj Close / Close` rather than hiding the factor behind an all-ones auto-adjusted file. `adjustment` is excluded from model features. Direct absolute price levels are excluded, canonical Alpha191 formulas are restricted to an explicit per-stock price-scale-invariant subset, and dollar turnover is checked against raw close times volume. Feature, preprocessing, and validation cache keys include the price convention and SHA256 fingerprints of cache-relevant source files; validation keys also include the random seed. Raw/adjusted runs, changed formulas, changed preprocessing, changed models, and changed seeds therefore cannot silently share cached artifacts.

## Canonical US300 Experiment Contract

| Item | Public configuration |
|---|---|
| Universe | Static snapshot of 300 U.S. large-cap equities; not point-in-time membership |
| Available local CSV period | 2022-01-03 to 2026-06-04 |
| Canonical sampled period | 2022-01-03 to 2026-06-04 |
| Pre-OOS sample period | 2022-01-03 to 2025-05-30; each boundary applies an additional 10-observation purge |
| Final OOS labels | 2025-06-02 to 2026-05-20; 244 shared-market trading dates before row-level missing labels |
| Target | Forward 10-trading-day close-to-close return |
| Candidate features | Determined by the clean manifest; direct absolute OHLCV/VWAP levels excluded |
| Canonical Alpha191 scope | 11 price-scale-invariant formulas: 001, 002, 004, 005, 006, 015, 018, 019, 020, 022, 023 |
| Selected features per fold | At most 50 |
| Feature filters | Missing ratio <= 60%, variance >= 0.001, pairwise correlation <= 0.95 |
| Models | Ridge and Lasso |
| Validation | 3 expanding-window walk-forward folds |
| Label-overlap control | Last 10 training observations purged per instrument at each boundary |
| Model-selection metric | Mean daily cross-sectional Pearson IC |
| Random seed | 42 |
| Canonical portfolio clock | Signal date `t`, close execution proxy at `t+1`, endpoint at `t+10` |

The current loader does not fabricate market capitalization from price and volume, so the clean run excludes all-empty market-cap and valuation columns. Its exact family counts come only from the run manifest. Fundamental features are not part of this canonical experiment.

### Current Evidence Status

No predictive or portfolio performance number is claimed in this README before the
canonical experiment is rerun with the current code and passes the public release
gate. Superseded artifacts remain available only in the local research worktree and
are excluded from the public Git tree. Those historical values used obsolete
label-boundary, missing-data, price, Sharpe, holding-clock, and transaction-cost
accounting rules.

The clean release package will report every walk-forward fold, untouched final OOS
metrics, the complete 64-cell portfolio grid, full sleeve entry/liquidation costs,
skipped incomplete return paths, anomaly attribution, data fingerprints, source
commit, dirty-worktree status, and exact runtime. Until then, citing a historical IC,
return, Sharpe, or drawdown as current MyQuant performance would be inaccurate.

## Feature Research

The feature layer includes:

- OHLCV-derived returns, momentum, volatility, volume-price, VWAP deviation, channel, oscillator, and range-position features;
- an explicit 11-formula scale-invariant Alpha191 subset for canonical training, with the wider implementation reserved for labelled diagnostics;
- optional macro and market-context proxies;
- optional point-in-time-style fundamental merge utilities for controlled experiments;
- formulaic factor-mining research using warm-start GP-style search, contextual bandits, probabilistic grammar, and PPO.

Formula mining is intentionally reported separately from the canonical result. The repository demonstrates the search and evaluation machinery; it currently makes no public claim that mined factors produce stable incremental OOS portfolio returns.

The strict incremental-factor experiment uses a progressive comparison under identical folds and model settings:

```text
technical/context baseline
-> technical/context + Alpha191
-> technical/context + Alpha191 + validation-selected mined factors
```

Its default model audit includes Ridge, Lasso, and XGBoost. The Alpha191 layer uses the same explicit 11-formula scale-invariant subset as the canonical baseline. Alpha191 is compared with the technical baseline; mined factors are compared with the Alpha191 baseline. This prevents the two feature increments from being combined into one misleading delta.

## Models and Evaluation

Supported model families include linear baselines, random forest, XGBoost, and LightGBM when their optional dependencies are available. The public canonical experiment uses Ridge and Lasso because they make incremental factor value easier to audit and reduce the risk of attributing flexible nonlinear fit to genuine alpha.

Primary prediction metrics:

- **Pearson IC:** mean daily cross-sectional Pearson correlation between signal and realized return;
- **Rank IC:** mean daily cross-sectional Spearman correlation;
- **IC stability:** daily IC median, standard deviation, non-annualized ICIR, and positive-date ratio;
- **RMSE / MAE:** point-forecast error diagnostics;
- **long-short spread:** same-date average realized return of the highest prediction bucket minus the lowest bucket.

Portfolio metrics are kept separate from prediction metrics and include cumulative return, equal-weight relative-wealth context, turnover, cost drag, Sharpe, and maximum drawdown. The equal-weight comparison is not labelled risk-matched alpha because the dollar-neutral and long-only books have different exposure budgets.

## Installation

Python 3.12 is the release target. Python 3.10-3.11 may remain source-compatible, but they are outside the current release gate.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c requirements-lock.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c requirements-lock.txt
```

Deep RL factor mining is optional:

```bash
python -m pip install -r requirements-mining.txt -c requirements-lock.txt
```

XGBoost and LightGBM model comparisons are optional:

```bash
python -m pip install -r requirements-tree.txt -c requirements-lock.txt
```

## Reproduction

### Real-Data Smoke Run

The smoke run downloads 30 real U.S. stocks and limits the workload to Ridge, two folds, and five Alpha factors. It validates the full data-to-prediction route without requiring the private US300 dataset.

```bash
./scripts/run_public_smoke.sh
```

This is an integration check, not a research benchmark. Runtime depends on Yahoo Finance and the local machine.

### Canonical US300 Run

The full public command is stored in one immutable entry point:

```bash
./scripts/run_canonical_us300.sh
```

Windows PowerShell users can run `./scripts/run_canonical_us300.ps1`.

It requires a locally prepared `data/us_large_cap_300_daily.csv`, which is excluded from Git. Exact schema and provenance requirements are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

After training, run `./scripts/run_canonical_us300_backtest.sh`, then use
`python scripts/export_public_evidence.py` and
`python scripts/build_public_figures.py` to regenerate the compact GitHub package.

The strict mined-factor experiment is intentionally separate from the canonical release result:

```bash
python main_mined_factor_strict_experiment.py \
  --data-path data/us_large_cap_300_daily.csv \
  --sample-start-date 2022-01-01 \
  --target-horizon 10 \
  --oos-start-date 2025-06-01 \
  --price-adjustment-mode vendor_adjusted \
  --max-alpha 0 \
  --alpha-factors alpha001 alpha002 alpha004 alpha005 alpha006 alpha015 \
                  alpha018 alpha019 alpha020 alpha022 alpha023 \
  --models ridge lasso xgboost \
  --mined-groups ppo \
  --skip-portfolio
```

The PPO zoo must be generated by `factor_mining_workspace/select_ppo_validation_factor_zoo.py`. Strict ablation requires its companion `validation_selected_factor_zoo_summary.json` and verifies the exact selection source, candidate/config hashes, source-mining data hash, current data hash, sample start, target horizon, price-adjustment mode, canonical formula-field allowlist, validation dates, OOS boundary, and final zoo hash. The required contract is `validation_factor_zoo_v3_scale_invariant_fields_bound`; a renamed historical CSV, an OOS-screened zoo, a raw-price formula, or a legacy PPO run is rejected. Warm-GP remains exploratory until it receives the same train-internal validation selector and is rerun.

## Entry-Point Status

| Status | Entry points | Evidence role |
|---|---|---|
| Canonical | `main.py`, `evaluate.py`, `main_long_short_backtest.py` | The only prediction and portfolio route eligible for the public baseline |
| Controlled diagnostics | `main_factor_diagnostics.py`, `main_alpha_diagnostics.py`, `main_mined_factor_strict_experiment.py` | Factor explanation and progressive technical/Alpha191/mined-factor ablation |
| Optional data extensions | `main_context_data.py`, `main_macro_data.py`, `main_fmp_fundamentals.py` | Build optional context or point-in-time-style inputs; not part of the canonical result |
| Historical exploratory comparisons | `main_experiments.py`, `main_ablation.py`, `main_portfolio_experiments.py`, `main_rolling_oos_backtest.py`, `main_mined_factor_incremental_experiment.py`, `main_strict_weighted_portfolio_experiment.py`, `run_relation_feature_incremental_experiment.py` | Retained for research audit only; their outputs cannot be mixed into the public baseline |

The root directory retains historical experiment drivers because their negative and superseded results remain useful audit evidence. A script's presence does not make its output release evidence; only the canonical wrappers and provenance-checked exporter establish that status.

## Repository Map

```text
MyQuant/
├── main.py                         # canonical training entry point
├── main_long_short_backtest.py     # portfolio diagnostic grid
├── evaluate.py                     # prediction evaluation
├── src/                            # data, features, models, validation, reporting
├── factor_mining_workspace/        # formula language and mining algorithms
├── scripts/                        # reproducible public entry points
├── results/public/us300_release_v1/ # clean public evidence target
├── docs/                           # methodology, results, limits, reproduction
├── requirements.txt                # core research dependencies
├── requirements-lock.txt           # candidate pinned direct dependencies
├── requirements-tree.txt           # optional XGBoost/LightGBM dependencies
├── requirements-mining.txt         # optional PyTorch mining dependencies
├── CONTRIBUTING.md                 # research-integrity contribution rules
└── LICENSE
```

Raw data, cached feature matrices, trained models, full predictions, and exploratory outputs are intentionally excluded from version control.

## Evidence and Limitations

- [Public result package](results/public/us300_release_v1/README.md)
- [Data card](docs/DATA.md)
- [Methodology](docs/METHODOLOGY.md)
- [Results and metric definitions](docs/RESULTS.md)
- [Research limitations](docs/LIMITATIONS.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)

The clean public evidence directory is currently a placeholder. Superseded local artifacts are excluded from the public tree because their original Git commit was not recorded and they predate the repaired loader, purge, price, Sharpe, and portfolio-accounting rules. The canonical command must be rerun before a formal tagged release.

## Data and License

Market data is downloaded by the user from Yahoo Finance for research and education. Users are responsible for the provider's terms and for obtaining any licensed point-in-time constituent, fundamental, or execution data needed for production research.

Code is released under the [MIT License](LICENSE).

Research-integrity and contribution rules are documented in [CONTRIBUTING.md](CONTRIBUTING.md).
