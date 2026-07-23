# Reproducibility Guide

## 1. Supported Environment

- Release target: Python 3.12
- Python 3.10-3.11: expected source compatibility, not part of the current release gate
- Operating systems: macOS, Linux, or Windows
- Core dependencies: [`requirements.txt`](../requirements.txt)
- Candidate release constraints: [`requirements-lock.txt`](../requirements-lock.txt)
- Optional XGBoost/LightGBM dependencies: [`requirements-tree.txt`](../requirements-tree.txt)
- Optional Deep RL dependencies: [`requirements-mining.txt`](../requirements-mining.txt)

Create an isolated environment before running any experiment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c requirements-lock.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c requirements-lock.txt
```

`requirements-lock.txt` pins direct dependencies for the release candidate. It
becomes the formal release lock only after the clean Python 3.12 CI/smoke
environment installs and runs successfully. XGBoost and LightGBM are installed
through `requirements-tree.txt`; PyTorch is installed through
`requirements-mining.txt`.

## 2. Data Policy

Raw market data is excluded from Git. The public smoke run downloads its own small real-data sample. The canonical US300 run requires a locally supplied CSV.

Required core columns:

```text
instrument_id,date,open,high,low,close,volume
```

Supported optional columns include:

```text
vwap,adjustment,market_cap,turnover,sector,next_open,y_1d,y_5d,y_10d,y
```

The loader validates and derives supported targets when enough market data is available. Never place target or future-price fields in a manually supplied feature list.

Canonical runs use `--price-adjustment-mode vendor_adjusted`. Yahoo downloads must use `auto_adjust=False` so the CSV retains raw OHLC and `adjustment = Adj Close / Close`. An all-ones auto-adjusted file is allowed only in a separately labelled compatibility experiment and is rejected by the public evidence exporter. The training and portfolio commands must use the same mode. The run manifest records this value, verifies raw-close dollar turnover, and cache keys isolate it. Cache keys also hash cache-relevant source files; validation cache keys include the random seed, preventing an old implementation or seed from being presented as a new run.

## 3. Real-Data Smoke Run

macOS or Linux:

```bash
./scripts/run_public_smoke.sh
```

Windows PowerShell:

```powershell
.\scripts\run_public_smoke.ps1
```

Expected local artifacts:

```text
data/smoke_us30_daily.csv
models/public_smoke_us30/
outputs/public_smoke_us30/
```

These paths are ignored by Git. The smoke experiment checks integration only and must not be compared with the canonical US300 metrics.

## 4. Canonical US300 Run

Place the prepared dataset at:

```text
data/us_large_cap_300_daily.csv
```

Then run:

```bash
./scripts/run_canonical_us300.sh
```

Windows PowerShell:

```powershell
.\scripts\run_canonical_us300.ps1
```

The canonical training and backtest wrappers stop before expensive work when Git
contains tracked or untracked source changes. This ensures the recorded commit can
reproduce the code that actually ran. `ALLOW_DIRTY_CANONICAL=1` is available only
for local debugging; evidence from that route is rejected by the public exporter.

The script fixes the public contract:

- sample start: 2022-01-01;
- OOS start: 2025-06-01;
- target horizon: 10 trading days;
- Alpha scope: the explicit price-scale-invariant set Alpha001, Alpha002, Alpha004, Alpha005, Alpha006, Alpha015, Alpha018, Alpha019, Alpha020, Alpha022, and Alpha023;
- models: Ridge and Lasso;
- walk-forward folds: 3;
- selected features: 50;
- feature filters: missing ratio 0.60, variance 0.001, pairwise correlation 0.95;
- feature score: train-only absolute mean daily cross-sectional Pearson IC;
- random seed: 42.
- price convention: vendor-adjusted OHLC and portfolio close returns.
- portfolio clock: signal at `t`, close execution proxy at `t+1`, endpoint at `t+10`.
- cache policy: `--refresh-caches`; the release run recomputes feature, preprocessing, and validation layers before writing new caches.

Changing any of these parameters creates a new experiment ID and must not overwrite the canonical evidence package.

## 5. Canonical Portfolio And Evidence Export

After training succeeds, run the fixed portfolio grid:

```bash
./scripts/run_canonical_us300_backtest.sh
```

Windows PowerShell:

```powershell
.\scripts\run_canonical_us300_backtest.ps1
```

Then export only reviewable evidence and rebuild figures:

```bash
python scripts/export_public_evidence.py
python scripts/build_public_figures.py
```

The exporter fails before touching `results/public/` unless it can prove that:

- the training run matches the fixed US300 configuration;
- training and backtest used the same clean Git commit and market-data SHA256;
- the backtest input is the exact saved canonical prediction artifact;
- all exported training/backtest files still match their manifest hashes;
- the complete 64-row cost/Top-K/neutralization grid uses `signal_horizon`;
- each effective holding period equals `hold_days - signal_delay_days`.

`--allow-pre-release` and `--allow-missing-backtest` exist only for local review. Evidence created with either exception cannot be published as the canonical result.

## 6. Required Artifact Manifest

A release-quality run must save:

- exact command and resolved configuration;
- Git commit hash and dirty-worktree flag;
- Python and dependency versions;
- input file fingerprint, row count, instrument count, and date range;
- train, validation, and OOS boundaries;
- feature-family counts and selected features;
- every fold metric;
- final OOS metrics;
- model weights and saved model paths;
- stage-level and total runtime;
- SHA256 fingerprints for predictions, fold tables, quality audits, selected-feature tables, model weights, and the portfolio grid;
- cache hit/miss status for feature engineering, preprocessing, and validation;
- known warnings and failed symbols;
- `data_quality_summary.json`, event-level `corporate_action_audit.csv`, and per-instrument `universe_coverage_audit.csv`;

`main_long_short_backtest.py` writes a separate `backtest_run_manifest.json`
containing prediction/data fingerprints, portfolio arguments, price convention,
holding clock and effective executable holding days,
runtime, dependency versions, and Git state. Portfolio results without this file
are historical diagnostics rather than release evidence.

Detailed portfolio runs also save `position_daily_contributions.csv`,
`extreme_return_days.csv`, and `instrument_return_attribution.csv`. The public
export compresses their grid-level warning fields into
`portfolio_anomaly_summary.csv` so unusually strong results remain auditable.

The superseded local package recorded most research fields, but its source commit
was missing and it predated the missing-OHLCV preservation rule, target-horizon
purge, unified adjusted-price policy, standard Sharpe definition, and
signal-horizon portfolio clock. It is ignored and cannot be exported as public
evidence. A clean run must build `us300_release_v1` from scratch before a tagged
release.

## 7. Public Evidence Package

The compact, reviewable package lives at:

```text
results/public/us300_release_v1/
```

It intentionally excludes raw data, full prediction matrices, caches, and binary model files. Every value used in the root README must be recoverable from its CSV or JSON files.

## 8. API Keys

Copy [`.env.example`](../.env.example) to a local `.env` only when an optional provider requires credentials. Do not commit `.env`, API keys, paid vendor data, or downloaded fundamentals.

## 9. Release Gate

Before tagging a release:

1. run the real-data smoke route in a clean environment;
2. establish the reviewed public-history strategy and create the clean source commit;
3. rerun the canonical US300 command from that public source commit;
4. regenerate the compact evidence package from that run;
5. stage the reviewed evidence and check that every README metric matches it;
6. confirm no secret, raw data, model binary, or machine-specific path is tracked;
7. record CI status and the final Git commit in the manifest.

Run the same public static gate used by CI before staging a release:

```bash
python scripts/check_public_repository.py
```

After the clean canonical evidence replaces every pre-release reference, run the stricter final gate:

```bash
python scripts/check_public_repository.py --release
```

The release mode also rejects pre-release labels, a missing source commit, a dirty source worktree, a non-release evidence manifest, a source commit that is not an ancestor of the release `HEAD`, and private/generated paths reachable from the release candidate's `HEAD`. Before pushing, manually confirm that no unrelated private branch or tag will be published to the remote.
Run release mode after staging the reviewed package because it also verifies that
all required tables, figures, manifests, and per-portfolio audit files are tracked.

## 10. Strict Mined-Factor Ablation

This experiment is outside the canonical result and must use a validation-selected factor zoo:

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

The factor zoo must be created by `factor_mining_workspace/select_ppo_validation_factor_zoo.py`. The strict command requires both `validation_selected_factor_zoo.csv` and `validation_selected_factor_zoo_summary.json`. Contract `validation_factor_zoo_v3_scale_invariant_fields_bound` verifies `selection_source=validation_reward_only`, candidate/config hashes, the PPO source-mining data hash, current data hash, sample start, target horizon, price-adjustment mode, the exact canonical formula-field allowlist, validation end date, OOS boundary, and the selected-zoo hash. Historical outputs, hand-renamed CSV files, and formulas referencing raw price intermediates fail closed. Warm-GP remains exploratory until a train-internal validation selector produces the same v3 contract. Its evidence package must retain the technical baseline, the canonical 11-formula scale-invariant Alpha191 baseline, mined-factor augmentation, progressive metric deltas, all folds, OOS subwindows, and `strict_increment_verdict.csv`.
