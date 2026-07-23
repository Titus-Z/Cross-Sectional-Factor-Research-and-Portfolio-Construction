# Contributing

MyQuant accepts changes that improve research validity, reproducibility, or clarity.

## Research Integrity Rules

- Split by date and keep all stocks from one date in the same block.
- Purge the final target-horizon observations per instrument before validation and OOS boundaries.
- Fit imputers, feature selectors, scalers, hyperparameters, and model weights on training or internal-validation data only.
- Treat final OOS as an audit. Do not use it to select formulas or portfolio settings.
- Keep prediction metrics, single-factor diagnostics, and portfolio results as separate evidence layers.
- Retain weak folds and failed ablations. Do not publish only the best window.
- Add transaction-cost and sample-length assumptions beside every portfolio result.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c requirements-lock.txt
```

Install [`requirements-mining.txt`](requirements-mining.txt) with
`-c requirements-lock.txt` only when changing PPO factor mining.
Install [`requirements-tree.txt`](requirements-tree.txt) with the same constraints
only when changing XGBoost or LightGBM comparison paths.

## Pull Request Scope

A focused pull request should explain:

1. the research or engineering problem;
2. the evidence boundary affected;
3. whether cache versions must change;
4. the exact command used for validation;
5. any metric or artifact that must be regenerated.

Never commit raw vendor data, cached matrices, binary models, local outputs, `.env`, API keys, machine-specific paths, or internal interview and product documents.

## Result Changes

If a code change can alter features, labels, splits, preprocessing, validation, predictions, or portfolio returns, mark existing public metrics as stale until the canonical run has been regenerated from a clean commit.

## Public Release Gate

Run the ordinary tracked-file hygiene check during development:

```bash
python scripts/check_public_repository.py
```

After clean canonical evidence has replaced every pre-release reference, use the stricter gate:

```bash
python scripts/check_public_repository.py --release
```

Release mode rejects pre-release markers, missing or dirty source provenance, a non-release evidence manifest, a source commit outside the release ancestry, and private/generated paths reachable from the release candidate's `HEAD`. Unrelated private local branches are outside this branch-level gate and must never be pushed to the public remote.
Stage the reviewed public evidence before running release mode; the gate verifies
that every required table, figure, manifest, and displayed portfolio detail is in
the Git index rather than merely present on one machine.
