# MyQuant Public Documentation

This directory contains the public research documentation for MyQuant.

## Read in This Order

1. [`DATA.md`](DATA.md): schema, universe construction, adjusted-price policy, and data biases.
2. [`METHODOLOGY.md`](METHODOLOGY.md): target, leakage controls, features, validation, and portfolio construction.
3. [`RESULTS.md`](RESULTS.md): current evidence status, release requirements, and metric interpretation rules.
4. [`LIMITATIONS.md`](LIMITATIONS.md): evidence boundaries, data biases, and execution assumptions.
5. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): environment, commands, and artifact checklist.
6. [`guides/`](guides/): focused guides for ablation, factor diagnostics, and optional fundamentals.

The machine-readable evidence target is stored in [`../results/public/us300_release_v1/`](../results/public/us300_release_v1/README.md).

## Public Scope

- Intended release baseline: `US300 + y_10d`; saved metrics await a clean rerun under the current loader.
- Scalable extension: US3000 data and training pipeline; formal benchmark results pending.
- Formula mining: implemented research capability; no stable incremental-alpha claim in the canonical package.
- Portfolio layer: cost-aware diagnostic simulation; no live-trading claim.

Internal learning notes, interview scripts, UI product documents, raw data, models, and exploratory outputs are intentionally outside the public documentation surface.

Root-level scripts are classified in the main README. Canonical, controlled-diagnostic, optional-data, and historical-exploratory entry points must not be treated as interchangeable result sources.
