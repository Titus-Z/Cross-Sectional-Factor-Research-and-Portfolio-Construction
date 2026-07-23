# Formulaic Factor Mining Workspace

This workspace studies whether a constrained search algorithm can generate interpretable formulas with incremental cross-sectional predictive value.

It is separated from the canonical training result for one reason: finding an attractive single factor and proving model-level OOS improvement are different research claims.

## Research Contract

A formula-mining run must follow this sequence:

1. define a safe seed feature pool using observable market variables;
2. generate formula abstract syntax trees (ASTs) under explicit depth and complexity limits;
3. reject forbidden fields and invalid numerical expressions;
4. score candidates using training-period or internal-validation data only;
5. purge the final target-horizon observations of every instrument before each internal validation boundary;
6. remove duplicates and highly correlated candidates without using final OOS performance;
7. retrain the baseline and baseline-plus-mined-factor models under identical folds;
8. use final OOS once for the model-level audit.

An OOS-screened factor zoo is exploratory. It cannot support a resume or public incremental-alpha claim.

## Formula Language

[`formula_language.py`](formula_language.py) represents formulas as structured trees instead of unvalidated strings.

Supported transformations include:

- unary operators such as `rank`, `abs`, `neg`, `tanh`, and signed square;
- binary operators such as weighted blend, spread, safe ratio, confirmation, and interaction;
- explicit metadata for source fields, family, depth, node count, and financial hypothesis.

The first public version composes existing leakage-audited features. It does not allow target columns, future prices, or arbitrary Python execution.

## Search Methods

| Method | Entry point | What is learned |
|---|---|---|
| Warm-start GP-style search | `auto_factor_mining.py` | populations improve through mutation, recombination, and survivor selection |
| Contextual bandit | `rl_factor_mining.py` | action values for formula mutations given a compact formula state |
| Probabilistic grammar | `generative_factor_mining.py` | sampling weights over fields, operators, and formula structures |
| PPO Deep RL | `deep_rl_formula_mining.py` | a policy and value function for multi-step formula construction |

The contextual bandit is a lightweight RL baseline. PPO is the full Deep RL implementation in this repository. Neither method directly chooses portfolio positions; both generate candidate factors.

## Reward Definitions

[`auto_alpha_reward.py`](auto_alpha_reward.py) contains two research objectives:

- `predictive_ic`: emphasizes single-factor Pearson IC, Rank IC, and long-short spread;
- `incremental_proxy`: places more weight on ranking quality, low redundancy, formula simplicity, and portfolio-oriented stability proxies.

Reward comparison is an ablation. A larger reward does not establish trading value; the final gate is a same-protocol model ablation.

## Main Components

- `alphaeval_style_evaluator.py`: multi-dimensional candidate diagnostics;
- `auto_alpha_benchmark.py`: compares miners under a common reporting schema;
- `mined_factor_model_ablation.py`: baseline versus baseline-plus-mined-factor training;
- `../main_mined_factor_strict_experiment.py`: progressive technical -> Alpha191 -> validation-selected mined-factor model and portfolio ablation;
- `residual_alpha_mining.py`: searches for formulas related to cross-fitted baseline residuals;
- `select_ppo_validation_factor_zoo.py`: constructs a validation-selected PPO zoo without OOS ranking and binds it to the canonical formula-field allowlist;
- `single_factor_case_study.py`: detailed IC, coverage, adjacent-date rank turnover, Top-20% retention, and grouping report for one factor.

## Minimal Examples

Warm-start search:

```bash
python factor_mining_workspace/auto_factor_mining.py search \
  --searcher warm_gp \
  --population-size 20 \
  --generations 2 \
  --target-horizon 10 \
  --oos-start-date 2025-06-01
```

Contextual-bandit baseline:

```bash
python factor_mining_workspace/rl_factor_mining.py \
  --episodes 20 \
  --seed-top-k 10 \
  --target-horizon 10
```

Probabilistic grammar:

```bash
python factor_mining_workspace/generative_factor_mining.py \
  --num-samples 20 \
  --seed-top-k 10 \
  --target-horizon 10
```

PPO smoke run:

```bash
python factor_mining_workspace/deep_rl_formula_mining.py \
  --data-path data/us_large_cap_300_daily.csv \
  --sample-start-date 2022-01-01 \
  --oos-start-date 2025-06-01 \
  --target-horizon 10 \
  --price-adjustment-mode vendor_adjusted \
  --total-updates 1 \
  --episodes-per-update 4 \
  --max-steps 2 \
  --selected-top-k 3 \
  --run-name smoke_ppo_formula_min
```

Install the optional PyTorch dependency first:

```bash
python -m pip install -r requirements-mining.txt
```

## Required Evaluation Outputs

A serious run should retain:

- formula and AST metadata;
- validation metrics and reward components;
- duplicate and correlation filters;
- selected-factor coverage, adjacent-date rank turnover, and Top-20% retention;
- baseline and augmented model metrics on identical rows;
- final OOS audit separated from selection;
- runtime and random seed;
- a limitation statement when improvement is absent.

## Strict Progressive Ablation

The public-facing incremental experiment fixes three explicit feature sets:

1. `strict_technical_baseline`: observable raw, technical, context, and eligible point-in-time fields; no Alpha191;
2. `strict_alpha191_baseline`: the same baseline plus the canonical 11-formula price-scale-invariant Alpha191 subset;
3. `strict_ppo` (or another requested mined group): the Alpha191 baseline plus a validation-selected factor zoo.

Default model families are Ridge, Lasso, and XGBoost. All groups share the same explicit Alpha001/002/004/005/006/015/018/019/020/022/023 scope, adjusted-price convention, split, purge, daily winsorization/z-score, available exposure neutralization, Top-N selector, walk-forward folds, model weighting, and final OOS rows. For mined-factor groups, those folds calibrate the downstream model after formulas have already been selected from pre-OOS validation data; they are not nested factor-mining estimates. Only the untouched final OOS delta is eligible evidence for mined-factor increment in the current implementation.

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

Required strict outputs include full model metrics, progressive deltas, every walk-forward fold, 3/6/12-month OOS subwindow diagnostics, a machine-readable `strict_increment_verdict.csv`, selected mined features, runtime, and optional same-protocol portfolio views. The strict loader requires the selector's `validation_factor_zoo_v3_scale_invariant_fields_bound` provenance contract and verifies the validation-only source, source candidate/config hashes, PPO source-mining data hash, current market-data hash, sample/target/OOS boundaries, adjusted-price convention, exact canonical formula-field allowlist, and selected-zoo hash before materializing any mined feature. The historical Warm-GP path still uses OOS-ranked research outputs and therefore remains outside strict public evidence until it receives an equivalent train-internal selector.

## Current Public Conclusion

The repository implements four formula-search families and a strict model-ablation path. Internal PPO and incremental-selection boundaries now use the same per-instrument target-horizon purge as the canonical model pipeline. Price convention is part of the mining and selection contract, and strict Alpha191 comparison uses the canonical 11-formula scale-invariant subset. Existing local mining artifacts predate these release fixes and must be regenerated before comparison. The canonical public evidence package does not yet contain a same-protocol mined-factor improvement. The defensible claim is implementation and controlled evaluation capability, not stable tradable alpha.
