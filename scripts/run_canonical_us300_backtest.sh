#!/usr/bin/env bash
set -euo pipefail

# Portfolio layer for the canonical US300 predictions. This is deliberately a
# separate command so prediction quality and trading assumptions remain auditable.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT_DIR"

if [[ "${ALLOW_DIRTY_CANONICAL:-0}" != "1" ]] && [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  printf '%s\n' \
    "Canonical release backtests require a clean Git worktree." \
    "Commit the reviewed source first, or set ALLOW_DIRTY_CANONICAL=1 only for local debugging." >&2
  exit 3
fi

if [[ ! -f outputs/public_us300_release_v1/test_predictions_with_actual.csv ]]; then
  printf '%s\n' \
    "Missing outputs/public_us300_release_v1/test_predictions_with_actual.csv." \
    "Run scripts/run_canonical_us300.sh first." >&2
  exit 2
fi

mkdir -p outputs/public_us300_release_v1_backtest

"$PYTHON_BIN" main_long_short_backtest.py \
  --predictions-paths outputs/public_us300_release_v1/test_predictions_with_actual.csv \
  --run-names canonical_us300_release_v1 \
  --data-path data/us_large_cap_300_daily.csv \
  --output-root-dir outputs/public_us300_release_v1_backtest \
  --hold-days-list 10 20 \
  --top-k-list 10 20 30 50 \
  --cost-bps-list 5 10 20 50 \
  --neutral-modes unconstrained sector_neutral \
  --signal-delay-days 1 \
  --holding-clock signal_horizon \
  --borrow-cost-bps 0 \
  --price-adjustment-mode vendor_adjusted \
  2>&1 | tee outputs/public_us300_release_v1_backtest/canonical_backtest.log
