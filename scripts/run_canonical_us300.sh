#!/usr/bin/env bash
set -euo pipefail

# Canonical public experiment. The explicit arguments are part of the public
# evidence contract; changing one creates a different experiment.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT_DIR"

if [[ "${ALLOW_DIRTY_CANONICAL:-0}" != "1" ]] && [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  printf '%s\n' \
    "Canonical release runs require a clean Git worktree." \
    "Commit the reviewed source first, or set ALLOW_DIRTY_CANONICAL=1 only for a non-release local debug run." >&2
  exit 3
fi

if [[ ! -f data/us_large_cap_300_daily.csv ]]; then
  printf '%s\n' \
    "Missing data/us_large_cap_300_daily.csv." \
    "See docs/REPRODUCIBILITY.md for the required schema and data policy." >&2
  exit 2
fi

mkdir -p outputs/public_us300_release_v1

"$PYTHON_BIN" main.py \
  --data-path data/us_large_cap_300_daily.csv \
  --universe-label us_large_cap_300_static_snapshot \
  --sample-start-date 2022-01-01 \
  --oos-start-date 2025-06-01 \
  --target-horizon 10 \
  --price-adjustment-mode vendor_adjusted \
  --max-alpha 0 \
  --alpha-factors alpha001 alpha002 alpha004 alpha005 alpha006 alpha015 \
                  alpha018 alpha019 alpha020 alpha022 alpha023 \
  --models ridge lasso \
  --n-splits 3 \
  --top-n 50 \
  --missing-threshold 0.6 \
  --variance-threshold 0.001 \
  --correlation-threshold 0.95 \
  --feature-score-method correlation \
  --validation-score-metric pearson_ic_mean \
  --refresh-caches \
  --random-state 42 \
  --model-dir models/public_us300_release_v1 \
  --output-dir outputs/public_us300_release_v1 \
  2>&1 | tee outputs/public_us300_release_v1/canonical_run.log
