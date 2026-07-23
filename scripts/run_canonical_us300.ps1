$ErrorActionPreference = "Stop"

# Windows counterpart of run_canonical_us300.sh. Keep the arguments identical;
# changing one parameter creates a different experiment contract.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($env:ALLOW_DIRTY_CANONICAL -ne "1") {
    $GitStatus = git status --porcelain --untracked-files=normal
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect Git status before the canonical run."
    }
    if ($GitStatus) {
        throw "Canonical release runs require a clean Git worktree. Set ALLOW_DIRTY_CANONICAL=1 only for a non-release local debug run."
    }
}

if (-not (Test-Path "data/us_large_cap_300_daily.csv")) {
    throw "Missing data/us_large_cap_300_daily.csv. See docs/REPRODUCIBILITY.md."
}

New-Item -ItemType Directory -Force -Path "outputs/public_us300_release_v1" | Out-Null

python main.py `
  --data-path data/us_large_cap_300_daily.csv `
  --universe-label us_large_cap_300_static_snapshot `
  --sample-start-date 2022-01-01 `
  --oos-start-date 2025-06-01 `
  --target-horizon 10 `
  --price-adjustment-mode vendor_adjusted `
  --max-alpha 0 `
  --alpha-factors alpha001 alpha002 alpha004 alpha005 alpha006 alpha015 alpha018 alpha019 alpha020 alpha022 alpha023 `
  --models ridge lasso `
  --n-splits 3 `
  --top-n 50 `
  --missing-threshold 0.6 `
  --variance-threshold 0.001 `
  --correlation-threshold 0.95 `
  --feature-score-method correlation `
  --validation-score-metric pearson_ic_mean `
  --refresh-caches `
  --random-state 42 `
  --model-dir models/public_us300_release_v1 `
  --output-dir outputs/public_us300_release_v1 `
  2>&1 | Tee-Object -FilePath "outputs/public_us300_release_v1/canonical_run.log"

if ($LASTEXITCODE -ne 0) {
    throw "Canonical US300 training failed with exit code $LASTEXITCODE."
}
