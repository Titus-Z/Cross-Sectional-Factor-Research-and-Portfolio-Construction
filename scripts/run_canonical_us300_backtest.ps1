$ErrorActionPreference = "Stop"

# Windows counterpart of the canonical portfolio grid.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($env:ALLOW_DIRTY_CANONICAL -ne "1") {
    $GitStatus = git status --porcelain --untracked-files=normal
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect Git status before the canonical backtest."
    }
    if ($GitStatus) {
        throw "Canonical release backtests require a clean Git worktree. Set ALLOW_DIRTY_CANONICAL=1 only for local debugging."
    }
}

if (-not (Test-Path "outputs/public_us300_release_v1/test_predictions_with_actual.csv")) {
    throw "Missing canonical predictions. Run scripts/run_canonical_us300.ps1 first."
}

New-Item -ItemType Directory -Force -Path "outputs/public_us300_release_v1_backtest" | Out-Null

python main_long_short_backtest.py `
  --predictions-paths outputs/public_us300_release_v1/test_predictions_with_actual.csv `
  --run-names canonical_us300_release_v1 `
  --data-path data/us_large_cap_300_daily.csv `
  --output-root-dir outputs/public_us300_release_v1_backtest `
  --hold-days-list 10 20 `
  --top-k-list 10 20 30 50 `
  --cost-bps-list 5 10 20 50 `
  --neutral-modes unconstrained sector_neutral `
  --signal-delay-days 1 `
  --holding-clock signal_horizon `
  --borrow-cost-bps 0 `
  --price-adjustment-mode vendor_adjusted `
  2>&1 | Tee-Object -FilePath "outputs/public_us300_release_v1_backtest/canonical_backtest.log"

if ($LASTEXITCODE -ne 0) {
    throw "Canonical US300 backtest failed with exit code $LASTEXITCODE."
}
