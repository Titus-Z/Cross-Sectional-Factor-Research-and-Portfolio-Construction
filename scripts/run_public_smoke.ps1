$ErrorActionPreference = "Stop"

# Real-data integration check for Windows. The workload is deliberately small:
# 30 liquid stocks, Ridge only, two folds, and Alpha001-Alpha005.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python main.py `
  --fetch-yfinance `
  --symbols AAPL MSFT NVDA AMZN GOOGL META BRK-B AVGO TSLA JPM LLY V WMT XOM MA UNH ORCL COST HD PG NFLX JNJ ABBV BAC KO CRM CVX AMD MRK PEP `
  --data-path data/smoke_us30_daily.csv `
  --universe-label smoke_us30_fixed_symbols `
  --start-date 2024-01-01 `
  --sample-start-date 2024-01-01 `
  --oos-start-date 2025-07-01 `
  --target-horizon 10 `
  --no-auto-adjust `
  --price-adjustment-mode vendor_adjusted `
  --max-alpha 5 `
  --models ridge `
  --n-splits 2 `
  --top-n 20 `
  --model-dir models/public_smoke_us30 `
  --output-dir outputs/public_smoke_us30
