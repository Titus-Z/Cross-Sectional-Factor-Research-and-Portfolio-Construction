# Manual Data Review

This review covers the canonical US300 release thresholds:

- every vendor-adjusted absolute daily return above 50%;
- every adjustment-change event whose residual adjusted return exceeds 20%.

## Verdict

All three threshold events were matched to same-day primary-source disclosures.
No unresolved split, reverse-split, or unexplained price-jump blocker remains
inside this narrowly defined release review.

| Instrument | Date | Adjusted return | Trigger | Classification | Primary evidence |
|---|---|---:|---|---|---|
| TGT | 2024-11-20 | -21.41% | Residual return above 20% on an adjustment-change date | Reviewed genuine market move | [Target Q3 results and revised guidance](https://corporate.target.com/press/release/2024/11/target-corporation-reports-third-quarter-earnings) |
| CVNA | 2023-06-08 | +56.02% | Adjusted return above 50% | Reviewed genuine market move | [Carvana Form 8-K](https://www.sec.gov/Archives/edgar/data/1690820/000169082023000195/cvna-20230608.htm) |
| SATS | 2025-08-26 | +70.25% | Adjusted return above 50% | Reviewed genuine market move | [EchoStar spectrum transaction announcement](https://ir.echostar.com/node/32621) |

The review is bound to:

- data SHA256: `84fdd8fc5a3836303b6bffd87f2edc11eb00bebe0aea26d40b015f60f97281b3`;
- corporate-action audit SHA256: `b07cfceee765436d6a241bc8f8486fd3fe99099474ef5048b6c9aba0338c06d3`.

## Boundary

This review does not establish complete corporate-action coverage. Yahoo Finance
remains a convenient research vendor rather than an institutional corporate-action
master. The project still lacks point-in-time constituent history, authoritative
delisting returns, real borrow availability, calibrated slippage, and nonlinear
market-impact data.
