# conf_blend Smoke Test — 2026-05-18

## Scope
Test whether `conf_blend` (Candidate A method) is viable on the current binary (`dist-newstyle` build, May 17 14:28).
Run single-row backtest with 60-second timeout.
Acceptance: completes in <60s and Sharpe ≈ +3.54 ±0.10.

## Binary under test
- Path: `haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs`
- mtime: 2026-05-17 14:28 UTC
- Size: 38,256,824 bytes

## Data under test
- Path: `data/BTCUSDT-4h-1000.csv`
- mtime: 2026-05-15 23:32 UTC
- Rows: 1000

## Test results

### conf_blend (primary test)
```
trader-hs --data data/BTCUSDT-4h-1000.csv --method conf_blend --vol-conf-gate vol_conf_v1_default --json
```
- **Result: FAIL — HANG**
- Timeout: 60s
- Exit: killed by timeout (no exit code)
- STDOUT: 0 bytes
- STDERR: 0 bytes

### conf_blend without vol-conf-gate
```
trader-hs --data data/BTCUSDT-4h-1000.csv --method conf_blend --json
```
- **Result: FAIL — HANG**
- Timeout: 10s (aborted early; same zero-output pattern)

### Control: ta_trend (known-good method)
```
trader-hs --data data/BTCUSDT-4h-1000.csv --method ta_trend --json
```
- **Result: OK**
- Elapsed: 2.58s
- Exit: 0
- JSON: parsed successfully

### Cross-check: other blend-family methods on current binary
| method | result | elapsed |
|--------|--------|---------|
| `blend` | HANG | >10s |
| `11` | HANG | >10s |
| `both` | HANG | >10s |
| `conf_blend` | HANG | >60s |

All Kalman+LSTM-family methods hang with zero output.
Only `ta_trend` completes.

### Baseline regression check (ta_trend + vol_conf_v1_default)
```
trader-hs --data data/BTCUSDT-4h-1000.csv --method ta_trend --vol-conf-gate vol_conf_v1_default --json
```
- Elapsed: 2.65s
- Sharpe: **0.1839**
- maxDD: 3.62%
- closedTrades: 4

**Key finding:** Even `ta_trend` has behavior regression vs. May 15 ratified baseline (Sharpe 3.5396 → 0.1839). This indicates a data- or model-layer change affecting all methods, but the complex methods additionally hit an infinite-loop or deadlock path.

### Old-binary cross-check
Tested `conf_blend` and `11` on `dist-newstyle-codex-review` binary (Feb 28 10:57). Both also hang (>15s, zero output). This binary predates the May 15 successful runs, suggesting the Feb 28 build never supported these methods on the current data format. It is not a viable fallback.

## Verdict
**FAIL — Candidate A (`conf_blend`) is not viable on the current binary.**

The hang is not isolated to `conf_blend`; it affects all Kalman+LSTM-family methods (`11`, `both`, `blend`). Only `ta_trend` executes, but its Sharpe regressed from +3.54 to +0.18, indicating deeper model/data drift.

## Required actions
| Action | Owner | Acceptance criteria |
|--------|-------|---------------------|
| 1. Bisect commits between last known-good `both`/`blend` run (May 15) and current HEAD to identify the hang-introducing commit | trader-firm-execution | Single backtest with `both` or `conf_blend` completes in <60s and produces non-zero stdout |
| 2. Audit `ta_trend` Sharpe regression (3.54 → 0.18) — data change or model change? | trader-firm-data / trader-firm-cto | Reproduce May 15 Sharpe 3.54 on current binary or identify the drift commit |
| 3. Provide ratified binary or revert hang-inducing commit | trader-firm-cto | Binary passes: `ta_trend` Sharpe ≈ 3.54 ±0.10, `conf_blend` completes <60s |

## Evidence
- Commit of this artifact: (to be appended)
- Reproduction commands are exact and copy-paste ready above.
