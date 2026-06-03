# Trend-Capture P1 Decision Packet — 2026-06-02

## Scope
Three exact reproductions on `data/BTCUSDT-4h-trend-1076.csv` for `ta_best` with `--trailing-stop 0.03`, per CIO P1.

## Dataset
- File: `data/BTCUSDT-4h-trend-1076.csv`
- Regime: contiguous BTCUSDT 4h up-trend window
- Reference market move in the window: `+20.15%`

## Commands
```bash
cd /Users/diegosaa/GitHub/trader
BIN=./haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs

$BIN --data data/BTCUSDT-4h-trend-1076.csv --price-column close --method ta_best --epochs 0 --trailing-stop 0.03 --json
$BIN --data data/BTCUSDT-4h-trend-1076.csv --price-column close --method ta_best --epochs 0 --trailing-stop 0.03 --positioning long-only --json
$BIN --data data/BTCUSDT-4h-trend-1076.csv --price-column close --method ta_best --epochs 0 --trailing-stop 0.03 --max-position-size 1 --json
```

## Results
- Default `maxPositionSize` at this row was `0.8`, so the higher-capture variant used `--max-position-size 1`.

| Row | Sharpe | Total Return | Max Drawdown | Closed Trades |
| --- | ---: | ---: | ---: | ---: |
| `ta_best --trailing-stop 0.03` | 1.484 | +0.97% | 6.50% | 5 |
| `+ --positioning long-only` | 1.484 | +0.97% | 6.50% | 5 |
| `+ --max-position-size 1` | 0.556 | +0.31% | 7.12% | 5 |

## Verdict
**hold** — `ta_best --trailing-stop 0.03` is still the only viable row here, but it captures just `+0.97%` on a `+20.15%` trend and the only supported higher-capture exposure bump (`--max-position-size 1`) made Sharpe, return, and drawdown worse.

## Notes
- `--positioning long-only` was numerically identical to the baseline on this dataset.
- The baseline captured about `4.8%` of the underlying move (`0.97 / 20.15`).
- Raw parsed metrics snapshot: `research-notes/tmp-2026-06-02-trend-p1-metrics.json`
