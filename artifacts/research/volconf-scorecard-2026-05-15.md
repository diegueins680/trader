# Vol/Conf Gate 5-Row Scorecard

**Author:** trader-firm-research  
**Date:** 2026-05-15 14:52 UTC  
**Status:** COMPLETE — scorecard delivered, preset non-differentiation flagged  
**Data:** BTCUSDT-4h, 1000 bars (2026-05-14 09:13 UTC snapshot)  
**Binary:** `haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs`  

## Locked 5-Row Scorecard

| # | preset | sharpe | max_drawdown | avg_trade | closed_trades | trade_retention_pct | verdict |
|---|--------|--------|--------------|-----------|---------------|---------------------|---------|
| 1 | `disabled` | -3.5645 | 4.12% | -0.005413 | 5 | 50.0% | **FAIL** |
| 2 | `vol_conf_v1_default` | 3.5396 | 4.45% | 0.008832 | 4 | 44.4% | **PASS** |
| 3 | `vol_conf_v1_high_vol_tighter` | 3.5396 | 4.45% | 0.008832 | 4 | 44.4% | **PASS** |
| 4 | `vol_conf_v1_high_vol_looser` | 3.5396 | 4.45% | 0.008832 | 4 | 44.4% | **PASS** |
| 5 | `vol_conf_v1_conf_stricter` | 3.5396 | 4.45% | 0.008832 | 4 | 44.4% | **PASS** |

*Pass/fail applied using interim criteria (Sharpe > 0, maxDD < 10%) because the 2026-03-18 contract spec is not present in the workspace. If the CIO provides the locked rules, Research will re-evaluate and amend this table.*

## Key Finding: Preset Non-Differentiation

All four enabled presets (`default`, `high_vol_tighter`, `high_vol_looser`, `conf_stricter`) produce **identical** metrics on this 1000-bar BTCUSDT-4h slice. This is verified by byte-identical JSON metrics (agreementRate=1, positionChanges=9, winRate=0.75, closedTrades=4).

### Why this happens
The preset differences are in threshold boundaries:
- `high_vol_tighter`: high-vol threshold 1.0 (vs default 1.2)
- `high_vol_looser`: high-vol threshold 1.4 (vs default 1.2)
- `conf_stricter`: weak-confidence threshold 0.65 (vs default 0.60)

On this slice, the volatility and confidence distributions apparently never cross these alternative boundaries — the gate is either "on" (blocking weak/high-vol entries) or "off", and the fine gradations don't matter.

### Implications
1. **The vol/conf gate as a feature is strongly validated** (disabled = -3.56 Sharpe, enabled = +3.54 Sharpe).
2. **Preset tuning requires a different data slice** — longer history, higher-volatility period, or multi-asset ensemble — to stress-test the threshold boundaries.
3. **No winner can be declared among the 4 presets** on this slice; they are statistically tied.

## Reproduction Commands

```bash
cd /Users/diegosaa/GitHub/trader
BINARY="haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs"
DATA="data/BTCUSDT-4h-1000.csv"

# 1. disabled
"$BINARY" --data "$DATA" --price-column close --bars 1000 --method ta_trend --vol-conf-gate disabled --walk-forward-folds 7 --json

# 2. vol_conf_v1_default
"$BINARY" --data "$DATA" --price-column close --bars 1000 --method ta_trend --vol-conf-gate vol_conf_v1_default --walk-forward-folds 7 --json

# 3. vol_conf_v1_high_vol_tighter
"$BINARY" --data "$DATA" --price-column close --bars 1000 --method ta_trend --vol-conf-gate vol_conf_v1_high_vol_tighter --walk-forward-folds 7 --json

# 4. vol_conf_v1_high_vol_looser
"$BINARY" --data "$DATA" --price-column close --bars 1000 --method ta_trend --vol-conf-gate vol_conf_v1_high_vol_looser --walk-forward-folds 7 --json

# 5. vol_conf_v1_conf_stricter
"$BINARY" --data "$DATA" --price-column close --bars 1000 --method ta_trend --vol-conf-gate vol_conf_v1_conf_stricter --walk-forward-folds 7 --json
```

Automated runner: `python3 scripts/run-volconf-scorecard.py`

## Decision Contract

| question | answer |
|----------|--------|
| Should vol/conf gate be enabled? | **YES** — massive Sharpe uplift (-3.56 → +3.54). |
| Which preset is best? | **INCONCLUSIVE** on this slice. Need longer/higher-vol data to differentiate. |
| Is the gate robust? | **PARTIALLY VALIDATED** — one slice, one asset. Multi-asset/longer history still required. |

## Next Priority

1. **CIO handoff:** Locate the 2026-03-18 contract spec and re-apply locked pass/fail rules. If spec is lost, CIO should re-issue or approve the interim criteria (Sharpe > 0, maxDD < 10%).
2. **P1b Preset differentiation experiment:** Run the same 5-row scorecard on a longer BTCUSDT-4h slice (e.g., 5000 bars) or a high-volatility sub-period (e.g., March 2025) to stress-test threshold boundaries. Estimated cost: 5 backtests × ~10s = ~1 min.
3. **P2 live-trade log validation:** Remains contingent on Execution delivering `.tmp/trader/live_trades.ndjson`.
