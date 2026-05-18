# Candidate A Execution Packet — `conf_blend` 6-Combo Grid

**Status:** BLOCKED by B5/B6. Do NOT execute until CIO clears B5.
**Reference:** Scoping memo a9befc84 §3.1, `artifacts/research/next-strategy-scope-2026-05-15.md`
**Prepared by:** trader-firm-research
**Date:** 2026-05-18 16:32 UTC

---

## 1. Dataset and Path

| Field | Value |
|-------|-------|
| Symbol | BTCUSDT |
| Interval | 4h |
| Bars | 1000 |
| File | `data/BTCUSDT-4h-1000.csv` |
| Columns | openTimeMs, open, high, low, close, volume, closeTimeMs, quoteAssetVolume, tradeCount, takerBuyBaseVolume, takerBuyQuoteVolume, ignore |
| mtime | 2026-05-15 23:32 UTC |
| Rows | 1001 (header + 1000) |

---

## 2. Base CLI Flags (constant across all 6 combos)

```
--data data/BTCUSDT-4h-1000.csv
--method conf_blend
--json
--positioning long-flat
--fee 0.0008
--slippage 0.0002
--spread 0.0002
--initial-balance 1.0
--backtest-ratio 0.2
--normalization standard
--predictors all
--kalman-process-var 1e-5
--kalman-measurement-var 1e-3
--hidden-size 16
--epochs 30
--lr 1e-3
--val-ratio 0.3
--patience 10
--seed 42
--vol-conf-gate vol_conf_v1_default
```

**Rationale for constants:**
- `conf_blend` requires both Kalman and LSTM contexts (enforced at runtime).
- Default LSTM config (`hidden-size 16`, `epochs 30`, `lr 1e-3`, `seed 42`) is the stable training baseline used in all prior scorecards.
- `vol_conf_v1_default` is the ratified gate preset from P1.
- `long-flat` is the conservative positioning for Candidate A validation.

---

## 3. 6-Combo Parameter Table

| Combo | `--blend-weight` | `--kalman-z-min` | `--kalman-z-max` | `--open-threshold` | `--close-threshold` | `--confidence-sizing` |
|-------|------------------|------------------|------------------|--------------------|---------------------|-----------------------|
| A1 | 0.50 | 0.5 | 3.0 | 0.002 | 0.002 | no |
| A2 | 0.50 | 0.5 | 3.0 | 0.004 | 0.004 | no |
| A3 | 0.50 | 1.0 | 3.0 | 0.002 | 0.002 | no |
| A4 | 0.70 | 0.5 | 3.0 | 0.002 | 0.002 | no |
| A5 | 0.50 | 0.5 | 3.0 | 0.002 | 0.002 | yes |
| A6 | 0.30 | 0.5 | 3.0 | 0.002 | 0.002 | no |

**Parameter rationale:**
- `blend-weight`: 0.5 is neutral Kalman/LSTM balance; 0.7 favors Kalman; 0.3 favors LSTM.
- `kalman-z-min`: 0.5 is default; 1.0 raises the directional threshold for Kalman confidence.
- `open-threshold`: 0.002 is default; 0.004 tests wider deadband (fewer trades, lower turnover).
- `confidence-sizing`: only A5 tests dynamic position sizing gated by Kalman z-score and LSTM confidence.

---

## 4. Exact Commands (copy-paste ready)

### A1 — Baseline
```bash
cd /Users/diegosaa/GitHub/trader/haskell && \
cabal run trader-hs -- \
  --data ../data/BTCUSDT-4h-1000.csv \
  --method conf_blend \
  --json \
  --positioning long-flat \
  --fee 0.0008 --slippage 0.0002 --spread 0.0002 \
  --initial-balance 1.0 --backtest-ratio 0.2 \
  --normalization standard --predictors all \
  --kalman-process-var 1e-5 --kalman-measurement-var 1e-3 \
  --hidden-size 16 --epochs 30 --lr 1e-3 --val-ratio 0.3 --patience 10 --seed 42 \
  --blend-weight 0.50 --kalman-z-min 0.5 --kalman-z-max 3.0 \
  --open-threshold 0.002 --close-threshold 0.002 \
  --no-confidence-sizing \
  --vol-conf-gate vol_conf_v1_default
```

### A2 — Wider Threshold
```bash
cd /Users/diegosaa/GitHub/trader/haskell && \
cabal run trader-hs -- \
  --data ../data/BTCUSDT-4h-1000.csv \
  --method conf_blend \
  --json \
  --positioning long-flat \
  --fee 0.0008 --slippage 0.0002 --spread 0.0002 \
  --initial-balance 1.0 --backtest-ratio 0.2 \
  --normalization standard --predictors all \
  --kalman-process-var 1e-5 --kalman-measurement-var 1e-3 \
  --hidden-size 16 --epochs 30 --lr 1e-3 --val-ratio 0.3 --patience 10 --seed 42 \
  --blend-weight 0.50 --kalman-z-min 0.5 --kalman-z-max 3.0 \
  --open-threshold 0.004 --close-threshold 0.004 \
  --no-confidence-sizing \
  --vol-conf-gate vol_conf_v1_default
```

### A3 — Higher Kalman Z-Min
```bash
cd /Users/diegosaa/GitHub/trader/haskell && \
cabal run trader-hs -- \
  --data ../data/BTCUSDT-4h-1000.csv \
  --method conf_blend \
  --json \
  --positioning long-flat \
  --fee 0.0008 --slippage 0.0002 --spread 0.0002 \
  --initial-balance 1.0 --backtest-ratio 0.2 \
  --normalization standard --predictors all \
  --kalman-process-var 1e-5 --kalman-measurement-var 1e-3 \
  --hidden-size 16 --epochs 30 --lr 1e-3 --val-ratio 0.3 --patience 10 --seed 42 \
  --blend-weight 0.50 --kalman-z-min 1.0 --kalman-z-max 3.0 \
  --open-threshold 0.002 --close-threshold 0.002 \
  --no-confidence-sizing \
  --vol-conf-gate vol_conf_v1_default
```

### A4 — Kalman-Favored Blend
```bash
cd /Users/diegosaa/GitHub/trader/haskell && \
cabal run trader-hs -- \
  --data ../data/BTCUSDT-4h-1000.csv \
  --method conf_blend \
  --json \
  --positioning long-flat \
  --fee 0.0008 --slippage 0.0002 --spread 0.0002 \
  --initial-balance 1.0 --backtest-ratio 0.2 \
  --normalization standard --predictors all \
  --kalman-process-var 1e-5 --kalman-measurement-var 1e-3 \
  --hidden-size 16 --epochs 30 --lr 1e-3 --val-ratio 0.3 --patience 10 --seed 42 \
  --blend-weight 0.70 --kalman-z-min 0.5 --kalman-z-max 3.0 \
  --open-threshold 0.002 --close-threshold 0.002 \
  --no-confidence-sizing \
  --vol-conf-gate vol_conf_v1_default
```

### A5 — Confidence Sizing Enabled
```bash
cd /Users/diegosaa/GitHub/trader/haskell && \
cabal run trader-hs -- \
  --data ../data/BTCUSDT-4h-1000.csv \
  --method conf_blend \
  --json \
  --positioning long-flat \
  --fee 0.0008 --slippage 0.0002 --spread 0.0002 \
  --initial-balance 1.0 --backtest-ratio 0.2 \
  --normalization standard --predictors all \
  --kalman-process-var 1e-5 --kalman-measurement-var 1e-3 \
  --hidden-size 16 --epochs 30 --lr 1e-3 --val-ratio 0.3 --patience 10 --seed 42 \
  --blend-weight 0.50 --kalman-z-min 0.5 --kalman-z-max 3.0 \
  --open-threshold 0.002 --close-threshold 0.002 \
  --confidence-sizing \
  --vol-conf-gate vol_conf_v1_default
```

### A6 — LSTM-Favored Blend
```bash
cd /Users/diegosaa/GitHub/trader/haskell && \
cabal run trader-hs -- \
  --data ../data/BTCUSDT-4h-1000.csv \
  --method conf_blend \
  --json \
  --positioning long-flat \
  --fee 0.0008 --slippage 0.0002 --spread 0.0002 \
  --initial-balance 1.0 --backtest-ratio 0.2 \
  --normalization standard --predictors all \
  --kalman-process-var 1e-5 --kalman-measurement-var 1e-3 \
  --hidden-size 16 --epochs 30 --lr 1e-3 --val-ratio 0.3 --patience 10 --seed 42 \
  --blend-weight 0.30 --kalman-z-min 0.5 --kalman-z-max 3.0 \
  --open-threshold 0.002 --close-threshold 0.002 \
  --no-confidence-sizing \
  --vol-conf-gate vol_conf_v1_default
```

---

## 5. Expected Output Schema (JSON fields to capture)

Each run emits a single JSON object to stdout. Research must extract these fields:

| Field | JSON Path | Type | Purpose |
|-------|-----------|------|---------|
| mode | `.mode` | String | always `"backtest"` |
| method | `.backtest.method` | String | always `"conf_blend"` |
| sharpe | `.backtest.sharpe` | Double | primary sort key |
| max_drawdown | `.backtest.max_drawdown` | Double | risk gate (must be < 0.10) |
| closed_trades | `.backtest.closed_trades` | Int | activity gate (must be ≥ 4) |
| avg_trade | `.backtest.avg_trade` | Double | per-trade economics |
| total_return | `.backtest.metrics.totalReturn` | Double | raw P&L |
| annualized_return | `.backtest.metrics.annualizedReturn` | Double | return scaling |
| annualized_volatility | `.backtest.metrics.annualizedVolatility` | Double | risk scaling |
| sortino | `.backtest.metrics.sortino` | Double | downside-adjusted return |
| calmar | `.backtest.metrics.calmar` | Double | return / maxDD |
| win_rate | `.backtest.metrics.winRate` | Double | hit rate |
| profit_factor | `.backtest.metrics.profitFactor` | Maybe Double | gross profit / gross loss |
| turnover | `.backtest.metrics.turnover` | Double | trading frequency |
| exposure | `.backtest.metrics.exposure` | Double | time in market |
| trade_count | `.backtest.metrics.tradeCount` | Int | total position changes |
| position_changes | `.backtest.metrics.positionChanges` | Int | total flips |
| agreement_rate | `.backtest.metrics.agreementRate` | Double | Kalman/LSTM agreement |
| blend_weight | `.backtest.blendWeight` | Double | param record |
| kalman_z_min | `.backtest.minEdge` | Double | param record (note: mapped to `minEdge` in JSON) |
| open_threshold | `.backtest.threshold` / `.backtest.openThreshold` | Double | param record |
| close_threshold | `.backtest.closeThreshold` | Double | param record |
| vol_conf_gate | `.backtest.vol_conf_gate` | String | gate preset used |
| train_size | `.backtest.split.train` | Int | training bars |
| backtest_size | `.backtest.split.backtest` | Int | backtest bars |
| runtime_sec | *measured externally* | Double | wall-clock per combo |

**Extraction one-liner (jq):**
```bash
jq '{
  mode: .mode,
  method: .backtest.method,
  sharpe: .backtest.sharpe,
  max_drawdown: .backtest.max_drawdown,
  closed_trades: .backtest.closed_trades,
  avg_trade: .backtest.avg_trade,
  total_return: .backtest.metrics.totalReturn,
  annualized_return: .backtest.metrics.annualizedReturn,
  annualized_volatility: .backtest.metrics.annualizedVolatility,
  sortino: .backtest.metrics.sortino,
  calmar: .backtest.metrics.calmar,
  win_rate: .backtest.metrics.winRate,
  profit_factor: .backtest.metrics.profitFactor,
  turnover: .backtest.metrics.turnover,
  exposure: .backtest.metrics.exposure,
  trade_count: .backtest.metrics.tradeCount,
  position_changes: .backtest.metrics.positionChanges,
  agreement_rate: .backtest.metrics.agreementRate,
  blend_weight: .backtest.blendWeight,
  open_threshold: .backtest.openThreshold,
  close_threshold: .backtest.closeThreshold,
  vol_conf_gate: .backtest.vol_conf_gate,
  train_size: .backtest.split.train,
  backtest_size: .backtest.split.backtest
}'
```

---

## 6. PASS/FAIL Criteria

### Per-Combo Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Runtime | < 120s | B5 baseline: `ta_trend` completes in ~2.6s; `conf_blend` is ~20× more complex due to LSTM training + dual predictor loop |
| Sharpe | > 0 | Hard gate from interim contract spec §2.1 |
| max_drawdown | < 0.10 (10%) | Hard gate from interim contract spec §2.1 |
| closed_trades | ≥ 4 | Hard gate from interim contract spec §2.1 |
| No NaN/Inf in metrics | all finite | Sanity gate |
| JSON parseable | jq exits 0 | Automation gate |

### Aggregate GO/NO-GO Verdict Logic

1. **NO-GO if any combo hangs** (runtime > 120s or zero stdout). This re-triggers B5 escalation.
2. **NO-GO if all combos fail Sharpe > 0**. Candidate A is disconfirmed.
3. **NO-GO if all combos fail maxDD < 10%**. Risk profile is unacceptable.
4. **GO if ≥ 3 combos pass all hard gates**. Proceed to locked 5-row scorecard replication.
5. **GO with qualification if 1–2 combos pass**. Run extended grid (12 combos) before scorecard.
6. **Best combo selection** (if GO): lexicographic tie-break per interim contract spec:
   - Primary: Sharpe ↓ (higher is better)
   - Secondary: max_drawdown ↑ (less negative is better)
   - Tertiary: avg_trade ↓ (less negative is better)
   - Quaternary: trade_retention_pct ↓
   - Quinary: turnover ↑

---

## 7. Estimated Runtime

| Component | Estimate |
|-----------|----------|
| LSTM training (30 epochs, 800 bars train, hidden=16) | ~15–30s |
| Kalman filter loop (1000 bars) | ~1–3s |
| Dual predictor backtest + metrics | ~5–10s |
| JSON serialization | < 1s |
| **Per combo total** | **~25–45s** |
| **6-combo grid total** | **~2.5–4.5 min** |
| **With 20% safety margin** | **~3–5.5 min** |

**Note:** If runtime exceeds 120s per combo, this is a B5 regression signal. Abort grid and escalate to trader-firm-execution.

---

## 8. Execution Checklist (for Execution team)

- [ ] B5 cleared by CIO (binary passes `conf_blend` smoke test in < 60s)
- [ ] B6 cleared by CIO/CTO (`ta_trend` Sharpe restored to ~3.54 ± 0.10)
- [ ] Binary built from known-good commit or HEAD with B5 fix
- [ ] Dataset `data/BTCUSDT-4h-1000.csv` present and checksum-verified
- [ ] Run each combo with `timeout 120` wrapper
- [ ] Capture stdout to `.json` file per combo
- [ ] Run jq extraction and append to `artifacts/research/candidate-a-scorecard-2026-05-18.csv`
- [ ] Apply aggregate GO/NO-GO verdict logic above
- [ ] Append results to `reports/trader-firm-research.md`

---

## 9. Blocker Status

| Blocker | Status | Owner | Clearance Criteria |
|---------|--------|-------|-------------------|
| B5 (Kalman+LSTM hang) | **OPEN** | trader-firm-execution / trader-firm-cto | `conf_blend` completes in < 60s with non-zero stdout |
| B6 (`ta_trend` Sharpe regression 3.54 → 0.18) | **OPEN** | trader-firm-cto / trader-firm-data | `ta_trend` Sharpe ≈ 3.54 ± 0.10 on current dataset + binary |

**Research stance:** Zero backtests will be run until both B5 and B6 are cleared. This packet is ready for immediate execution once cleared.

---

## 10. Revision History

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0 | 2026-05-18 16:32 UTC | trader-firm-research | Initial packet: 6 combos, exact CLI, schema, criteria, runtime estimate |
