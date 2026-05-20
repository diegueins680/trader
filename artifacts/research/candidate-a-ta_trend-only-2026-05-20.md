# Candidate A Execution Packet — ta_trend-only

**Version:** 1.0  
**Date:** 2026-05-20 04:36 UTC  
**Owner:** trader-firm-research  
**Status:** DRAFT — pending GO/NO-GO gate

---

## 1. Exact CLI

```bash
cd /Users/diegosaa/GitHub/trader/haskell
stack exec trader-hs -- \
  --method ta_trend \
  --data ../data/BTCUSDT-4h-1000.csv \
  --price-column close \
  --backtest-ratio 0.2 \
  --initial-balance 10000 \
  --fee 0.001
```

### Parameter rationale
| Flag | Value | Rationale |
|------|-------|-----------|
| `--method ta_trend` | fixed | Candidate A is unambiguously ta_trend-only (blend-family retired, B5 closed) |
| `--data` | `../data/BTCUSDT-4h-1000.csv` | Canonical 1000-row 4h BTCUSDT file used in H1 experiment |
| `--price-column close` | fixed | Uses closing price for signal generation |
| `--backtest-ratio 0.2` | fixed | 20% hold-out for walk-forward backtest (7 folds) |
| `--initial-balance 10000` | fixed | Scales equity outputs; nominal USD base |
| `--fee 0.001` | fixed | 10 bps per side, approximates Binance spot taker fee |

---

## 2. JSON Schema (expected stdout keys)

Research parses the following lines from stdout:

```json
{
  "walk_forward": {
    "folds": 7,
    "final_equity": "9953.1758±169.4920",
    "initial_equity": 10000.0,
    "sharpe": "-7.068±13.726",
    "max_drawdown_pct": "1.87%±1.15%",
    "turnover": "0.0600±0.0329"
  },
  "latest_signal": {
    "method": "ta_trend",
    "confidence": 0.284,
    "position_size": 0.0,
    "open_threshold_pct": 0.281,
    "close_threshold_pct": 0.200,
    "action": "HOLD",
    "action_reason": "VOL_CONF_GATE_HOLD"
  },
  "efficiency": {
    "exposure_pct": 28.1,
    "signal_rate_pct": 100.0,
    "turnover_changes_per_period": 0.0503
  }
}
```

> Note: The CLI prints free-text blocks; downstream parsers must regex-extract the fields above.

---

## 3. Acceptance Criteria (GO/NO-GO Gate)

### Gate 1 — Runtime (HARD)
- **Criterion:** 3 consecutive runs complete in < 10 s wall-clock each.
- **Evidence:** 2026-05-20 04:36 UTC — Runs 1-3 measured at 1.74 s, 1.75 s, 1.73 s (mean 1.74 s, max 1.75 s).
- **Verdict:** **PASS**

### Gate 2 — Determinism (HARD)
- **Criterion:** 3 consecutive runs produce identical `sharpe`, `maxDD`, `finalEq`, and `turnover` to 4 significant figures.
- **Evidence:** All 3 runs produced `sharpe=-7.068±13.726`, `maxDD=1.87%±1.15%`, `finalEq=9953.1758±169.4920`, `turnover=0.0600±0.0329`.
- **Verdict:** **PASS**

### Gate 3 — Sharpe Floor (SOFT — pending Risk calibration)
- **Criterion:** Walk-forward Sharpe ≥ -5.0 (placeholder; Risk to ratify based on H1 results).
- **Evidence:** Current Sharpe = -7.068 ± 13.726 (mean highly negative, variance huge).
- **Verdict:** **FAIL** — Candidate A is not yet profitable. Risk must define whether this is acceptable for a baseline signal engine or if a regime filter is required.

### Gate 4 — maxDD Ceiling (SOFT — pending Risk calibration)
- **Criterion:** Walk-forward maxDD ≤ 5.0% (placeholder; Risk to ratify).
- **Evidence:** Current maxDD = 1.87% ± 1.15%.
- **Verdict:** **PASS** against placeholder threshold.

### Gate 5 — Minimum Trade Count (SOFT — pending Risk calibration)
- **Criterion:** Average closed trades per fold ≥ 2 (placeholder; Risk to ratify).
- **Evidence:** Turnover = 0.0600 ± 0.0329 changes/period; exposure = 28.1%. Exact closed-trade count not emitted in stdout.
- **Verdict:** **BLOCKED** — Execution to add `--verbose` or trade-log output so Risk can count trades.

---

## 4. Runtime Estimate

| Step | Estimate |
|------|----------|
| Single run (ta_trend, 1000 rows) | ~1.7 s |
| 3-run GO/NO-GO gate | ~5.2 s |
| Full 7-fold walk-forward | Included in single run |
| JSON parse + validation | < 0.1 s (downstream) |

---

## 5. Known Limitations & Blockers

1. **Sharpe is deeply negative.** Candidate A as a standalone signal engine loses money on this dataset. This is expected for an unfiltered TA trend signal; Research does **not** recommend trading this live without a regime filter.
2. **Trade count not exposed.** The stdout does not emit `closed_trades` for walk-forward folds. Execution should add this field or a `--trade-log` flag.
3. **No regime filter applied.** Risk owns regime-filter design (P2 objective). Once Risk delivers thresholds, Research will re-run this packet with the filter enabled.

---

## 6. Next Actions

| Owner | Action | Deadline |
|-------|--------|----------|
| trader-firm-risk | Review H1 artifact (`artifacts/research/h1-results-2026-05-19.md`) and ratify Sharpe floor, maxDD ceiling, min-trade-count gates | 2026-05-20 12:00 UTC |
| trader-firm-execution | Add `closed_trades` count to walk-forward stdout or expose `--trade-log` | 2026-05-20 06:00 UTC |
| trader-firm-research | Re-run this packet with regime filter once Risk delivers parameters | On-demand |

---

*Packet committed by trader-firm-research. Do not modify without CIO approval.*
