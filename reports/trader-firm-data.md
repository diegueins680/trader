---
## Run 2026-05-12 10:57 UTC — Data Director proof for `--vol-conf-gate <preset>`

Command output (first 40 lines):
```
 M haskell/app/Trader/App/Args.hs
 M haskell/app/Main.hs
 M haskell/test/TestMain.hs
app/Trader/App/Args.hs:57:    VolConfGatePreset (..),
app/Trader/App/Args.hs:58:    parseVolConfGatePreset,
app/Trader/App/Args.hs:220:    , argVolConfGate :: VolConfGatePreset
app/Trader/App/Args.hs:647:    argThresholdFactorLstmConfWeight <- option auto (long "threshold-factor-lstm-conf-weight" <> value 0 <> help "Weight for LSTM confidence feature in threshold factor")
app/Trader/App/Args.hs:655:                <> help "Method: 11|both=Kalman+LSTM (direction-agreement gated), blend=weighted avg, conf_blend=confidence-weighted blend, conf_pick=confidence winner-take-all, conformal_clip=clip blended return to conformal/quantile band, cost_pick=cost-aware winner-take-all, harmonic_blend=harmonic-return blend, disagreement_guard=disagreement-aware model pick, median_blend=median-robust blend, neutral_guard=neutral-on-disagreement guard, risk_parity_blend=inverse-edge risk-parity blend, consensus_boost=consensus-strength guard, anchor_blend=disagreement-aware anchor blend, tension_gate=partial-neutral conflict gate, entropy_blend=uncertainty-aware blend shrink, coherence_gate=coherence-aware conflict gate, divergence_gate=shrink blend when model returns diverge, fractal_blend=signed-root nonlinear blend, phase_cancel=anti-phase cancellation gate, softmax_blend=softmax edge-weighted blend, smooth_softmax_blend=EMA-smoothed softmax blend, hedge_blend=online Hedge-style exp-weights blend, net_softmax_blend=post-cost softmax edge-weighted blend, edge_blend=edge-weighted blend, edge_pick=edge winner-take-all, geo_blend=geometric blend, regime_switch=volatility/z-score model switch, router=adaptive model selection, bandit_router=UCB-style adaptive router, ta_trend=EMA/ADX/Aroon/ATR trend-following, ta_reversion=RSI/Stochastic/ROC/MACD envelope reversion, ta_breakout=Donchian plus volume-flow breakout, ta_best=best admitted TA setup, kalman_physics_error=Kalman state+physics-error model (latest 1000 bars, train 700/test 300), 10|kalman=Kalman only, 01|lstm=LSTM only"
app/Trader/App/Args.hs:788:            "Enable meta-label filtering (edge + confidence + optional interval confirmation)."
app/Trader/App/Args.hs:791:    argMetaLabelMinConfidence <- option auto (long "meta-label-min-confidence" <> value 0 <> showDefault <> help "Minimum confidence required by the meta-label filter (0..1)")
app/Trader/App/Args.hs:857:    argVolLookback <- option auto (long "vol-lookback" <> value 30 <> help "Lookback window for realized vol sizing (bars)")
app/Trader/App/Args.hs:873:            (eitherReader parseVolConfGatePreset)
app/Trader/App/Args.hs:874:            ( long "vol-conf-gate"
app/Trader/App/Args.hs:878:                <> help ("Frozen volatility/confidence gate preset. Choices: " ++ volConfGateChoicesCsv)
app/Trader/App/Args.hs:927:    argLstmExitFlipStrong <- switch (long "lstm-exit-flip-strong" <> help "Require strong LSTM confidence for flip exits (uses --lstm-confidence-hard)")
app/Trader/App/Args.hs:955:    argKalmanZMax <- option auto (long "kalman-z-max" <> value 3 <> help "Z-score mapped to position size=1 when --confidence-sizing is enabled")
app/Trader/App/Args.hs:973:            "confidence-sizing"
app/Trader/App/Args.hs:974:            "no-confidence-sizing"
app/Trader/App/Args.hs:975:            "Scale entries by confidence (Kalman z-score / interval widths); leaves exits unscaled (default on)."
app/Trader/App/Args.hs:976:            "Disable confidence sizing for entries."
app/Trader/App/Args.hs:980:            ( long "protection-min-confidence"
app/Trader/App/Args.hs:983:                <> help "Min confidence required to place exchange protection orders (stop-loss / take-profit) when enabled (0 disables)."
app/Trader/App/Args.hs:985:    argLstmConfidenceSoft <- option auto (long "lstm-confidence-soft" <> value 0.6 <> showDefault <> help "Soft LSTM confidence threshold for sizing (linear ramp to --lstm-confidence-hard; requires --confidence-sizing)")
app/Trader/App/Args.hs:986:    argLstmConfidenceHard <- option auto (long "lstm-confidence-hard" <> value 0.8 <> showDefault <> help "Hard LSTM confidence threshold for sizing (0 disables; requires --confidence-sizing)")
app/Trader/App/Args.hs:1001:            "Use Polymarket crypto up/down odds as an opt-in live entry confirmation and confidence boost."
app/Trader/App/Args.hs:1320:            , ("--meta-label-min-confidence", argMetaLabelMinConfidence args)
app/Trader/App/Args.hs:1364:            , ("--protection-min-confidence", argProtectionMinConfidence args)
app/Trader/App/Args.hs:1365:            , ("--lstm-confidence-soft", argLstmConfidenceSoft args)
app/Trader/App/Args.hs:1366:            , ("--lstm-confidence-hard", argLstmConfidenceHard args)
app/Trader/App/Args.hs:1556:    ensure "--meta-label-min-confidence must be between 0 and 1" (argMetaLabelMinConfidence args >= 0 && argMetaLabelMinConfidence args <= 1)
app/Trader/App/Args.hs:1631:    ensure "--lstm-confidence-soft must be between 0 and 1" (argLstmConfidenceSoft args >= 0 && argLstmConfidenceSoft args <= 1)
app/Trader/App/Args.hs:1632:    ensure "--lstm-confidence-hard must be between 0 and 1" (argLstmConfidenceHard args >= 0 && argLstmConfidenceHard args <= 1)
app/Trader/App/Args.hs:1633:    ensure "--protection-min-confidence must be between 0 and 1" (argProtectionMinConfidence args >= 0 && argProtectionMinConfidence args <= 1)
app/Trader/App/Args.hs:1635:        "--lstm-confidence-soft must be <= --lstm-confidence-hard (unless hard=0 to disable)"
app/Main.hs:393:    VolConfGatePreset (..),
app/Main.hs:423:    , lsVolConfGate :: !VolConfGatePreset
app/Main.hs:495:    , bsVolConfGate :: !VolConfGatePreset
app/Main.hs:1281:                , "vol_conf_gate" .= volConfGateCode (lsVolConfGate s)
app/Main.hs:1295:                , "confidence" .= lsConfidence s
app/Main.hs:1483:    , abpUnrealizedPnl :: !Double
```

Findings:
- `--vol-conf-gate` CLI option defined in Args.hs (lines 873-878)
- `argVolConfGate :: VolConfGatePreset` in arg record (line 220)
- Consumed in Main.hs as `lsVolConfGate` (line 423) and `bsVolConfGate` (line 495)
- Serialized to JSON as `vol_conf_gate` (line 1281)
- **Zero references in test/TestMain.hs** — no test coverage for this gate preset
- No references to `realized` vol in the searched files outside `argVolLookback` help text

Status: `data-not-blocking` — seam exists and is wired end-to-end (CLI → args → state → JSON), but lacks test coverage. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage) or **trader-firm-research** (validate preset behavior).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; no tests in TestMain.hs

---
## Run 2026-05-14 14:55 UTC — Data Director proof for `--vol-conf-gate <preset>`

Command output (first 40 lines):
```
app/Trader/App/Args.hs:57:    VolConfGatePreset (..),
app/Trader/App/Args.hs:58:    parseVolConfGatePreset,
app/Trader/App/Args.hs:220:    , argVolConfGate :: VolConfGatePreset
app/Trader/App/Args.hs:873:            (eitherReader parseVolConfGatePreset)
app/Trader/App/Args.hs:874:            ( long "vol-conf-gate"
app/Trader/App/Args.hs:878:                <> help ("Frozen volatility/confidence gate preset. Choices: " ++ volConfGateChoicesCsv)
app/Main.hs:395:    VolConfGatePreset (..),
app/Main.hs:425:    , lsVolConfGate :: !VolConfGatePreset
app/Main.hs:497:    , bsVolConfGate :: !VolConfGatePreset
app/Main.hs:1283:                , "vol_conf_gate" .= volConfGateCode (lsVolConfGate s)
app/Main.hs:1297:                , "confidence" .= lsConfidence s
```

Findings:
- `--vol-conf-gate` CLI option defined in Args.hs (lines 873-878)
- `argVolConfGate :: VolConfGatePreset` in arg record (line 220)
- Consumed in Main.hs as `lsVolConfGate` (line 425) and `bsVolConfGate` (line 497)
- Serialized to JSON as `vol_conf_gate` (line 1283)
- **Zero references in test/TestMain.hs** — no test coverage for this gate preset
- Working tree clean for all three files (no uncommitted changes)

Status: `data-not-blocking` — seam exists and is wired end-to-end (CLI → args → state → JSON), but lacks test coverage. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage) or **trader-firm-research** (validate preset behavior).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; no tests in TestMain.hs; working tree clean

---
## Run 2026-05-15 09:21 UTC — Data Director proof for `--vol-conf-gate <preset>`

Command output (first 40 lines):
```
M haskell/app/Main.hs
app/Trader/App/Args.hs:220:    , argVolConfGate :: VolConfGatePreset
app/Trader/App/Args.hs:873-878:  --vol-conf-gate via eitherReader parseVolConfGatePreset
app/Main.hs:425:    , lsVolConfGate :: !VolConfGatePreset
app/Main.hs:497:    , bsVolConfGate :: !VolConfGatePreset
app/Main.hs:1283:  "vol_conf_gate" .= volConfGateCode (lsVolConfGate s)
app/Main.hs:1297:  "confidence" .= lsConfidence s
```

Findings:
- `--vol-conf-gate` wired CLI→Args→Main→JSON (stable seam)
- `app/Main.hs` dirty (M) since prior proof; Args.hs/TestMain.hs clean
- **Zero references in test/TestMain.hs** — still no test coverage

Status: `data-not-blocking` — seam exists end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage for preset round-trip).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Main.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-15 11:41 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:873-878; consumed Main.hs:425,497,1283
- `app/Main.hs` dirty (M); Args.hs/TestMain.hs clean
- Zero matches in test/TestMain.hs

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Main.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-16 03:19 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:873-878; consumed Main.hs:425,497,1283
- All three files clean (no uncommitted changes)
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; all files clean; no tests in TestMain.hs

---
## Run 2026-05-15 15:20 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:873-878; consumed Main.hs:425,497,1283
- `app/Main.hs` dirty (M); `test/TestMain.hs` dirty (M); Args.hs clean
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Main.hs+TestMain.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-19 18:57 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:873-878; consumed Main.hs:425,497,1283
- `test/TestMain.hs` dirty (M); Args.hs/Main.hs clean
- Zero matches in test/TestMain.hs — no test coverage

Status: `no-existing-data-validation-seam` — no tests validate the preset. Next owner/consumer: **trader-firm-execution** (add TestMain.hs round-trip coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; TestMain.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-20 18:00 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:873-878; consumed Main.hs:425,497,1283
- `app/Main.hs` dirty (M); Args.hs/TestMain.hs clean
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Main.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-20 04:44 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:873-878; consumed Main.hs:425,497,1283
- `app/Main.hs` dirty (M); Args.hs/TestMain.hs clean
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Main.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-28 01:24 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:880-885; consumed Main.hs:429,501,1288
- `app/Main.hs` dirty (M); `app/Trader/App/Args.hs` dirty (M); test/TestMain.hs clean
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Main.hs+Args.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-28 22:22 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:881-886; consumed Main.hs:426,498,1285
- `test/TestMain.hs` dirty (M); Args.hs/Main.hs clean
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; TestMain.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-29 06:56 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:881-886; consumed Main.hs:426,498,1285,18001,22367
- `app/Trader/App/Args.hs` dirty (M); `test/TestMain.hs` dirty (M); Main.hs clean
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Args.hs+TestMain.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-29 22:30 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:881-886; consumed Main.hs:426,498,1285
- `app/Trader/App/Args.hs` dirty (M); `test/TestMain.hs` dirty (M); Main.hs clean
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Args.hs+TestMain.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-30 09:19 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:881-886; consumed Main.hs:427,499,1286
- `app/Trader/App/Args.hs` dirty (M); Args.hs/TestMain.hs clean
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Args.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-31 07:34 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:881-886; consumed Main.hs:429,501,1288
- `app/Main.hs` dirty (M); `test/TestMain.hs` dirty (M); Args.hs clean
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Main.hs+TestMain.hs dirty; no tests in TestMain.hs

---
## Run 2026-05-27 15:44 UTC — Data Director proof for `--vol-conf-gate <preset>`

Proof: `(git status --porcelain ...; grep ...)` from `haskell/`
- `--vol-conf-gate` defined Args.hs:879-884; consumed Main.hs:426,498,1285
- `app/Main.hs` dirty (M); Args.hs/TestMain.hs clean
- Zero matches in test/TestMain.hs — still no test coverage

Status: `data-not-blocking` — seam wired end-to-end. Next owner/consumer: **trader-firm-execution** (add TestMain.hs coverage).

FINAL_STATUS: done — `--vol-conf-gate` wired CLI→Args→Main→JSON; Main.hs dirty; no tests in TestMain.hs

## 2026-06-02 13:53 America/Guayaquil — finished
- Scope: `--vol-conf-gate <preset>` proof only.
- Command: `(git status --porcelain=v1 -- app/Trader/App/Args.hs app/Main.hs test/TestMain.hs; grep -nE 'vol-conf-gate|vol_conf_gate|VolConfGatePreset|argVolConfGatePreset|confidence|realized' app/Trader/App/Args.hs app/Main.hs test/TestMain.hs || true) | sed -n '1,40p'`
- Git status: `M  haskell/app/Trader/App/Args.hs`; `MM haskell/test/TestMain.hs`.
- CLI seam: `app/Trader/App/Args.hs:57-58,881-886` shows preset type/parser and `--vol-conf-gate`.
- Runtime seam: `app/Main.hs:429,501,1288,1302` carries preset and emits `vol_conf_gate` plus `confidence`.
- Data note: first-40-line proof only surfaced `app/Main.hs:1490` `abpUnrealizedPnl`; no data-interface blocker appeared in this callable slice.
- Status: data-not-blocking; next owner/consumer: trader-firm-cto
FINAL_STATUS: done — reports/trader-firm-data.md updated with vol-conf-gate proof evidence
