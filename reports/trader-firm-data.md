
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
