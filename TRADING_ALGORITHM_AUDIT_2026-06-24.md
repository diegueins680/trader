# Trading Algorithm Audit — 2026-06-24

Scope: the signal → optimization → execution pipeline in `haskell/app/Trader`.
Method: read the core modules, ran three parallel deep-reads of the signal,
execution/risk, and optimizer/backtest layers, then **independently verified
every high-severity claim against the source** before recording it here.
Findings the verification disproved are listed in §5 so they are not chased.

Headline: the dominant, real risk is **selection overfitting in the optimizer/
leaderboard pipeline** — and it is already empirically visible in the repo's own
`ENGINEERING_REVIEW_2026-06-14.md` (leaderboard median = 4 trades, median
in-sample Sharpe = 8.52, 0/500 combos carry a walk-forward summary). The signal
and execution layers are, on inspection, in better shape than a first pass
suggests; most of the "critical" indicator findings did not survive verification.

---

## 1. Critical / High — verified

### 1.1 The optimizer rewards overfit combos by construction
Evidence (code, before the hardening recorded in §6):
- `Trader/Optimization.hs` `defaultTuneConfig` shipped
  `tcWalkForwardFolds = 1`, `tcWalkForwardEmbargoBars = 0`, `tcMinRoundTrips = 0`.
  Any caller that uses the default (tests, internal sub-optimizations, anything
  not going through the CLI) scores **purely in-sample, with no minimum trade
  count**. The CLI (`OptimizeEquityMain.hs`) overrides folds to 7, so production
  is better — but the safe default is the unsafe one.
- Objective `TuneSharpe` (`Optimization.hs:168`) and `TuneRoi` are computed on
  the *same* backtest being maximized, with no penalty for sample size or for
  cross-fold Sharpe instability.

Evidence (production data, from `ENGINEERING_REVIEW_2026-06-14.md` §0):
- Median `tradeCount` across the 500-combo leaderboard = **4**.
- Median in-sample Sharpe = **8.52** (the review itself calls this
  "statistically meaningless given n=4 trades").
- Combos with a `walkForwardSummary` = **0 / 500** — the walk-forward Sharpe
  gate has nothing to fire on at adoption time.

Why it matters: a 4-trade backtest with Sharpe 8 is noise. Ranking 500 of these
by in-sample score and promoting the top is pure selection bias — you are
sampling the right tail of a noise distribution.

Recommended fixes (in priority order):
1. Make `tcMinRoundTrips` a default selection floor (e.g. >= 20-30 round trips)
   inside `defaultTuneConfig`, not just at the CLI, so convenience callers do
   not rank a 4-trade run as an eligible candidate. Separately decide whether
   all-below-floor searches should fail closed instead of returning the existing
   base-threshold fallback.
2. Make walk-forward mandatory in the default (`tcWalkForwardFolds ≥ 5`,
   `tcWalkForwardEmbargoBars ≥ 1`) and **reject** combos that lack a
   `walkForwardSummary` rather than letting them onto the leaderboard.
3. Score on **out-of-fold** Sharpe/ROI (mean across folds) and **penalize the
   cross-fold std** (`TuneStats` already carries `tsStdScore` —
   `Optimization.hs:113` — wire it into the objective).
4. Deflate the selection: with N candidates evaluated, the expected max
   in-sample Sharpe under the null is large. Apply a deflated-Sharpe / Bonferroni
   style haircut keyed off the number of trials before ranking.

### 1.2 Threshold sweep is not fold-aware
`Optimization.hs` `sweepThresholdWithHLWith` selects entry/exit thresholds over
the price series it is handed. Walk-forward protects the *strategy params* but
the threshold grid search should run **inside each training fold** and be frozen
for the test fold; otherwise the swept threshold has seen the test data. Confirm
the call sites pass fold-local slices, not the full series, and add a test that
asserts the test fold is untouched during the sweep.

### 1.3 The pipeline currently produces nothing deployable
Also from the 06-14 review: **0 / 500** combos clear the 18 bp `venueMinEdgeFloor`
cost floor, and the system has not opened a live trade in ~17 days. The cost
floor is the correct guard (see §3), but combined with 1.1 it means the optimizer
emits combos whose edge is entirely consumed by realistic costs. The fix is not
to lower the floor — it is to optimize **net-of-cost edge** as the objective so
the search stops finding gross-edge mirages. Verify `roiImplementationScore`
and the sweep both subtract `roundTripCostAt` before ranking.

---

## 2. Medium — verified

### 2.1 Aroon uses the earliest extreme on ties
`Indicators.hs:318` `highestIdx = V.maxIndex highWindow` returns the *first*
occurrence of the max; canonical Aroon uses the *most recent* extreme. On
repeated equal highs/lows (common after rounding or on flat ranges) this biases
`aroonUp/aroonDown` downward and makes the indicator less responsive. Fix: scan
for the last index of the max/min within the window.

### 2.2 Silent staleness from `latestJust` fallback
Strategy entries pull indicators via `latestJust (… series)` (e.g.
`Strategies.hs:408-414`). When the most recent value is `Nothing` (flat
Stochastic range, zero-prev-price ROC, etc.) `latestJust` walks backward and
returns an older value with **no staleness flag**. During illiquid/halted
periods a signal can fire on indicator values several bars old. Recommend
capping the look-back (reject if the latest valid value is > k bars old) and
surfacing the age in telemetry.

### 2.3 Library default for walk-forward is a footgun (same root as 1.1)
Worth calling out separately: prefer to delete the "safe-looking" default of
1 fold / 0 embargo entirely and require callers to pass an explicit
`TuneConfig`, so an in-sample-only score can never happen by omission.

### 2.4 Magic numbers are pervasive and undocumented
`VolConfGate.hs` (vol thresholds 0.5/1.2, confidence 0.60/0.80, size mults
0.35–1.00), `SignalGates.hs` (headroom ×1.5, edge-spike ×1000 cap 5.0, regime
mass tol 1e-3), `Method.hs` (divergence 0.02, magnitude floor 1e-12 vs sign
tol 1e-9). None are individually bugs, but they are unsourced and many are *not*
swept by the optimizer, so they are silent priors. Recommend: move the
trade-impacting ones into the optimizer's search space or document the
calibration that produced them.

### 2.5 Inconsistent epsilons
`1e-9` in `OrderExecution.hs`, `1e-12` in `Formal/Execution.hs` /
`VolConfGate.hs` / cost code. Positions below `1e-9` are silently zeroed in
`applyExecutedQuantity`. For most assets this is harmless, but pin a single
documented dust threshold rather than scattering three.

---

## 3. What is actually well-built (do not "fix")

These were flagged by the automated pass but verification showed they are
correct or are deliberate, good design:

- **No look-ahead in the core sim loop.** `Trading.hs:1994-1997` decides at bar
  `t` (`prev = pricesV ! t`) and fills/evaluates at `t+1` (`hi/lo = barHigh/Low
  (t+1)`). This is the correct anti-look-ahead pattern.
- **Cost model is realistic.** `costPerSideBreakdown` (`Trading.hs:1179`) charges
  fee + fixed + min + base slippage + vol-scaled slippage + power-law size
  impact + spread, plus funding via `applyFundingWithTotals`. More complete than
  most retail backtesters.
- **Reduce-only IS applied live.** `applyReduceOnlyExecutedQuantity` is used on
  the live close path (`Main.hs:10355, 13994, 14127`) and orders send
  `reduceOnly=True` (`Main.hs:21291`). It is correctly absent from the backtest
  simulator only.
- **Risk-per-trade IS used.** `riskScaleAt` (`Trading.hs:1721`) scales size by
  `risk / stopLossFrac`.
- **A stress-shock penalty exists** in the objective (`scoreBacktest`,
  `Optimization.hs:142-155`) — combos are penalized for fragility under a vol
  multiplier + shock.
- **Donchian breakout uses the prior-bar channel** (`Strategies.hs:892`) — this
  is the *correct* non-look-ahead construction, not a bug.

---

## 4. Suggested priority order

1. **§1.1** — floor min round-trips + mandatory walk-forward in the *default*
   config; reject combos without a walk-forward summary. (Highest leverage; the
   leaderboard is currently noise.)
2. **§1.3 / §1.2** — optimize net-of-cost edge and make the threshold sweep
   fold-local. (Makes the surviving combos actually deployable.)
3. **§1.1.3/1.1.4** — penalize cross-fold Sharpe std and apply a
   multiple-testing haircut before ranking.
4. **§2.2** — staleness cap on indicator fallbacks.
5. **§2.1** — Aroon tie fix.
6. **§2.4 / §2.5** — document/centralize magic numbers and epsilons.

---

## 5. Claims that did NOT survive verification (recorded so they are not chased)

- "ATR seed is off by one." `Indicators.hs:209` averages TR[1..period]
  deliberately because TR[0] has no prior close — this is standard Wilder ATR,
  not a bug.
- "Bollinger uses population variance → bands too tight (critical)." ÷N is the
  **standard** Bollinger convention (StockCharts and most libraries). Not a bug.
- "Reversed envelope multipliers create a long bias." `lower*1.02` and
  `upper*0.98` (`Strategies.hs:415-416`) are *symmetric* — both loosen the
  trigger band toward the center by 2%. No directional bias.
- "Reduce-only never applied / risk-per-trade never used / funding not modeled."
  All three are used in the live path (see §3); the automated pass conflated the
  backtest simulator with the live trader in `Main.hs`.
- "Regime scoring mixes incompatible scales." Each component is normalized by its
  own floor/span to ~[0,1] before averaging (`Strategies.hs:317-319`), which is
  exactly what makes them comparable. The equal-weight average is a defensible
  design choice, not an error.

---

## 6. Implementation status (2026-06-24)

Implemented in this change:

- **§1.1 / §2.3 — `defaultTuneConfig` is now safe by default**
  (`Optimization.hs`): walk-forward scoring defaults to `folds 1->5` and
  `embargo 0->1`, with a `minRoundTrips 0->20` selection floor. The production
  CLI path passes an explicit `TuneConfig`, so its behavior is unchanged. The
  existing threshold sweep fallback still returns the configured base thresholds
  if every candidate misses the activity floor; treat that as a separate
  fail-closed policy decision.
- **§2.1 — Aroon now uses the most-recent extreme** (`Indicators.hs`):
  `lastIndexOfMax/Min` replace `V.maxIndex/minIndex`, which returned the oldest
  extreme on ties.
- **Bonus (found while implementing) — `.env.example` no longer ships the
  walk-forward adoption gates disabled.** It set
  `TRADER_OPTIMIZER_MIN_WF_SHARPE_MEAN=0` and `…MAX_WF_SHARPE_STD=0`, which turn
  OFF the very gates (code defaults 0.3 / 1.5) that stop single-fold overfit
  combos from being adopted. Now set to the safe defaults with a warning comment.

Deliberately NOT changed, with reasons:

- **§1.1.3 cross-fold std penalty** — already implemented in the auto-optimizer
  as `wfSharpeStdScorePenalty` (default 0.05, `Main.hs:12090`), alongside
  `minWfSharpeMean=0.3` and `maxWfSharpeStd=1.5` gates. No need to duplicate.
- **Auto-optimizer `minRoundTrips=3`** (`Main.hs:12022`) — left as-is. The
  auto-optimizer is a *discovery* engine that is permissive by design and gated
  at *adoption*; raising the discovery floor is the wrong layer. The real defect
  the 06-14 review found is that **0/500 combos carry a `walkForwardSummary`**, so
  the adoption-time WF gate has nothing to fire on. That is a plumbing fix
  (compute + persist the WF summary for every candidate) that needs broader
  optimizer/top-combo changes — flagged, not attempted here.
- **§1.1.4 multiple-testing haircut, §1.3 net-of-cost objective, §2.2 indicator
  staleness cap** — each changes scoring or signal semantics and must be
  validated as separate optimizer/signal work before deployment.

## 7. Follow-up status (2026-11-07)

Landed after the original change (#310):

- **#377** — `scripts/verify.sh` pins a UTF-8 locale (the test suite aborted
  decoding `Main.hs` under a bare locale) and 116 stale `tmp/` fixtures that
  predate the walk-forward-summary feature are untracked. The "0/500 combos
  carry a `walkForwardSummary`" figure quoted above came from those stale
  snapshots, not from a plumbing defect: current runs emit the summary and the
  adoption gate already rejects candidates without one (`walk-forward-missing`).
- **#378 — §1.1.4 multiple-testing haircut** (opt-in). `optimize-equity
  --wf-sharpe-trials-haircut SCALE` / `TRADER_OPTIMIZER_WF_SHARPE_TRIALS_HAIRCUT`
  deflates each trial's walk-forward Sharpe mean by
  `SCALE·√(2 ln trials)·sharpeStd/√evaluatedFolds` before the
  `--min-wf-sharpe-mean` gate. Default `0` leaves production unchanged; `1.0`
  is the recommended starting point once its effect on the leaderboard has
  been observed.
- **#379 — §2.2 indicator staleness cap.** `latestJust`/`laggedJust` reject
  readings older than `defaultMaxIndicatorStaleBars` (2) and fail closed; the
  bounded primitives expose the reading's age for telemetry.

Verified against the source and closed without a code change:

- **§1.2 threshold sweep is fold-local.** The CLI tunes on
  `tunePrices = drop fitSize trainPrices` (the trailing `--tune-ratio` share of
  `splitTrain`), and the walk-forward report simulates the frozen thresholds
  over `splitBacktest` fold by fold (`mkFold` in `Main.hs`). The sweep never
  receives a backtest bar. `testSplitKeepsTuneWindowOutOfBacktest` pins the
  split contract (ordered partition, tune window ending at `splitTrainEnd`,
  every tune bar before every backtest bar). The live re-tune path hands the
  sweep the trailing live window, where the "test fold" is the future.
- **§1.3 net-of-cost objective.** Fees, slippage, spread and impact are applied
  inside the simulator, so every tune/backtest metric the sweep ranks on is
  already net of cost, and `minEdge` is floored at `venueMinEdgeFloor`.

Still open (documentation/calibration, no behavior defect): §2.4 magic numbers
and §2.5 epsilon consolidation — closed in §8 below.

## 8. Constants review (2026-11-14): §2.4 and §2.5 closed

Re-read against the current source; the picture is better than §2.4/§2.5
describe.

**§2.4 — the trade-impacting constants are already configurable and swept.**

| Constant (audit wording) | Where it lives now | CLI | Swept by the optimizer |
| --- | --- | --- | --- |
| Vol thresholds 0.5 / 1.2, confidence 0.60 / 0.80, size multipliers 0.35–1.00 | `VolConfGateConfig` defaults in `VolConfGate.hs` (presets `vol_conf_v1_*`) | `--vol-conf-*` | yes: `torVolConf*` ranges in `Optimizer/Optimize.hs` (evidence max, both vol thresholds, both confidence thresholds, all five size multipliers) |
| Entry-edge headroom ×1.5, spike ×1000 / cap 5.0 / 3 consecutive, trend slack, directionality lookback/efficiency/weak-band, predictor tracking floor | `SignalGateConfig` defaults in `SignalGates.hs` | `--signal-*` (wired at `Main.hs` `argSignal*`) | yes: `torSignal*` ranges (headroom, spike multiple, cap and consecutive limit, trend slack multiple and cap, directionality lookback, chop/MR efficiency, weak-band z, tracking floor) |
| Regime mass tolerance 1e-3, efficiency tolerance 1e-12 | `directionalityRegimeMassTol`, `directionalityEfficiencyTol` in `SignalGates.hs` | fixed | no — numerical tolerances (probability mass must sum to 1 within 1e-3; efficiency comparisons), not trading priors |
| Divergence 0.02, magnitude floor 1e-12, sign tolerance 1e-9 | `regimeDivergenceThreshold`, `momentumMagnitudeFloor`, `momentumSignTolerance` in `Method.hs`, each named and documented | fixed | no — the only trading prior left fixed is the 2 % "strong divergence" cut used by the divergence-aware methods; the other two are numerical guards |

So the "silent priors" concern reduces to one constant (`regimeDivergenceThreshold`), and the real risk the section pointed at — thresholds that trading depends on but the search cannot touch — does not exist: every gate parameter the optimizer promotes is part of its search space and is persisted with the trial.

**§2.5 — the execution epsilons were consistent; they are now named.**

The audit counted three epsilons. Reading them: `OrderExecution.hs` uses a 1e-9 dust threshold for fills and residual positions plus a 1e-12 *relative* tolerance in the fill-conservation assertion; `Formal/Execution.hs` restates the same 1e-9 in its naive spec (its 1e-12 appears only as a probe quantity in the verification grids); `VolConfGate.hs`'s 1e-12 is a division guard on the volatility evidence maximum, unrelated to dust. There was no disagreement to reconcile. `OrderExecution.hs` now exports `executionDustQuantity` (1e-9) and `executionRelativeTolerance` (1e-12) and uses them in both accounting paths, with haddocks stating their roles; the formal spec keeps its own literal by design (it must stay naive and independent of the implementation), and `verifyFormalExecution` already checks implementation and spec agree on quantities straddling the threshold (0, 1e-12, 1e-9, 1e-6, …).

With this, every item in §1 and §2 is either implemented (#310, #377–#381) or verified against the source and recorded here.
