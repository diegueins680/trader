# Engineering Review — 2026-06-14 (Sunday morning, America/Guayaquil)

> Daily engineering review per `AGENTS.md` workflow. Trades observed today,
> hypotheses with falsification, measured failure modes, code changes,
> validation. Verified by `stack build` (green), `stack test` (green:
> `Test suite trader-tests passed`), `fourmolu --mode check` (clean on
> touched files), `hlint app/Trader/BotStartSemantics.hs app/Main.hs
> test/TestMain.hs` (no hints), and a `trader-hs` JSON backtest smoke run
> against `data/sample_prices.csv`. The review was started at 09:18 local
> on 2026-06-14 (America/Guayaquil), driven by the daily cron job
> `Trader daily engineering review`.

## 0. Summary

| Metric | Value | Δ vs 2026-06-12 |
|---|---|---|
| Trades executed today | **0** | unchanged |
| Live positions opened today | 0 | unchanged |
| `live_trades.ndjson` last write | 2026-05-28 (`haskell/.tmp`), 2026-05-24 (`.tmp`) | unchanged (system has not opened a live trade for ~17 days) |
| Running binary | `Build: 0.1.0.0 (3d32f58cc08d)` since 2026-06-13 | binary now includes yesterday's cost-floor + adoption-cap + WF-Sharpe shipset (post-fix code) |
| Total `/binance/positions` requests today (current slice) | ~8,300 lines | dominant traffic; **100% returned HTTP 401** (operator key/IP issue, see §6.1) |
| `live_trades` opened today | **0** | the cost-floor/position-cap/WF-Sharpe guards correctly hold the system flat |
| `Top combos sync reconciled db, s3` (current slice) | 45 | unchanged baseline behaviour |
| `Top-combo startup backtest aborted` (current slice) | 0 | the H10/H11 fix from 2026-06-12 is shipping and the guard never aborted today |
| `Live bot auto-start failed` (current slice) | 0 visible (no signal layer ran) | — |
| `auth circuit OPEN` (current slice) | 0 | the classifier still returns `unknown` for the live 401s; circuit never trips |
| Total leaderboard combos | **500** | unchanged |
| Combos with `minEdge >= venueMinEdgeFloor` (18 bp) | **0 / 500** | unchanged — cost floor filters all of them out |
| **Median `tradeCount` across leaderboard** | **4** | **NEW finding: a 4-trade sample cannot produce a meaningful Sharpe** |
| **Combos with `walkForwardSummary`** | **0 / 500** | **NEW finding: the WF Sharpe gate has nothing to fire on at adoption time** |
| Median `maxPositionSize` on leaderboard | 0.771 | clamped to ≤ 0.25 at adoption by yesterday's `capAdoptedMaxPositionSize` |
| Median in-sample Sharpe on leaderboard | 8.52 | statistically meaningless given n=4 trades (§1.3) |
| Code changes today | 3 files + CHANGELOG + this review | — |
| Tests added | 6 new invariants (adoption min-trade-count, adoption WF-Sharpe-mean) | — |

The system continues to place no live trades today. The signal layer has
not run because Binance still returns HTTP 401 on every preflight; that
is an operator action (key rotation / IP allow-list) and remains outside
the scope of code changes (see §6.1, carried forward from 2026-06-12).

The good news is that **yesterday's cost-floor + position-cap + WF-Sharpe
shipset is live** (the running binary is `3d32f58c`, which includes
`venueMinEdgeFloor`, `capAdoptedMaxPositionSize`, and the
`minWfSharpeMean = 0.3` / `maxWfSharpeStd = 1.5` optimizer defaults).
The cost floor mathematically prevents the prior `Sharpe = -3.20`
disaster from recurring: every one of the 500 leaderboard combos has
`minEdge` below the floor, so adoption refuses all of them. Flat is
strictly better than `Sharpe = -3.20`.

The bad news, exposed only because the cost floor is now filtering the
leaderboard, is that **the same 500 combos also have median `tradeCount = 4`
and zero of them carry a `walkForwardSummary`**. The cost-floor filter
catches them today, but if a future predictor improvement justifies
loosening the floor, those low-evidence combos will be the first to leak
through. Today's fix mirrors the optimizer's two production gates at
**adoption time** so the two gates stay falsifiably equal:

1. **H1 (adoption min-trade-count gate):** `topComboTradeCountBelowFloor`
   rejects any combo whose stored `metrics.tradeCount < 20`, matching the
   documented `TRADER_OPTIMIZER_MIN_ROUND_TRIPS` production gate. Closes
   the 2026-06-14 leaderboard pathology: 500/500 combos with median
   `tradeCount=4` and an inflated median Sharpe of 8.52.
2. **H2 (adoption walk-forward Sharpe-mean gate):** `topComboWalkForwardSharpeBelowFloor`
   rejects any combo whose `metrics.walkForwardSummary.sharpeMean` is
   missing, non-finite, or below `0.3` — exactly the optimizer default
   that yesterday's fix turned on (`runAutoOptimizerLoop`'s
   `minWfSharpeMean = 0.3`). Closes the gap where the optimizer-side
   gate fires on freshly-produced trials but the adoption path has
   never enforced it on legacy leaderboard combos.

Both filters are wired into the two adoption call sites:
`selectCompatibleTopComboArgs` (single-symbol adoption) and
`topCombosTopTargets` (multi-symbol fleet adoption). The pure predicates
live in `Trader.BotStartSemantics` next to `adoptionMaxPositionSizeCap`
and `capAdoptedMaxPositionSize`, so the adoption surface is one module.

---

## 1. Trade Analysis

### 1.1 Raw counts

The launchd log `/tmp/trader-api-launchd.log` is 16.94 MB / 113,853 lines
and has seen 59 distinct `Build:` lines across the file's history. The
running binary as of this review is `3d32f58cc08d` (line 105,547 — the
last process restart), so the "today" slice is the 8,307 lines since
that timestamp.

```
$ grep -n "Build: 0.1.0.0" /tmp/trader-api-launchd.log | tail -5
102595:Build: 0.1.0.0 (b8fd10dd1f9b)
102609:Build: 0.1.0.0 (50f33e6bda72)
104273:Build: 0.1.0.0 (3d32f58cc08d)
105373:Build: 0.1.0.0 (3d32f58cc08d)
105547:Build: 0.1.0.0 (3d32f58cc08d)
```

Counts in the current process slice (8,307 lines from the running
binary):

```
   ~8,300  Request GET /binance/positions -> 401 (the dominant traffic)
        6  Request POST /binance/trades   -> 200 (recent UI activity)
       45  Top combos sync reconciled db, s3
        0  Top-combo startup backtest aborted
        0  Live bot auto-start failed
        0  auth circuit OPEN
```

The signal layer did not run today (no `auto-start`, no `BACKTEST` log
lines, no `BOT_START` lines), which means there were no startup
backtest guards to record a verdict — the H10/H11 fix from 2026-06-12
is shipping but had nothing to operate on today.

### 1.2 Why the signal layer did not run

The 401 storm on `/binance/positions` is the same auth/IP failure mode
flagged in the 2026-06-12 review's §6.1. The 2026-06-08 backoff
classifier (`Trader.App.AutoStartBackoff.classifyError`) does not
recognize today's error body shape, so every failure is logged as
`unknown (treated as transient)` instead of `ErrAuth`. With every
failure classified as transient, the per-symbol exponential backoff
does throttle the retry cadence, but the global auth circuit
(`TRADER_BOT_AUTOSTART_AUTH_CIRCUIT_THRESHOLD`) never trips because no
symbol is ever classified as `ErrAuth`. **The traffic we are seeing
today is the throttled-but-not-silenced loop**: 401 every 0–8 ms, which
is the Warp request log line, not the auto-start loop directly.

The operator action remains: either rotate `BINANCE_API_KEY` /
`BINANCE_API_SECRET` and re-add the current egress IP to the Binance
allow-list, or capture the exact 401 body so the classifier can be
extended. This is an operator decision because it requires touching
production secrets and the production Binance allow-list, which is not
something the daily review should automate.

### 1.3 The leaderboard pathology that today's fix closes

The running binary's adoption path filters on `minEdge >=
venueMinEdgeFloor`, `liveQuarantined`, and (via `capAdoptedMaxPositionSize`)
`maxPositionSize <= 0.25`. The top-combos JSON store on the local box
(`haskell/.tmp/optimizer/top-combos.json`, 2.3 MB) has 500 combos.
Aggregate statistics (computed via `python3` against the JSON, see
the snippet at the bottom of §4):

| Metric | Value |
|---|---|
| Total combos | **500** |
| Combos with `params.minEdge >= venueMinEdgeFloor` (18 bp) | **0** |
| Combos with `metrics.tradeCount >= adoptionMinTradeCount` (20) | **?** |
| Median `metrics.tradeCount` | **4** |
| Min / max `metrics.tradeCount` | 1 / 182 |
| Median `metrics.sharpe` | **8.522** |
| Median `metrics.annualizedReturn` | **8.355** |
| Median `params.minEdge` | 4.8 bp |
| Median `params.maxPositionSize` | 0.771 |
| Combos with any `metrics.walkForwardSummary` | **0** |
| `liveStats.quarantined == true` | 0 |

The Sharpe of 8.52 is the headline number, and it is statistically
meaningless. A 4-trade window of per-trade returns with σ ≈ 1% produces
a Sharpe estimate with standard error ~`1 / sqrt(4) = 0.5`. An observed
Sharpe of 8 is therefore ~16 standard errors above zero **assuming the
true Sharpe is zero**, and the sample is small enough that the *sampling
distribution* of the Sharpe estimator under a true Sharpe of zero is
heavy-tailed — the "observed 8.5" is consistent with the heavy-tail
mode of the estimator, not with a true edge. The right test is the
walk-forward Sharpe mean on a held-out window with n ≥ 20 trades per
fold; today's 500 combos have zero walk-forward summaries.

Both failure modes — n=4 backtest and missing walk-forward summary —
are mathematically incompatible with the optimizer's *current* default
production gates. They survive on the leaderboard only because they
were generated before those gates landed. The cost-floor filter catches
them today; the new adoption gates catch them when the cost-floor is
relaxed.

---

## 2. Hypotheses (falsifiable)

### H1 — Adoption-side minimum trade count

**Hypothesis:** combos with `metrics.tradeCount < 20` cannot be
adopted, regardless of their (inflated) backtest Sharpe.

**Falsification:** the new pure predicate
`comboTradeCountMeetsAdoptionFloor` is `False` for `Nothing`, `Just 0`,
`Just 19`, and `True` from `Just 20` upward. The new `Main.hs` helper
`topComboTradeCountBelowFloor` is wired into both
`selectCompatibleTopComboArgs` (the per-symbol adoption path) and
`topCombosTopTargets` (the fleet-wide adoption path). Today's
production-regression test pins the failure of a combo with
`tradeCount = 4` (the observed leaderboard median) explicitly so a
future relaxation cannot regress to the pre-fix behavior without
breaking the test.

**Why 20?** The optimizer's documented production gate is
`TRADER_OPTIMIZER_MIN_ROUND_TRIPS = 20` (the auto-loop default is
`3` for discovery, but production sweeps document `20` as the
production-quality floor; see `OptimizeEquityMain.hs` defaults at
the `--min-round-trips` flag). Adoption must be at least as strict
as the gate that produced the combo, otherwise the optimizer's
production-mode work is wasted.

### H2 — Adoption-side walk-forward Sharpe-mean gate

**Hypothesis:** combos missing a walk-forward summary, or whose
walk-forward summary reports `sharpeMean < 0.3`, cannot be adopted.

**Falsification:** the new pure predicate
`comboWalkForwardSharpeMeetsAdoptionFloor` is `False` for `Nothing`,
non-finite (NaN, ±Inf), and below-floor readings; `True` at `>= 0.3`
exactly. The threshold is pinned by
`testAdoptionMinWalkForwardSharpeMatchesOptimizerDefault` to equal
the optimizer's `minWfSharpeMean` default, so a future change to the
optimizer must be mirrored in adoption (and vice versa) or the test
fails. The new `Main.hs` helper `topComboWalkForwardSharpeBelowFloor`
reads `metrics.walkForwardSummary.sharpeMean` directly from the
combo's stored metrics blob and is wired into the same two adoption
call sites as H1.

**Why fail closed on missing?** The optimizer turned on the WF gate
yesterday. Combos that predate the gate's introduction never produced
a walk-forward summary, so accepting them on `Nothing` would silently
opt them out of the gate. The adoption path's job is to make a
positive assertion about the combo's evidence, not to permit through
on the absence of evidence.

### H3 — Adoption surface is one module

**Hypothesis:** all adoption-time gates (cost-floor, position-cap,
trade-count, WF-Sharpe-mean) live in one module so a future audit
finds them in one place.

**Falsification:** today's PR adds
`adoptionMinTradeCount = 20`, `adoptionMinWalkForwardSharpeMean = 0.3`,
`comboTradeCountMeetsAdoptionFloor`, and
`comboWalkForwardSharpeMeetsAdoptionFloor` to `Trader.BotStartSemantics`,
next to the existing `adoptionMaxPositionSizeCap` and
`capAdoptedMaxPositionSize`. The two new `Main.hs` helpers are pure
delegations to those predicates so the wire layer holds no thresholds
of its own.

---

## 3. Code changes

```
haskell/app/Trader/BotStartSemantics.hs   +90  4 new exports (predicates + thresholds)
haskell/app/Main.hs                       +60  2 new TopCombo helpers + import + 2 filter wires
haskell/test/TestMain.hs                  +95  6 new invariants + import update + runner registration
CHANGELOG.md                              +1   new "Unreleased" entry
ENGINEERING_REVIEW_2026-06-14.md          new  this file
```

The Haskell change is small (~150 LOC across two app files and the test
runner) and falsifiable: every new behavior has a test. The cost-floor
filter from 2026-06-13 stays as-is; today's two new filters are
*additive* — they reject the same combos the cost-floor filter rejects,
plus a strict superset.

### 3.1 Adoption-time decision table (new today)

| `tradeCount` reading | `wfSharpeMean` reading | `minEdge` | adopt? |
|---|---|---|---|
| `Nothing` | anything | anything | No (H1 fails closed) |
| `Just n, n < 20` | anything | anything | No (H1) |
| `Just n, n >= 20` | `Nothing` | anything | No (H2 fails closed) |
| `Just n, n >= 20` | `Just s, s < 0.3` | anything | No (H2) |
| `Just n, n >= 20` | `Just s, s >= 0.3` (finite) | below floor | No (cost floor, 2026-06-13) |
| `Just n, n >= 20` | `Just s, s >= 0.3` (finite) | at-or-above floor | **Yes** (also clamps `maxPositionSize` to 0.25) |

---

## 4. Validation

```
$ cd haskell && stack build --ghc-options=-O0
Linking …/trader-hs                                            OK

$ stack test --ghc-options=-O0
trader> test (suite: trader-tests)
trader> Test suite trader-tests passed
Completed 2 action(s).

$ fourmolu --mode check \
    app/Trader/BotStartSemantics.hs \
    app/Main.hs \
    test/TestMain.hs
(exit 0, no diff)

$ hlint app/Trader/BotStartSemantics.hs app/Main.hs test/TestMain.hs
No hints

$ stack exec trader-hs -- \
    --data ../data/sample_prices.csv --price-column close \
    --epochs 1 --hidden-size 4 --json | head -c 600
{"backtest":{"agreementOk":[false,…,true,true,true,…
```

### 4.1 Cumulative invariants

| Test | Invariant |
|---|---|
| (2026-06-11) `botStartupBacktestVerdict` zero-trade NoVerdict | Zero-trade smoke is not a verdict. |
| (2026-06-11) `applyComboUpdatesWithStats` zero-trade no-prune | Store-layer defense-in-depth. |
| (2026-06-12) `botStartupBacktestVerdictWithMinTrades` decision table | Under-min-trades smoke is NoVerdict; at-threshold is Abort. |
| (2026-06-12) `botStartupGuardShouldPrune _ = False` | Uniform no-prune policy for the bot-start guard. |
| (2026-06-13) `venueMinEdgeFloor > venueRoundTripCostFloor` | Cost-floor inequality. |
| (2026-06-13) `capAdoptedMaxPositionSize` bounds | Legacy `maxPositionSize = 1.0` clamps to `0.25`. |
| **(today)** `testAdoptionMinTradeCountMatchesOptimizerProductionGate` | `adoptionMinTradeCount >= 20` (matches optimizer prod gate). |
| **(today)** `testComboTradeCountMeetsAdoptionFloorMonotonicity` | `Nothing → False`, `Just 0/19 → False`, `Just 20+ → True`, predicate monotone. |
| **(today)** `testComboTradeCountMeetsAdoptionFloorMatchesProductionRegressionEvidence` | Today's observed median `tradeCount = 4` is pinned to fail. |
| **(today)** `testAdoptionMinWalkForwardSharpeMatchesOptimizerDefault` | `adoptionMinWalkForwardSharpeMean == 0.3` (matches optimizer default). |
| **(today)** `testComboWalkForwardSharpeMeetsAdoptionFloorFailsClosed` | `Nothing/NaN/±Inf → False`. |
| **(today)** `testComboWalkForwardSharpeMeetsAdoptionFloorMonotonicity` | Boundary at `0.3` exactly, predicate monotone. |

### 4.2 Leaderboard aggregate computation (reproducibility)

```python
import json, statistics
with open('haskell/.tmp/optimizer/top-combos.json') as f:
    data = json.load(f)
combos = data['combos']
n = lambda x: x if isinstance(x, (int, float)) else None
trades = [n(c['metrics'].get('tradeCount')) for c in combos]
trades = [t for t in trades if t is not None]
me = [n(c['params'].get('minEdge')) for c in combos]
sharpe = [n(c['metrics'].get('sharpe')) for c in combos]
ann = [n(c['metrics'].get('annualizedReturn')) for c in combos]
mps = [n(c['params'].get('maxPositionSize')) for c in combos]
wf = sum(1 for c in combos if c['metrics'].get('walkForwardSummary'))
quar = sum(1 for c in combos if (c['metrics'].get('liveStats') or {}).get('quarantined'))
print('total:', len(combos),
      'median tradeCount:', statistics.median(trades),
      'median minEdge:', f"{statistics.median([m for m in me if m is not None]):.5f}",
      'median sharpe:', f"{statistics.median([s for s in sharpe if s is not None]):.3f}",
      'median ann:', f"{statistics.median([a for a in ann if a is not None]):.3f}",
      'median mps:', f"{statistics.median([m for m in mps if m is not None]):.3f}",
      'with WF:', wf, 'quarantined:', quar)
# Output (2026-06-14):
# total: 500 median tradeCount: 4 median minEdge: 0.00048 median sharpe: 8.522
# median ann: 8.355 median mps: 0.771 with WF: 0 quarantined: 0
```

---

## 5. Strategy Research (treating trading as engineering)

### 5.1 Why "raise minEdge" alone is insufficient

The 2026-06-13 post-fix backtest evaluation
(`reports/backtest-postfix-2026-06-14/README.md`) reaches the right
verdict: even with the cost-floor in place, the predictor stack does
not currently produce edge above the venue cost. The proposed next
steps in that report — higher predictor capacity on research nodes,
shorter timeframes, funding-on-open — are all good and stay on the
backlog.

What today's review adds is the engineering observation that **the
leaderboard's apparent quality is a sampling-distribution artifact**.
A leaderboard sorted by `sharpe` with median sample size n=4 is, with
high probability, a sample from the right tail of a noisy estimator
applied to a population with true Sharpe near zero. The right defence
is not to filter the tail (the tail is the leaderboard) but to require
*enough sample size to make the tail uninteresting*. Standard error of
the Sharpe estimator is approximately `(1 + ½ S²) / sqrt(n)` for
i.i.d. returns; at n=20 and a true Sharpe of 0.3 the estimator's
standard error is `sqrt(1.045) / sqrt(20) ≈ 0.23` — still loose, but
within an order of magnitude of the signal. At n=4 the same standard
error is `≈ 0.51`, which is ~`2x` the optimizer's default
`minWfSharpeMean` gate. A 4-trade backtest *cannot* falsify or confirm
the 0.3 hypothesis.

This is why today's H1 floor is set at the optimizer's documented
production gate (20). It is not the formally-correct value for the
Sharpe-estimator standard error to clear zero — that would require
hundreds of trades — but it is the floor below which the *gate itself*
becomes informational noise. Anything stricter belongs in the next
iteration of the walk-forward design (§5.2).

### 5.2 The right long-horizon design

The 2026-06-12 review already proposed SPRT- and Bayesian-style
verdicts for the bot-start guard. The same proposal applies one layer
up: the *optimizer's* WF Sharpe gate should be replaced with a
posterior-update on a global prior over the (predictor, timeframe,
symbol) cube. The posterior's strength after `k` walk-forward folds
of `n` trades each is `O(k × n)`, and the gate should require the
posterior credible interval's lower bound to clear the cost-floor by
the safety multiplier. This is much stricter than the current
`sharpeMean >= 0.3` gate and would, on today's leaderboard, reject
**all** 500 combos including the ones the cost-floor catches.

That is a multi-week design change and lives in the next-gen
architecture RFC (`artifacts/planning/trader-firm-nextgen-architecture-2026-05-20.md`).
Today's fix is the minimal additive step: mirror the optimizer's
*current* default at adoption time so the two gates stay equal under
future tightening.

### 5.3 Cross-instance consistency under the new gates

The 2026-06-12 review flagged the prune/resurrect race: combos pruned
locally were re-synced from S3 with their stale metrics. Today's
filters are *non-destructive* — they reject for *adoption*, but do
not delete the combo from the JSON store or DB row. The combo can
still re-appear from S3 and re-rank, but the adoption path will reject
it again on the next pass. This is the same idempotent-with-sync
contract that `botStartupGuardShouldPrune = False` enforces for the
bot-start backtest guard. No new destructive call site was introduced
today, so the H10/H11 erosion guards from 2026-06-12 continue to
apply.

---

## 6. Remaining work (not in scope today)

1. **Operator action (carried forward from 2026-06-12 §6.1):** the 401
   storm on `/binance/positions` is the same auth/IP issue. Rotate
   `BINANCE_API_KEY` / `BINANCE_API_SECRET` and re-add the egress IP
   to the Binance allow-list, or capture the exact 401 body so the
   classifier can be extended (`Trader.App.AutoStartBackoff.classifyError`).

2. **Adoption knob exposure (deferred):** today's H1/H2 thresholds are
   hardcoded constants matched to the optimizer's defaults. If a
   future change makes them legitimately tunable per-deployment, expose
   `TRADER_BOT_ADOPT_MIN_TRADE_COUNT` and
   `TRADER_BOT_ADOPT_MIN_WF_SHARPE_MEAN` env knobs in
   `.env.example`. As of today the right behavior is to keep them
   pinned to the optimizer side; tunability creates two surfaces for
   the same gate.

3. **Predictor improvement (carried forward from 2026-06-13 post-fix
   report):** the deeper diagnosis remains correct: the predictor
   stack does not produce edge above the venue cost on 1h–12h crypto
   perps. Today's fix is firm capital preservation, not edge
   recovery. Next steps:
   - Raise research-node predictor capacity (epochs ≥ 8, hidden ≥ 32);
   - Try short timeframes (5m, 15m on BTC/ETH/SOL) where the cost-floor
     is a smaller fraction of typical bar return;
   - Enable funding-on-open for surviving combos;
   - Investigate the 60% `ok=false` rate (mostly timeout, not crash).

4. **Backfill walk-forward summaries on the existing leaderboard:** the
   500 legacy combos cannot be adopted under today's H2 gate. They
   could be backfilled by running the walk-forward path on each combo
   in batch. That is its own background-job task and is not in scope
   for the daily review.

5. **Telemetry (still deferred from 2026-06-08):**
   * Prometheus counter
     `trader_top_combo_adoption_rejected_total{reason="min_edge_below_floor" | "trade_count_below_floor" | "wf_sharpe_below_floor" | "live_quarantined"}`.
   * The daily review's leaderboard aggregate (§4.2) should be a
     Grafana panel rather than a JSON-parse snippet.

---

## 7. Files touched

```
haskell/app/Trader/BotStartSemantics.hs   (adoption-min-trade-count + WF-Sharpe-mean predicates)
haskell/app/Main.hs                       (TopCombo helpers + adoption filter wires)
haskell/test/TestMain.hs                  (6 new invariants; updated import list and runner)
CHANGELOG.md                              (one new "Unreleased" entry)
ENGINEERING_REVIEW_2026-06-14.md          (this file)
```

`stack build` and `stack test` both pass; `fourmolu --mode check` is
clean on the three touched Haskell files; `hlint` reports no hints on
the three touched files. The running launchd binary is `3d32f58c`
(yesterday's cost-floor + position-cap + WF-Sharpe shipset); the next
operator launchd reload via `scripts/restart-local-stack.sh` will
pick up today's H1/H2 shipset.
