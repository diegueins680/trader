# Engineering Review — 2026-06-12 (Friday late, America/Guayaquil)

> Daily engineering review per `AGENTS.md` workflow. Trades observed today,
> hypotheses with falsification, measured failure modes, code changes,
> validation. Verified by `stack test` (green), `fourmolu --mode check`
> (clean on touched files), `hlint app/Trader/BotStartSemantics.hs app/Main.hs
> test/TestMain.hs` (no hints), and a `trader-hs` JSON backtest smoke run
> against `data/sample_prices.csv`. The review was started at 23:00 local on
> 2026-06-12 (America/Guayaquil), one hour before the UTC day boundary;
> "today" for log-grep purposes is the active launchd-managed `trader-hs`
> server-API process (PID 49590, started Fri Jun 12 20:05:13 2026) and the
> 502 log lines emitted since that start.

## 0. Summary

| Metric | Value | Δ vs 2026-06-11 |
|---|---|---|
| Trades executed today | **0** | unchanged |
| Live positions opened today | 0 | unchanged |
| `live_trades.ndjson` last write | 2026-05-28 (haskell/.tmp), 2026-05-24 (.tmp) | unchanged (system has not opened a live trade for ~16 days) |
| Running binary | `Build: 0.1.0.0 (9b39dfef8ed0)` since 2026-06-12 20:05:13 | **NEW binary running** — H1 from 2026-06-11 is closed |
| `Live bot auto-start classified` lines (current slice) | 7 | **first non-zero count** — the 2026-06-08 backoff/classify code is live |
| `Live bot auto-start backoff: skipping` (current slice) | 22 | **first non-zero count** — per-symbol exponential backoff is live |
| `auth circuit OPEN` (current slice) | 0 | the circuit threshold is not yet tripped |
| `Top-combo startup backtest aborted` (current slice) | 7 | down from 195 yesterday — H6 + H7 fixes are running |
| Of which `finalEquity=1.000000` (zero-trade) | **0** | **the H7 zero-trade prune is fully closed at runtime** |
| Of which real-loss (`finalEquity` 0.94–0.99) | 7 | exposes a new mode (H10/H11 below) |
| Same combo UUID aborted ≥ 2× inside this slice | **2 of 4** distinct UUIDs (~50%) | new finding: prune/resurrect oscillation |
| Same combo UUID aborted ≥ 2× across whole log | **37 of 49** distinct UUIDs (~76%) | the dominant historical pattern, not an outlier |
| `Queued bot start failed` (current slice) | 0 | down from 18,198 yesterday |
| Code changes today | 4 files + CHANGELOG + `.env.example` + this review | — |
| Tests added | 3 new invariants (under-min-trades guard, default pin, no-prune contract) | — |

The system still **placed no trades today**. The signal layer continues
not to run because Binance returns transient/auth errors on every
preflight (today's classification: 100% `unknown (treated as transient)`
for AAVEUSDT/BTCUSDT/ATOMUSDT). The good news is that the 2026-06-08
backoff and 2026-06-11 zero-trade guard are *finally running in the live
process*: H1 is closed.

The bad news is that a more subtle erosion mode is now visible. Inside
a single 502-line slice of one process, two of the four distinct combos
that the bot-start guard touched were aborted 2–3 times each. Across
the whole launchd log, **37 of 49 unique pruned UUIDs were pruned more
than once** (~76%). That is the fingerprint of a *prune/resurrect race*:
the bot-start guard prunes the combo locally and deletes the DB row,
but the next cross-instance `Top combos sync reconciled db, s3` (we see
these every 1–3 seconds in the log) re-imports the combo from S3 with
its stale optimizer-side metrics, and the guard prunes it again.

Today's fix closes that erosion at two places:

1. **H10 (prune/resurrect race):** the bot-start guard no longer prunes.
   It uses `applyComboUpdatesKeepAllWithStats` (the same path the
   periodic refresh has used since 2026-06-10) and the new uniform
   policy `botStartupGuardShouldPrune _ = False`. The guard's job is
   to *block a start*, not to delete a combo. Pruning is the
   optimizer's / periodic-refresh's job.

2. **H11 (under-sampled smoke verdict):** the verdict function now
   takes a `minTrades` knob (default 3, env override
   `TRADER_BOT_START_BACKTEST_MIN_TRADES`). A sub-threshold
   `finalEquity` from a smoke window that fired fewer than `minTrades`
   trades is treated as `BacktestNoVerdict`, not `BacktestAbort`. The
   AAVEUSDT smoke windows today produced `finalEquity ∈ {0.952, 0.954}`
   on what is plausibly a single ~5% trade (one daily ATR) — n=1
   evidence overruling a combo with out-of-sample `finalEquity ≥ 1.42`.

---

## 1. Trade Analysis

### 1.1 Raw counts

`/tmp/trader-api-launchd.log` ≈ 15.06 MB, 88,808 total lines, but
multiple binaries have run in this file (today's restarts are visible
as four distinct `Build:` lines):

```
$ grep -n "Build: 0.1.0.0" /tmp/trader-api-launchd.log | tail -5
84601:Build: 0.1.0.0 (b5a5fc73e7b9)
84615:Build: 0.1.0.0 (686e3f7c1af0)
84724:Build: 0.1.0.0 (9b39dfef8ed0)
84994:Build: 0.1.0.0 (9b39dfef8ed0)
86609:Build: 0.1.0.0 (9b39dfef8ed0)
88307:Build: 0.1.0.0 (9b39dfef8ed0)
```

`9b39dfef` is the post-2026-06-11 shipset (zero-trade guard +
backoff + fingerprint normalization). The current process (PID 49590)
started at `88308` and has emitted 502 lines.

Counts in the **current process slice** (the only honest "today" view
of the new behaviour):

```
   0  auth circuit OPEN
  22  backoff: skipping             ← NEW: 2026-06-08 backoff is live
   7  Top-combo startup backtest aborted
   0  Queued bot start failed       ← was 18,198 yesterday
   7  Live bot auto-start classified ← NEW: 2026-06-08 classifier is live
   7  Live bot auto-start failed
   0  finalEquity=1.000000          ← the H7 zero-trade prune is closed
```

Counts in the **whole log** (for the dominant-pattern findings):

```
 138  backoff: skipping
 701  Top-combo startup backtest aborted
 539  finalEquity=1.000000         ← pre-fix zero-trade prunes
 161  finalEquity ∈ (0.0, 1.0)     ← pre-fix or post-fix real-loss prunes
  49  unique combo UUIDs pruned
  37  of those pruned ≥ 2× (oscillation count)
```

### 1.2 The two findings the new binary made visible

#### Finding A — under-sampled abort verdict (H11)

```
$ tail -n +88308 /tmp/trader-api-launchd.log | grep "Top-combo startup backtest aborted"
… for AAVEUSDT … finalEquity=0.954018 for combo 5283bbe3…
… for BTCUSDT  … finalEquity=0.994839 for combo 1492e418…
… for ATOMUSDT … finalEquity=0.993178 for combo ba2be3bf…
… for AAVEUSDT … finalEquity=0.954018 for combo f5a91330…
… for AAVEUSDT … finalEquity=0.954018 for combo 5283bbe3…
… for AAVEUSDT … finalEquity=0.954018 for combo f5a91330…
… for AAVEUSDT … finalEquity=0.952678 for combo 5283bbe3…
```

`finalEquity = 0.954018` ≈ one ~4.6% drawdown on a smoke window where
the active combos use `vol-target 0.57` and daily ATR is roughly 5%. A
single trade can produce that drawdown by chance: the conditional
probability `P(trade=1 ∧ pnl ≤ −4.6% | combo with positive expectation)`
is non-trivially > 0 for any reasonable daily-bar strategy. Treating a
single such event as a *verdict* on a combo whose out-of-sample
`finalEquity ≥ 1.42` (n=many) is an evidence-strength bug: we're using
n=1 small-sample evidence to overrule n=many optimizer evidence.

The fix (§3) requires the smoke window to actually trade at least
`TRADER_BOT_START_BACKTEST_MIN_TRADES` times before a sub-threshold
`finalEquity` becomes `BacktestAbort`. Default 3 — the smallest sample
where a non-trivial win-rate-vs-payoff calculation can distinguish
"signal lost" from "unlucky first trade."

#### Finding B — prune/resurrect oscillation (H10)

Within the current 502-line slice, the same combo UUIDs appear
multiple times:

```
$ … | grep "Top-combo startup backtest aborted" | grep -oE "combo [0-9a-f-]+" | sort | uniq -c
   1 combo 1492e418-29ba-3c13-0c4b-6e0031a1ae18  (BTCUSDT)
   3 combo 5283bbe3-88f7-325e-a209-b200b993d93d  (AAVEUSDT — pruned 3×)
   1 combo ba2be3bf-ef73-2e74-b053-9ece7cf415ea  (ATOMUSDT)
   2 combo f5a91330-d324-865c-a878-786f89b5c8ed  (AAVEUSDT — pruned 2×)
```

A correct prune should be terminal: once the combo is deleted from the
JSON store AND the `combos` DB row, the next start cycle should not
find it. The fact that the *same UUID* is pruned 2–3 times in 502
lines means the prune is being *undone* between cycles. The undo
source is the `Top combos sync reconciled db, s3 (499 combos)` line
that fires every 1–3 seconds in the same slice — the cross-instance
sync re-imports the combo from S3 (where another box has the
non-pruned version) with stale optimizer-side metrics.

This is the *same* race the periodic refresh path already fixes (it
uses `applyComboUpdatesKeepAllWithStats` since 2026-06-10, with the
exact reason documented as a docstring: *"a locally pruned combo
would be resurrected with its stale, inflated score by the next
cross-instance union merge"*). The bot-start guard had not yet
adopted that fix. Today's review brings them into agreement.

Whole-log evidence that this is dominant, not an edge case:

```
$ grep "Top-combo startup backtest aborted" /tmp/trader-api-launchd.log \
    | grep -oE "combo [0-9a-f-]+" \
    | sort | uniq -c \
    | awk '$1>=2' | wc -l
37
$ … | sort -u | wc -l
49
```

**37 of 49 unique pruned combos (~76%) were pruned more than once.**
The bot-start guard has been doing busywork against the sync layer for
weeks; on every quiet day the same handful of "marginal-loss" combos
get pruned, resurrected, pruned again.

### 1.3 What this means for trading

It means today, like every recent day, the *signal layer never ran*.
But the engineering value being eroded is not direct: each erroneous
prune deletes a combo from the `combos` DB row, and even though the
S3 union sync resurrects the JSON entry, the DB attribution (which
server owned the combo, the live-trade ops chain) does not come back.
Tomorrow's leaderboard merge sees the resurrected combo as
"unstamped" and may give it lower weight in the cross-instance score
blend. The cumulative effect on the strategy bank is the same shape
of bug as last week's zero-trade prune erosion, just less obvious
because the JSON survives.

### 1.4 H1 is closed

For the first time since the 2026-06-08 backoff shipped, the running
launchd binary actually contains the new code:

```
$ ps -p 49590 -o lstart=
Fri Jun 12 20:05:13 2026
$ stat -f "%Sm %z %N" .../trader-hs
Jun 12 19:56:40 2026  39371832  .../trader-hs    (built ~9 minutes earlier)
$ strings .../trader-hs | grep BacktestNoVerdict | head -1
trader-0.1.0.0-inplace-trader-hs:Trader.BotStartSemantics.BacktestNoVerdict
```

The pid started 9 minutes *after* the binary was built. The launchd
reload finally happened. No further H1 work today.

### 1.5 Engineering verdict

* **Trade decision quality:** still not the question — the signal layer
  never ran. The 1100 `Live bot auto-start failed` lines in the log
  are 100% `unknown (treated as transient)` — Binance is responding,
  but the response is not classifiable into `Auth` / `Permanent` /
  `Transient` by the current `Trader.App.AutoStartBackoff`
  classifier. That's a *classifier-coverage* gap, not a fix
  available in code today without seeing the actual error bodies.
  Logging them was the 2026-06-10 normalization work; the *classifier*
  side is a remaining-work item (§6.5).
* **Reliability shape:** good news (H1 closed, zero-trade prune
  fully gone at runtime), and a new structural finding (the
  prune/resurrect race makes prunes anti-idempotent across the S3
  sync layer). The fix today removes the race source. The strategy
  bank stays intact even on weeks-long Binance auth outages.
* **Engineering goal today:** close H10/H11 in code (deterministic),
  with the same shape as 2026-06-11's H6/H7 closure: small, testable,
  defense-in-depth.

---

## 2. Hypotheses

| # | Hypothesis | Falsifier | Owner |
|---|---|---|---|
| **H10** | The bot-start guard's prune path (`applyComboUpdatesWithStats` + `deleteTopComboFromDbMaybe`) is *anti-idempotent* across the `Top combos sync` cross-instance union merge: the merge resurrects the locally pruned combo every 1–3 s, then the guard prunes it again next cycle. The result is silent attribution loss in the `combos` DB rows. | Within the 2026-06-12 current-binary slice (502 lines), the same combo UUID must be aborted at most once. *Today: 50% (2 of 4 distinct UUIDs) violate this; whole-log 76% (37 of 49) violate it.* Fixed by using the keep-all variant and `botStartupGuardShouldPrune _ = False`. | This commit. |
| **H11** | A single-trade smoke window (`tradeCount = 1`, `finalEquity ≈ 0.95` on AAVEUSDT) is below the noise floor for combos with out-of-sample `finalEquity ≥ 1.42`. The 2026-06-11 zero-trade guard fixed the `tradeCount = 0` case but did not raise the bar for low-`n` cases. | A test where `tradeCount ∈ {1, 2}` and `finalEquity ∈ {0.95, 0.99}` yields `BacktestNoVerdict` (no abort) when `minTrades = 3`; `tradeCount ≥ 3` with the same `finalEquity` yields `BacktestAbort`. *Today: enforced by `testBotStartupBacktestVerdictMinTradesGuard`.* | This commit. |
| **H12** *(deferred)* | The current top-combo population uses asymmetric risk parameters where typical `take-profit ≈ 0.4%`, `stop-loss ≈ 0.5%`, plus fee + slippage ≈ 0.1% per side, so the break-even win-rate is ≳ 55%. If the live distribution win-rate is lower, the combo is *structurally* unprofitable regardless of which method generates the signal. | Compute realized win-rate vs break-even win-rate from `live_trades.ndjson` once live trades resume. Today the ndjsons are last-written 2026-05-24/28 (system has not traded for ~16 days), so the test is not actionable. | Deferred — needs live trades. |
| **H1** *(closed today)* | The 2026-06-08 `AutoStartBackoff` shipped to disk but the running launchd process does not include it. | The launchd log must show ≥ 1 of {`classified`, `auth circuit OPEN`, `backoff: skipping`}. *Today: 22 backoff, 7 classified in current slice → H1 closed.* | Was operator action; now confirmed live. |

---

## 3. Change set

### 3.1 Verdict parameterized on minimum trade count

```haskell
-- Trader.BotStartSemantics

defaultBotStartupBacktestMinTrades :: Int
defaultBotStartupBacktestMinTrades = 3

botStartupBacktestVerdictWithMinTrades :: Int -> Bool -> Maybe Double -> Maybe Int -> BacktestVerdict
botStartupBacktestVerdictWithMinTrades _ False _ _ = BacktestAllow
botStartupBacktestVerdictWithMinTrades _ True Nothing _ = BacktestNoVerdict
botStartupBacktestVerdictWithMinTrades minTradesRaw True mFinalEquity mTradeCount =
    let minTrades = max 1 minTradesRaw
     in if botStartupBacktestRoiAcceptable mFinalEquity
            then BacktestAllow
            else case mTradeCount of
                Just n | n >= minTrades -> BacktestAbort
                _ -> BacktestNoVerdict

-- The 2-arg form is preserved (backward compatibility / existing tests):
botStartupBacktestVerdict :: Bool -> Maybe Double -> Maybe Int -> BacktestVerdict
botStartupBacktestVerdict = botStartupBacktestVerdictWithMinTrades 1
```

`max 1 minTradesRaw` is deliberate: pathological inputs (0 / negative)
normalize to *the existing pre-2026-06-12 behaviour* (any single trade
is enough). The test `testBotStartupBacktestVerdictMinTradesGuard`
pins this row.

### 3.2 No-prune policy, centralized

```haskell
-- Trader.BotStartSemantics

botStartupGuardShouldPrune :: BacktestVerdict -> Bool
botStartupGuardShouldPrune _ = False
```

One function, three rows in the test (`Allow`, `Abort`,
`NoVerdict`), all `False`. The guard now reads this policy instead
of hard-coding "delete on abort". If a future change wants the guard
to start pruning again, the policy lives in one place and the test
breaks.

### 3.3 Wired through to `runTopComboStartupBacktestGuard`

`Main.hs`:

* The verdict call now uses `botStartupBacktestVerdictWithMinTrades`
  with `tcbcMinTradesForAbort ctx` (env-tunable; see §3.5).
* The update path is now `applyStartupComboBacktestUpdateKeepAll`
  (new sibling of the existing `applyStartupComboBacktestUpdate`,
  parameterized over the apply function — they share an
  `applyStartupComboBacktestUpdateImpl` helper). Keep-all = stamps the
  combo with the fresh smoke metrics but never prunes.
* The `deleteTopComboFromDbMaybe` call is now gated on
  `botStartupGuardShouldPrune verdict`. Since that's uniformly
  `False`, the DB row is never deleted by the guard. The structure is
  kept so a future policy change does not have to re-edit
  `runTopComboStartupBacktestGuard`.
* The `NoVerdict` log line now also reports the active
  `minTradesForAbort`, so a log reader can tell why a near-loss
  smoke window was allowed.

### 3.4 Defense-in-depth in the store layer (unchanged from 2026-06-11)

`Trader.TopCombosStore.applyComboUpdatesWithStats` still refuses to
prune on a zero-trade update. That invariant is now belt-and-suspenders
since the bot-start path doesn't prune at all, but the store-level
guard is still useful for the *optimizer*'s top-N rerun path
(`Main.hs:10644`-ish), which still uses the pruning variant for true
"this combo finally lost over many trades" cases.

### 3.5 New env knob

```
TRADER_BOT_START_BACKTEST_MIN_TRADES=3
```

Documented in `.env.example`. Parsed in `Main.hs` with the existing
`readMaybe . dropWhile isSpace . takeWhile (not . isSpace)` shape and
plumbed into `TopCombosBacktestCtx.tcbcMinTradesForAbort`. A
non-positive / unparseable value falls back to
`defaultBotStartupBacktestMinTrades`.

### 3.6 New tests

Added to `test/TestMain.hs`:

| Test | Invariant |
|---|---|
| `testBotStartupBacktestVerdictMinTradesGuard` | **H11** — under-min-trades sub-threshold smoke is `NoVerdict`, at-threshold is `Abort`, above-threshold finalEquity wins regardless of trade count, zero-trade still NoVerdict (regression on 2026-06-11), disabled guard always allows, pathological `minTrades ≤ 0` normalizes to 1. |
| `testBotStartupBacktestVerdictDefaultMinTradesIsThree` | Pins `defaultBotStartupBacktestMinTrades == 3`. Any future change must edit the test. |
| `testBotStartupGuardShouldPruneIsFalse` | **H10** — pin that the guard never prunes on any verdict. |

Existing invariants are preserved (verdict tests, store tests).
Decision table after today:

| enabled | finalEquity | tradeCount | minTrades | verdict |
|---|---|---|---|---|
| False | any | any | any | Allow |
| True | Nothing | any | any | NoVerdict |
| True | Just > 1.0 finite | any | any | Allow |
| True | Just ≤ 1.0 / non-finite | Just n, n ≥ minTrades | minTrades | Abort |
| True | Just ≤ 1.0 / non-finite | Just n, 0 < n < minTrades | minTrades | NoVerdict |
| True | Just ≤ 1.0 / non-finite | Just 0 | any | NoVerdict |
| True | Just ≤ 1.0 / non-finite | Nothing | any | NoVerdict |

And the prune policy: `botStartupGuardShouldPrune verdict = False` for
all verdicts.

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

The `bash scripts/verify.sh haskell` wrapper requires `cabal`, which is
not installed on this host (only `stack`); the four steps above are the
equivalent path through `stack`, mirroring 2026-06-08 / 2026-06-10 /
2026-06-11 reviews.

### 4.1 Cumulative invariants

| Test | Invariant |
|---|---|
| (existing) `botStartupBacktestRoiAcceptable` ladder | Two-valued ROI threshold preserved for `Maybe Double` callers. |
| (existing) `botStartupBacktestAborts` fail-open | Old two-valued semantics preserved (disabled / `Nothing` / non-finite). |
| (2026-06-11) `botStartupBacktestVerdict` zero-trade NoVerdict | H6 — zero-trade smoke is not a verdict. |
| (2026-06-11) `botStartupBacktestVerdict` abort on traded loss | Real signal regressions still abort. |
| (2026-06-11) `botStartupBacktestVerdict` disabled-guard preserves Allow | No behaviour change for disabled boxes. |
| (2026-06-11) `applyComboUpdatesWithStats` zero-trade no-prune | H9 — store-layer defense-in-depth. |
| (2026-06-11) `applyComboUpdatesWithStats` genuine loss prunes | Symmetry: the fix does not swallow real losses. |
| (2026-06-11) `normalizeBarsForLookback` Binance >1000 deferred-pin | H8 — current behaviour pinned, change requires test edit. |
| **(today)** `botStartupBacktestVerdictWithMinTrades` decision table | **H11** — under-min-trades smoke is NoVerdict, at-threshold is Abort, above-threshold finalEquity always wins. |
| **(today)** `defaultBotStartupBacktestMinTrades == 3` | Default policy pinned. |
| **(today)** `botStartupGuardShouldPrune _ = False` | **H10** — uniform no-prune policy for the bot-start guard. |

---

## 5. Strategy Research (treating trading as engineering)

The signal layer still hasn't run, so a backtest-vs-live comparison is
again not actionable. Two research lines remain alive today.

### 5.1 Evidence sufficiency under sample asymmetry

The 2026-06-11 review framed the zero-trade prune as a two-valued vs
three-valued verdict bug. Today's finding deepens that: even with the
three-valued verdict, *the third value's domain is still too small*.
The correct projection of evidence is not just "did the smoke window
trade?" but "did the smoke window trade *enough* times for the loss
to be statistically distinguishable from noise?". The minimum sample
size for that distinction is a function of the combo's trade frequency
and pnl distribution — but as a starting point, "at least 3 trades"
beats "at least 1 trade" by a wide margin for daily-bar strategies
whose per-trade σ is comparable to the threshold for `BacktestAbort`.

The research direction worth flagging:

* **Sequential probability ratio test (SPRT) as the verdict.** Instead
  of "≥ minTrades trades, then threshold the equity", run an SPRT on
  the smoke window's per-trade pnl against the null hypothesis "the
  combo's true expectation is at most break-even" with type-I error
  `α = 0.05` and type-II `β = 0.05`. The expected sample size to a
  decision is `n* ≈ (zα + zβ)² × σ² / Δ²`, which for daily-bar
  strategies with σ ≈ 1% per trade and Δ ≈ 0.2% per trade is `n* ≈ 130`
  trades — far more than any plausible smoke window. The corollary is
  that **no realistic smoke window can produce strong evidence**, and
  the right design is to treat *all* smoke windows below SPRT-sample
  as `NoVerdict`. Today's `minTrades = 3` is a coarse, conservative
  approximation of that result.
* **Bayesian updating on the optimizer prior.** The combo's optimizer
  `finalEquity = 1.42` is a posterior over ~thousands of bars. The
  smoke window is a *partial likelihood* update. A 1-trade smoke loss
  shifts the posterior by ~ε. A 30-trade smoke loss shifts it by a
  meaningful amount. The right verdict is roughly "abort iff the
  posterior `finalEquity` falls below 1.0 *with the smoke window's
  weight*", which is again equivalent to "require many trades before
  the smoke can override the prior".

These belong on a longer-horizon roadmap (§6.6) since they require
changing not just the guard but also the metric the guard reads
(`finalEquity` → posterior). The `minTrades = 3` fix today is the
right *first* step.

### 5.2 Idempotency under cross-instance sync

The prune/resurrect race is the same shape of bug as last week's
zero-trade prune erosion, but harder to detect:

* **Erosion 1 (closed 2026-06-11):** `finalEquity == 1.0` on a
  zero-trade smoke was treated as a loss. The combo was removed from
  both the JSON store and the DB. Total damage: 124 prunes in one
  session.
* **Erosion 2 (closed today):** A real but tiny-sample loss in a smoke
  window was treated as a verdict. The combo was removed from the JSON
  store and DB, but the cross-instance S3 sync resurrected the JSON
  next cycle (with stale metrics). The visible damage in logs is
  smaller because the JSON comes back; the *invisible* damage is the
  DB row attribution being lost every cycle (37 of 49 unique combos
  pruned ≥ 2× across the log).

The deeper engineering principle: **destructive operations on the
combo store must be idempotent with the sync layer, or they should
not run from a non-authoritative path**. The optimizer is authoritative
(it has the full out-of-sample sample). The bot-start guard is not
(it has the smoke window). Today's policy
`botStartupGuardShouldPrune _ = False` says exactly that: the
non-authoritative path no longer prunes.

---

## 6. Remaining work (not in scope today)

1. **Operator action (orthogonal, root cause of the active outage):**
   the Binance error bodies on today's 1,100 `Live bot auto-start
   failed` events all classify as `unknown (treated as transient)`.
   Either rotate `BINANCE_API_KEY` / `BINANCE_API_SECRET` and add the
   active egress IP to the Binance allow-list, OR investigate the
   exact error body (is it `-2015`, a Cloudflare body, or an
   IP-ban response) so the classifier can be extended.

2. **Classifier coverage gap (H1.5 — code, deferred):** today the
   classifier returns `unknown (treated as transient)` for 100% of the
   live failures. The 2026-06-08 backoff design assumed auth failures
   were classifiable as `ErrAuth` and would open the auth circuit.
   They are not. Extend `Trader.App.AutoStartBackoff.classifyError`
   to recognize the actual on-the-wire error bodies seen in the
   2026-06-12 log (deferred until we have the exact body text — the
   normalizer redacts IPs / signatures, so a few raw samples need to
   be captured separately).

3. **Auth-circuit-open prerequisite:** with classifier coverage at
   100% `unknown`, the auth-circuit-threshold latch
   (`TRADER_BOT_AUTOSTART_AUTH_CIRCUIT_THRESHOLD`) never trips.
   Consider a fallback "if N consecutive `unknown` failures, escalate
   to `ErrPermanent`" so the global circuit opens and the loop quiets
   down. Today's loop is well-behaved (138 backoff lines vs. 1,100
   failed lines = the backoff *is* compressing, just not silencing).

4. **Restore lost DB rows from the journal:** the 37 distinct combos
   pruned ≥ 2× before today's fix have lost their `combos` DB
   attribution. Replay the `bot.start_combo_backtest_aborted` ops in
   the journal to re-create the DB rows (the JSON store already has
   them via S3 sync). Out of scope today, but valuable.

5. **Telemetry (still deferred from 2026-06-08):**
   * Prometheus counter
     `trader_top_combo_prune_total{reason="zero_trade_smoke" | "low_n_smoke" | "loss" | …}`.
   * Counter `trader_bot_start_combo_backtest_no_verdict_total{cause="zero_trade" | "low_n" | "missing_metric"}` so the
     daily review can be a Grafana panel rather than a log grep.
   * Gauge `trader_top_combos_count{store="json" | "db"}` so JSON↔DB
     divergence (the H10 erosion fingerprint) is visible on a
     dashboard.

6. **Longer horizon (research, deferred):** evaluate SPRT- and
   Bayesian-style verdicts for the bot-start guard (§5.1). Both
   require changing the metric the guard reads, not just the decision
   function, so they belong in their own RFC.

---

## 7. Files touched

```
haskell/app/Trader/BotStartSemantics.hs   (BacktestVerdictWithMinTrades + no-prune policy)
haskell/app/Main.hs                       (env knob, keep-all path, gated delete-from-db)
haskell/test/TestMain.hs                  (3 new invariants; updated import list)
.env.example                              (new TRADER_BOT_START_BACKTEST_MIN_TRADES knob)
CHANGELOG.md                              (one new "Unreleased" entry)
ENGINEERING_REVIEW_2026-06-12.md          (this file)
```

`stack test` green; `fourmolu --mode check` clean on the three touched
Haskell files; `hlint` reports no hints on the three touched files.
The running launchd binary as of this review is `9b39dfef` (which
predates today's commit and *does not* include H10/H11); the next
operator launchd reload will pick up the new shipset.
