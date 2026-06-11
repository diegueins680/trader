# Engineering Review — 2026-06-11 (Wednesday late, UTC day boundary)

> Daily engineering review per `AGENTS.md` workflow. Trades observed today,
> hypotheses with falsification, measured failure modes, code changes,
> validation. Verified by `stack test` (green), `fourmolu --mode check`
> (no diff on the touched files), and `hlint app test bench` (no new
> hints; only 7 pre-existing hints in `test/TestMain.hs` lines 326-440 and
> `app/Trader/LSTM.hs` line 326 remain — none in code I touched). The
> review was started at 23:00 local on 2026-06-10 (America/Guayaquil)
> just after the UTC day boundary; "today" for log-grep purposes is the
> single launchd session running since 2026-06-03 00:10 UTC, scoped to
> the most recent ~24h slice via the line counts below.

## 0. Summary

| Metric | Value | Δ vs 2026-06-10 |
|---|---|---|
| Trades executed today | **0** | unchanged |
| Live positions opened today | 0 | unchanged |
| `Live bot auto-start failed` log lines | **497**+ (still growing) | up from 476 — fix-on-disk-only state unchanged (H1 below) |
| `Live bot auto-start auth circuit OPEN` lines | **0** | unchanged — running binary still predates 2026-06-08 |
| `Live bot auto-start backoff: skipping` lines | **0** | unchanged |
| `Live bot auto-start classified` lines | **0** | unchanged |
| `Top-combo startup backtest aborted ...: refreshed ROI is not positive` lines | **195**+ | **new failure mode, more aggressive than yesterday's pattern** |
| Of which `finalEquity=1.000000` exactly (zero-trade smoke) | **193** | **99.0 %** of all "ROI not positive" verdicts |
| Of which a genuine loss (`< 1.0` and finite) | 2 | 1.0 % |
| Combos pruned from top-combos JSON + `combos` DB this session | **~195** | this is the silent bug |
| `Queued bot start failed` lines | **18,198**+ | knock-on of the same prune-on-zero-trade loop |
| `Need at least N price rows for lookback=...` | 2 (FILUSDT 3m, XRPUSDT 3m) | bounded |
| `hiddenSize too high for this API instance (max 32)` | 16 | bounded |
| Code changes today | 4 files + CHANGELOG + this review | — |
| Tests added | 6 new invariants (3 verdict, 2 store, 1 lookback pin) | — |
| Running launchd PID start | Wed Jun 3 00:10:24 2026 | unchanged — still pre-2026-06-08 binary |
| New binary on disk | rebuilt 2026-06-10 23:31 | from `stack build` in this run |

The system **placed no trades again today**. The signal layer still
never runs because the Binance preflight (`futures/positionRisk`)
returns Binance `-2015` for every cycle. That is the same operator
action item from 2026-06-08 / 2026-06-10. *Underneath* that, a new and
much more damaging failure mode is visible in the logs: every time a
queued start runs the startup top-combo smoke-backtest, the backtest
fires zero trades, lands `finalEquity == 1.0` exactly, and the code
treats that as a sub-threshold ROI verdict and **prunes the combo from
disk and from the DB**. 193 of today's 195 prune lines are zero-trade
smoke windows masquerading as losses; only 2 are real. The strategy
bank is being eroded one quiet day at a time.

Today's fix removes that erosion at two levels:

1. The bot-start guard reads `tradeCount` from the smoke-backtest metrics
   and returns a three-valued verdict (`Allow` / `Abort` /
   `NoVerdict`). Zero-trade smoke windows produce `NoVerdict`: the start
   is allowed, smoke metrics are NOT persisted onto the combo, and the
   prune-update is NOT applied.
2. The top-combos JSON store's `applyComboUpdatesWithStats` defends the
   same invariant: even if an upstream caller still tries to apply a
   zero-trade update with `finalEquity ≤ 1.0`, the store will NOT prune
   the target combo.

H1 from yesterday (the pre-2026-06-08 binary is still running under
launchd) remains an operator action item and is unchanged.

---

## 1. Trade Analysis

### 1.1 Raw counts

`/tmp/trader-api-launchd.log` ≈ 5.85 MB, ~44,828 lines, running
`trader-hs` PID 41339 started **Wed Jun 3 00:10:24 2026**.

```
$ wc -l /tmp/trader-api-launchd.log
44828

$ grep -c "Live bot auto-start failed"    /tmp/trader-api-launchd.log
497
$ grep -c "Queued bot start failed"        /tmp/trader-api-launchd.log
18198
$ grep -c "Top-combo startup backtest aborted" /tmp/trader-api-launchd.log
195
$ grep -c "; combo removed from top combos" /tmp/trader-api-launchd.log
195
$ grep -c "Live bot auto-start classified" /tmp/trader-api-launchd.log
0
$ grep -c "auth circuit OPEN"              /tmp/trader-api-launchd.log
0
$ grep -c "backoff: skipping"              /tmp/trader-api-launchd.log
0
```

Per-symbol `Live bot auto-start failed` distribution (yesterday's IP
rotation again):

```
   4 AAVEUSDT     3 ATOMUSDT     3 ETCUSDT      1 UNIUSDT
  97 ADAUSDT     96 DOGEUSDT    94 FILUSDT     95 XRPUSDT
   3 ETHUSDT    101 SOLUSDT      3 SUIUSDT
```

Per-symbol *prune* distribution (today's new finding):

```
  13 AAVEUSDT    16 ADAUSDT     10 ARBUSDT     10 ATOMUSDT
  14 AVAXUSDT     7 BCHUSDT     13 BNBUSDT     13 BTCUSDT
  11 DOGEUSDT     9 DOTUSDT     12 ETHUSDT     11 LINKUSDT
   9 LTCUSDT      8 NEARUSDT     7 OPUSDT      18 SOLUSDT
  14 UNIUSDT
```

Note SOLUSDT lost **18 distinct combos** in one session.

### 1.2 The zero-trade prune finding (new)

The decisive diagnostic:

```
$ grep "refreshed ROI is not positive" /tmp/trader-api-launchd.log \
    | grep -oE "finalEquity=[0-9.]+" \
    | sort | uniq -c
   1 finalEquity=0.947245
   1 finalEquity=0.969753
 193 finalEquity=1.000000
```

`finalEquity == 1.000000` exactly is **not a loss**. It is the
deterministic output of running the backtester through the smoke window
and emitting zero trades (the simulator starts at equity 1.0 and never
moves). The backtester sets `tradeCount = 0` in the metrics in that
case, and that field is the falsifier: a real loss-without-trades is
impossible.

What `runTopComboStartupBacktestGuard` (haskell/app/Main.hs:7287) was
doing pre-fix:

```haskell
let mFinalEq = comboMetricDouble "finalEquity" metricsVal
    update   = ComboBacktestUpdate { ..., cbuFinalEquity = mFinalEq, ... }
updateResult <- applyStartupComboBacktestUpdate ctx comboKey update
let acceptable = not (botStartupBacktestAborts (tcbcEnabled ctx) mFinalEq)
...
when pruned (deleteTopComboFromDbMaybe (tcbcOps ctx) comboUuid)
if acceptable then pure (Right ()) else pure (Left msg)
```

And `botStartupBacktestAborts True (Just 1.0) == True`. So:

1. Every queued start ran the smoke backtest.
2. The smoke window is short (a single signal-gated slice) — on a quiet
   day the dominant outcome is "no trade fired".
3. The smoke metrics were persisted onto the combo, `finalEquity` was
   updated from the optimizer's out-of-sample reading (≥ 1.42 for the
   active leaders) to **1.0 exactly**.
4. The store layer
   (`Trader.TopCombosStore.applyComboUpdatesWithStats`) saw
   `finalEquity ≤ 1.0` and removed the combo from the in-memory combos
   array.
5. Main.hs then deleted the combo from the `combos` Postgres table.
6. The bot start was aborted.
7. Next cycle the symbol resolved a *different* combo, and the loop
   continued — eroding the leaderboard one prune at a time.

This is not the dedup/auth circuit class of bug. This is a *correctness*
bug in the engineering of the guard: it treats `tradeCount == 0` and
`tradeCount > 0 && finalEquity == 1.0` as the same evidence, when they
are different events with different epistemic content.

### 1.3 The IP-rotation finding (recap, unchanged)

Yesterday's review documented that the running launchd binary predates
the 2026-06-08 backoff/circuit/normalization shipset, and that the
2026-06-10 fingerprint fix is therefore on disk but not running. That
remains true today:

```
$ ls -la haskell/.stack-work/install/.../bin/trader-hs
-rwx------ ...  Jun 10 23:31  trader-hs   (built today)
$ ps -p 41339 -o lstart=
Wed Jun  3 00:10:24 2026
```

The running process predates the new binary by 7 days, 23h. The
`Live bot auto-start failed` distribution today has the same per-symbol
shape (94–101 events for the auth-failing five, 1–4 for everything
else) and the same multi-IP signature (`148.227.107.{16,97,145,172,253}`
plus `157.100.191.150`), confirming the pre-2026-06-08 dedup path is
still active. No further code work today closes H1; that is an
operator action item (§6.1).

### 1.4 Engineering verdict

* **Trade decision quality:** still not the question — the signal layer
  never ran. But the *strategy infrastructure quality* has now been
  shown to degrade silently with every quiet day. The right primary
  engineering objective today is to stop that erosion, which is
  deterministic and testable, even without live trading.
* **Reliability shape:** same regression as 2026-06-10 (binary not
  reloaded), plus a new, previously-unfound combo-prune erosion bug
  that quietly removed ~195 ranked combos in a single session. The
  prune count exceeded the auth-failure count.
* **Engineering goal today:** close H6/H7 in code (deterministic),
  pin H8 with a regression test (deferred behaviour), call out H1
  again as the operator action item.

---

## 2. Hypotheses

| # | Hypothesis | Falsifier | Owner |
|---|---|---|---|
| **H6** | `runTopComboStartupBacktestGuard` aborts on `finalEquity ≤ 1.0` without checking `tradeCount`. When the smoke backtest fires zero trades, `finalEquity == 1.000000` exactly — the deterministic fingerprint of "no trade fired" — and the guard treats it identically to a losing backtest. | A test where `tradeCount == 0` and `finalEquity == 1.0` must produce `BacktestNoVerdict` (do not abort, do not prune). *Today: enforced by 3 new verdict tests + 1 store test.* | This commit. |
| **H7** | Most of today's `refreshed ROI is not positive` lines were zero-trade smoke windows, not real losses, and they silently deleted healthy combos from the top-combos JSON file and the `combos` Postgres table. | Inspect the `finalEquity=...` field of each line; expect a dominant peak at exactly `1.000000`. *Today: 193 of 195 (99.0%) hit `finalEquity=1.000000` exactly; the other two were 0.947 and 0.970.* | This commit. |
| **H8** | `normalizeBarsForLookback` declines to clamp when `requiredBars > 1000` for Binance, so requests with `--lookback-bars 3360` against a 3m feed surface as `Need at least 3361 price rows ... (got 500)` and the bot fails open without ever starting. | A test pinning the current behaviour and explicitly marking the question as deferred to the optimizer / paging layer. *Today: pinned, not changed.* | Deferred. |
| **H9** | `Trader.TopCombosStore.applyComboUpdatesWithStats` is the second prune path (the optimizer top-N rerun uses it too), so even if the bot-start guard never produced a zero-trade update, a separate caller could. The defensive layer should refuse to prune on any inbound update whose own `tradeCount == 0`. | A direct store-layer test where a zero-trade update with `finalEquity ≤ 1.0` is applied to a combo with healthy stored metrics; the combo must remain in the array and not appear in `cbasPrunedKeys`. *Today: enforced.* | This commit. |
| **H1** *(unchanged)* | The 2026-06-08 `AutoStartBackoff` shipped to disk but the running launchd process does not include it; therefore the live system today behaves like 2026-06-04. | The launchd log shows ≥ 1 of {`classified`, `auth circuit OPEN`, `backoff: skipping`}. *Today: 0 of each → H1 not falsified.* | Operator (reload launchd after deploy). |

The dedup/fingerprint regression from 2026-06-10 (H2/H3/H4/H5) is
unchanged: that code is still on disk only.

---

## 3. Change set

### 3.1 New three-valued verdict

```haskell
-- | Trader.BotStartSemantics
data BacktestVerdict
    = BacktestAllow      -- enabled & finalEquity acceptable & finite
    | BacktestAbort      -- enabled & traded & sub-threshold/non-finite
    | BacktestNoVerdict  -- enabled & (no finalEquity OR no tradeCount OR tradeCount == 0)

botStartupBacktestVerdict :: Bool -> Maybe Double -> Maybe Int -> BacktestVerdict
backtestVerdictAborts     :: BacktestVerdict -> Bool
```

The semantics are deliberately conservative on missing evidence: any
case where we can't *confirm* that the smoke window actually traded is
`NoVerdict`. This includes `Nothing` for either `finalEquity` or
`tradeCount`. Concretely:

* `botStartupBacktestVerdict True (Just 1.5) (Just 0)   = BacktestAllow`
  (profitable wins regardless of trade count — already-acceptable
  evidence).
* `botStartupBacktestVerdict True (Just 1.0) (Just 0)   = BacktestNoVerdict`
* `botStartupBacktestVerdict True (Just 0.5) (Just 0)   = BacktestNoVerdict`
* `botStartupBacktestVerdict True (Just 1.0) (Just 1)   = BacktestAbort`
* `botStartupBacktestVerdict True (Just 0.85) (Just 12) = BacktestAbort`
* `botStartupBacktestVerdict True (Just 1.0) Nothing    = BacktestNoVerdict`
* `botStartupBacktestVerdict True Nothing _             = BacktestNoVerdict`
* `botStartupBacktestVerdict False _ _                  = BacktestAllow`

The existing `botStartupBacktestAborts` two-valued function is kept
unchanged (still tested) so the existing semantics are visibly preserved
for the disabled / `Nothing` cases.

### 3.2 Guard wired through

`runTopComboStartupBacktestGuard` in `app/Main.hs`:

* Reads `tradeCount` from the metrics via the new
  `Trader.TopCombosStore.comboMetricInt`.
* Calls `botStartupBacktestVerdict` once.
* On `BacktestNoVerdict`:
  * Records a new ops event `bot.start_combo_backtest_no_verdict` with
    a human-readable reason that includes both `tradeCount` and
    `finalEquity` for forensics.
  * **Does NOT** call `applyStartupComboBacktestUpdate`, so the smoke
    metrics are not persisted onto the combo at all (the optimizer's
    out-of-sample metrics remain authoritative).
  * **Does NOT** call `deleteTopComboFromDbMaybe`.
  * Returns `Right ()` — the start is allowed.
* On `BacktestAllow` or `BacktestAbort`: the existing path runs
  (persist update, prune if applicable, abort if applicable). This
  preserves the current behaviour for the cases where the verdict is
  actionable.

### 3.3 Defense-in-depth at the store layer

`Trader.TopCombosStore.applyComboUpdatesWithStats` now reads
`comboMetricInt "tradeCount" (cbuMetrics upd)` from the inbound update.
If `tradeCount == 0`, the combo is kept regardless of its post-update
`finalEquity`. This closes both the bot-start guard path and the
optimizer top-N rerun path (`app/Main.hs:10412`), without requiring the
two callers to coordinate.

The store now also exports `comboMetricInt` for the same reason
`comboMetricDouble` was already exported: it is a small, idempotent
helper that the guard layer needs.

### 3.4 Deferred lookback-clamp pin

`testNormalizeBarsForLookbackBinanceClampsAtPageCap` pins the current
behaviour: when `--lookback-bars 3360` is requested with `--interval 3m`
on Binance, `argBars` is left unchanged (the optimizer's job is to page,
not the bot-starter's). The test exists so any future change is
deliberate. The two FILUSDT / XRPUSDT failures today are bounded and
fail open (the bot-start path treats them as infrastructure failures,
not aborts).

### 3.5 New tests

Added to `test/TestMain.hs`:

| Test | Invariant |
|---|---|
| `testBotStartupBacktestVerdictZeroTradeIsNoVerdict` | H6 baseline: zero-trade smoke yields `BacktestNoVerdict`; profitable yields `Allow`. Also pins `backtestVerdictAborts` on each constructor. |
| `testBotStartupBacktestVerdictAbortOnLossWithTrades` | Symmetry: a backtest that traded and lost (or finished flat with `tradeCount > 0`) still aborts. We do not want the fix to swallow real regressions. |
| `testBotStartupBacktestVerdictPreservesDisabledBehaviour` | The disabled-guard short-circuit is identical to the existing `botStartupBacktestAborts` semantics. |
| `testApplyComboUpdatesZeroTradeDoesNotPrune` | H9: applying a zero-trade update with `finalEquity = 1.0` to a healthy combo (stored `finalEquity = 1.42`, `tradeCount = 8`) leaves the combo in the array and `cbasPrunedCount = 0`. |
| `testApplyComboUpdatesGenuineLossStillPrunes` | Symmetry: a genuine loss with `tradeCount = 12` still prunes (`cbasPrunedCount = 1`). |
| `testNormalizeBarsForLookbackBinanceClampsAtPageCap` | Pin the deferred H8 behaviour: `--lookback-bars 3360 --interval 3m` leaves `argBars` unchanged today. |

Verdict-function decision table (every row covered by tests):

| enabled | finalEquity | tradeCount | verdict        |
|--------|-------------|------------|----------------|
| False  | any         | any        | Allow          |
| True   | Nothing     | any        | NoVerdict      |
| True   | Just > 1.0 finite | any  | Allow          |
| True   | Just ≤ 1.0 or non-finite | Just > 0 | Abort   |
| True   | Just ≤ 1.0 or non-finite | Just 0  | NoVerdict |
| True   | Just ≤ 1.0 or non-finite | Nothing | NoVerdict |

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
    app/Trader/TopCombosStore.hs \
    app/Main.hs \
    test/TestMain.hs
(exit 0, no diff)

$ hlint app test bench
(7 pre-existing hints, none in code touched today)

$ .stack-work/install/.../bin/trader-hs \
    --data data/sample_prices.csv --price-column close \
    --epochs 1 --hidden-size 4 --json | head -c 300
{"backtest":{"agreementOk":[false,...
```

The `bash scripts/verify.sh haskell` wrapper requires `cabal`, which is
not installed on this host (only `stack`); the four steps above are the
equivalent path through `stack`, mirroring 2026-06-08 / 2026-06-10
reviews.

### 4.1 Invariants enforced (cumulative)

| Test | Invariant |
|---|---|
| (existing) `botStartupBacktestRoiAcceptable` ladder | Two-valued ROI threshold preserved for `Maybe Double` callers. |
| (existing) `botStartupBacktestAborts` fail-open | Old two-valued semantics preserved (disabled / `Nothing` / non-finite). |
| (new) `botStartupBacktestVerdict` zero-trade NoVerdict | **H6 today** — zero-trade smoke is not a verdict. |
| (new) `botStartupBacktestVerdict` abort on traded loss | Real signal regressions still abort. |
| (new) `botStartupBacktestVerdict` disabled-guard preserves Allow | No behaviour change for disabled boxes. |
| (new) `applyComboUpdatesWithStats` zero-trade no-prune | **H9 today** — store-layer defense-in-depth. |
| (new) `applyComboUpdatesWithStats` genuine loss prunes | Symmetry: the fix does not swallow real losses. |
| (new) `normalizeBarsForLookback` Binance >1000 deferred-pin | **H8 today** — current behaviour pinned, change requires test edit. |
| (2026-06-10) `normalizeAutoStartErrorMessage` fingerprint stability | H2/H3/H4/H5 from yesterday — code on disk, not yet running. |
| (2026-06-08) auth circuit / classify / backoff | Shipped but still not running (H1 unresolved). |

---

## 5. Strategy Research (treating trading as engineering)

The signal layer still hasn't run, so a backtest-vs-live comparison is
again not actionable. The relevant research direction today, given the
observed failure mode, is **evidence sufficiency in operating-decision
guards** — a sub-domain that recurs wherever a trading stack uses one
metric to gate a different operating decision:

* **Two-valued vs three-valued verdicts.** A binary
  "abort / don't abort" verdict has to project an unknown into one of
  two buckets. That projection is fine when the unknown is "no data" and
  the cost of failing open is small. It is *not* fine when the unknown
  is "no trade fired in the smoke window," because the cost of failing
  closed is a **permanent state change** (prune from disk + DB). Treating
  `finalEquity == 1.0` ∧ `tradeCount == 0` as identical to a loss is the
  same shape of bug as treating "no fill" as "rejected" in an
  order-routing gateway: the right answer is a third state for "didn't
  happen yet."
* **Smoke backtests are inherently zero-trade-biased.** A smoke window
  is, by construction, short, signal-gated, and uses fresh exchange data.
  Three properties combine to make zero-trade the dominant outcome on
  quiet days: (a) the window is too short for slow signals (e.g. ADX
  /EMA200 trend filters with 200-bar lookbacks); (b) volatility filters
  often hold below their entry edge through the entire window; (c) the
  fresh Binance fetch is bounded at 1000 bars (today's H8), so the most
  active combos (which need ≥ 3360 bars) can't run at all. The right
  engineering response is to refuse to update the combo's authoritative
  metrics from a smoke run unless that run actually traded — which is
  exactly what 3.2 does on the bot-start path.
* **Erosion vs failure.** The combo-prune bug is an erosion bug, not a
  failure bug. Each individual prune looks like a correct application of
  the policy: the combo's last reading is "not profitable," so we drop
  it. The aggregate effect is to ablate the leaderboard. Without
  3.3 (store-layer defense-in-depth), any callsite that builds a
  `ComboBacktestUpdate` from a fresh-but-non-trading backtest would
  erode the leaderboard. With 3.3, the same protective rule lives next
  to the data structure that holds the leaderboard, so future callers
  cannot bypass it by accident.
* **What this is not.** This is *not* a research-strategy win. The
  improved guard does not change which combos win; it changes which
  combos are *kept*. The strategy bank is the input to every future
  improvement; preserving it is a precondition.

---

## 6. Remaining work (not in scope today)

1. **Operator action (high priority, H1 still open):** reload the
   `ai.openclaw.trader.api` LaunchAgent so the binary that includes
   2026-06-08 backoff + 2026-06-10 fingerprint + 2026-06-11 zero-trade
   guard actually runs:

   ```
   launchctl bootout gui/501/ai.openclaw.trader.api
   launchctl bootstrap gui/501 \
     ~/Library/LaunchAgents/ai.openclaw.trader.api.plist
   ```

   Until this happens, the launchd log will continue to show
   pre-2026-06-08 loop behaviour AND the pre-2026-06-11 prune behaviour.
   The 195 combos lost today are gone from disk; the fix prevents
   further loss but does not restore them. (Restoration would require
   replaying `optimizer.combos.snapshot` ops from the journal — out of
   scope.)

2. **Operator action (orthogonal, root cause of H1's loop):** rotate
   `BINANCE_API_KEY` / `BINANCE_API_SECRET` or add the active egress IP
   (currently `148.227.107.253`) to the Binance allow-list.

3. **Optimizer-side audit (deferred):** sweep recent
   `optimizer.combos.snapshot` ops from the journal to:
   * Quantify how many of today's 195 prunes had positive optimizer
     scores at last write (this is the "erosion harm" number).
   * Decide whether to ship a `bot.start_combo_backtest_restore` op
     that re-adds combos for which the prune-cause was zero-trade.

4. **Lookback / paging (H8 deferred):** decide explicitly whether
   `normalizeBarsForLookback` should page or shrink `argLookback` when
   Binance requests `requiredBars > 1000`. Today the test pins the
   current "leave alone" behaviour. Two FILUSDT / XRPUSDT errors per
   session is bounded but the failure mode is silent.

5. **Telemetry (deferred from 2026-06-08):**
   * Add Prometheus counter
     `trader_top_combo_prune_total{reason="zero_trade_smoke"|"loss"|...}`.
   * Add gauge `trader_top_combos_count` so erosion is visible on a
     dashboard, not only in log archaeology.

6. **Journal normalization (deferred from 2026-06-10):** apply
   `normalizeAutoStartErrorMessage` to `bot.start_queued_failed` /
   `bot.start_failed` op messages before they enter the journal/DB.

---

## 7. Files touched

```
haskell/app/Trader/BotStartSemantics.hs        (BacktestVerdict + verdict function)
haskell/app/Trader/TopCombosStore.hs           (export + zero-trade no-prune defense)
haskell/app/Main.hs                            (wire verdict into runTopComboStartupBacktestGuard)
haskell/test/TestMain.hs                       (6 new invariants + Aeson .= import)
CHANGELOG.md                                   (one new "Unreleased" entry)
ENGINEERING_REVIEW_2026-06-11.md               (this file)
```

`stack test` green; `fourmolu --mode check` clean on the four touched
files; `hlint` produces only pre-existing hints in `app/Trader/LSTM.hs`
and `test/TestMain.hs` lines 326–440 (unrelated to today's changes).
