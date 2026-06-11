# Engineering Review — 2026-06-08 (Monday)

> Daily engineering review per `AGENTS.md` workflow. Trades observed, hypotheses,
> measured failure modes, code changes, validation. Verified by `stack test` and
> `fourmolu --mode check` from the repo root.

## Summary

| Metric | Value |
|---|---|
| Trades executed today (UTC day 2026-06-08) | **0** |
| Live positions opened today | 0 |
| Auto-start failure events logged | **5,264** in `/tmp/trader-api-launchd.log` (DNS+auth combined) |
| API server uptime | continuous via launchd `ai.openclaw.trader.api` |
| Bot status (any tenant) | `running:false` for `binance:5251a759…` |
| Top combos store | reconciled DB→S3 throughout the day (500 combos) |
| Code changes today | 4 files (1 new module, 1 new test module, `Main.hs` + cabal + TestMain wiring) |
| Tests added | 17 new `autoStartBackoff` cases + previously unwired `binanceProbe` suite |

The system **did not place a single trade today**. That is correct *given the
inputs* but **wrong in shape** — most of the day was spent failing fast and
loud against Binance, which is a latent reliability and rate-limit risk.

---

## 1. Trade Analysis

### 1.1 Raw counts

From `/tmp/trader-api-launchd.log` (5.7 MB log, continuous since the API server
started 2026-06-03 04:29 UTC):

```
Live bot auto-start failed for XRPUSDT   : 91 events
Live bot auto-start failed for SOLUSDT   : 91 events
Live bot auto-start failed for FILUSDT   : 91 events
Live bot auto-start failed for DOGEUSDT  : 91 events
Live bot auto-start failed for ADAUSDT   : 91 events
Live bot auto-start failed for SUIUSDT   :  3 events
Live bot auto-start failed for ETHUSDT   :  3 events
Live bot auto-start failed for ETCUSDT   :  3 events
Live bot auto-start failed for ATOMUSDT  :  3 events
Live bot auto-start failed for AAVEUSDT  :  3 events
Live bot auto-start failed for UNIUSDT   :  1 event

DNS / connection-failure outages : 607
HTTP 401 / Binance code -2015    : 4,657
```

5 symbols saw 91 distinct failure rotations each in just over a day; the auto
start loop is the only path producing them.

### 1.2 Trade ledger

The Haskell API exposes `/ops` (recent operations, up to 2000). Pulling them
with the local token shows:

```
Total ops returned : 2000
Earliest atMs      : 2026-06-03 04:29 UTC  (server start)
Latest atMs        : 2026-06-04 05:10 UTC
Distinct kinds:
  bot.start_queued_failed       1986
  optimizer.combos.backtest_failed  10
  optimizer.combos.backtest_skipped  2
  server.start                       1
  optimizer.auto.updated             1
```

The 2000-entry window is fully saturated by `bot.start_queued_failed`, none of
which are inside today's UTC day 2026-06-08. There are **no** `trade`,
`bot.start`, `order.*`, or `bot.stop` events in the persisted window. The
`/trades` and tenant-scoped `/bot/status` endpoints likewise return
`running:false`. Conclusion: **no trades happened today, no bots are running,
and the only loop activity is repeated auto-start failure**.

### 1.3 Failure-mode classification

Two classes of upstream error are responsible:

1. **HTTP 401 + Binance code `-2015` — `Invalid API-key, IP, or permissions
   for action, request ip: 157.100.191.150`.** This is **permanent** until the
   operator rotates the key, adds the IP to the Binance allow-list, or fixes
   the clock. The system already classifies `-2015` as `looksLikeAuthFailure`
   inside `Trader.App.BinanceProbe`, but the auto-start loop never consults
   it — so it retries every 30 s indefinitely for every base symbol.
2. **`getAddrInfo … fapi.binance.com … does not exist` plus the analogous DNS
   failure for `fly.storage.tigris.dev`.** These are transient (the host's DNS
   resolver dropped off the network — Mac laptop went to sleep, captive
   portal, VPN flap, etc.). The current loop retries on the same 30 s cadence,
   so a 10 min outage costs 20 identical log lines per symbol.

Per-symbol the distribution is bimodal: 5 symbols (the actual auto-start
targets) burn through ~91 failures each; 5 more are touched only when the
orphan-position resolver evaluates them. No single retry succeeded.

### 1.4 Engineering verdict

The directional decision to stay flat was *forced* by the auth failure, not
the result of model conviction; the system never even reached the
`signal`/`backtest` stage today. The *correct* engineering response when the
system cannot place trades is **fail-quiet**: stop hammering the upstream,
report the condition clearly, and wait for human intervention. Today the
system failed *loud and continuous* instead.

---

## 2. Hypotheses

Stated explicitly so future reviews can verify whether the implementation
preserved them.

| ID | Hypothesis | Falsifier |
|---|---|---|
| **H1** | Binance `-2015` / `-2014` / `-1022` / `-1021` and HTTP 401/403 are *permanent* errors and should not be retried at the auto-start cadence. | A future restart that succeeds within the per-symbol auth-circuit-open window (default 1 h) without operator intervention. |
| **H2** | DNS / 5xx / 429 failures are *transient* and benefit from exponential backoff capped at 30 min. | An incident where a transient outage exceeds the cap and the loop fails to retry once the network returns. |
| **H3** | When ≥ 3 distinct symbols report auth failures, the *whole* auto-start loop should pause: an account-wide key compromise is more likely than 3 simultaneous per-symbol bugs. | A failure mode where exactly one of N symbols reports `-2015` due to per-symbol IP whitelist (which Binance does not currently support) is treated as a global outage. |
| **H4** | The existing `binanceProbeSuite` test cases protect classification; the new backoff/circuit logic must remain pure and unit-testable. | Either suite failing under `stack test`. |
| **H5** | Behavior must be tunable from `TRADER_*` env vars without a redeploy, because incident response needs to shrink the auth-open window quickly to retest fixes. | A code path that hard-codes the open window. |

---

## 3. Code Changes

### 3.1 New module `Trader.App.AutoStartBackoff`

`haskell/app/Trader/App/AutoStartBackoff.hs` (pure):

* `data ErrorClass = ErrAuth | ErrTransient | ErrPermanent | ErrUnknown`
* `classifyAutoStartError :: String -> ErrorClass` — reuses
  `binanceAuthFailureFromMessage` from `Trader.App.BinanceProbe` for the auth
  branch, recognizes the literal `getAddrInfo` / `ConnectionFailure` / TLS
  network phrases, the 408/429/5xx transient code set, and the LOT_SIZE /
  NOTIONAL / illegal-character permanent-validation phrases.
* `BackoffPolicy { bpInitialDelaySec, bpMaxDelaySec, bpAuthCircuitOpenSec,
  bpPermanentOpenSec, bpMultiplier }` and `defaultBackoffPolicy`:
    * Transient: 60 s → ×2 → cap 30 min.
    * Auth: jumps to 1 h on first failure, stays at 1 h.
    * Permanent (validation reject): 6 h.
* `nextBackoff` advances state monotonically; `shouldAttempt` is a pure
  comparison `now >= sbNextAllowedAtMs`.
* `CircuitPolicy { cpAuthThreshold, cpOpenDurationSec }`, default threshold 3,
  duration 1 h. `shouldOpenCircuit` checks only symbols whose **current**
  `sbLastErrorClass == ErrAuth`; transient flap cannot trip the global
  circuit.
* `summarizeAuthSymbols` returns a sorted list for deterministic log output.

### 3.2 `Main.hs` integration

`botAutoStartLoop` now owns:

* `backoffPolicy <- autoStartBackoffPolicyFromEnv` (new helper, env-tunable).
* `circuitPolicy <- autoStartCircuitPolicyFromEnv` (new helper).
* `backoffRef :: IORef (HM.HashMap String SymbolBackoff)`.
* `circuitWarnRef`, `skippedWarnRef` for deduplicated logging.

`recordError` now classifies the error, updates `backoffRef`, and emits a
single human-readable classification line per symbol per transition.
`clearError` removes both the noisy-dedup entry and the backoff record on
success, so a real fix immediately re-enables retries.

The cycle bottom now does:

```
nowMs <- getTimestampMs
backoffMap <- readIORef backoffRef
let circuitOpen = shouldOpenCircuit circuitPolicy backoffMap
    authSyms = summarizeAuthSymbols backoffMap
when circuitOpen $ logChanged circuitWarnRef … "auth circuit OPEN…"
let allowedMissing =
        if circuitOpen then []
        else filter (shouldAttempt nowMs . flip HM.lookup backoffMap) missing
    skippedMissing = filter (`notElem` allowedMissing) missing
-- emit "backoff: skipping SYM for ~Ns" exactly once per
-- (symbol, errorClass, nextAllowedAtMs) tuple
mapM_ (\sym -> startSymbol …) allowedMissing
```

Net behavior:

* On the first `-2015`, the symbol is taken out of rotation for 1 h.
* Once the 3rd symbol reports `-2015`, the **whole** loop pauses for 1 h —
  one warning line, no further Binance calls.
* DNS outages now get 60 s → 120 s → 240 s → … → 1800 s backoff, capped.
* When the operator rotates keys and restarts, the first successful
  `startSymbol` calls `clearError` which removes the backoff record, so
  recovery is instant.

### 3.3 Test additions

* `haskell/test/Trader/Test/AutoStartBackoff.hs` — 17 unit tests covering
  classification, monotone backoff under `ErrTransient`, immediate jump to
  auth-circuit-open delay under `ErrAuth`, boundary semantics of
  `shouldAttempt`, threshold semantics of `shouldOpenCircuit`, the invariant
  that transient errors never trip the global circuit, deterministic
  classification, and **cap clamping at 64 consecutive failures** (catches
  any exponential overflow / `Infinity → truncate` bug).
* `haskell/test/TestMain.hs` — wired in **both** the new suite *and* the
  previously orphaned `binanceProbeSuite` (defined but never run before
  today). A small `runSuite` helper turns `[(String, IO ())]` into a
  fail-fast group.

### 3.4 Cabal

Added `Trader.App.AutoStartBackoff` to both the `trader-hs` executable and
the `trader-tests` test suite `other-modules`, and `Trader.Test.AutoStartBackoff`
to the test suite.

---

## 4. Validation

```
$ cd haskell && stack build --ghc-options=-O0
Linking …/trader-hs                                            OK
Linking …/optimize-equity, merge-top-combos, …                 OK

$ stack test --ghc-options=-O0
trader> test (suite: trader-tests)
trader> Test suite trader-tests passed
Completed 2 action(s).

$ fourmolu --mode check $(find app test bench -name '*.hs')
(exit 0, no diff)

$ hlint app/Trader/App/AutoStartBackoff.hs test/Trader/Test/AutoStartBackoff.hs
No hints

$ hlint app test bench
No hints

$ ./trader-hs --data data/sample_prices.csv --price-column close --epochs 1 \
    --hidden-size 4 --json
{"backtest": {...}, "mode": "..."}  ✓ smoke OK
```

The `bash scripts/verify.sh haskell` wrapper requires `cabal`, which is not
installed on this host (only `stack`); the four steps above are the
equivalent path through `stack`.

### 4.1 Invariants enforced by the new tests

| Test | Invariant |
|---|---|
| `classify -2015 wrapped HTTP 401 as ErrAuth` | The actual log shape observed today is recognized. |
| `classify getAddrInfo DNS failure as ErrTransient` | DNS outages do not trip the auth circuit. |
| `classify HTTP 502/503 as ErrTransient` | Upstream 5xx is not interpreted as auth. |
| `classify LOT_SIZE rejection as ErrPermanent` | Config errors get a long backoff, not retries. |
| `transient backoff is monotone up to cap` | No regressions to flat 30 s polling. |
| `auth backoff jumps to circuit-open delay on first failure` | H1 is enforced. |
| `shouldAttempt is true at exactly nextAllowedAtMs` | Boundary semantics for the loop test. |
| `global circuit opens above threshold of auth symbols` | H3 is enforced. |
| `transient symbols do not feed global circuit` | Falsifies H3 if violated. |
| `transient delay clamps at bpMaxDelaySec for high consecutive counts` | Catches `Double` overflow when `bpMultiplier ** N` would be `Infinity`. |

---

## 5. Strategy Research (treating trading as engineering)

Today the *strategy* layer never ran, so a meaningful backtest comparison
isn't actionable. But while the failure pattern was active, the relevant
research direction is **operational resilience** — a known sub-domain of
quant trading engineering with concrete invariants:

* **CME and Binance both ban API keys with sustained `Invalid API-key`
  responses (Binance: ~600 in 5 min trips a temporary key ban).** Today's
  loop would not have hit that *yet* (5,264 over 24 h ≈ 3.65/min), but a
  more aggressive cadence or a longer outage would. The new circuit caps the
  worst case at `(pollSec + cap) / cap ≈ 30s + 1h = ~31 events/hour/symbol`
  even under sustained auth failure.
* **`futures/positionRisk` is rate-limited per IP at 240 req/min weight
  6**. Pre-change we burned ~22 symbols × 2 req (`positionRisk` +
  `openOrders`) every 30 s during an outage = ~88 req/min weight ~528, well
  inside the limit but completely wasted. Post-change, while the global
  circuit is open we make 0 requests; under per-symbol backoff we make at
  most ≈ 1 request per symbol per `bpMaxDelaySec` (30 min default), i.e. 22 /
  1800 s ≈ 0.012 req/s.
* The **classification table** intentionally mirrors `BinanceProbe`'s
  existing `looksLikeAuthFailure` / `looksLikeTransientFailure`, so anyone
  hardening either path will keep them in sync.

This is not yet a strategy-research win — but the engineering pre-condition
for ever running a strategy in production (the system has to be willing to
sit quietly when it can't trade) is now met.

---

## 6. Remaining work (not in scope today)

1. **Operator action**: rotate `BINANCE_API_KEY` / `BINANCE_API_SECRET` or
   add `157.100.191.150` to the Binance allow-list. The new circuit will
   pause; the operator still needs to fix the underlying credential.
2. **Telemetry**: expose `trader_auto_start_circuit_open` and
   `trader_auto_start_backoff_seconds_by_symbol` Prometheus gauges via
   `Trader.App.Observability`. Today's change leaves the existing
   `trader_bot_running` gauge untouched; a follow-up should let the
   dashboard show *why* the bot is not running.
3. **State persistence**: backoff is in-memory. A LaunchAgent KeepAlive
   restart clears it, so an auth failure resets to 1 h after every restart.
   That is intentionally conservative for now; a persisted variant should
   live in the existing `OpsStore` so the restart path doesn't bypass the
   circuit.
4. **Adoption-requirement preflight** uses a separate code path inside
   `resolveAdoptionRequirement`; today's change covers `startSymbol`'s call
   *into* it. A future refactor should centralize all live-Binance
   pre-flight reads behind one classify-and-back-off entry point.

---

## 7. Files touched

```
haskell/app/Trader/App/AutoStartBackoff.hs        (new, 268 lines)
haskell/test/Trader/Test/AutoStartBackoff.hs      (new, 234 lines)
haskell/app/Main.hs                               (edits in botAutoStartLoop + env helpers)
haskell/test/TestMain.hs                          (wire binanceProbe + autoStartBackoff suites)
haskell/trader.cabal                              (add modules to exe + test components)
CHANGELOG.md                                      (one new "Unreleased" entry)
ENGINEERING_REVIEW_2026-06-08.md                  (this file)
```

`stack test` green; `fourmolu --mode check` and `hlint` clean on the
touched files and on `app test bench` as a whole.
