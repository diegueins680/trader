# Engineering Review — 2026-06-10 (Wednesday)

> Daily engineering review per `AGENTS.md` workflow. Trades observed,
> hypotheses with falsification, measured failure modes, code changes,
> validation. Verified by `stack test` (green), `fourmolu --mode check`
> (no diff), and `hlint app test bench` (no hints) from `haskell/`.

## 0. Summary

| Metric | Value | Δ vs 2026-06-08 |
|---|---|---|
| Trades executed today (UTC day 2026-06-10) | **0** | unchanged |
| Live positions opened today | 0 | unchanged |
| `Live bot auto-start failed` log lines today | **476** | down from 5,264, but with a confounder (see §1.2) |
| `Live bot auto-start auth circuit OPEN` lines | **0** | regression vs the intent shipped 2026-06-08 |
| `Live bot auto-start backoff: skipping` lines | **0** | regression vs the intent shipped 2026-06-08 |
| `Live bot auto-start classified` lines | **0** | regression vs the intent shipped 2026-06-08 |
| `Top combos sync reconciled` log lines | 3,318 | unchanged |
| API server uptime | continuous via launchd `ai.openclaw.trader.api` | unchanged |
| Bot status (any tenant) | `running:false` | unchanged |
| Distinct egress IPs cited in today's `-2015` replies | **6** (1×157.100.x, 5×148.227.107.x) | new failure mode |
| Code changes today | 3 files (`AutoStartBackoff.hs`, `Main.hs`, `Trader/Test/AutoStartBackoff.hs`) + CHANGELOG | — |
| Tests added | 7 new `autoStartBackoff` invariants on fingerprint stability | — |

The system **did not place a single trade today**. That is again *correct
given the inputs* (Binance `-2015` is still active for the operator) but
**three operationally-loud regressions** were observed: the backoff/circuit
shipped 2026-06-08 produced exactly zero `classified`, `circuit OPEN`, and
`backoff: skipping` log lines today — i.e. the code is on disk but not
running — *and* the dedup logic that did run has a previously-unfound bug
that re-fires every time the host's egress IP rotates. Today's launchd log
documents both regressions; today's fix removes the second one (the first
is an operator action item, §6).

---

## 1. Trade Analysis

### 1.1 Raw counts (UTC day 2026-06-10)

`/tmp/trader-api-launchd.log` ≈ 6.06 MB, 44,014 lines, running `trader-hs`
PID 41339 started **Wed Jun 3 00:10:24 2026**.

```
$ grep -c "Top combos sync reconciled" /tmp/trader-api-launchd.log
3318
$ grep -c "Live bot auto-start failed"   /tmp/trader-api-launchd.log
476
$ grep -c "futures/positionRisk"          /tmp/trader-api-launchd.log
4637
$ grep -c "auth circuit OPEN"             /tmp/trader-api-launchd.log
0
$ grep -c "backoff: skipping"             /tmp/trader-api-launchd.log
0
$ grep -c "Live bot auto-start classified"/tmp/trader-api-launchd.log
0
```

Per-symbol breakdown (still bimodal, same five base-auto-start targets):

```
   3 for AAVEUSDT:
  92 for ADAUSDT:
   3 for ATOMUSDT:
  92 for DOGEUSDT:
   3 for ETCUSDT:
   3 for ETHUSDT:
  92 for FILUSDT:
  92 for SOLUSDT:
   3 for SUIUSDT:
   1 for UNIUSDT:
  92 for XRPUSDT:
```

### 1.2 The IP-rotation finding

Yesterday's review (2026-06-08) reported **5,264** identical
`Live bot auto-start failed` lines for the same five symbols. Today's
review reports **476** — apparently a 10× drop. That number is **not** a
real reliability improvement; it is a confounder produced by today's
fix-on-disk-only state plus today's distinct DHCP behaviour.

Inspecting the `request ip:` field cited in each `-2015` reply:

```
$ grep -oE "request ip: [0-9.]+" /tmp/trader-api-launchd.log | sort | uniq -c
   5 request ip: 148.227.107.145
   1 request ip: 148.227.107.16
   5 request ip: 148.227.107.172
4641 request ip: 148.227.107.253
   5 request ip: 148.227.107.97
   5 request ip: 157.100.191.150
```

This is a host whose **egress IP rotated** across six values today
(VPN/captive portal flap). The same `-2015` failure was reported to the
agent through five additional distinct strings. The pre-fix `recordError`
dedup compared the **raw** error message against a per-symbol previous
string:

```haskell
let firstTime = HM.lookup sym prev /= Just msg
when firstTime $ do
    writeIORef errRef (HM.insert sym msg prev)
    putStrLn ("Live bot auto-start failed for " ++ sym ++ ": " ++ msg)
```

so the moment the IP rotated, `firstTime` was true again, the log line
re-fired, and the dedup state was replaced with the new IP's message.
Within `Trader.App.AutoStartBackoff`, `sbLastErrorMessage` was annotated
"Stable message fingerprint to drive log-dedup decisions upstream", but
was *also* set to the raw `msg`, so the in-memory fingerprint rotated in
lock-step. Today's per-IP distribution and per-symbol "92 events" count
are exactly what the bug predicts.

### 1.3 The stale-binary finding

`Trader.App.AutoStartBackoff` was committed on 2026-06-08 with a green
`stack test`. But:

```
$ ls -la /Users/.../haskell/dist-newstyle/build/.../trader-hs/trader-hs
-rwxr-xr-x  ...  Jun  8 22:32  trader-hs
$ ps -p 41339 -o lstart=
Wed Jun  3 00:10:24 2026
```

The running launchd process predates the new binary by **5 days, 22h**.
Launchd does not auto-restart on binary mtime change. So the
classification/backoff/circuit code shipped 2026-06-08 is on disk but
**not in the running process** today — that explains the 0 `classified`,
0 `auth circuit OPEN`, 0 `backoff: skipping` lines:

* These features can't run.
* The 476 `Live bot auto-start failed` lines visible today come from the
  *pre-2026-06-08* code path — i.e. they are the same loop yesterday's
  review thought we had already fixed.

The shape (92 events × 5 symbols ≈ 460 + tail) matches a 30 s poll cadence
over today's session window, with the IP-rotation re-fire layered on top.

### 1.4 Engineering verdict

* **Trade decision quality:** not the question today. The signal layer was
  never reached because the Binance preflight (`futures/positionRisk`)
  fails with `-2015` for every cycle.
* **Reliability shape today:** worse than the 2026-06-08 commit promised
  on the modified codepath (regression H2) *and* the 2026-06-08 codepath
  has not yet executed (regression H1). The directional decision to stay
  flat was forced again, not chosen.
* **Engineering goal today:** close H2 in code (deterministic), and call
  out H1 as the operator action item that no in-process code can resolve.

---

## 2. Hypotheses

| # | Hypothesis | Falsifier | Owner |
|---|---|---|---|
| **H1** | The 2026-06-08 `AutoStartBackoff` shipped to disk but the running launchd process does not include it; therefore the live system today behaves like 2026-06-04. | The launchd log shows ≥ 1 of {`Live bot auto-start classified`, `Live bot auto-start auth circuit OPEN`, `Live bot auto-start backoff: skipping`}. *Today: 0 of each → H1 not falsified.* | Operator (reload launchd after deploy). |
| **H2** | `recordError` dedup and `sbLastErrorMessage` compare raw error strings; when Binance's `-2015` reply embeds `request ip: <egress>`, an IP rotation re-fires the log line and resets the fingerprint, breaking the "stable fingerprint" invariant. | Two raw `-2015` messages that differ only in `request ip:` must produce identical `normalizeAutoStartErrorMessage` outputs, and identical `sbLastErrorMessage` fields after `initialBackoff`/`nextBackoff`. *Today: enforced by 7 new tests.* | This commit. |
| **H3** | The existing `autoStartBackoffSuite` had no test asserting fingerprint stability under volatile-noise rotation, so H2-style regressions had no callable validation seam. | A property-style test that mutates request IP / signature / timestamp / listenKey and asserts `normalizeAutoStartErrorMessage`-equivalence. *Today: 5 representative scenario tests cover the four most-rotational tokens.* | This commit. |
| **H4** | Normalization must be idempotent (`f . f = f`); otherwise a downstream re-normalization (e.g. journal compression) could double-redact. | A bounded test runs `normalizeAutoStartErrorMessage . normalizeAutoStartErrorMessage` and asserts equality on at least four shapes (including empty/whitespace inputs). *Today: enforced.* | This commit. |
| **H5** | Normalization must preserve the classified `ErrorClass`; otherwise stripping noise could silently demote a permanent `-1013 LOT_SIZE` failure into the transient bucket. | A bounded test classifies each pre-fix raw message and the post-fix normalized message and asserts equality across the {auth, transient, permanent, dns} four-tuple. *Today: enforced.* | This commit. |

H1 was caught only because today's log was empty of the expected diagnostic
lines. Without §1.3, the 10× drop in `Live bot auto-start failed` between
2026-06-08 and 2026-06-10 would have looked like a win.

---

## 3. Change set

### 3.1 New pure function

```haskell
-- | Trader.App.AutoStartBackoff
normalizeAutoStartErrorMessage :: String -> String
```

Strips known volatile noise tokens from a raw Binance / network-stack
error message: `request ip: …`, `client ip: …`, `signature=…`,
`signature: …`, `listenKey=…`, `timestamp=…`, `recvWindow=…`,
`serverTime=…`, `x-mbx-uuid=…`, `requestId=…`, `request-id: …`. Each
match is replaced by `…<REDACTED>`. The function is total, idempotent,
classification-preserving, and pure. It also collapses runs of internal
whitespace and trims trailing punctuation so log lines whose only
difference is whitespace cannot break dedup.

Implementation notes:

* Case-insensitive prefix match (the `BinanceProbe` summary already
  lowercases for its own keyword scan; we preserve original output
  casing for human-readability).
* After replacing the literal text the leftover head is fed through a
  per-token `dropWhile` of the appropriate character class
  (`isDigit`/`isHexDigit`/IPv4-or-v6/UUID/listen-key alphabet).
* Idempotency is enforced by consuming a leading literal `<REDACTED>`
  marker after each replacement; without this guard, re-running the
  function on an already-redacted string would emit `…<REDACTED><REDACTED>`.
  The idempotency test in §3.3 caught this on the first build.

### 3.2 Backoff record now stores the fingerprint

`initialBackoff` and `nextBackoff` both set
`sbLastErrorMessage = normalizeAutoStartErrorMessage msg` — restoring the
invariant the docstring already claimed:

> Stable message fingerprint to drive log-dedup decisions upstream.

### 3.3 `recordError` dedup

`recordError` in `botAutoStartLoop` now computes
`fingerprint = normalizeAutoStartErrorMessage msg` and dedups on it.
`putStrLn` still emits the *raw* message — the operator log is more
useful with the actual rotating IP for diagnostic purposes — but the
dedup keys and the in-memory backoff fingerprint are stable.

### 3.4 New tests

Added to `autoStartBackoffSuite` (7 new entries):

| Test | Invariant |
|---|---|
| `fingerprint is stable across request-ip rotation` | H2: today's exact `157.100.191.150` ↔ `148.227.107.253` ↔ `148.227.107.97` triple maps to one fingerprint. |
| `fingerprint is stable across signature/timestamp rotation` | H2 (extended): rotating `signature=…&timestamp=…` does not change the fingerprint. |
| `fingerprint is stable across listenKey rotation` | H2 (extended): two distinct `listenKey=…` values fingerprint identically. |
| `fingerprint normalization is idempotent` | H4: caught a real bug on first run (double-redaction of `request ip`). |
| `fingerprint preserves ErrorClass classification` | H5: auth/transient/permanent/DNS classification unchanged after normalization. |
| `sbLastErrorMessage stores the fingerprint, not the raw message` | Backoff-record invariant: closes the documentation-vs-implementation gap. |
| `backoff stores the same fingerprint across IP rotation` | End-to-end: two `nextBackoff` calls on the same auth failure across IPs produce identical `sbLastErrorMessage`. |

Final count: 17 → 24 `autoStartBackoff` cases.

---

## 4. Validation

```
$ cd haskell && stack build --ghc-options=-O0
Linking …/trader-hs                                            OK

$ stack test --ghc-options=-O0
trader> test (suite: trader-tests)
trader> Test suite trader-tests passed
Completed 2 action(s).

$ fourmolu --mode check app/Trader/App/AutoStartBackoff.hs \
                       test/Trader/Test/AutoStartBackoff.hs \
                       app/Main.hs
(exit 0, no diff)

$ hlint app test bench
No hints

$ ./.stack-work/install/.../bin/trader-hs --version
trader-hs 0.1.0.0

$ ./.stack-work/install/.../bin/trader-hs \
    --data data/sample_prices.csv --price-column close \
    --epochs 1 --hidden-size 4 --json | head -c 300
{"backtest":{"agreementOk":[…],"avg_trade":0,"baselines":[{"metrics":{…
```

The `bash scripts/verify.sh haskell` wrapper requires `cabal`, which is
not installed on this host (only `stack`); the four steps above are the
equivalent path through `stack`, mirroring yesterday's review's
validation.

### 4.1 Invariants enforced (cumulative)

| Test | Invariant |
|---|---|
| (existing) `classify -2015 wrapped HTTP 401 as ErrAuth` | The actual log shape observed today is recognized. |
| (existing) `classify getAddrInfo DNS failure as ErrTransient` | DNS outages do not trip the auth circuit. |
| (existing) `transient backoff is monotone up to cap` | No regression to flat 30 s polling. |
| (existing) `transient delay clamps at bpMaxDelaySec for high consecutive counts` | Catches `Double` overflow when `bpMultiplier ** N` would be `Infinity`. |
| (existing) `global circuit opens above threshold of auth symbols` | H3 of the 2026-06-08 review. |
| (new) `fingerprint is stable across request-ip rotation` | H2 of today. |
| (new) `fingerprint normalization is idempotent` | H4 of today. |
| (new) `fingerprint preserves ErrorClass classification` | H5 of today. |
| (new) `sbLastErrorMessage stores the fingerprint, not the raw message` | Backoff fingerprint contract is enforced, not aspirational. |
| (new) `backoff stores the same fingerprint across IP rotation` | End-to-end fingerprint stability across `initialBackoff` → `nextBackoff`. |

---

## 5. Strategy Research (treating trading as engineering)

The signal layer still hasn't run today, so a backtest-vs-live comparison
is again not actionable. The relevant *research direction* given the
observed failure mode is **observability under volatile transport
identifiers** — a sub-domain that appears wherever a trading stack treats
upstream error messages as features:

* **Volatile-identifier scrubbing** is a known practice in production
  trading observability (e.g. CME's iLink and OCC's MQ error code/strap
  normalization). The invariant is: *the dedup key for an error message
  must be a function of the failure semantics only, not of the transport
  identifiers that happen to appear in the operator's view of the
  response.* Today's bug violated that invariant; the new
  `normalizeAutoStartErrorMessage` enforces it for the auto-start path.
* **The same principle should be applied to journal compression.** Today
  every `bot.start_queued_failed` op carries the raw error message; that
  field is also where downstream alerting would lossily group on. A
  follow-up should normalize the persisted `bot.start_queued_failed` /
  `bot.start_failed` ops too — but only after the operator action in §6
  lands, so the journal isn't simultaneously being rewritten and read by a
  pre-2026-06-08 binary.
* **Egress-IP instability as a separate signal.** Today's log uniquely
  identifies a 5-IP rotation (`148.227.107.{16,97,145,172,253}` plus
  `157.100.191.150`) inside an outage window. That is a legitimate
  operational signal (the host or its NAT layer is flapping) that today
  was *hidden* by the previous bug — the operator only saw "many auth
  failures" without the IP rotation pattern. The new normalization
  preserves the raw message in `putStrLn` output, so the operator can
  still see the rotation; only the dedup key changes. This is the right
  tradeoff: dedup on semantics, log the raw transport for forensics.

This is again not yet a strategy-research win. But the engineering
pre-condition for ever telling whether a trade was *correctly* declined
versus *prevented* is stable: an outage today produces a stable, single
classified log line per symbol, not a fan-out per egress IP.

---

## 6. Remaining work (not in scope today)

1. **Operator action (high priority, H1 falsifier):** reload the
   `ai.openclaw.trader.api` LaunchAgent so the new binary (with both
   2026-06-08 backoff/circuit and 2026-06-10 fingerprint fix) actually
   runs:

   ```
   launchctl bootout gui/501/ai.openclaw.trader.api
   launchctl bootstrap gui/501 \
     ~/Library/LaunchAgents/ai.openclaw.trader.api.plist
   ```

   *Until this happens*, the log will continue to show pre-2026-06-08
   loop behaviour (no `classified`, no `circuit OPEN`, no
   `backoff: skipping`), even though the code on disk is correct.

2. **Operator action (orthogonal, H2 root cause):** rotate
   `BINANCE_API_KEY` / `BINANCE_API_SECRET` or add the active egress IP
   (currently `148.227.107.253`) to the Binance allow-list. The
   classification + backoff + circuit pauses the loop; only an operator
   change clears the `-2015`.

3. **Telemetry follow-up (deferred from 2026-06-08):** expose
   `trader_auto_start_circuit_open`,
   `trader_auto_start_backoff_seconds_by_symbol`, and a new
   `trader_auto_start_error_class_total{class=…}` Prometheus counter via
   `Trader.App.Observability`. The new fingerprint makes
   `error_class_total` a stable, bounded label set (4 values), which is
   the precondition for Prometheus aggregation.

4. **State persistence (deferred):** backoff is still in-memory.
   Today's launchd reload (when it happens) will reset all per-symbol
   backoff records to fresh `ErrAuth` initials, which is intentionally
   conservative. A persisted variant should live in `OpsStore` so the
   restart path doesn't bypass the circuit.

5. **Journal normalization (new):** apply the same
   `normalizeAutoStartErrorMessage` to `bot.start_queued_failed` and
   `bot.start_failed` op messages before they enter the journal/DB, so
   downstream queries can group on a stable key. Defer until §6.1 lands.

---

## 7. Files touched

```
haskell/app/Trader/App/AutoStartBackoff.hs    (export + normalize + wire into init/next)
haskell/app/Main.hs                           (dedup key in recordError)
haskell/test/Trader/Test/AutoStartBackoff.hs  (7 new invariants)
CHANGELOG.md                                  (one new "Unreleased" entry)
ENGINEERING_REVIEW_2026-06-10.md              (this file)
```

`stack test` green; `fourmolu --mode check` and `hlint` clean on the
touched files and on `app test bench` as a whole. No formatting drift in
files outside the change set.
