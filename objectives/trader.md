# Trader Objectives

## 2026-03-27

### Findings
- The freshest local bot-state artifacts were saved after midnight (`2026-03-28 01:09-01:10 America/Guayaquil`), so today’s review was explicitly truncated to the task cutoff `2026-03-27 23:44 America/Guayaquil` to avoid counting future exits.
- Three completed trades had exited by the cutoff and all three lost money:
  - `ETCUSDT` `3m` short, `2026-03-27 09:48 -05` to `10:09 -05`, return `-0.02727%`, `TRAILING_STOP`
  - `SOLUSDT` `4h` long, `2026-03-27 07:00 -05` to `15:00 -05`, return `-0.01803%`, `TRAILING_STOP`
  - `BTCUSDT` `1d` short, `2026-03-26 19:00 -05` to `2026-03-27 19:00 -05`, return `-0.08248%`, `STOP_LOSS`
- Day-scoped realized compound return at the cutoff was `-0.12774%`; wins/losses were `0/3`; average trade return was `-0.04260%`.
- Three positions had entered on the same local date and were still open at the cutoff: `ATOMUSDT` long, `DOTUSDT` short, and `DOGEUSDT` short.
- The dominant decision pathology remained the already-known implausible edge cluster (`ETCUSDT` completed at `88.54%` edge; `DOGEUSDT` and `DOTUSDT` open at `100.34%` and `106.66%`), but today exposed a separate engineering gap: saved bot-state still showed raw configured thresholds above `1`, while the runtime trading logic was already normalizing thresholds internally. That made the daily review ambiguous at exactly the point where threshold scale mattered.

### Hypotheses
- Do not stack a second trading heuristic on top of the in-flight absolute `50%` entry-edge cap from the dirty tree; today’s artifacts still primarily validate that existing fix.
- Treat the remaining gap as an observability invariant: persisted `/bot/status` snapshots must expose both configured and effective normalized thresholds so future reviews can verify what the engine actually used without reverse-engineering it from multiple fields.
- Preserve API compatibility by keeping the existing top-level threshold fields unchanged and adding an explicit nested threshold-boundary block instead of rewriting current consumers in one cron run.

### Metrics
- Before the new observability patch (cutoff-scoped):
  - `trades=3`
  - `winRate=0.0%`
  - `avgReturn=-0.04260%`
  - `compound=-0.12774%`
  - `openPositionsEnteredToday=3`
- Counterfactual under the already-dirty absolute `50%` edge cap, using the same cutoff:
  - Blocked completed trade: `ETCUSDT`
  - Blocked open positions: `DOGEUSDT`, `DOTUSDT`
  - `trades=2`
  - `compound=-0.10050%`
  - `openPositionsEnteredToday=1` (`ATOMUSDT`)
- New measurable boundary examples from saved state:
  - `DOGEUSDT openThreshold: 1.6034807339 -> effective 0.999999`
  - `DOTUSDT openThreshold: 1.0167245813 -> effective 0.999999`

### Changes Made
- Used Codex to add a pure `SignalThresholdBoundary` helper in `haskell/app/Trader/SignalGates.hs`.
- Updated `haskell/app/Main.hs` so `/bot/status` and persisted bot snapshots include `thresholds.configured` and `thresholds.effective` alongside the existing top-level threshold fields.
- Added regression coverage for the threshold-boundary behavior in `haskell/test/TestMain.hs`.
- Updated `README.md` and `CHANGELOG.md` for the new status/snapshot field.

### Validation Results
- Artifact replay used `python3` over the saved bot-state JSON files with an explicit cutoff at `2026-03-27 23:44 America/Guayaquil`.
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal test`
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal build`
- Result:
  - `cabal test` passed.
  - `cabal build` recompiled the affected executables and linked `optimize-equity`, but the final `trader-hs` link did not return before the bounded run stopped waiting; no compiler error surfaced before that stall.
  - The required `openclaw system event ...` completion ping was attempted exactly once from Codex and failed because the local gateway closed with `1006 abnormal closure`.

### Remaining Risks
- Today’s analysis still depends on bot-state reconstruction plus a manual cutoff because the available snapshots were saved after midnight and no DB-backed day ledger was available here.
- The primary decision fix for the implausible-edge cluster is still the uncommitted absolute `50%` cap already present in the dirty tree; today’s code change improves measurement and reviewability, not realized PnL by itself.
- The gateway abnormal closure means the auto-notify path is currently unreliable even though the code/test work completed.

## 2026-03-26

### Findings
- Today’s freshest local artifacts are the tenant bot snapshots under `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/`, saved at `2026-03-26 23:27-23:28 America/Guayaquil`.
- Four completed trades touched local date `2026-03-26`, all in `ETCUSDT` on the `3m` bot and all exited via `TRAILING_STOP`:
  - short `2026-03-26 18:54 -05` to `19:12 -05`, return `-0.0229%`
  - short `2026-03-26 19:18 -05` to `20:15 -05`, return `+0.0138%`
  - short `2026-03-26 20:21 -05` to `22:21 -05`, return `-0.1439%`
  - short `2026-03-26 22:27 -05` to `23:21 -05`, return `+0.0993%`
- Daily realized compound return from those four trades was `-0.05382%`; wins/losses were `2/2`; average trade return was `-0.01342%`.
- Five positions were still open at save time and had entered on the same local date: `ATOMUSDT` long, `DOGEUSDT` short, `DOTUSDT` short, `ETCUSDT` short, and `SOLUSDT` short.
- The dominant engineering failure mode was an absolute edge-credibility gap in the existing `EDGE_SPIKE` rule. All four `ETCUSDT` entries reused the same implied one-bar LSTM edge of about `88.75%` with `openThreshold=46.05%`, so the existing relative guard still passed at only `1.927x` threshold. Fresh same-day `DOGEUSDT` and `DOTUSDT` shorts were even larger at `100.33%` and `106.47%` implied edge.

### Hypotheses
- Treat entry magnitude as an invariant: when `openThreshold > 0`, the effective edge must satisfy `edge <= min(4 * normalizedOpenThreshold, 0.5)`.
- Keep the existing relative `4x` `EDGE_SPIKE` rule, but add an absolute `50%` credibility cap so high thresholds cannot legitimize obviously broken one-bar forecasts.
- Keep the change focused on entry sanitation rather than adding a new cooldown or exit heuristic; today’s completed churn and three of five new openings are explained by implausible forecast magnitude, not by exit timing.
- Preserve credible higher-timeframe trades: today’s saved `ATOMUSDT` (`8.14%`), `SOLUSDT` (`6.56%`), and prior `BTCUSDT` (`5.63%`) entry edges remain below the proposed cap.

### Metrics
- Day-scoped baseline from the local snapshot:
  - `trades=4`
  - `winRate=50.0%`
  - `avgReturn=-0.0134%`
  - `compound=-0.05382%`
- Order-flow and exposure context from the same artifact:
  - `orderEventsTouching2026-03-26=21`
  - `openPositionsEnteredToday=5`
  - `openPositionsWithEdge>50%=3` (`DOGEUSDT`, `DOTUSDT`, `ETCUSDT`)
- Inferred result after adding an absolute `50%` entry-edge cap and replaying the saved entries:
  - Blocked completed trades: `ETCUSDT x4`
  - Blocked fresh openings: `DOGEUSDT`, `DOTUSDT`, `ETCUSDT`
  - `trades=0`
  - `compound=0.0%`
  - `openPositionsEnteredToday=2` (`ATOMUSDT`, `SOLUSDT`)

### Changes Made
- Added an absolute `50%` credibility cap to `signalEntryEdgeSpikeOk` in `haskell/app/Trader/SignalGates.hs`.
- Added regression tests for the `50%` boundary and the observed `ETCUSDT`-style `88.7%` edge case.
- Updated `README.md` and `CHANGELOG.md` for the new `EDGE_SPIKE` invariant.

### Validation Results
- Artifact replay command:
  - `python3` over `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/bot-state-*.json`
- Haskell validation:
  - `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal test`
  - `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal build`
- Result:
  - `cabal test` passed.
  - `cabal build` recompiled the touched executables through `Trader.SignalGates`, `Trader.Trading`, and `Main`; this shell again did not return a clean final completion line for `trader-hs` after the last link-stage output, although it did surface no new compile errors.

### Remaining Risks
- The review still depends on bot-state reconstruction rather than persisted `ops`/fill tables because no DB client is available here.
- The new `50%` cap is an engineering sanity bound derived from today’s artifacts, not a full historical calibration study across every timeframe and market.
- The change only affects threshold-enabled entries; threshold-disabled sign-only flows still rely on their existing semantics.

## 2026-03-22

### Findings
- Today’s best local artifact is the tenant snapshot set under `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/`, saved at `2026-03-22 23:18 America/Guayaquil`.
- Four completed trades touched local date `2026-03-22`: `AVAXUSDT` short `+4.3346%` (`TAKE_PROFIT`), `DOGEUSDT` long `-3.5081%` (`TRAILING_STOP`), `LINKUSDT` long `-2.7484%` (`STOP_LOSS`), and `TRXUSDT` long `+1.2763%` (`SIGNAL`).
- Daily realized compound return from those four trades was `-0.84296%`; wins/losses were `2/2`.
- The dominant new failure mode was invalid threshold scale, not just stale edge magnitude. Current snapshot thresholds include `DOGEUSDT openThreshold=9.6053`, `DOTUSDT openThreshold=1.0167`, and `UNIUSDT openThreshold=0.9890`, which violates the intended fractional-deadband semantics.
- `DOGEUSDT` is the concrete pathological case: the saved entry edge was `9.7573`, so the existing `EDGE_SPIKE` gate only failed because the threshold itself was already nonsensical.

### Hypotheses
- Treat live/backtest thresholds as an invariant: effective `openThreshold` / `closeThreshold` must remain in `[0, 1)`.
- Preserve the existing `EDGE_SPIKE` gate, but normalize thresholds first so a bad combo/profile cannot neutralize that guard by setting `threshold >= 1`.
- Keep the change focused on threshold sanitation rather than adding another overlapping exit heuristic; today’s measured loser was preventable at entry.

### Metrics
- Day-scoped baseline from the local snapshot:
  - `trades=4`
  - `winRate=50.0%`
  - `avgReturn=-0.1614%`
  - `compound=-0.84296%`
- Threshold pathology in the saved live snapshot:
  - `openThreshold >= 1`: `DOGEUSDT`, `DOTUSDT`
  - `openThreshold >= 0.95`: `DOGEUSDT`, `DOTUSDT`, `UNIUSDT`
- Inferred result after clamping thresholds below `100%` and replaying the existing `EDGE_SPIKE` filter on the same saved entries:
  - Blocked trade: `DOGEUSDT`
  - `trades=3`
  - `winRate=66.7%`
  - `avgReturn=0.9542%`
  - `compound=2.7621%`
- Broader completed-trade context from the same artifact:
  - Before: `trades=10`, `compound=-2.2640%`
  - After blocking `ADAUSDT`, `DOGEUSDT`, and `SOLUSDT` under normalized thresholds plus `EDGE_SPIKE`: `compound=4.7271%`

### Changes Made
- Added `normalizeSignalThreshold` and applied it to effective live/backtest thresholds plus threshold-factor history replay.
- Preserved the in-flight `EDGE_SPIKE` entry gate and made it operate on normalized thresholds.
- Added regression tests for pathological threshold normalization and the observed `DOGEUSDT`-style spike case.
- Updated `README.md` / `CHANGELOG.md` for the user-facing threshold invariant.

### Validation Results
- Artifact replay command:
  - `python3` over `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/bot-state-*.json`
- Haskell validation:
  - `PATH=$HOME/.ghcup/bin:$PATH /Users/diegosaa/.ghcup/bin/cabal test`
- Result:
  - `cabal test` passed.
  - `cabal build` reached executable compilation during verification but this shell did not surface a clean completion line after `Main` compilation; the earlier build output still showed the pre-existing `Trader.SignalGates` missing-home-modules warning in the `optimize-equity` stanza.

### Remaining Risks
- The day review still uses snapshot reconstruction rather than persisted `ops`/fill tables because no DB client is available in this environment.
- `LINKUSDT` remains a separate carry-trade loss that threshold normalization does not address; if that pattern persists, the next iteration should test tighter age-based exits against saved replays before touching defaults.
- The threshold cap is an engineering invariant derived from the repo’s fractional-deadband semantics and today’s artifacts, not a full historical optimizer audit.

## 2026-03-21

### Findings
- Today’s usable local execution artifact was the bot snapshot set under `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/`.
- Exact local-date normalization matters: only 1 completed trade in that artifact touched `2026-03-21 America/Guayaquil` (`ARBUSDT`, `-0.2105%`, trailing-stop exit at `2026-03-21 15:00 -05`).
- The same snapshot recorded 10 order decisions on `2026-03-21`, which is the best available local representation of today’s decision flow without direct Postgres `ops` access.
- The day’s dominant failure mode was exhausted-move entry: extreme LSTM edge relative to `openThreshold`, followed by reversal within `2-4` bars.

### Hypotheses
- Keep the new `EDGE_SPIKE` entry invariant and treat it as a first-line no-trade condition for pathological LSTM predictions.
- Preserve exit independence: close signals must still flatten risk even if an entry would now be blocked.

### Metrics
- Today-only completed trade baseline: `1` trade touching `2026-03-21`, `-0.2105%`.
- Broader bot-state context: `6` trades, `33.3%` win rate, `-1.4331%` compounded.
- Edge-spike filtered inference: `4` trades, `1.9122%` compounded.

### Changes Made
- Added regression coverage for `signalEntryEdgeSpikeOk`.
- Updated user-facing docs for the `4x open-threshold` entry gate.

### Validation Results
- Completed local verification commands:
  - `cd haskell && cabal build`
  - `cd haskell && cabal test`
- Result: both commands succeeded; the test suite passed and build output included the pre-existing `Trader.SignalGates` missing-home-modules warning in the `optimize-equity` stanza.

### Remaining Risks
- Persisted `ops`/fill history was not directly queryable from this shell because no `psql` client is installed.
- Today’s artifacts are snapshot-oriented rather than a day-scoped realized-trade ledger, so same-day analysis depends on timestamp normalization and snapshot reconstruction.
- The core gate implementation was already present in the dirty tree before this review started, so this cycle validated and documented it rather than introducing a second overlapping trading-engine change.

### Next Objectives
- Re-run the review against persisted `ops` rows or `/ops` API output once a DB client is available, and confirm that the saved bot-state trade set matches actual fills.
- Add lightweight instrumentation for blocked entry reasons (`EDGE_SPIKE`, `TREND_FILTER`, `MAX_VOLATILITY`, `VOL_TARGET_WARMUP`) so future daily reviews can measure gate hit rates without reconstructing from snapshots.
- Build a narrow replay harness from saved bot-state arrays (`prices`, `openTimes`, `lstmPredNext`) so daily before/after gate comparisons can be executed automatically.
