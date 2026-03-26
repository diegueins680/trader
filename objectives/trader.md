# Trader Objectives

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
