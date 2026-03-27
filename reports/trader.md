# Trader Reports

## 2026-03-22

### Findings
- Primary local data source: `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/bot-state-*.json`, saved at `2026-03-22 23:18 America/Guayaquil`.
- Four completed trades touched local date `2026-03-22` after timezone normalization:
  - `AVAXUSDT` short, entered `2026-03-18 19:00 -05`, exited `2026-03-22 07:00 -05`, return `+4.3346%`, `TAKE_PROFIT`
  - `DOGEUSDT` long, entered `2026-03-21 11:00 -05`, exited `2026-03-22 15:00 -05`, return `-3.5081%`, `TRAILING_STOP`
  - `LINKUSDT` long, entered `2026-03-20 18:00 -05`, exited `2026-03-22 03:00 -05`, return `-2.7484%`, `STOP_LOSS`
  - `TRXUSDT` long, entered `2026-03-18 19:00 -05`, exited `2026-03-22 19:00 -05`, return `+1.2763%`, `SIGNAL`
- Day-scoped aggregate metrics from those four trades: win rate `50.0%`, average return `-0.1614%`, compounded return `-0.84296%`.
- Order-flow on the same local date contained 6 order events and 0 persisted `operations` rows in the snapshot payloads; 3 positions were still open at save time (`ATOMUSDT` long, `DOTUSDT` short, `SOLUSDT` short).
- The main engineering failure mode was invalid threshold scale in current live state. Current `latestSignal` snapshots show:
  - `DOGEUSDT`: `openThreshold=9.6053`, `closeThreshold=9.2821`
  - `DOTUSDT`: `openThreshold=1.0167`, `closeThreshold=1.0167`
  - `UNIUSDT`: `openThreshold=0.9890`, `closeThreshold=0.9890`
- Those values contradict the repo’s documented semantics that thresholds are fractional deadbands. At `threshold >= 1`, the engine is effectively asking for a `>= 100%` one-bar move before the threshold test changes state.
- The `DOGEUSDT` loser is the concrete measurable case:
  - saved LSTM edge at entry: `9.7573`
  - saved threshold at review time: `9.6053`
  - without threshold normalization, `EDGE_SPIKE` cannot classify that signal as pathological because the threshold itself is already inflated

### Research Notes
- Trend-following / time-series momentum remains viable over long samples, but its implementation quality matters. Source: Hurst, Ooi, Pedersen, “A Century of Evidence on Trend-Following Investing” (`https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026`).
- Momentum strategies are vulnerable to concentrated crash states and benefit from dynamic risk controls when volatility/panic rises. Source: Daniel, Moskowitz, “Momentum Crashes” (`https://www.nber.org/papers/w20439`).
- Volatility-managed signals improve when risk scaling reacts to volatility rather than letting raw signal magnitudes dominate. Source: Moreira, Muir, “Volatility Managed Portfolios” (`https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2773438`).
- Inference from those sources plus today’s artifacts: the bot should keep momentum-style direction logic, but the thresholds that gate it must stay on a sane fractional-return scale. Allowing `threshold >= 1` breaks that invariant and makes risk filters, edge filters, and close logic less meaningful.

### Hypotheses
- Normalize effective `openThreshold` / `closeThreshold` below `100%` everywhere they are consumed, not just at CLI boundaries.
- Reuse the existing `EDGE_SPIKE` gate rather than inventing a second DOGE-specific heuristic.
- Defer age-based exit changes until there is repeated evidence beyond today’s single `LINKUSDT` carry loss.

### Metrics
- Before threshold normalization:
  - Day-scoped: `trades=4`, `compound=-0.84296%`
  - Broader completed-trade snapshot: `trades=10`, `compound=-2.2640%`
- After replaying the same artifact with normalized thresholds plus the existing `EDGE_SPIKE` rule:
  - Day-scoped blocked trade: `DOGEUSDT`
  - Day-scoped: `trades=3`, `compound=2.7621%`
  - Broader snapshot blocked trades: `ADAUSDT`, `DOGEUSDT`, `SOLUSDT`
  - Broader snapshot: `compound=4.7271%`
- This before/after remains an inference from saved snapshot state, not a DB-backed fill replay.

### Changes Made
- Added `normalizeSignalThreshold` in `haskell/app/Trader/SignalGates.hs`.
- Applied threshold normalization to:
  - live latest-signal computation in `haskell/app/Main.hs`
  - threshold-factor history replay in `haskell/app/Main.hs`
  - backtest/simulation thresholds in `haskell/app/Trader/Trading.hs`
- Added regression coverage in `haskell/test/TestMain.hs` for:
  - pathological threshold normalization
  - `EDGE_SPIKE` rejection on the observed DOGE-style input
- Updated `README.md` and `CHANGELOG.md` for the threshold invariant.

### Validation Results
- Artifact replay used `python3` over the saved bot-state JSON files to enumerate same-day trades, threshold pathologies, and inferred blocked trades.
- `PATH=$HOME/.ghcup/bin:$PATH /Users/diegosaa/.ghcup/bin/cabal test` passed.
- `PATH=$HOME/.ghcup/bin:$PATH /Users/diegosaa/.ghcup/bin/cabal build` compiled through the modified executables during review, again surfacing the existing `Trader.SignalGates` missing-home-modules warning in the `optimize-equity` stanza; this shell did not emit a clean final completion line for the standalone `trader-hs` link step.

### Remaining Risks
- No persisted Postgres `ops` replay was available, so artifact analysis still depends on bot-state reconstruction.
- `LINKUSDT` remains an unaddressed stale-carry loser. If the next day shows the same multi-bar decay pattern, the next focused experiment should be a replayed age-based close rule, not another threshold change.
- Threshold normalization is intentionally conservative (`< 100%`) to restore semantics without inventing a tighter regime-specific cap from one day of data.

## 2026-03-21

### Findings
- Primary local data source: `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/bot-state-*.json`.
- After normalizing timestamps to `America/Guayaquil`, only 1 completed trade touched local date `2026-03-21`:
  - `ARBUSDT` short, entered `2026-03-20 23:00 -05`, exited `2026-03-21 15:00 -05`, return `-0.2105%`, `TRAILING_STOP`
- Today’s local decision set also contained 10 order attempts/messages on `2026-03-21` across `ARBUSDT`, `AVAXUSDT`, `DOGEUSDT`, `DOTUSDT`, `LINKUSDT`, `SOLUSDT`, and `TRXUSDT`.
- Because the bot-state artifact does not expose a day-isolated realized-PnL ledger, I used the broader completed-trade snapshot in the same files to identify repeated failure modes.
- That broader snapshot contains 6 completed trades across `ADAUSDT` (1), `ARBUSDT` (3), `ETCUSDT` (1), and `SOLUSDT` (1).
- Aggregate trade metrics from that artifact: win rate `33.3%` (`2/6`), average trade return `-0.2327%`, compounded result `-1.4331%`, median holding period `2.5` bars.
- Exit mix: `TRAILING_STOP=3`, `STOP_LOSS=1`, `TAKE_PROFIT=1`, `SIGNAL=1`.
- The loss pattern is fast reversal after entry, not slow bleed. Four of six trades closed within `2-4` bars and the two worst losses were:
  - `ADAUSDT` long, `12h`, `2026-03-17 00:00 UTC` to `2026-03-18 00:00 UTC`, return `-1.5751%`, `TRAILING_STOP`
  - `SOLUSDT` short, `4h`, `2026-03-17 16:00 UTC` to `2026-03-18 08:00 UTC`, return `-1.7347%`, `STOP_LOSS`
- Entry-edge ratios for the completed LSTM-only trades, computed as `abs(pred/current - 1) / openThreshold`, were:
  - `ADAUSDT`: `138.653x`
  - `SOLUSDT`: `14.269x`
  - `ARBUSDT`: `1.254x` to `1.480x`
  - `ETCUSDT`: `3.866x`

### Hypotheses
- LSTM-only entries with edge far above the configured open threshold are likely stale or outlier forecasts rather than tradeable momentum, and they are entering exhausted moves.
- A hard invariant on entry edge magnitude is safer than loosening stops because the observed losers reversed almost immediately after entry.
- Close-direction logic should remain independent of the new entry gate so the bot can still flatten risk even when the open signal is implausibly large.

### Metrics
- Today-only completed trade metric:
  - `completedTradesTouching2026-03-21=1`
  - `compound=-0.2105%`
- Broader artifact context before gate:
  - `trades=6`
  - `wins=2`
  - `losses=4`
  - `winRate=33.3%`
  - `avgReturn=-0.2327%`
  - `compound=-1.4331%`
- After applying the `4x open-threshold` edge-spike filter to the same completed-trade entries:
  - Blocked trades: `ADAUSDT`, `SOLUSDT`
  - Kept trades: `ARBUSDT x3`, `ETCUSDT x1`
  - `trades=4`
  - `avgReturn=0.4785%`
  - `compound=1.9122%`
- This before/after is an inference from the saved bot-state replay inputs, not a full persisted-ops replay.

### Changes Made
- Validated the in-flight `signalEntryEdgeSpikeOk` gate against today’s local bot-state artifact instead of rewriting unrelated dirty trading-engine code.
- Added unit-test coverage for the `4x open-threshold` invariant in `haskell/test/TestMain.hs`.
- Documented the new entry invariant in `README.md` and `CHANGELOG.md`.

### Validation Results
- Artifact analysis commands:
  - `python3` scripts over `haskell/.tmp/bot/.../bot-state-*.json` to enumerate trades, compute entry-edge ratios, and compare before/after compounded returns.
- Build/test commands run:
  - `cd haskell && cabal build`
  - `cd haskell && cabal test`
- Results:
  - `cabal build`: completed successfully with linked `optimize-equity` and `trader-hs` executables; Cabal emitted an existing `-Wmissing-home-modules` warning for `Trader.SignalGates` in the `optimize-equity` stanza.
  - `cabal test trader-tests --test-show-details=direct`: passed.

### Remaining Risks
- Confidence is limited by the pre-existing dirty tree in trading-engine files (`haskell/app/Main.hs`, `haskell/app/Trader/SignalGates.hs`, `haskell/app/Trader/Trading.hs`); I treated those edits as in-flight candidate work and only layered tests/docs on top.
- I could not query persisted Postgres `ops` rows directly because `psql`/`pg_isready` are not installed in this environment, so the review used bot-state snapshots instead of DB-backed execution history.
- The bot-state files were written on `2026-03-21`, but most completed trades in those snapshots were opened and closed before `2026-03-21`; only one completed trade touched the target local date.
- The before/after improvement is based on saved signal inputs and completed-trade filtering; a full engine replay against persisted fills would tighten confidence further.
