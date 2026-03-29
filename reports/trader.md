# Trader Reports

## 2026-03-27

### Findings
- Primary local data source: `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/bot-state-*.json`.
- Those artifacts were saved at `2026-03-28 01:09-01:10 America/Guayaquil`, so I truncated the review to the task cutoff `2026-03-27 23:44 America/Guayaquil` rather than counting post-midnight exits.
- Three completed trades had exited by the cutoff and all three lost money:
  - `ETCUSDT` short, `3m`, entered `2026-03-27 09:48 -05`, exited `2026-03-27 10:09 -05`, return `-0.02727%`, `TRAILING_STOP`
  - `SOLUSDT` long, `4h`, entered `2026-03-27 07:00 -05`, exited `2026-03-27 15:00 -05`, return `-0.01803%`, `TRAILING_STOP`
  - `BTCUSDT` short, `1d`, entered `2026-03-26 19:00 -05`, exited `2026-03-27 19:00 -05`, return `-0.08248%`, `STOP_LOSS`
- Day-scoped aggregate metrics at the cutoff: win rate `0.0%`, average return `-0.04260%`, compounded return `-0.12774%`.
- Same-day open exposure at the cutoff still concentrated in the already-familiar edge pathology:
  - `ATOMUSDT` long, entry edge `9.217%`, raw `openThreshold=3.197%`
  - `DOTUSDT` short, entry edge `106.66%`, raw `openThreshold=101.67%`
  - `DOGEUSDT` short, entry edge `100.34%`, raw `openThreshold=160.35%`
- The main decision problem remained implausible one-bar forecast magnitude. The `ETCUSDT` loser still matched the prior-day failure mode (`88.54%` edge against raw `46.05%` threshold), and the open `DOGEUSDT` / `DOTUSDT` positions were even less credible. But today’s new engineering issue was observability drift: saved bot-state continued surfacing raw threshold values above `1`, while the live decision logic had already been normalizing thresholds internally. That made it impossible to tell from the snapshot alone whether a large threshold was a configured value, an effective runtime value, or both.

### Research Notes
- The same literature from the 2026-03-26 review still fits today’s conditions: clipping or winsorizing extreme machine-learning forecasts improves robustness when raw model magnitudes become implausible. Source: Buncic, Caner, Matthies, “Pooling and Winsorizing Machine Learning Forecasts” (`https://www.sciencedirect.com/science/article/pii/S0169207024000640`).
- Trend-following evidence supports keeping directional momentum logic, but only with disciplined implementation and sane signal scaling. Source: Hurst, Ooi, Pedersen, “A Century of Evidence on Trend-Following Investing” (`https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026`).
- Volatility-managed trading work reinforces the same point: raw signal magnitude should not be trusted blindly when risk state changes or model scale drifts. Source: Moreira, Muir, “Volatility Managed Portfolios” (`https://ssrn.com/abstract=2773438`).
- Inference for today: the trading fix remains forecast clipping via the in-flight `50%` cap, while the code change worth making tonight is an explicit configured-vs-effective threshold boundary in runtime state so future reviews can verify whether that guardrail is actually being fed sane inputs.

### Hypotheses
- Keep the dirty absolute `50%` `EDGE_SPIKE` cap as the primary decision fix; today did not justify inventing a second overlapping gate.
- Add one narrow observability invariant at the bot-status boundary: every saved status/snapshot should expose both configured thresholds and normalized effective thresholds.
- Preserve backward compatibility by adding a nested `thresholds` block instead of mutating the existing top-level fields in a cron-sized change.

### Metrics
- Before the new observability patch (cutoff-scoped):
  - Completed trades: `3`
  - Completed-trade compounded return: `-0.12774%`
  - Open positions entered today at cutoff: `3`
  - Same-day order events by cutoff: `9`
- Counterfactual under the already-dirty `50%` edge cap using the same cutoff:
  - Blocked completed trade: `ETCUSDT`
  - Blocked open positions: `DOGEUSDT`, `DOTUSDT`
  - Completed trades: `2`
  - Completed-trade compounded return: `-0.10050%`
  - Open positions entered today at cutoff: `1` (`ATOMUSDT`)
- New explicit threshold-boundary examples after the code change:
  - `DOGEUSDT`: configured `1.6034807339`, effective `0.999999`
  - `DOTUSDT`: configured `1.0167245813`, effective `0.999999`

### Changes Made
- Added `SignalThresholdBoundary` and `mkSignalThresholdBoundary` to `haskell/app/Trader/SignalGates.hs`.
- Updated `haskell/app/Main.hs` so `/bot/status` and persisted bot snapshots emit:
  - `thresholds.configured.threshold/openThreshold/closeThreshold`
  - `thresholds.effective.threshold/openThreshold/closeThreshold`
- Added regression coverage in `haskell/test/TestMain.hs` to lock the configured-vs-effective threshold behavior.
- Updated `README.md` and `CHANGELOG.md` for the new bot-status/snapshot field.

### Validation Results
- Artifact replay used `python3` over the saved bot-state JSON files with an explicit cutoff at `2026-03-27 23:44 America/Guayaquil`.
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal test`
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal build`
- Result:
  - `cabal test` passed.
  - `cabal build` recompiled the changed path through `Trader.SignalGates`, `Main`, and the linked executables, but the final `trader-hs` link never returned in this sandbox after several minutes; no compiler error surfaced before the bounded run stopped waiting.
  - The required completion ping (`openclaw system event ...`) was attempted exactly once from Codex and failed because the local gateway closed with `1006 abnormal closure`.

### Remaining Risks
- Today’s review still depends on snapshot reconstruction rather than a persisted day ledger, and the available artifacts required manual truncation to the task cutoff because they were saved after midnight.
- The observability patch makes threshold normalization auditable, but it does not change PnL directly; actual trade improvement still depends on deploying the already-dirty `50%` entry-edge cap.
- If tomorrow still shows losses after the cap is live and auditable, the next focused experiment should inspect exit behavior on BTC/SOL rather than tightening entry filters further.

## 2026-03-26

### Findings
- Primary local data source: `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/bot-state-*.json`, saved at `2026-03-26 23:27-23:28 America/Guayaquil`.
- Four completed trades touched local date `2026-03-26` after timezone normalization, and every one of them came from the same `ETCUSDT` `3m` bot:
  - short, entered `2026-03-26 18:54 -05`, exited `2026-03-26 19:12 -05`, return `-0.0229%`, `TRAILING_STOP`
  - short, entered `2026-03-26 19:18 -05`, exited `2026-03-26 20:15 -05`, return `+0.0138%`, `TRAILING_STOP`
  - short, entered `2026-03-26 20:21 -05`, exited `2026-03-26 22:21 -05`, return `-0.1439%`, `TRAILING_STOP`
  - short, entered `2026-03-26 22:27 -05`, exited `2026-03-26 23:21 -05`, return `+0.0993%`, `TRAILING_STOP`
- Day-scoped aggregate metrics from those four trades: win rate `50.0%`, average return `-0.0134%`, compounded return `-0.05382%`.
- Same-day order flow was concentrated in the same failure cluster: `21` order events touched the day, all from `ETCUSDT` (`16`), `DOGEUSDT` (`3`), or `DOTUSDT` (`2`).
- Five positions were still open at save time and had entered on `2026-03-26`: `ATOMUSDT` long, `DOGEUSDT` short, `DOTUSDT` short, `ETCUSDT` short, and `SOLUSDT` short.
- The main engineering failure mode was an absolute edge-credibility gap in the existing `EDGE_SPIKE` filter. The `ETCUSDT` churn loop repeatedly opened shorts on an implied one-bar LSTM edge of about `88.75%` with `openThreshold=46.05%`; the existing relative guard only saw `1.927x` threshold, so it allowed the trade even though an `88%` move on a `3m` bar is not a credible live forecast. Fresh same-day `DOGEUSDT` and `DOTUSDT` openings were even more extreme at `100.33%` and `106.47%`.
- The measurable consequence was cost-dominated churn: two `ETCUSDT` winners were too small to offset two losers, and all four completed trades exited via the same trailing-stop path.

### Research Notes
- Forecast pooling work finds that clipping or winsorizing extreme machine-learning forecasts improves stability and guards against outlier magnitudes dominating decisions. Source: Buncic, Caner, Matthies, “Pooling and Winsorizing Machine Learning Forecasts” (`https://www.sciencedirect.com/science/article/pii/S0169207024000640`).
- Trend-following has long-run evidence, but implementation discipline matters more than raw signal magnitude. Source: Hurst, Ooi, Pedersen, “A Century of Evidence on Trend-Following Investing” (`https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026`).
- Volatility-managed signals improve when risk control reacts to changing volatility instead of trusting raw forecast amplitude. Source: Moreira, Muir, “Volatility Managed Portfolios” (`https://ssrn.com/abstract=2773438`).
- Inference from those sources plus today’s artifacts: keep the directional momentum logic, but clip clearly non-credible one-bar forecast magnitudes before they can create high-turnover churn.

### Hypotheses
- Add one narrow invariant to the shared entry gate: when `openThreshold > 0`, require `edge <= min(4 * normalizedOpenThreshold, 0.5)`.
- Reuse the existing `EDGE_SPIKE` mechanism instead of inventing a new ETC-specific rule or touching exit logic.
- Leave today’s credible higher-timeframe entries alone: `ATOMUSDT` (`8.14%` edge), `SOLUSDT` (`6.56%`), and `BTCUSDT` (`5.63%`) stay under the new cap.

### Metrics
- Before the new absolute cap:
  - Day-scoped completed trades: `4`
  - Day-scoped compounded return: `-0.05382%`
  - Day-scoped mean entry edge on completed trades: `88.75%`
  - Day-scoped mean edge/open-threshold ratio on completed trades: `1.927x`
  - Same-day fresh open positions: `5`
- After replaying the same artifact with the absolute `50%` edge cap layered onto the existing `EDGE_SPIKE` rule:
  - Blocked completed trades: `ETCUSDT x4`
  - Blocked fresh openings: `DOGEUSDT`, `DOTUSDT`, `ETCUSDT`
  - Day-scoped completed trades: `0`
  - Day-scoped compounded return: `0.0%`
  - Same-day fresh open positions: `2` (`ATOMUSDT`, `SOLUSDT`)
- This before/after remains an inference from saved snapshot state, not a DB-backed fill replay.

### Changes Made
- Added `maxCredibleSignalEdge = 0.5` to `haskell/app/Trader/SignalGates.hs` and applied it inside `signalEntryEdgeSpikeOk`.
- Added regression coverage in `haskell/test/TestMain.hs` for:
  - equality at the new `50%` cap
  - rejection above the cap
  - rejection of the observed `ETCUSDT`-style `88.7%` edge even though it is below the existing `4x threshold` multiplier
- Updated `README.md` and `CHANGELOG.md` for the new user-visible `EDGE_SPIKE` invariant.

### Validation Results
- Artifact replay used `python3` over the saved bot-state JSON files to enumerate same-day trades, open positions, order churn, entry edges, and the counterfactual blocked set under the new cap.
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal test`
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal build`
- Result:
  - `cabal test` passed.
  - `cabal build` recompiled the changed path through `Trader.SignalGates`, `Trader.Trading`, and `Main`; this shell again stopped emitting output after the final `trader-hs` link-stage output, so the compile path is verified but the final exit line remained inconclusive here.

### Remaining Risks
- No persisted Postgres `ops` replay was available, so the review still depends on bot-state reconstruction.
- The `50%` cap is intentionally conservative and evidence-backed for today’s liquid Binance symbols, but it is still a heuristic rather than a globally optimized calibration.
- The change does not address unrelated exit-timing questions; if future days show losses with sane entry magnitudes, the next cycle should inspect close rules rather than tightening entry caps further.

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
