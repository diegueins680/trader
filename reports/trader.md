# Trader Reports

## 2026-04-06 (23:13 trading review)

### Findings
- Primary local data source: `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/bot-state-*.json`, replayed for local date `2026-04-06 America/Guayaquil` with task cutoff `2026-04-06 23:13 -05`.
- Only one completed trade touched the bounded window:
  - `UNIUSDT` `1h` long, adopted at `2026-04-06 12:00 -05`, exited `2026-04-06 14:00 -05`, return `+0.02320%`, `exitReason=SIGNAL`, `entrySource=adopted`, `provenance=startup_adopted_position`
- One carry position remained open at the cutoff:
  - `BTCUSDT` `1d` short, entry `2026-04-05 19:00 -05`, MTM `+0.08742%`, entry regime `high-vol`, efficiency `0.05622`
- The important pathology was in the review tooling, not in live entry logic. Before this patch, the replay counted `nonDirectionalOrderAttempts=2`, which would imply two fresh low-quality entry attempts. But today’s two `UNIUSDT` order events were both `SELL` close/flatten flow on the adopted long:
  - `2026-04-06 12:16:57 -05`: ack-only `Order sent.`
  - `2026-04-06 14:00:51 -05`: `No order: already flat.`
- The live engine therefore looked worse than it actually was. Today did not show two new non-directional entries slipping through the gate; it showed an inherited position being flattened in a non-directional regime while the review layer mislabeled that as fresh entry intent.

### Research Notes
- Trend following still has strong long-run support across asset classes and macro environments, which argues against retuning the alpha engine from one small profitable adopted-position exit. Source: Hurst, Ooi, Pedersen, *A Century of Evidence on Trend-Following Investing* (`https://research.cbs.dk/en/publications/a-century-of-evidence-on-trend-following-investing`).
- Volatility-managed portfolio research supports cutting risk more aggressively when uncertainty/volatility rises faster than expected return. That matches today’s engineering interpretation: close/flatten flow in weak-direction or uncertain state should be read as risk reduction, not as evidence of a bad fresh entry. Source: Moreira, Muir, *Volatility Managed Portfolios* (`https://www.nber.org/papers/w22208`).
- Inference for this run: the strategy lesson is not “add another entry heuristic.” It is “preserve semantic separation between seeking risk and reducing risk,” especially for startup-adopted positions where local entry provenance is weaker.

### Hypotheses
- The daily review should satisfy a simple invariant: non-directional *attempt* metrics count only opening/additive flow, never close/flatten flow.
- Startup-adopted positions that close today should remain visible as carry-management outcomes, but they should not be allowed to inflate the metric used to judge whether the live entry gate failed.
- Today’s sample does not justify a new live trading-rule change beyond the existing low-directionality gate; the highest-leverage fix is report correctness.

### Metrics
- Pre-fix replay at the start of this pass:
  - `completedTrades=1`
  - `completedCompoundPct=+0.02320%`
  - `openPositionsEnteredToday=0`
  - `openPositionsCarriedIn=1`
  - `sameDayOrderEvents=2`
  - `ackOnlyOrderEvents=1`
  - `nonDirectionalOrderAttempts=2` **(false-positive entry count)**
  - `fillEvidenceGaps=1`
- Post-fix replay on the same bounded window:
  - `nonDirectionalOrderAttempts=0`
  - `nonDirectionalExitOrFlattenEvents=2`
  - `nonDirectionalUnknownRoleEvents=0`
- Explicit order-flow classification after the patch:
  - `UNIUSDT` `2026-04-06 12:16:57 -05` `SELL` -> `exit_or_flatten` via `completed_trade_entry_side`
  - `UNIUSDT` `2026-04-06 14:00:51 -05` `SELL` -> `exit_or_flatten` via `message_already_flat`

### Changes Made
- Used Codex to add `flowRole` classification to every saved order event in `haskell/scripts/review_bot_day.py`.
- Split non-directional event reporting into:
  - `nonDirectionalOrderAttempts` for `entry_or_add`
  - `nonDirectionalExitOrFlattenEvents` for closes/flattening
  - `nonDirectionalUnknownRoleEvents` when replay evidence is insufficient
- Extended markdown/json reporting so order rows surface both `flowRole` and `flowRoleEvidence`.
- Added deterministic regression coverage in `haskell/scripts/test_review_bot_day.py` for the adopted-`UNIUSDT` close/flatten case.
- Updated `README.md` and `CHANGELOG.md` for the new report semantics.

### Validation Results
- `python3 -m py_compile haskell/scripts/review_bot_day.py haskell/scripts/test_review_bot_day.py` passed.
- `python3 -m unittest haskell/scripts/test_review_bot_day.py` passed (`8` tests).
- `python3 haskell/scripts/review_bot_day.py --date 2026-04-06 --timezone America/Guayaquil --end-local 2026-04-06T23:13:00-05:00 --format json` passed and now reports zero non-directional entry attempts for today while exposing two non-directional exit/flatten events.

### Remaining Risks
- Today’s `UNIUSDT` exit still has ack-only fill evidence, so end-to-end execution provenance is not yet as strong as an exchange-native fill log.
- `BTCUSDT` remains open in a high-vol / low-efficiency regime; if repeated reviews show startup-adopted exposure lingering into these states, the next focused experiment should be a dedicated adoption-policy rule rather than another generic entry filter.
- Snapshot drift after the review window remains present (`snapshotsUpdatedAfterWindow=11`), so bounded replays still need cutoff-aware handling.

## 2026-04-06

### Findings
- This bounded CTO pass was an engineering-lane audit, not a new trading-rule experiment. I read the current repo-native autoloop contract (`README.md`, `scripts/autoloop-forever.sh`, `scripts/autoloop-forever.mjs`), the current objective/report files, and the latest data report section (`2026-04-05`) before making lane-level changes.
- `mission.md` and `org.md` were absent at the start of the pass, which meant future "resume" requests had no explicit mission or ownership contract to read.
- The repo-side forever contract was already materially improved in the dirty tree:
  - stale PID detection in `./scripts/autoloop-forever.sh status`
  - status heartbeat updates in `scripts/autoloop-forever.mjs`
  - a repo-native LaunchAgent installer in `scripts/install-autoloop-launchagent.sh`
  - matching contract tests in `test/autoloop.test.mjs`
- The concrete repo defect still preventing a usable forever operator path was mundane but real: `scripts/install-autoloop-launchagent.sh` was not executable, so the README-documented install/status path failed before doing any useful work.
- Runtime truth at audit time:
  - repo-local forever supervisor is alive and heartbeating
  - runner state is `blocked`
  - blocker is the dirty worktree
  - LaunchAgent keepalive is not installed
- The latest data report (`2026-04-05`) still stands as the latest trading-engine truth: low-directionality entries were the actual live decision pathology. Today’s pass did not overturn that; it made the standing autoloop mission explicit and operationally honest.

### Metrics
- `npm run test:autoloop`: `33/33` passing
- `./scripts/install-autoloop-launchagent.sh status`:
  - LaunchAgent plist: not installed
  - repo autoloop status: `live=true`, `state=blocked`, fresh `heartbeatAt`
- Current blocker set reported by the runner:
  - dirty tracked files in Haskell trading logic/docs/tests
  - untracked `scripts/install-autoloop-launchagent.sh`
  - untracked `haskell/.build-commit`

### Changes Made
- Fixed the executable bit on `scripts/install-autoloop-launchagent.sh`.
- Created `mission.md` and `org.md` to establish a resumable repo mission/org contract.
- Created persistent `.openclaw` CTO lane files:
  - `.openclaw/trader-firm-cto.repo-autoloop-forever.objective.md`
  - `.openclaw/trader-firm-cto.repo-autoloop-forever.report.md`
- Appended this report and rewrote the current objective so the standing lane is explicit: `trader-firm-cto.repo-autoloop-forever`.

### Validation Results
- `npm run test:autoloop` passed.
- `./scripts/install-autoloop-launchagent.sh status` passed after the permission fix.
- Repo-local autoloop status now reports fresh-heartbeat liveness truth instead of blindly trusting stale `status.json`.

### Engineering Truth
- The forever runner is **live right now**, but only in the narrow sense that the supervisor process is up and reporting a heartbeat.
- The forever lane is **not healthy enough to call done** because it is blocked by the dirty worktree and is not yet LaunchAgent-managed.
- That means the current blocker is operational, not conceptual: the repo contract is now mostly in place, but the machine-level enablement and worktree hygiene are not.

### Blocker Ownership
- **CTO lane owner:** closed the repo-side executable-bit gap and wrote the missing mission/org/.openclaw memory.
- **Operator owner (Diego):** must decide how to resolve the dirty worktree and whether to install the LaunchAgent keepalive now that the installer path is usable.

### Remaining Risks
- If the current manual/live supervisor dies, there is no installed LaunchAgent to bring it back automatically.
- Because the dirty worktree remains intentional or unresolved, the running supervisor will keep blocking bounded cycles rather than executing improvements.
- Until those two conditions are cleared, the lane should be reported as live-but-blocked, not green.

## 2026-04-05

### Findings
- Primary local data source: `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/bot-state-*.json`, replayed for local date `2026-04-05 America/Guayaquil` with task cutoff `2026-04-05 23:07 -05`.
- The raw Binance trade dump was unavailable here because `tmp/today-binance-trades.raw` reported `Missing BINANCE_API_KEY`, so the review used persisted bot-state artifacts instead of exchange fill rows.
- No completed trades exited before the cutoff and no positions were open at the cutoff:
  - `completedTrades=0`
  - `openPositionsEnteredToday=0`
  - `openPositionsCarriedIn=0`
- Same-day decision intent was limited to two ack-only order attempts:
  - `BTCUSDT` `1d` `SELL` at `2026-04-05 19:00 -05`, price `69212.2`, `status=NEW`, `executedQty=0`
  - `UNIUSDT` `1h` `BUY` at `2026-04-05 22:00 -05`, price `3.171`, `status=NEW`, `executedQty=0`
- Both attempted entries were structurally low-directionality decisions:
  - `BTCUSDT`: `high-vol`, efficiency `0.06009`, realized vol `2.12952%`, net return over lookback `-2.37434%`
  - `UNIUSDT`: `chop`, efficiency `0.19639`, realized vol `0.46675%`, net return over lookback `+1.79775%`, HMM regime probs `mr=0.89790`, `trend=0.04494`, `highVol=0.05717`
- The repeated engineering pattern now spans multiple reviews. Earlier reports already showed `BNBUSDT` (`2026-04-03`) plus `ATOMUSDT` / `LINKUSDT` (`2026-03-31`) opening or attempting in low-efficiency `chop` / `range-drift` states. Today added two more attempts from the same class, but with zero realized fills.
- The actionable failure mode was not exit logic or fill attribution today; it was allowing fresh directional intent after the path had already become non-directional, while saved diagnostics were still too weak to audit that systematically from persisted order events.

### Research Notes
- Trend following still has strong long-run support, but that support is conditional on implementation quality and state discrimination. Hurst, Ooi, and Pedersen’s long-horizon evidence argues for keeping a trend engine, not for forcing it through chop. Source: https://research.cbs.dk/en/publications/a-century-of-evidence-on-trend-following-investing
- Volatility-managed portfolio research points in the same direction: after volatility shocks, exposure should fall faster than one-step expected return rises. That supports a control-layer veto in high-vol / weak-direction states before adding more micro-heuristics. Source: https://www.nber.org/papers/w22208
- Kaufman-efficiency / choppiness-style methods are directly relevant here: when the realized path meanders with low efficiency, large forecast magnitude is more likely a noisy edge estimate than a tradeable directional move.
- Inference for this run: the right trading change is a narrow low-directionality veto layered after direction selection, plus persisted diagnostics so the daily review can prove the gate is hitting the intended setups.

### Hypotheses
- Add one explicit invariant to directional entries: if recent realized path efficiency is too low, hold flat regardless of raw predicted edge.
- Keep the change narrow and deterministic:
  - `NON_DIRECTIONAL_CHOP` when 24-bar efficiency `<= 0.25`
  - `NON_DIRECTIONAL_MR` when efficiency `<= 0.40` and mean-reversion probability is dominant by at least the existing `regimeBankHysteresis` gap
- Persist the gate inputs (`efficiency`, `realizedVolPct`, regime leader/gap, veto reason) into `latestSignal` and order artifacts so future reviews can validate the live decision boundary from saved state instead of reconstructing it ad hoc.

### Metrics
- Bounded replay summary (`2026-04-05 00:00-23:07 -05`):
  - Completed trades: `0`
  - Completed-trade compounded return: `0.0%`
  - Open positions entered today: `0`
  - Open positions carried in: `0`
  - Same-day order events: `2`
  - Ack-only order events: `2`
  - Non-directional order attempts: `2`
  - Active-trade fill-evidence gaps: `0`
  - Snapshots updated after review window: `11`
- Counterfactual replay under the new gate:
  - blocked today: `2/2` attempted entries
  - `BTCUSDT`: `NON_DIRECTIONAL_CHOP`
  - `UNIUSDT`: `NON_DIRECTIONAL_CHOP`
- Historical spot checks replayed against the current saved artifacts:
  - `BNBUSDT` `2026-04-03` short candidate: blocked, efficiency `0.02106`
  - `ATOMUSDT` `2026-03-31` long candidate: blocked, efficiency `0.16509`
  - `LINKUSDT` `2026-03-31` long candidate: blocked, efficiency `0.16916`
  - aggregate spot-check coverage: `3/3` blocked

### Changes Made
- Added `DirectionalitySnapshot` + `signalDirectionalitySnapshot` in `haskell/app/Trader/SignalGates.hs` to compute a 24-bar directionality diagnostic from realized path efficiency, realized volatility, z-score, and optional HMM regime probabilities.
- Inserted a new post-direction gate in the shared signal pipeline so live/backtest entries are vetoed as `NON_DIRECTIONAL_CHOP` or `NON_DIRECTIONAL_MR` before order execution.
- Extended `haskell/app/Main.hs` so `latestSignal` and `BotOrderEvent` persist the new directionality snapshot into saved bot-state artifacts.
- Extended `haskell/scripts/review_bot_day.py` to replay/report those diagnostics directly and to count `nonDirectionalOrderAttempts` even for older snapshots via fallback reconstruction.
- Added Haskell regression tests for chop, MR-drift, and clean-trend cases, and Python regressions for report-level non-directional-order counting.
- Updated `README.md` and `CHANGELOG.md` for the new live/backtest rule and diagnostics surface.

### Validation Results
- `python3 -m py_compile haskell/scripts/review_bot_day.py haskell/scripts/test_review_bot_day.py` passed.
- `python3 -m unittest haskell/scripts/test_review_bot_day.py` passed (`7` tests).
- `python3 haskell/scripts/review_bot_day.py --date 2026-04-05 --timezone America/Guayaquil --end-local 2026-04-05T23:07:00-05:00 --format json` passed and now emits per-order directionality plus `nonDirectionalOrderAttempts=2`.
- Counterfactual replay over today’s saved order attempts plus three previously reviewed low-efficiency examples reported:
  - `blocked_today=2/2`
  - `blocked_historical_checks=3/3`
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal test` passed.
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal build exe:trader-hs` recompiled through `Main`, but the final link did not finish before the bounded timeout; no compiler error surfaced before termination.

### Remaining Risks
- Because Binance trade export was unavailable in this shell, the day review still depends on bot-state reconstruction rather than exchange-native fill history.
- The low-directionality thresholds are intentionally conservative engineering bounds derived from repeated reviewed failures, not a full-symbol historical optimization study.
- Persisted HMM regime probabilities were unavailable for the `BTCUSDT` attempt, so that veto used realized path diagnostics only.
- The `trader-hs` link step remains the slowest part of bounded verification in this sandbox, so end-to-end executable completion is still less certain than the compile path and passing tests.

## 2026-04-03

### Findings
- Primary local data source: `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/bot-state-*.json`, replayed for local date `2026-04-03 America/Guayaquil` with task cutoff `2026-04-03 23:37 -05`.
- Those live tenant snapshots continued updating after the task timestamp while this review ran, so the metrics below are anchored to the captured cutoff replay from the task window rather than to a later rerun after midnight.
- One completed trade exited before the cutoff:
  - `BNBUSDT` short, `2h`, entered `2026-04-03 17:00 -05`, exited `2026-04-03 21:00 -05`, return `-0.00730%`, `SIGNAL`
- No same-day or carried-in positions were still open at the cutoff:
  - `openPositionsEnteredToday=0`
  - `openPositionsCarriedIn=0`
- Same-day order activity was minimal and still lacked direct fill proof:
  - ack-only `Order sent.` record at `2026-04-03 18:30:28 -05` with `status=NEW`, `executedQty=0`, `quantity=0.14`
  - `No order: already flat.` record at `2026-04-03 21:00:53 -05`
  - `fillEvidenceGaps=1`
- The observed regime was weak-directional to choppy rather than strongly trending. The `BNBUSDT` entry replayed as `range-drift` (`efficiency=0.36259`, `realizedVolPct=0.65850`, `zScore=-1.18733`) and the exit replayed as `chop` (`efficiency=0.10368`, `realizedVolPct=0.47985`, `zScore=-0.33109`). Saved regime probabilities at review time leaned mean-reversion (`mr=0.82061`, `trend=0.08102`).
- The main engineering failure mode was review leakage, not signal selection. The bounded replay at `23:37 -05` now reports `snapshotsUpdatedAfterWindow=8`, proving that several tenant snapshots were updated after the task cutoff and that an unbounded day review could silently incorporate post-cutoff state.

### Research Notes
- Long-run trend following remains a defensible core strategy, but the evidence argues for applying it conditionally rather than forcing it through every market texture. Hurst, Ooi, and Pedersen report positive average time-series-momentum returns across many macro environments, which supports keeping the trend engine but not retuning it from one tiny losing chop trade. Source: https://research.cbs.dk/en/publications/a-century-of-evidence-on-trend-following-investing
- Volatility-management research supports using regime/risk controls as a separate layer over alpha logic. Moreira and Muir show that after volatility shocks, optimal exposure initially falls because volatility rises faster than expected return. That points toward control-layer improvements before heuristic micromanagement. Source: https://www.nber.org/papers/w22208
- Recent crypto-specific regime work also points toward volatility-structure plus normalized-momentum classification as a robust control layer rather than a direct forecasting model. Banerjee’s 2025/2026 SSRN note frames regime detection as an input to downstream trading/risk systems, which matches today’s need for better bounded replay and regime-aware diagnostics. Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5920642
- Inference for this run: the most relevant strategy method for today’s `range-drift -> chop` sample is stronger regime discrimination, potentially including an explicit flat/no-trade path when efficiency stays low. But the data quality problem was larger than the alpha problem today, so the implemented change stayed in replay correctness.

### Hypotheses
- The correct cron-sized improvement for today is a bounded-review correctness patch, not a live-rule change.
- A daily engineering review should make two things explicit whenever it runs before midnight:
  - whether any saved snapshots were updated after the requested review window
  - what positions were actually open at the cutoff, even if they closed later in the terminal snapshot
- If future confirmed-fill days continue to show low-efficiency `range-drift/chop` entries, the next strategy experiment should be a replayed regime gate or alternate-policy switch for low-direction states.

### Metrics
- Bounded replay summary (`2026-04-03 00:00-23:37 -05`):
  - Completed trades: `1`
  - Completed-trade compounded return: `-0.00730%`
  - Completed-trade average return: `-0.00730%`
  - Open positions entered today: `0`
  - Open positions carried in: `0`
  - Same-day order events: `2`
  - Ack-only order events: `1`
  - Active-trade fill-evidence gaps: `1`
  - Snapshots updated after review window: `8`
- Window-leakage details:
  - symbols updated after `2026-04-03 23:37 -05`: `ADAUSDT`, `ARBUSDT`, `BTCUSDT`, `DOGEUSDT`, `ETCUSDT`, `FILUSDT`, `UNIUSDT`, `XRPUSDT`
- Trade-level regime detail:
  - `BNBUSDT`: `range-drift -> chop`, entry efficiency `0.36259`, exit efficiency `0.10368`

### Changes Made
- Extended `haskell/scripts/review_bot_day.py` with `--end-local` so bounded daily reviews no longer need ad hoc truncation logic outside the tool.
- Reconstructed window-end open positions from the saved `positions` vector at the cutoff, which preserves same-day open exposure even when the terminal snapshot later shows the trade closed.
- Added `entry_index_before_window` carry classification plus `snapshotsUpdatedAfterWindow` reporting, and withheld post-window latest-signal fields for those future-updated rows.
- Added deterministic regression coverage in `haskell/scripts/test_review_bot_day.py` for the cutoff-open/later-close case.
- Updated `README.md` and `CHANGELOG.md` for the new review-tool behavior.

### Validation Results
- `python3 -m py_compile haskell/scripts/review_bot_day.py haskell/scripts/test_review_bot_day.py` passed.
- `python3 -m unittest haskell/scripts/test_review_bot_day.py` passed (`3` tests).
- `python3 haskell/scripts/review_bot_day.py --date 2026-04-03 --timezone America/Guayaquil --format json` passed.
- `python3 haskell/scripts/review_bot_day.py --date 2026-04-03 --timezone America/Guayaquil --end-local 2026-04-03T23:37:00-05:00 --format json` passed and exposed `snapshotsUpdatedAfterWindow=8`.
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal build` passed.
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal test` passed.
- No separate backtest was run because the implemented change is a replay/observability fix rather than a trading-rule change.

### Remaining Risks
- The replay is still reconstructive: it can identify cutoff-open positions and future-updated snapshots, but it cannot prove fills when saved order history only shows ack-level `NEW` responses.
- Intrabar state between the last saved bar open and the exact cutoff minute is still approximated by the snapshot data available at review time.
- The next live strategy change should wait for repeated, fill-backed low-efficiency entries; today’s one tiny loss is not enough to justify that retune.

## 2026-03-31

### Findings
- Primary local data source: `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/bot-state-*.json`, replayed for local date `2026-03-31 America/Guayaquil`.
- One completed trade touched the target date after timezone normalization:
  - `ADAUSDT` long, `2h`, entered `2026-03-31 13:00 -05`, exited `2026-03-31 23:00 -05`, return `+0.12177%`, `SIGNAL`
- Two same-day entries were still open in the snapshot:
  - `ATOMUSDT` long, `12h`, entry `2026-03-31 07:00 -05`, MTM `+0.03302%`
  - `LINKUSDT` long, `1h`, entry `2026-03-31 13:00 -05`, MTM `+0.03470%`
- All three entries were taken in low-directional `chop` conditions. The replayed 24-bar efficiency ratios were `0.0213` (`ADA`), `0.2372` (`ATOM`), and `0.1208` (`LINK`), and the saved latest regime probabilities were consistently low on `trend` and high on `mr` where present.
- Same-day order-flow evidence was thin and partly ambiguous:
  - `sameDayOrderEvents=4`
  - `ackOnlyOrderEvents=3`
  - `fillEvidenceGaps=2`
- The most important engineering lesson from today is that the persisted artifacts still show intent more reliably than confirmed execution. `ADAUSDT` and `ATOMUSDT` both had ack-only same-day order records (`status=NEW`, `executedQty=0`) while the replay also showed an active/completed trade, so any strategy diagnosis from this day needs to be treated as provisional until fill provenance is first-class in the saved state.

### Research Notes
- In low-directional markets, trend systems usually improve more from regime discrimination than from threshold micro-tuning. Kaufman-style efficiency-ratio thinking points the same way as today’s replay: low directional efficiency is usually noise/chop, not strong trend continuation.
- Choppiness-index / ADX-style filters are standard engineering tools for this exact condition: when directionality is weak and mean-reversion probability is high, either route to a mean-reversion policy or stay flat rather than forcing trend entries.
- Volatility- and cost-aware trading research still matters here, but today’s realized edge was small enough that measurement error was the bigger risk than missing an obvious alpha opportunity.
- Inference for this run: do not overfit live rules to one mildly positive chop day. First fix observability so future strategy changes are reacting to trustworthy fill-attributed evidence.

### Hypotheses
- The correct immediate change is a review/diagnostics improvement, not a live-trading rule change.
- A daily engineering review should be able to answer four deterministic questions from persisted artifacts alone: what trades completed, what positions remain open, what regimes those entries/exits occurred in, and where order intent still lacks fill evidence.
- Once that observability invariant is in place, repeated low-efficiency chop days should be evaluated with a replayed trend-entry gate rather than by manually eyeballing individual symbols.

### Metrics
- Day-scoped replay summary:
  - Completed trades: `1`
  - Completed-trade compounded return: `+0.12177%`
  - Completed-trade average return: `+0.12177%`
  - Open positions entered today: `2`
  - Same-day order events: `4`
  - Ack-only order events: `3`
  - Active-trade fill-evidence gaps: `2`
- Regime diagnostics at entry:
  - `ADAUSDT`: `chop`, efficiency `0.0213`
  - `ATOMUSDT`: `chop`, efficiency `0.2372`
  - `LINKUSDT`: `chop`, efficiency `0.1208`

### Changes Made
- Used Codex to add `haskell/scripts/review_bot_day.py`.
- The script reconstructs one local trading day from persisted `bot-state-*.json` snapshots, outputs completed/open trades plus same-day order events, classifies the latest regime with explicit 24-bar thresholds, and flags ack-only order events that still lack direct fill evidence for active same-day trades.
- Updated `README.md` and `CHANGELOG.md` to document the new review utility and its output.

### Validation Results
- `python3 -m py_compile haskell/scripts/review_bot_day.py` passed.
- `python3 haskell/scripts/review_bot_day.py --date 2026-03-31 --timezone America/Guayaquil` passed and reproduced the expected summary (`completed=1`, `open_entered_today=2`, `ack_only=3`, `fill_gaps=2`).
- Codex also ran `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal build` and `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal test trader-tests --test-show-details=direct`; both completed successfully in the agent log.

### Remaining Risks
- The replay utility improves explainability, but it does not yet close the execution-attribution gap because the underlying snapshots still do not persist enough direct fill provenance.
- Today’s sample is too small and too mildly positive to justify a live strategy retune by itself.
- If several future reviews show the same low-efficiency chop pattern after fill provenance is fixed, the next justified code experiment should be a measured regime gate for trend entries rather than another broad signal-threshold adjustment.

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
