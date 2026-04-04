# Trader Objectives

## 2026-04-03

### Findings
- Source of truth for this bounded review was `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/`, replayed for local date `2026-04-03` with task cutoff `2026-04-03 23:37 America/Guayaquil`.
- The live tenant snapshots continued updating after the task timestamp while this review ran, so the metrics below are anchored to the captured cutoff replay taken during the task window rather than to a later rerun after midnight.
- One completed trade exited before the cutoff: `BNBUSDT` `2h` short, entered `2026-04-03 17:00 -05`, exited `2026-04-03 21:00 -05`, return `-0.00730%`, exit reason `SIGNAL`.
- No positions were still open at the cutoff: `openPositionsEnteredToday=0` and `openPositionsCarriedIn=0`.
- Same-day order evidence was limited to two `BNBUSDT` records: an ack-only `Order sent.` event at `2026-04-03 18:30:28 -05` with `status=NEW` / `executedQty=0`, and a `No order: already flat.` event at `2026-04-03 21:00:53 -05`. That leaves `fillEvidenceGaps=1`, so the day still does not justify a live-rule retune.
- The observed regime was weakly directional and increasingly choppy. The entry sat in `range-drift` (`efficiency=0.36259`, `realizedVolPct=0.65850`, `zScore=-1.18733`) and the exit sat in `chop` (`efficiency=0.10368`, `realizedVolPct=0.47985`). Saved latest regime probabilities also leaned strongly mean-reversion (`mr=0.82061`, `trend=0.08102`).
- The most important engineering failure mode today was bounded-review leakage, not strategy logic. The prior day-review tool had no explicit cutoff mode, and the same replay now shows `snapshotsUpdatedAfterWindow=8`, meaning several symbols had persisted state after the task timestamp that could silently contaminate a before-midnight review.

### Hypotheses
- When the realized sample is one tiny losing trade with ack-only entry evidence and zero open exposure at the cutoff, the highest-value change is review correctness, not another live-trading heuristic.
- A bounded daily review should satisfy two invariants:
  - it must expose whether snapshots were updated after the requested review window
  - it must reconstruct open-position membership from cutoff-state history instead of relying only on the final `openTrade` snapshot
- If several future confirmed-fill days show the same low-efficiency `range-drift/chop` entries, the next strategy experiment should be a stricter no-trade or alternate-policy gate when directional efficiency stays low, not a threshold tweak fit to this one sample.

### Metrics
- Bounded replay summary (`2026-04-03 00:00-23:37 -05`):
  - `completedTrades=1`
  - `completedCompoundPct=-0.00730%`
  - `completedAveragePct=-0.00730%`
  - `openPositionsEnteredToday=0`
  - `openPositionsCarriedIn=0`
  - `sameDayOrderEvents=2`
  - `ackOnlyOrderEvents=1`
  - `fillEvidenceGaps=1`
  - `ambiguousOpenPositionOrigins=0`
  - `snapshotsUpdatedAfterWindow=8`
- Trade-level diagnostic:
  - `BNBUSDT 2h` short
  - entry `2026-04-03T17:00:00-05:00` @ `587.77`
  - exit `2026-04-03T21:00:00-05:00` @ `587.69`
  - entry regime `range-drift`, exit regime `chop`
- Measurable observability improvement:
  - before: bounded reviews required ad hoc truncation and had no explicit leakage metric
  - after: the replay emits `snapshotsUpdatedAfterWindow` and reconstructs cutoff-open positions from the saved `positions` vector even if they close later

### Changes Made
- Updated `haskell/scripts/review_bot_day.py` to accept `--end-local` for bounded local-day reviews.
- Reworked open-position replay so window-end positions are reconstructed from the saved `positions` history at the cutoff instead of depending only on the terminal `openTrade` snapshot.
- Added a carry classification for positions already open before the review window and a new `snapshotsUpdatedAfterWindow` anomaly/summary metric.
- Extended `haskell/scripts/test_review_bot_day.py` with a deterministic cutoff regression where a same-day position is open at the cutoff but closes later in the saved snapshot.
- Updated `README.md` and `CHANGELOG.md` for the new review-tool behavior.

### Validation Results
- `python3 -m py_compile haskell/scripts/review_bot_day.py haskell/scripts/test_review_bot_day.py` passed.
- `python3 -m unittest haskell/scripts/test_review_bot_day.py` passed (`3` tests).
- `python3 haskell/scripts/review_bot_day.py --date 2026-04-03 --timezone America/Guayaquil --format json` passed and reported the one completed `BNBUSDT` short with `snapshotsUpdatedAfterWindow=0` for the full-day window.
- `python3 haskell/scripts/review_bot_day.py --date 2026-04-03 --timezone America/Guayaquil --end-local 2026-04-03T23:37:00-05:00 --format json` passed and reported the same trade set plus `snapshotsUpdatedAfterWindow=8` for the bounded review.
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal build` passed.
- `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal test` passed.
- No separate backtest was run because this change improves replay/diagnostics correctness rather than live decision rules.

### Remaining Risks
- The review still depends on bot-state snapshots, so an ack-only order record remains weaker than a persisted exchange-fill ledger.
- The cutoff replay now reports future-updated snapshots and reconstructs open-position membership from history, but it still cannot recreate intra-bar prices/signals at an arbitrary minute inside the current bar.
- Strategy research still points toward stricter low-direction regime gating when confirmed fills accumulate, but today’s single tiny loss is not enough evidence to deploy that change.

## 2026-04-02

### Findings
- Task cutoff was `2026-04-02 00:32 America/Guayaquil`. A cutoff-scoped replay over the latest tenant snapshots showed `completedTrades=0`, `openPositionsEnteredBeforeCutoff=0`, and `orderEventsBeforeCutoff=0`; there were no fill-backed same-day decisions to retune yet.
- To avoid a null engineering review, I also replayed the just-finished local day `2026-04-01` with the patched review tool. That day had `completedTrades=0`, `openPositionsEnteredToday=0`, `openPositionsCarriedIn=1`, `sameDayOrderEvents=3`, `ackOnlyOrderEvents=2`, `fillEvidenceGaps=0`, and `ambiguousOpenPositionOrigins=0`.
- The concrete defect was attribution drift: the persisted `BTCUSDT` `1d` short had been counted as a same-day open even though the same-day order message was `No order: already short.` and the nearest supporting sent order evidence was prior-day `2026-03-31 19:00 -05`. After the fix, it is classified as `openPositionsCarriedIn` with provenance `prior_day_order_evidence`.
- The carried BTC short sat in a high-vol / no-direction regime (`efficiency=0.01346`, `realizedVolPct=2.04114`, `zScore=-0.05078`), while the only same-day intents on `2026-04-01` were ack-only `ADAUSDT` / `LINKUSDT` orders with no persisted fill evidence. That is too little confirmed signal to justify another live strategy change tonight.

### Hypotheses
- When the day contains zero fill-backed same-day entries/exits, the highest-leverage improvement is review correctness, not another live-rule tweak.
- Treat “fresh entry” as an invariant requiring either same-day sent-order evidence or explicit carry/ambiguity classification; adoption messages like `already short` / `already long` must never inflate same-day entry counts.
- If several future reviews show confirmed entries opening in high-vol + ultra-low-efficiency conditions, test a tighter regime-strength / volatility-managed gate for trend entries; do not fit that rule off today’s carry-only sample.

### Metrics
- Task-cutoff replay (`2026-04-02 00:00-00:32 -05`):
  - `completedTrades=0`
  - `openPositionsEnteredBeforeCutoff=0`
  - `orderEventsBeforeCutoff=0`
- Full-day `2026-04-01` replay after the attribution fix:
  - `completedTrades=0`
  - `openPositionsEnteredToday=0`
  - `openPositionsCarriedIn=1`
  - `sameDayOrderEvents=3`
  - `ackOnlyOrderEvents=2`
  - `fillEvidenceGaps=0`
  - `ambiguousOpenPositionOrigins=0`
- Carried-position diagnostic:
  - `BTCUSDT 1d` short
  - supporting prior order: `2026-03-31T19:00:36.514000-05:00`
  - adoption event: `2026-04-01T19:00:47.094000-05:00`
  - provenance: `prior_day_order_evidence`
  - entry regime: `high-vol`, efficiency `0.01346`

### Changes Made
- Used Codex to update `haskell/scripts/review_bot_day.py` so open positions are split into `openPositionsEnteredToday`, `openPositionsCarriedIn`, and ambiguous-origin anomalies using saved order evidence plus same-day `already short` / `already long` adoption messages.
- Added `haskell/scripts/test_review_bot_day.py` with deterministic synthetic coverage for both carried-in and ambiguous-adoption cases.
- Updated `README.md` and `CHANGELOG.md` for the new daily-review provenance behavior.

### Validation Results
- `python3 -m py_compile haskell/scripts/review_bot_day.py` passed.
- `python3 -m unittest haskell/scripts/test_review_bot_day.py` passed (`2` tests).
- `python3 haskell/scripts/review_bot_day.py --date 2026-04-01 --timezone America/Guayaquil --format json` passed and now reports `openPositionsEnteredToday=0`, `openPositionsCarriedIn=1` for the BTC carry.
- A one-off cutoff-scoped Python replay for `2026-04-02 00:00-00:32 America/Guayaquil` confirmed zero completed trades, zero same-day entries, and zero order events before the task timestamp.
- No separate backtest was run because this change is an execution-attribution / diagnostics fix, not a live strategy-rule modification.

### Remaining Risks
- Carry/adoption provenance still depends on saved order history; a `NEW` ack without later fill evidence remains weaker than an exchange fill ledger.
- The review tool now avoids a false positive on same-day entries, but it still cannot prove fills end-to-end when snapshots omit execution details.
- Strategy research still points toward stronger trend-entry discrimination in high-vol / low-efficiency regimes, but today’s cutoff-scoped sample did not justify deploying that live-rule change.

## 2026-03-31

### Findings
- Source of truth for this review was the latest active bot snapshot tenant under `haskell/.tmp/bot/tenants/binance-dc286605a9946343b18aeb2670e23ce51f6d9e0e1b37f50205f1945c6c54016a/`, replayed for local date `2026-03-31 America/Guayaquil`.
- One completed trade exited on the target date: `ADAUSDT` `2h` long, entered `2026-03-31 13:00 -05`, exited `2026-03-31 23:00 -05`, realized return `+0.12177%`, exit reason `SIGNAL`.
- Two positions were opened on the same local date and still open in the snapshot: `ATOMUSDT` `12h` long (`+0.03302%` mark-to-market on equity) and `LINKUSDT` `1h` long (`+0.03470%` mark-to-market on equity).
- All three entries were made in explicitly low-directional `chop` conditions by a 24-bar regime replay. Entry efficiency was low (`ADA 0.0213`, `ATOM 0.2372`, `LINK 0.1208`), while the saved regime probabilities also leaned mean-reversion (`mr` high, `trend` low where available).
- The concrete engineering problem exposed today was execution observability, not an obvious signal-rule blowup: same-day order events totaled `4`, of which `3` were ack-only (`status=NEW`, `executedQty=0`) and `2` still lacked direct fill evidence even though the replay also showed an active/completed trade (`ADAUSDT`, `ATOMUSDT`).
- There was also a same-day `BTCUSDT` ack-only `SELL` order with no corresponding same-day completed/open trade, reinforcing that the persisted review artifacts are currently better at showing intent than proving exchange execution.

### Hypotheses
- Do not retune live entry/exit logic off this day alone. The realized sample is one mildly profitable completed trade in a low-efficiency chop regime, so strategy-level tuning would be mostly noise fitting.
- The highest-leverage improvement for tomorrow is to close the measurement gap: every daily review should be able to deterministically reconstruct completed/open trades, same-day order intents, regime labels, and explicit fill-evidence gaps from persisted artifacts.
- Once fill provenance is trustworthy, the most plausible strategy experiment for repeated days like this is an explicit chop/no-trend gate for momentum entries (for example: require higher directional efficiency or lower mean-reversion probability before opening new trend trades).
- Treat `fillEvidenceGaps > 0` as a review-quality invariant violation: PnL attribution and trade-decision diagnosis are provisional until that count is driven to zero.

### Metrics
- Day-scoped replay summary:
  - `completedTrades=1`
  - `completedCompoundPct=+0.12177%`
  - `completedAveragePct=+0.12177%`
  - `openPositionsEnteredToday=2`
  - `sameDayOrderEvents=4`
  - `ackOnlyOrderEvents=3`
  - `fillEvidenceGaps=2`
- Entry-regime diagnostics:
  - `ADAUSDT 2h`: `chop`, efficiency `0.0213`, net return over lookback `+0.4172%`, realized vol `0.9827%`
  - `ATOMUSDT 12h`: `chop`, efficiency `0.2372`, net return over lookback `-6.2431%`, realized vol `1.4451%`
  - `LINKUSDT 1h`: `chop`, efficiency `0.1208`, net return over lookback `+1.6299%`, realized vol `0.7690%`
- Same-day open-position context at review time:
  - `ATOMUSDT`: entry `1.697`, current `1.705`, MTM `+0.03302%`
  - `LINKUSDT`: entry `8.792`, current `8.835`, MTM `+0.03470%`

### Changes Made
- Used Codex to add `haskell/scripts/review_bot_day.py`, a dependency-free daily replay tool for persisted `bot-state-*.json` snapshots.
- The new script auto-selects the latest tenant (or accepts `--tenant-dir`), reconstructs completed trades and open positions for a local calendar day, classifies the recent regime with explicit 24-bar thresholds (`high-vol`, `trend-up`, `trend-down`, `chop`, `range-drift`), and flags ack-only same-day order events that still lack direct fill evidence.
- Added both Markdown and JSON output modes so the review can become a stable input to future automation.
- Updated `README.md` and `CHANGELOG.md` because this introduces a new user-facing review utility.

### Validation Results
- `python3 -m py_compile haskell/scripts/review_bot_day.py` passed.
- `python3 haskell/scripts/review_bot_day.py --date 2026-03-31 --timezone America/Guayaquil` passed and reproduced the metrics above, including `ack_only=3` and `fill_gaps=2`.
- Codex also ran `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal build` and `cd haskell && PATH=$HOME/.ghcup/bin:$PATH cabal test trader-tests --test-show-details=direct`; both completed successfully in the agent log.
- No separate backtest was required for this change because the implemented improvement is a deterministic post-trade replay/diagnostics tool rather than a live trading-rule change.

### Remaining Risks
- The new replay script still depends on bot-state snapshots; it cannot prove exchange fills unless direct fill provenance is persisted into the saved artifacts.
- Today’s review therefore improves diagnosis quality more than realized PnL. It tells us that the day was dominated by chop and small returns, but it does not yet let us audit every fill path end-to-end.
- If multiple future days continue to show low-efficiency chop entries after fill attribution is fixed, the next narrow experiment should be a replayed regime gate for trend entries instead of another broad threshold tweak.

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
