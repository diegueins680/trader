# Environment and action contract: `exposure_env_v1`

Registered in `research-notes/registrations/exposure-rl-control-v1.json`. Code: `scripts/research/exposure_rl/`.

## Decision process

A constrained, partially observed decision process. Prices and funding are exogenous: the account is a price-taker and its trades do not move the market. Only the account's own position, costs and equity depend on the action. Finite features do not make the market observable or stationary, and no such claim is made.

| Element | Definition |
|---|---|
| Decision time | Close of each completed 6h bar *t*. |
| Observation | 15 market features of bars ≤ *t* and funding events stamped ≤ close *t*, plus 4 account-state features. 60-bar burn-in. |
| Action | Raw proposal in [-1, 1]: an absolute target, not an increment. Target exposure = `trunc(raw × 1666)` in units of 1e-4 of equity. |
| Transition | Fill at the open of bar *t+1*; equity marked at close *t+1*. |
| Reward | `100 × (E[t+1] − E[t]) / E[t]`, net of costs and funding. No shaping. |
| Discount | 0.99 per bar (about 100 bars, 25 days). |
| Initial condition | Flat, equity 1, at a random symbol and bar (training) or the start of the window (evaluation). |

## Observations

Market features use fixed transforms (returns over 1, 4, 12 and 28 bars scaled by 56-bar volatility, a volatility ratio and level, 20-bar channel position, EMA20/EMA50 gap in ATRs, RSI14, ADX14, volume ratio, last and 21-event mean funding rate, BTC 4- and 28-bar returns), clipped to ±5. **No scaler is fitted**, so no fitted statistic can carry later data into an earlier observation. Account state: exposure, trailing drawdown, a reduce-only flag and bars since the last fill.

There are no recurrent policies, so there is no hidden state to initialise or isolate.

## Action stages

Every decision logs four stages and the intervention reasons.

1. **Proposed**: the raw policy output (untrusted).
2. **Shield-permitted**: an integer target, or no intent.
3. **Submitted**: a simulated intent to move exposure to that target.
4. **Filled**: units actually changed, after partial or missed fills.

Training and evaluation both use the composed path *proposal → shield → execution*. Learners see the clipped raw proposal; rewards, bootstrap targets and log-probabilities all refer to that same proposal, and nothing is relabelled with the projected action. Known consequence: the TD3 actor gradient is zero wherever the shield projects.

## Shield (`exposure_shield_v1`)

Integer arithmetic on a validated snapshot (exposure, drawdown, freshness, headroom).

| Rule | Effect |
|---|---|
| Decode | Non-finite or out-of-range proposals are rejected; valid ones truncate toward zero. |
| Cap | `|target| ≤ 1666` (16.66% of equity). |
| Headroom | `|target| ≤ min(|current|, cap) + reserved headroom`. |
| Reduce-only | If drawdown from the trailing 120-bar peak is ≥ 10%, or the snapshot is not fresh, the target cannot increase absolute exposure or change side. |
| Minimum size | A change under 25 units forms no intent. It is never rounded up. |
| Invalid proposal | Rejected; the fallback proposes the current exposure, which still forces a reduction to the cap when above it. |
| Invalid snapshot | Recovery required; no intent. |

The drawdown peak is trailing, not all-time. An all-time peak left a buy-and-hold account reduce-only for 61% of decisions in a pilot, a permanent lockout.

## Execution and funding timing

1. Funding stamped exactly at open *t+1* settles on the **pre-fill** position.
2. The intent fills at open *t+1* at a cost of 8 bp of traded notional.
3. Funding stamped inside the bar settles on the **post-fill** position.
4. Equity is marked at close *t+1*.

The sizing rule sets units to `target × equity_at_open / open`. Because cost is deducted afterwards, exposure just after a fill can exceed the target by a factor of at most `1 / (1 − cost × turnover)`; price drift between bars adds more. Both are state effects that the next snapshot reports, not admitted actions.

Stresses change one assumption each: cost ×1.5 and ×2, 25 bp with a one-bar delay, a one-bar delay (fill at close *t+1*), 50% partial fills, 10% missed fills, funding ×2, 5% stale snapshots, 1% corrupt snapshots, and cost widening with the size of the last return.

**Not modelled:** queue position, intrabar paths, market impact, exchange liquidation and margin, pending orders, exchange precision and minimum notional, outages. OHLCV replay supports a price-taker exposure experiment and nothing finer.

## Ledger

`E[t+1] − E[t] = gross P&L − cost − funding`, with fees, slippage and spread entering once through cost and unrealised P&L entering once through the mark. There are no external cash flows. The economic ledger is kept separately from the learner's scaled reward. Log-wealth rewards are not used.

## Episode ends

| End | Meaning | Bootstrap | Liquidation |
|---|---|---|---|
| Time limit (256 decisions, training) | Truncation | From the true successor state | None |
| End of the permitted data range | Truncation | From the true successor state | None in training; charged once in evaluation |
| Equity ≤ 0.5 | Economic termination | None | At cost |

Window-end liquidation in evaluation is accounting, not a training signal.

## Fallback by state

"Do nothing" is not treated as automatically safe. An invalid proposal holds the position if it is within the cap and reduces it to the cap otherwise. A stale snapshot allows reductions only. An invalid snapshot forms no intent and would hand the position to the established risk controller; this campaign simulates that as a hold and performs no real recovery action.
