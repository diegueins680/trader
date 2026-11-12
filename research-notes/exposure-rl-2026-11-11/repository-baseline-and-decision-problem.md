# Repository baseline and selected decision problem

Campaign `exposure_rl_control_v1`. Base commit `b79dfa78cd42ef7e94540c5066aa73e5eb1f7d05` (2026-11-11). Findings below come from reading the code at that commit, not from documentation claims.

## Current champion

- **Decision policy.** The six live bots run technical-analysis rule sets, `ta_regime_switch` (five symbols) and `ta_best` (one), from `haskell/app/Trader/TechnicalAnalysis/Strategies.hs`. The rule picks a trend, reversion or breakout sub-strategy from ADX/EMA/Bollinger state and admits an entry when the implied edge clears a cost-based floor. It is not the Kalman or LSTM path.
- **Universe and timeframe.** Binance USDT-M perpetuals, 6h bars: SOL, BCH, NEAR, ARB, DOGE, ETH. Each bot is capped at 16.66% of equity (`TRADER_BOT_START_ADOPTION_MAX_POSITION_SIZE_CAP=0.1666`).
- **Parameters.** Each bot adopts a pinned optimizer "combo" (stop-loss, take-profit, size bounds, minimum edge). The pinned combos are not in git. Untracked operations evidence from 2026-10-27 holds partial numbers for four symbols, and three of today's six pins differ from it.
- **Consequence for this campaign.** An exact champion replay is unavailable. The comparator `champion_method` runs the repository's real `ta_regime_switch` code through the `trader-hs` CLI with documented defaults and fleet-median stop/take-profit, then replays its positions on the common ledger. It is code-faithful and parameter-approximate, and the registration blocks any superiority claim over the pinned production combos.

## Interfaces that matter

| Area | Finding |
|---|---|
| Backtest fill timing | `Trader/Trading.hs` decides on bar *t* and fills at the same close. Live trading acts on closed klines, so real fills are near the next open. This campaign fills at the next open. |
| Backtest funding | A flat annual rate per bar (`--funding-rate`), not historical settlements. This campaign uses the actual settlements. |
| Cost floors | `Trader/CostCalibration.hs`: taker 5 bp, slippage 0.5 bp, spread 1 bp. Pinned combos carry about 8 bp per side; this campaign uses 8 bp. |
| Per-bar export | No CLI flag exports features or per-sensor outputs. `--json` backtests export `positions`, `openTimes` and predictions, which is what the champion comparator uses. |
| DSR / PBO | The Haskell `deflatedSharpeProxy` and `pboProxy` are proxies. The canonical forms are in `scripts/research/diagnostics.py` and are the ones used here. |

## Model-name fidelity audit

The code's own implementation identifiers already state that these are stand-ins. They are not inputs to this campaign.

| Name | What the code does | Identifier in code |
|---|---|---|
| `TCN` | Ridge regression on returns at dilated lags. No convolution stack, nonlinearity or residual block. | `dilated_lag_ridge_v1` |
| `PatchTST` | Ridge regression on five summary statistics per patch. No patch embedding or attention. | `patch_summary_ridge_v1` |
| `Transformer` | One softmax similarity weighting over stored training rows (kernel regression). No learned projections or layers. | `similarity_attention_v1` |

No identifier or saved-artifact meaning was changed.

## Prior RL and adaptive-selection work

- **`sequential_control_screen_v1` (2026-09-17), rejected.** NumPy PPO, Double DQN and discrete CQL over a three-action target `{-0.25, 0, +0.25}` on 8h bars for ten perpetuals. 686 of 1,080 base-cost RL paths breached a risk limit, largely because its shield recorded a breach as a failure instead of projecting the action. All 108 OPE batches were invalid and DSR/PBO were null. The champion was never compared.
- **Later successors** (`ppo_successor_v2`, `causal_ppo_v5`, related accounting and OPE helpers) are default-disabled engineering work with zero financial trials.
- **Production adaptive logic** that is not RL: a UCB-style forecaster router, Hedge blending, and a "neural governor" that is an online regressor on taken trades with no arms or propensities.
- **Never implemented before this campaign:** TD3, SAC, IQL, TD3+BC, FQE, a real behaviour-cloning baseline.

## Data partitions already consumed or reserved

| Window (UTC) | Status |
|---|---|
| 2020-09-23 → 2025-03-17 | Contaminated development data: 45 earlier trials plus the sequential screen. Used here as development evidence only. |
| 2025-03-17 16:00 → 2026-05-01 | Sealed holdout reserved by earlier registrations. Never fetched by this campaign. |
| 2026-05-01 → 2026-07-16 | Claimed by no earlier registration. Registered here as a small, one-shot confirmation window. |
| 2026-07-17 → 2027-01-20 | Embargoed for the prospective funding-carry test. Never fetched. |

## Selected decision problem

**Bounded target exposure for one perpetual per account** (position management), decided at each completed 6h bar. One primary use case; no secondary use case was started.

Why it is sequential: proportional costs make the value of a target depend on the exposure already held; funding accrues to the position carried across a settlement; and the trailing-drawdown gate makes later admissible actions depend on earlier losses. A myopic rule ignores those couplings, which is exactly what the contextual-bandit comparator tests.

What differs from the rejected screen: a continuous action, a shield that projects instead of failing, next-open fills, 6h bars and fleet symbols matched to the champion, off-policy and offline learners that were missing, a champion comparator, and validated OPE on simulated data.

## Formal tooling found

Custom Python: Z3 bindings for SMT, hand-written breadth-first state models, and `ghc`-compiled conformance drivers, guarded by SHA-256 pins and an exact-match receipt. There is no TLA+ or SBV. This campaign reuses that pattern as a scoped sub-ledger under `formal/research/exposure-rl/`, run by the same `bash scripts/verify.sh formal` target.

## Compute and dependencies

CPU only, NumPy 2.3.5 and pandas 2.3.3 (the repository's existing research pins). No PyTorch wheel exists for the development host, so all learners are NumPy with hand-written gradients checked against finite differences. No paid data or compute was used.
