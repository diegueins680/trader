# Policy cards: `exposure_rl_control_v1`

All policies: observation `exposure_features_v1` (19 inputs), action a raw proposal in [-1, 1] decoded and projected by `exposure_shield_v1`, environment `exposure_env_v1`. Networks have two tanh layers of 64 units. Operating mode: `research_replay` only. Artifacts are strict JSON with a content hash and live under the ignored run directory; none is committed. Figures are the fleet aggregate over four pooled test folds at base cost, seed mean unless stated.

| | PPO | TD3 | IQL |
|---|---|---|---|
| **Regime** | On-policy simulator | Off-policy simulator | Offline, simulated dataset |
| **Training** | 600k steps, 64 lanes, GAE 0.95, clip 0.2 | 120k steps, 29k updates, twin critics, delayed smoothed targets | 300k simulated transitions, 30k updates, expectile 0.7, β 30 |
| **Seeds** | 11, 23, 47, 71, 97 | same | same |
| **Annualised mean** | 6.5% | 23.9% | 41.4% |
| **Sharpe (seed range)** | 0.18 (−0.17 to 0.58) | 0.62 (−0.44 to 0.80) | 1.02 (0.79 to 1.31) |
| **Max drawdown** | 42.1% | 32.1% | 38.0% |
| **Turnover per year** | 37.7 | 73.9 | 41.6 |
| **Mean absolute exposure** | 8.1% | 14.1% | 11.0% |
| **At 2× cost** | −10.4% | −10.2% | +22.1% |
| **Deflated Sharpe probability** | 0.00 | 0.01 | 0.04 |
| **Development gates** | fail | fail | fail |
| **Disposition** | Reject | Reject | Reject; best of the three, keep for reproduction |

## Extension and ablation

| | PPO, variance penalty | PPO without funding and BTC context |
|---|---|---|
| **Purpose** | Registered reward-shaping extension | Registered input ablation |
| **Annualised mean / Sharpe** | 4.4% / 0.61 | 27.6% / 0.83 |
| **Max drawdown** | 9.8% | 32.2% |
| **Mean absolute exposure** | 1.6% | 8.1% |
| **Reading** | Lower risk through much lower exposure, not added return | The extra inputs did not help PPO |
| **Disposition** | Reject | Diagnostic only |

## Non-RL comparators

| Policy | What it is | Ann. mean | Sharpe | Disposition |
|---|---|---|---|---|
| Buy-and-hold | Constant target at the cap | 49.5% | 0.70 | Reference; highest return, 61% drawdown |
| Ridge forecast | Ridge regression of next-bar return, cost-aware target | 26.5% | 0.50 | Reference |
| Behaviour cloning | Regression onto the simulated behaviour mixture | 13.4% | 0.59 | Reference |
| Regime rule | Unfitted trend/reversion/breakout rule on the same inputs | 3.8% | 0.08 | Reference |
| Contextual bandit | Greedy on a one-step reward model | −31.7% | −0.90 | Reference; shows the cost of ignoring inventory |
| Champion method, own ledger | `ta_regime_switch` through the CLI, its own fills | 4.9% | 0.25 | Parameter-approximate proxy, favourable ledger |
| Champion method, replayed | Same positions on the common ledger | −45.6% | −3.55 | Not a valid champion estimate: drops intrabar take-profit exits |

## Use restrictions for every card

- Development evidence on previously exposed data. Not confirmed out of sample.
- Not eligible for shadow, paper or live use. No policy here is exposed to any production selection path.
- The shield's guarantees are about admitted actions for a truthful snapshot. They do not bound drawdown or loss.
