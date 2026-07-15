# Historical Funding Campaign Development Result - 2026-07-15

## Decision

**Reject this trial family and do not open the final holdout.** The fixed residual-momentum campaign failed its development promotion gates. The registered `1,227` final returns remain reserved and unconsumed.

## Protocol

- Campaign: `residual_momentum_funding_only_v1`
- Registration commit: `723d43981372681939a7e0b2fea4bbc8781cf339`
- Merge commit: `0d698f6eefc2d48fe6b783661b9f8897245ccac8`
- Registration SHA-256: `cedaea5af05c880af732ceb1a78c39d4056efc2b1065157bc7a1ce31ff684d9f`
- Snapshot acquisition completed: `2026-07-15T06:05:24.791656Z`
- Command: `python3 scripts/research/run_historical_funding_campaign.py --acquire`
- Final-holdout flag: absent

The campaign used the committed ten-symbol, eight-hour grid from `2020-09-23T00:00:00Z` through `2026-04-30T16:00:00Z`. It evaluated six fixed `24h`/`72h`/`168h` base-versus-funding-only trials on the `4,910`-bar development window.

## Integrity

- Snapshot manifest SHA-256: `0e970ef24bbda0a2ceff24af5b83bc1a70de60a96ab4940a729c505e9a9c801e`
- Campaign manifest canonical SHA-256: `96dccaed245597d0a1f8ab1eb86dd058da59fabf44d03c58f061dc00d3541549`
- Campaign manifest file SHA-256: `686ddb5ae44c4b2e461ffedd5c2b8199da4399b75d14c9075636ac062a008776`
- Summary file SHA-256: `eb6d6a588d063ea3b392ab799cb1be83d8b3cc5c8f4c13ec9aaf58d71cf3bd44`
- Implementation SHA-256: `0da130ac2975d4c27af517457a41dd499317736520e17ca4b2760700f2280053`
- Registered full panel SHA-256: `11d0af89e1603fad91bceffae3847dfdaed819bf56a45fe6f9d6bec45c953c0a`
- Registered settlements SHA-256: `cc0daa4d86d8c2d64285cf12baae86dcb1d9cb5606525074cf023f77bd59c795`

All `30` raw artifacts sealed. The runner acquired `61,455` endpoint-returned funding events and resolved all of them; `34,075` used the registered causal hourly-open fallback. The maximum observed gap was `28,800,047` ms against the registered `28,860,000` ms tolerance.

No `final-holdout-returns.csv` or `final-holdout-opened.json` exists, and the shared holdout registry contains no JSON entry for this campaign.

## Development Results

The full-development champion was `resmom_168h_base`. Its whole-window Sharpe was `0.847` with total return `+386.2%`, but that number is not untouched evidence because the same development history participated in model selection.

The nested rolling-origin result reversed the conclusion:

| Measure | Result |
| --- | ---: |
| Outer-OOS observations | 2,443 |
| Outer-OOS Sharpe | -1.296 |
| Block-bootstrap 95% Sharpe CI | [-2.674, 0.035] |
| Outer-OOS total return | -76.69% |
| Outer-OOS max drawdown | 81.53% |
| Losing outer folds | 6 of 7 |
| Current-campaign DSR probability | 0.6282 |
| PBO probability | 0.1349 |
| Lifetime Bonferroni PSR probability | 0.4082 |

The three registered funding-minus-base comparisons all failed their simultaneous confidence requirement:

| Horizon | Simultaneous Sharpe CI |
| --- | ---: |
| 24h | [-0.828, 1.404] |
| 72h | [-2.264, 0.132] |
| 168h | [-2.084, 0.226] |

Frozen stresses also failed:

| Stress | Outer-OOS Sharpe | 95% Sharpe CI | Total return |
| --- | ---: | ---: | ---: |
| Additional 1-bar delay | -1.418 | [-2.718, -0.119] | -79.32% |
| 2x cost | -2.059 | [-3.440, -0.705] | -88.86% |

Only the sample-size, activity, symbol, funding-resolution, regime-coverage, and PBO gates passed. The outer-OOS confidence, funding improvement, DSR, lifetime correction, regime-loss, worst-fold, doubled-cost, and delay gates failed.

## Interpretation

The positive whole-development result was selection-sensitive and did not survive nested chronological evaluation. Funding crowding exclusion did not add a demonstrable edge at any registered horizon.

Turnover is also economically material. On the nested path, mean gross return was `-2.09` bps per eight-hour bar and mean modeled cost was `3.02` bps. A simple sign inversion would change gross to about `+2.09` bps but still leave approximately `-0.93` bps net under the same turnover. Reversing the signal without changing execution is therefore not a sufficient next experiment.

## Next Direction

Treat the following as exploratory development generated after observing this result:

1. Test residual reversal rather than momentum.
2. Reduce turnover ex ante with fixed slower rebalance intervals or rank hysteresis.
3. Do not reuse the failed funding filter unless a separate economic mechanism is registered.
4. Count every new configuration in the lifetime trial family, increasing the current count above `21`.
5. Keep the same final chronological holdout sealed until one pre-registered development campaign passes every gate.

The next campaign should isolate reversal direction from turnover control rather than introduce another broad feature sweep.
