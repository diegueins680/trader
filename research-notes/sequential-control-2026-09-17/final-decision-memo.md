# Decision memo — 2026-09-17

**No candidate passed; preserve the current champion. Reject the tested PPO, Double DQN and discrete CQL configurations for integration.** The useful deliverable is a reproducible negative mechanism screen and isolated research infrastructure. This is not completion of the requested untouched-market confirmation program.

Preregistration was committed before fitting. All 108 neural fits completed; 19,440 registered replay paths and all 19,548 training/evaluation registry entries reached a recorded terminal state. Seeds 11, 23 and 47 are all included. Four algorithm configurations include the one CQL inventory-penalty ablation. No candidate was integrated into production, shadow or paper trading.

## Evidence and contamination boundary

The only market inputs are the previously exposed 4,910-bar, ten-symbol Binance USD-M development panel and settlement file. Forty-five earlier adaptive development attempts remain disclosed, plus the separate prospective attempt 46. The three chronological folds are exploratory out-of-training-prefix evaluations; they are **not independent out-of-sample confirmation**. No final holdout was allocated or opened. The historical 1,227-return holdout remains sealed; the active carry campaign remains embargoed through 2027-01-20T13:00Z; existing HAR/missingness/OFI future-data registrations are unchanged.

| Fold | First decision close (UTC) | Last permitted outcome close (UTC) | Training stop index (exclusive) |
|---|---|---|---:|
| 0 | 2022-03-12T15:59:59.999+00:00 | 2022-09-28T07:59:59.999+00:00 | 1600 |
| 1 | 2023-02-08T23:59:59.999+00:00 | 2023-08-27T15:59:59.999+00:00 | 2600 |
| 2 | 2024-01-08T07:59:59.999+00:00 | 2024-07-25T23:59:59.999+00:00 | 3600 |

The first outcome occurs one bar after the first decision. Each complete fold path has 599 outcome bars. All symbols use identical folds and costs. UNI/SUI/ETC from the reviewed fleet, point-in-time membership, delisted assets, Coinbase/Kraken equivalent execution and matched current-champion/adopted-combo exports are missing. None is approximated by a different model and called an identical comparison.

## All seeds at baseline costs

Each cell aggregates ten symbols × three folds. Return is mean terminal-or-stopped net return, in percent of initial equity. Early-stopped paths have unequal endpoints: **do not rank these averages as economic superiority**. Failed/30 is an independent hard rejection criterion.

| Algorithm | Cadence bars | Seed 11: return%; failed | Seed 23: return%; failed | Seed 47: return%; failed |
|---|---:|---:|---:|---:|
| ppo | 1 | 2.425; 29/30 | -2.386; 30/30 | -3.181; 25/30 |
| ppo | 3 | -1.875; 27/30 | -4.600; 23/30 | -1.276; 0/30 |
| ppo | 6 | -2.987; 14/30 | -1.114; 17/30 | -1.252; 23/30 |
| double_dqn | 1 | -0.298; 28/30 | 0.309; 16/30 | -4.754; 21/30 |
| double_dqn | 3 | -3.947; 17/30 | -3.148; 22/30 | -1.940; 14/30 |
| double_dqn | 6 | -4.123; 25/30 | -4.319; 29/30 | -1.505; 12/30 |
| cql | 1 | -3.176; 20/30 | -1.839; 17/30 | -3.080; 22/30 |
| cql | 3 | -1.369; 15/30 | -1.496; 15/30 | -1.074; 17/30 |
| cql | 6 | -0.600; 17/30 | -1.962; 17/30 | -1.146; 17/30 |
| cql_no_inventory_penalty | 1 | -3.176; 20/30 | -1.839; 17/30 | -3.080; 22/30 |
| cql_no_inventory_penalty | 3 | -1.369; 15/30 | -1.496; 15/30 | -1.074; 17/30 |
| cql_no_inventory_penalty | 6 | -0.638; 17/30 | -1.962; 17/30 | -1.225; 17/30 |

At baseline costs 686/1,080 RL paths fail; the worst across all RL stress paths reaches 25.7378% drawdown and 2.9923% one-bar ES95. These are worst individual paths, not equal-endpoint portfolio comparisons. Cash has zero return, zero drawdown and zero ES throughout. Logistic baseline completes all paths at cadences 1 and 6 (mean returns 0.075% and 0.055%); cadence 3 has one failure. Neither this nor favorable stopped means proves a statistically credible advantage for any baseline. Current-champion drawdown and tail comparisons are unavailable under matched assumptions.

## Costs, inference, statistics and deployment

Baseline turnover cost is 10 bp plus signed historical funding. RL failures rise to 688 at 1.5× costs, 697 at 2×, 700 at 25 bp and 693 with another bar of delay. Partial/missed fills, funding and impact sensitivities also reject candidates. See the full [cost and robustness report](cost-execution-and-robustness.md).

All 108 OPE batches are invalid because failed episodes cannot be excluded. DSR, PBO, SPA and paired confidence intervals are null: no complete admissible matched-champion selection matrix or independent confirmation exists. Canonical DSR ≥0.95, PBO ≤0.20 and positive lower-bound gates were not relaxed. No statistical significance or economic superiority is claimed.

Wall time: 3,005.917 s; aggregate neural training: 494.684 s; observed peak RSS: 128.199 MiB; policy artifacts: 6,741–6,913 bytes. Maximum observed per-path inference p99: 2.945 ms, exceeding the preregistered 2 ms target; two replay paths failed the 20 ms timer. Measurements were on macOS Intel CPU with other verification work running, not an isolated production benchmark. Cold start, reload, concurrent-symbol service and preemptive timeout budgets are unmeasured. Small model size alone is not production readiness.

The Haskell proposal boundary is default-disabled, requires every deterministic evidence guard and cannot authorize an order. Tests cover 2,688 guard combinations. Market drawdown/exposure thresholds can still be crossed by a gap before detection; no real-world invariant stronger than the implementation is claimed. The HIGH/OPEN RL-OFFLINE-001 risk records simulator, support, gap and evidence deficiencies.

## Recommendation and next admissible step

General: no adoption. RL: reject these configurations and continue offline research only if a separately preregistered question addresses a demonstrated gap. Prefer deterministic optimization or a supervised policy as the complexity baseline; this screen does not establish a winner among them. No shadow/paper candidate passed, so no activation plan is authorized. A future protocol needs independently available data, full reviewed-fleet coverage, exchange-calibrated execution, matched Haskell champion, credible OPE, complete statistical controls and new untouched confirmation. Do not reopen existing protected data to finish this report.

Production authorization remains absent. No live flags, adopted UUIDs, leverage, margin, ownership, deployment identity, live caps or capital-preservation settings changed. No exchange order or authenticated endpoint was used, no live-money exploration occurred, and no policy can authorize an order. Existing production authorization in tracked profiles was preserved; this report does not falsely claim the pre-existing fleet was disabled.
