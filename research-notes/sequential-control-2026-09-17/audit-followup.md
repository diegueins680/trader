# Draft PR #256 follow-up audit

The audit preserves **no candidate passed** and rejects integration of the tested
policies. The larger confirmation program remains incomplete. This is a review
of engineering boundaries and frozen negative evidence, not another financial
trial or independent replication of economic performance.

## Revision and evidence integrity

The existing draft starts from latest main `f669c7e9`; audited head was
`0484060fd8f4bd44962dd34796e80a8024830a35`. A separate worktree preserved the user's
uncommitted checkout. No open GitHub issues or PR review comments were returned;
related drafts #249–251 already cover feature/artifact work, which is not duplicated.
The original registration and all trial results remain unchanged.

The exporter hashed the full indexed archive and reproduced all seven compact
files byte-for-byte. [audit-receipt.json](audit-receipt.json) records their hashes
and the pinned external index. This establishes archive consistency, not that
historical data, simulator fills or statistical assumptions are correct.

## Reproduced OPE numerical defect and repair

At the audited head, a NaN discount emitted a NaN IS estimate; a negative discount
was accepted; finite behavior probabilities of 1e-300 overflowed trajectory
weights and produced infinite IS and NaN effective sample size. The existing
shape/probability checks were insufficient. These are deterministic numerical
fixtures and are not counted as new financial experiments.

The helper now requires finite numeric aligned nonempty inputs, valid integer
action indices, discounts in [0,1], probabilities in their declared domains and
zero terminal bootstrap. Overflow in ratios, products, moments, estimates or
bootstrap means raises ValueError. No importance weights are clipped. Zero target
support still yields ESS zero, absent WIS and unreliable status. Valid estimates
still confer no promotion authority.

A complete two-action, two-decision behavior tree independently supplies a known
2.5 target value for IS, PDIS, WIS and DR. With exact Q/V control variates, every
DR sample is 2.5. The fixture checks off-policy weights and discount timing,
where the previous test used identical behavior/target probabilities. The
[sequential DR source](https://proceedings.mlr.press/v48/jiang16.html) provides the
methodological basis; numerical admissibility is an engineering obligation.

All 108 original OPE batches already returned invalid before these calculations
because episodes failed. They remain invalid; no ESS, confidence interval or
new favorable value is inferred and no original result is rewritten.

## Economic evidence checked, not extended

| Algorithm | Seed 11 failures / 90 | Seed 23 failures / 90 | Seed 47 failures / 90 |
|---|---:|---:|---:|
| PPO | 70 | 70 | 48 |
| Double DQN | 70 | 67 | 47 |
| CQL | 52 | 49 | 56 |
| CQL without inventory penalty | 52 | 49 | 56 |

Counts aggregate the already recorded baseline-cost paths across three folds,
three decision cadences and ten symbols; they are not seed-selected results.
There are 686 failures in 1,080 RL baseline-cost paths. At 1.5x, 2x, extreme costs
and another bar delay, failures are respectively 688, 697, 700 and 693. Stopped
paths have unequal endpoints and cannot support economic rankings. Worst recorded
RL stress-path drawdown is 25.7378%; worst one-bar ES95 is 2.9923%. Matched champion
risk comparisons are unavailable. DSR, PBO, SPA and paired confidence remain absent.

The market folds span 2022-03-12 through 2022-09-28, 2023-02-08 through 2023-08-27,
and 2024-01-08 through 2024-07-25; exact decision/outcome timestamps are in the
[decision memo](final-decision-memo.md). All are contaminated development evidence.
No final holdout was read. The prior sealed holdout and prospective restrictions
remain in force; no new registration, training or checkpoint selection occurred.

## Fidelity, safety and operational scope

Direct source inspection confirms patch-summary ridge for PatchTST, dilated-lag
ridge for TCN and similarity-weighted memory for Transformer. Existing aliases
and legacy serialized semantics remain unchanged. Haskell's private proposal
constructor and always-false order authorization remain isolated from production
callers; this audit does not claim a theorem about real market gap risk.

The Python simulator's invalid-state rejection, terminal accounting and bounded
actions are research mechanics. Its cost/impact/delay assumptions do not prove
exchange execution fidelity. No live exploration, order, authenticated trading
endpoint, deployment, merge, self-modification or policy promotion occurred.
Production fleet, champion, UUIDs, live authorizations and risk limits are unchanged.
The pre-existing fleet is not represented as globally disabled.

The formal registry adds explicit OPE domain and failure clauses plus enumerated
and regression evidence. RL-OFFLINE-001 stays HIGH/OPEN; its Markdown mitigation
now names numerical rejection. Canonical severity/status and Haskell risk
projection remain unchanged. README, CHANGELOG and formal documentation explain
the behavior; no environment configuration or new dependency is introduced.

## Literature refresh and limits

The focused matrix now has 51 rows after adding the September 12 JumpStart
preprint as Monitor. BFQ's venue field records its author-reported ICML 2026
journal reference without claiming independent proceedings verification. The
[literature addendum](literature-review.md#audit-source-refresh) links primary
records. This refresh is not a fresh exhaustive review of every paper.

The [deliverables index](deliverables-index.md) remains authoritative about
missing matched-champion evidence, causal execution data, state-joint support,
regime and feature ablations, calibration, independent confirmation and production
benchmarks. Current engineering repairs do not satisfy those missing conditions.
General recommendation: no candidate passed. RL recommendation: reject these
configurations; continue offline work only under a justified new registration,
with simpler controls as the complexity baseline.

Verification results and exact environmental failures are recorded in
[verification.md](verification.md).
