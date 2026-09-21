# Research and formal-assurance decision update — 2026-09-20

**No candidate is adopted. The requested full research/formal-verification mission
is not complete.** The independently useful new deliverable is a scoped formal
gate and an explicit proof/assumption/traceability ledger. It does not repair the
failed economic/OPE evidence or establish whole-system implementation refinement.

The [September 21 verification receipt](formal-verification-2026-09-21.md) records
passing final local formal/full checks and both Linux CI runs at their exact
revisions, along with resolved failures and the still-failing acceptance gate.

## Preparation and fidelity

The isolated branch starts from freshly fetched `main` at `dbd45e26`; the original
workspace's unrelated modifications are preserved. Open issues were empty. Open
PRs #249–251 contain existing artifact/missingness research and #253–255 contain
web dependency work; none is duplicated. Commit history includes the prior RL
screen and numerous admission, numerical, accounting and provenance fixes. The
September 17 research packet is inherited evidence, not work newly trained here.

Directly rereading `TCN.hs`, `PatchTST.hs` and `Transformer.hs` confirms dilated-lag
ridge, patch-summary ridge and stored-example similarity attention, respectively.
`git diff f669c7e9..dbd45e26 -- haskell/app/Trader/Predictors` is empty. Accurate
versioned aliases already exist. The [full model-fidelity audit](model-fidelity-audit.md)
and paper-to-code gaps remain current for that source range. No identifier,
serialized config or artifact semantics change. The prior review inventories
GBDT/tree/KNN/HMM/LSTM/Kalman/online neural/quantile/conformal/ensembles/regimes,
market context, ranking, optimizer and bandit routing; none is silently replaced.

## Existing RL experiment, archive revalidated without a financial rerun

The [registration](../registrations/sequential-control-screen-v1.json) was committed
before the original experiment. It defines partially observed inventory control:
12 observations, targets {-0.25,0,+0.25}, decisions every 1/3/6 bars, old-inventory
P&L and funding before next-close execution, fees/spread/slippage, inventory reward
penalty, explicit terminal liquidation and failure stops. This has delayed
inventory/cost consequences, but does not establish a Markov market observation.

PPO, Double DQN, discrete CQL and the CQL inventory-penalty ablation were already
trained with seeds 11/23/47. The registered 108 fits and 19,440 replays are retained
in the [registry](experiment-registry.csv); no new financial trial, seed, reward,
checkpoint rule or economic replay was run in this follow-up. Engineering fixture
checks are not counted as fresh market experiments. See [every seed](all-seed-results.csv),
[training](multi-seed-training.json), [summary](evaluation-summary.json),
[cost/robustness](cost-execution-and-robustness.md), and [policy cards](policy-cards.md).

The data remain 4,910 exposed 8h Binance USD-M bars per symbol: ADA, AVAX, BNB,
BTC, DOGE, ETH, LINK, LTC, SOL, XRP. Exact data hashes and source/licensing
limitations remain in [the manifest](experiment-manifest.json) and
[data-source ledger](data-source-license-manifest.json). UNI/SUI/ETC, point-in-time
membership, delistings and equivalent Coinbase/Kraken execution evidence are absent.
No historical publication timestamp is upgraded into an observed first-seen time.

| Exploratory fold | First decision close UTC | Last permitted outcome close UTC |
|---|---|---|
| 0 | 2022-03-12 15:59:59.999 | 2022-09-28 07:59:59.999 |
| 1 | 2023-02-08 23:59:59.999 | 2023-08-27 15:59:59.999 |
| 2 | 2024-01-08 07:59:59.999 | 2024-07-25 23:59:59.999 |

These are out-of-training-prefix slices of contaminated development history,
**not independent out-of-sample confirmation**. Baseline cost is 10 bp per turnover
plus signed actual funding. RL base failures: 686/1,080; at 1.5x/2x/extreme25bp/
extra-one-bar-delay: 688/697/700/693. Maximum across RL stress paths: 25.7378%
drawdown and 2.9923% one-bar ES95. Stopped paths have different endpoints and cannot
be ranked as equal-period superior investments. Cash remains zero return/risk.
Matched champion/adopted-combo drawdown, tail-risk and net-return comparisons are
unavailable. Favorable baseline means do not establish an accepted replacement.

Uniform simulated behavior has nominal target propensity 1/3. It is not actual
exchange action support. All 108 OPE batches are invalid because failed episodes
cannot be dropped. No reliable ESS-based OPE inference, joint live support,
simulator-to-independent-history gap or statistical superiority is available.
DSR, PBO, SPA and paired confidence remain null, not zero or passing. No threshold
is relaxed. Observation-support ranges are not a joint state/action coverage proof.

Prior runtime evidence: 3,005.917 s overall, 494.684 s aggregate training, 128.199
MiB peak RSS, policy sizes 6,741–6,913 bytes, worst observed inference p99 2.945 ms
against the 2 ms target; two replay paths exceeded 20 ms. Cold start, reload,
concurrent production inference and preemptive timeout remain unmeasured. These
are inherited CPU measurements, not benchmarks rerun by this follow-up.

## Holdout and adoption status

No final holdout is opened or evaluated. The 1,227-return historical holdout stays
sealed. The carry protocol remains embargoed through 2027-01-20T13:00Z; future
HAR/missingness/OFI registrations remain intact. No fabricated holdout metric is
reported. Existing hard development/evidence failures block proceeding to a final
holdout; opening one solely to complete a deliverables checklist is prohibited.

No policy passed shadow/paper gates, so no activation plan or loader is added.
General recommendation: **no candidate passed; preserve the champion**. RL
recommendation: **reject the tested configurations; continue offline research only
with a separately justified preregistration**, using deterministic and supervised
policies as complexity baselines. No universal conclusion that RL cannot work is
supported by this screen.

## New formal evidence and its limits

The [canonical contract](../../formal/research/contract.md) was committed before
new checker implementation. It resolves coverage-versus-proof terminology,
retained proposals after disable, historical-data contamination, existing proxy
aliases and pre-existing production authorization. This is a scoped consistency
audit, not a proof of global consistency of every prose requirement.

The [ledger](../../formal/research/proof-ledger.json) maps canonical requirements
to assumptions, formal source, proof result, implementation, tests and CI, with
reverse critical-file coverage. Nine manually encoded SMT obligations are UNSAT
for negations; IEEE binary64 bounds cover the complete numeric comparison domain,
while causality/split lemmas use mathematical integers and accounting uses exact
reals. These are not universal Haskell/compiler or Python-simulator proofs.

The abstract two-caller model exhaustively reaches 75 states and checks 349
transitions (maximum shortest-path depth 6). Under completion fairness it drains
within two completion events. No actual production server/ownership refinement is
claimed. `CE-RL-001` refutes the stronger idea that disable erases caller-owned
immutable proposals; the corrected contract retains no order authority. The trace
is stored and replayed deterministically. It is a specification counterexample,
not an observed live-order incident.

Compiled Haskell matches 16,384 representative boundary combinations and 4,096
seeded arbitrary-bit/generated cases; all 240 admitted proposals preserve target
bits and deny authority. The compiler also rejects direct use of the private
proposal constructor. These are conformance and boundary checks, not proof of all
possible source programs. No neural-network region is verified. Probabilistic
checking is not applied without defensible transition probabilities.

Six named assumptions cover binary64/defined values, trusted evidence, solver and
compiler correctness, bounded interleavings/fairness, causal-index semantics and
exact-real accounting. All 38 broader requested implementation obligations remain
open or partial. Production lifecycle/ownership, true source availability,
floating-point ledger refinement, hard-gap loss bounds, artifact authenticity and
universal model-to-code correspondence are unresolved blockers. `RL-OFFLINE-001`
stays HIGH/OPEN. No full-system verification or readiness claim is warranted.

## Deliverable navigation and outstanding work

The [existing deliverable index](deliverables-index.md) locates literature,
scorecards, preregistration, datasets, training, all-seed results, OPE, rejected
policy cards, simulator/reward limitations, artifacts and reproducibility. The
[new formal runbook](../../formal/research/README.md) locates the canonical model,
assumptions/obligations/status ledgers, toolchain pins, counterexample, executable
SMT/model/conformance sources, deterministic results and verification limitations.
The [literature refresh](formal-literature-2026-09-20.md) adds primary-source formal
and recent RL/forecasting screening. These links cover the requested reporting
dimensions without relabeling absent empirical evidence as completed work.

New production/model implementation: none. Existing research implementations remain
available; their original financial evidence is not regenerated. New configuration
is only an optional local verification interpreter. No API/CLI trading, database,
web, saved-model or deployment identity changes. README, CHANGELOG, formal docs
and both human/machine risk records explain the verification scope.

No live-order setting was enabled or modified, no authenticated exchange endpoint
was called, no order was submitted, and no live-money exploration occurred. The
work introduces no self-promotion, production learning or order capability. It
preserves any pre-existing fleet authorization rather than falsely reporting that
the entire production fleet is disabled. No proof source contains a proof
placeholder or an ignored failed solver result. Neither draft is a deployment or
merge request, and neither is declared ready for candidate integration.

## Archive revalidation receipt

The current exporter streamed and verified the frozen archive against index SHA-256
`764fd123a1570c6b31ecc7e0729ef5dcc1fe19a39c29aee48efc1614c289974b`,
reconciled the roster and result summaries, and wrote to a new external directory.
All seven compact reports matched committed bytes, including all-seed results,
experiment registry, manifest, training, OPE, per-symbol/fold outcomes and summary.
The [receipt](formal-recheck-receipt-2026-09-20.json) pins the checker revision,
exporter and report hashes. No source archive or protected data was modified, and
no return path or policy was regenerated. Internal consistency and hash identity
do not establish economic truth, statistical significance or untouched evidence.
