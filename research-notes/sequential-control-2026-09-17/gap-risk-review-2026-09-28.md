# Research and assurance decision — 2026-09-28

**No candidate passed. Preserve the champion. Reject the tested RL configurations
for integration. The full research mission remains incomplete.** This follow-up
adds useful scoped verification and current literature surveillance; it does not
claim untouched confirmation, a production safety proof or deployment readiness.

## Repository preparation and scope

Latest fetched main was `dbd45e26`. The original workspace was dirty on an unrelated
branch; it was preserved. Work uses `research/sequential-review-2026-09-28` in a
separate worktree. Existing [PR #281](https://github.com/diegueins680/trader/pull/281)
supplies the reviewed formal tooling at `1817742d`; this follow-up is stacked on
that branch, without merging it. [PR #282](https://github.com/diegueins680/trader/pull/282)
already owns the September 20 formal-literature refresh. Open issues were empty.
Open PRs #249–251 own market-context, HAR and missingness artifacts; they are not
duplicated. Dependency PRs #253–255 and #283 are unrelated.

The current [fidelity inventory](model-fidelity-audit.md) and earlier financial
audit remain valid: no files under `Trader/Predictors` or `Predictors.hs` changed
between the audited `f669c7e9` and latest main. Source inspection confirms TCN
fits ridge on fixed dilated lags; PatchTST fits ridge on patch summaries;
Transformer is softmax similarity-weighted memory. None is its namesake neural
architecture. Existing accurate aliases and saved semantics remain unchanged;
no redundant migration is introduced. GBDT, trees, KNN, HMM, LSTM, online MLP,
Kalman, quantile/conformal, technical sensors, ensembles, regime/context,
cross-sectional naming, optimizer and promotion gaps are retained in that audit.

The [literature extension](literature-update-2026-09-28.md) records the execution-date
search and five supplementary entries, including newly published FX RRL and
conditional offline shielding. It corrects one forecast-horizon field. The
existing 50/51-paper maps, detailed reviews, four-family shortlist and algorithm
scorecards remain the canonical broad review. No new financial candidate,
hyperparameter, reward, seed, checkpoint or selection experiment was run.

## Reproduced empirical evidence, with its limits

The complete external archive index was verified and all seven compact reports
re-exported byte-for-byte; see [receipt](gap-risk-evidence-receipt-2026-09-28.json).
This is integrity/reproducibility evidence, not a rerun or a fresh replication.

- Prior screen: 108 neural fits, 19,440 replays, 19,548 terminal registry records.
- All seeds **11, 23, 47**, cadences **1, 3, 6**, PPO, Double DQN, discrete CQL,
  and CQL without inventory penalty remain in [all-seed results](all-seed-results.csv).
- Inputs: already exposed 4,910-bar, ten-symbol Binance USD-M 8h development panel,
  with open times 2020-09-23T00:00Z through 2025-03-17T08:00Z. BTC/ETH are included;
  UNI/SUI/ETC from the reviewed AVAX/UNI/SUI/ETC/ADA fleet are missing.
- No independently untouched out-of-sample result exists. Later folds are outside
  each fit's prefix, but their development data was previously inspected.
- No final holdout opened: the historical 1,227-return reserve stays sealed;
  carry outcomes remain embargoed through 2027-01-20T13:00Z. This task adds no
  authority to access them. Existing prospective registrations remain unchanged.

| Fold | First decision close UTC | Last permitted outcome close UTC | Training stop, exclusive |
|---|---|---|---:|
| 0 | 2022-03-12 15:59:59.999 | 2022-09-28 07:59:59.999 | 1600 |
| 1 | 2023-02-08 23:59:59.999 | 2023-08-27 15:59:59.999 | 2600 |
| 2 | 2024-01-08 07:59:59.999 | 2024-07-25 23:59:59.999 | 3600 |

Each complete path has 599 outcome bars. Stopped paths have unequal endpoints;
their means cannot be ranked as matched-period investment performance.

Costs are assumed 5 bp fee + 0.5 bp half-spread + 4.5 bp slippage per turnover,
plus signed historical funding and terminal costs. Fills use the next close;
stress adds one bar. No actual maker/queue/fill calibration is established.

| Scenario | Failed RL paths / 1,080 |
|---|---:|
| Baseline | 686 |
| 1.5x costs | 688 |
| 2x costs | 697 |
| 25 bp extreme | 700 |
| Added bar delay | 693 |
| Double funding | 700 |
| 10 bp impact surcharge | 688 |
| Miss every tenth fill | 687 |
| Half fills | 454 |

Worst individual RL stress-path drawdown is **25.7378%**, one-bar ES95 **2.9923%**.
Cash has zero return/drawdown/ES. Matched champion drawdown/tail comparisons are
unavailable. A lower failure count with half fills does not prove an execution
improvement. The sole trained reward ablation does not rescue CQL. Full feature,
architecture and calibration ablations remain unperformed, not passed.

DSR, PBO, SPA, paired confidence bounds and economic superiority are unavailable.
The thresholds DSR>=0.95 and PBO<=0.20 were not weakened. All **108 OPE batches are
invalid**; no empirical ESS, uncertainty interval or estimator agreement is
admissible. Uniform simulated behavior (probability 1/3) does not establish actual
exchange support. Coordinate support is not joint state/action coverage.

## RL interpretation and engineering evidence

The implemented environment is partially observed inventory control: old inventory,
funding, delayed fills and turnover couple decisions. Its twelve observations are
six trailing price/volatility features and six inventory/equity/pending/episode
features. Targets are -0.25, 0, +0.25. Reward is 100 times relative equity change,
less an inventory-variance penalty; the penalty is not a cash debit. Discount is
0.99 per bar. The [versioned environment contract](environment-contract.md) gives
training-only normalization, episodes, terminal rules, artifacts and exact deviations.

Existing deterministic cost-aware ridge, supervised/logistic, contextual regression,
momentum/reversal, cash and constant-exposure baselines remain in the reports. No
matched current-champion/adopted-combo replay exists. SAC/TD3, distributional/CVaR,
model-based/robust MPC, imitation/offline variants and Decision Transformer remain
literature comparisons, not claimed implementations. No new policy was trained.

The simulator-to-independent-history/actual-execution gap is **unmeasured**.
Missing L2, historical filters, partial-fill identification, delistings, first-seen
bar witnesses and equivalent Coinbase/Kraken data block credible deployment claims.
Reward-hacking checks cover timing, funding, terminal costs, invalid data and
retained failures; they do not exclude every simulator exploit.

Original run benchmarks: wall 3,005.917 s, aggregate training 494.684 s, peak RSS
128.199 MiB, artifacts 6,741–6,913 bytes, maximum path inference p99 2.945 ms.
The 2 ms target was exceeded; two paths timed out at 20 ms. These are shared-host
observations, not guarantees. Production cold start, reload, concurrency and
preemptive timeout are unmeasured. This follow-up adds no inference work.

## Canonical specification, proofs and counterexamples

The [gap contract](../../formal/research/gap-risk-contract.md) was committed before
implementation (`1b2b9a1e`). It extends the existing canonical scope and resolves
the distinction between target bounds, cost-adjusted exposure and pathwise loss.
The existing simulator specifies endpoint monitoring; it cannot promise a strict
capital floor over arbitrary positive future prices. The resolution preserves
its behavior and refutes the stronger claim, rather than inventing a market axiom.

| Claim | Evidence class | Scope/result |
|---|---|---|
| Conditional gap-loss bound | `smt_verified` | Exact reals; bounded pre-bar exposure, price move and total cash debit |
| Drawdown composition | `smt_verified` | Prior and one-bar drawdown combine as d0+d-d0*d |
| Post-cost exposure | `smt_verified` | Full fill with explicit debit reserve; no floating-point universality |
| Unconditional 0.80 capital floor | `refuted` | Two SAT rational witnesses reproduced by actual Replay |
| Accounting conformance | `exhaustively_checked` | 180 finite two-bar cases; tolerance 2e-15, not all inputs |
| Haskell proposal contract | `smt_verified` / `property_tested` | Inherited abstract IEEE lemmas plus compiled conformance; no universal refinement |
| Two-caller lifecycle | `model_checked` | Inherited 75 states, 349 transitions, max shortest depth 6; fixed point |
| Whole-system obligations | `open` / `partially_verified` | All 38 retain gaps; acceptance must fail |

There are **12 UNSAT obligations**, including three new exact-real lemmas with
individually satisfiable premises. Z3 is 4.15.4; Python 3.13.3, NumPy 2.3.5 and
GHC 9.4.8 are checked/pinned. Each solver call has a 10 s timeout. Unexpected
SAT, UNKNOWN, source drift, missing evidence and stale receipts fail verification.
NumPy is the existing research dependency, now wheel-hash locked for this gate.
The official PyPI version metadata listed no NumPy 2.3.5 advisories on the
execution date; this is a bounded advisory check, not a security guarantee.
The first lemma's drifted-exposure premise is deliberately stronger than the
0.35 endpoint monitor: even a 0.25 full target becomes greater than 0.25 after
fees at an unchanged price. Thus it does not certify a maximal default fill.
The separate post-cost theorem states the reserve needed for a chosen bound;
no reserve or new sizing rule is installed in the trading system.

The abstract lifecycle has two callers and atomic model steps; under completion
fairness at most two completions drain it. It is not a verified production server.
Haskell conformance checks 16,384 representative combinations and 4,096 generated
binary64 cases; the 240 accepted proposals retain bits and false order authority.
Private-constructor rejection is compiled. No neural-network region or probabilistic
market model is certified, and no theorem-prover/compiler refinement is claimed.

New counterexamples: after an accepted 25% target, the long price path 100→12.5
ends at **24991/32000 = 0.78096875** equity; the short path 100→400 ends at
**199/800 = 0.24875**. Both include entry/exit costs and stop as `capital_floor`.
Pending proposals are cancelled and solvent inventory liquidated. Losses are not
clipped or discarded. Two invalid-price fixtures reject unaccounted learning
samples. These are engineering witnesses, not frequencies or financial trials.

Existing CE-RL-001 remains: disable cannot erase a caller-owned immutable proposal;
the retained proposal has no authority. The specification is resolved; actual
future risk-bound feasibility remains open. No production bug is inferred from
these offline counterexamples.

[Proof ledger](../../formal/research/proof-ledger.json) supplies bidirectional
requirement→statement→assumptions→model→artifact→implementation→test→CI links and
all 38 mission obligations. [Results](../../formal/research/results.json) and
[toolchain/source lock](../../formal/research/toolchain.json) bind reproducible
checks. `A-GAP` explicitly records unestablished return/debit bounds alongside
runtime/compiler, guard-truth, fairness, availability and real-arithmetic
assumptions. `RL-OFFLINE-001` stays HIGH/OPEN. Full proof coverage is not claimed.

## Delivery and remaining work

README, CHANGELOG, formal documentation, canonical clauses, risk evidence and
runbook are updated. No new configuration is needed; this delta does not change
`.env.example`. Only small deterministic fixtures and manifests enter Git.
The [verification receipt](verification-2026-09-28.md) records actual commands,
tool versions, successes/failures and limitations. A passing scoped formal/full
check does not discharge the deliberately open acceptance obligations.

General recommendation: **no candidate passed; no adoption**. RL recommendation:
**reject these configurations; continue offline research only under a new justified
registration**, preferring deterministic/supervised controls as complexity baselines.
Do not interpret this as proof that all RL methods fail. Missing matched champion,
fresh confirmation, OPE, market realism, full robustness/ablation and implementation
refinement prevent claiming mission completion. No shadow/paper candidate passes.

No orders, authenticated exchange endpoints, live exploration, merge or deployment.
No live flag, fleet UUID, leverage, margin, exposure cap, ownership, risk limit,
production identity or champion was changed. New research authority remains absent;
pre-existing production authorization was preserved, not falsely called disabled.
No policy directly authorizes an exchange order through this isolated boundary.
No self-promotion or production learning is introduced. Executable proof sources
contain no proof placeholders; that is distinct from having no open obligations.
