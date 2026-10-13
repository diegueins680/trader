# Current-main RL replication, 2026-10-13

The decision remains **no RL adoption**. This continuation reuses the existing
causal target-exposure research implementation and reproduces its five-seed
budget study on verified main. It is engineering replication of consumed
development evidence, not a new independent financial trial. It cannot establish
champion superiority, and the user explicitly authorized continuing without the
missing historically faithful champion export.

The starting SHA, recorded before changes, is
`0b76f2b879f75f5c669fe944a14ea6700d446240`. GitHub's `CI` workflow reported success
for this exact main commit on discovery ([run 38313512581](https://github.com/diegueins680/trader/actions/runs/38313512581)). Branch:
`research/rl-causal-campaign-2026-10-13`. The 15 research-only commits through
`f512d7a8` were transplanted from the existing RL branch; original registrations,
trial identities, failed policies and source SHAs are retained. These are reused
deliverables, not claimed as newly authored or newly observed research. The
separate deployment changes in existing [PR342](https://github.com/diegueins680/trader/pull/342)
were excluded. The accounting sign-case proof decomposition from `1890739c` is
reused with its original premises, claim and ten-second per-query deadline.

The [replication registration](../registrations/rl-reproduction-2026-10-13.json)
was committed as `8e83b2d9` before the new run. It freezes all five seeds
(11, 23, 47, 71, 101), 50 fits, 1,218 stress replays, the original checkpoint
rule, and absolute/relative comparison tolerances of 1e-12. No tuning, new
checkpoint choice, best-seed choice, holdout opening or new selection trial is
permitted. The original 180 fits across the October studies remain distinct
from the 50 repeated fits; the September 108-fit screen remains older evidence.

## Decision problem and implementation

One primary use case: propose target exposure −25%, 0%, or +25% for independent
single-symbol futures accounts. Position inventory, transaction costs, delayed
fills and funding make this sequential. Market observations do not establish a
fully observable or stationary MDP. There is no allocation or high-frequency
execution claim and no new predictor research.

The runnable path is causal close-history observations → sampled/greedy policy
proposal → deterministic support/turnover admission → next-close simulated fill
→ exact rational unit/funding/cost ledger → archived evaluation. Policy actions
and likelihoods describe **proposals**; many proposals can have the same shielded
outcome. Labels are not overwritten with executed targets. Gamma 1 and the
unshaped reward `100 * (equity_after - equity_before)` preserve the finite-horizon
economic sum. Fees/funding enter only through equity. A terminal step pays
liquidation costs, clears pending targets and stops; a rollout batch cut retains
the episode and bootstraps. Training episodes have 32 transitions, evaluation 64;
that horizon shift is disclosed, not treated as continuing-operation validation.

| Learner | Actual mechanism and training regime |
|---|---|
| PPO | Fresh 128-step sampled simulator rollouts; categorical actor, separate critic, GAE λ=.95, clipped surrogate .2, four epochs; 4,096 interactions, 128 actor updates |
| Double DQN | Fresh epsilon-greedy simulation plus replay; online action selection/target evaluation; batch 64, target synchronization every 100 updates; 4,096 interactions and updates |
| CQL | Scalar discrete CQL(H)-inspired penalty α=.1 plus Double-Q targets; 4,096 updates on a frozen 512-transition simulated dataset per fold; no training collection |
| CQL α=0 | Offline extrapolation diagnostic, not an additional suitable offline RL learner |
| Behavior cloning | Supervised proposal likelihood on the same fixed buffer; not RL |

All use the same twelve causal observations, 16-unit tanh network and admission
constraints. Cash, constant-long, normalized momentum and normalized reversal
are deterministic controls. The normalized rules read observation[1], avoiding
the earlier raw-versus-normalized comparison mismatch. PPO's registered frozen
evaluation extracts argmax rather than sampling; this is an explicit policy
extraction choice. Neural inference and training remain isolated Python research;
Haskell retains the existing production safety/authorization boundary and exact
conformance oracle. No research artifact is accepted by a production loader.

The current tracked fleet is 18 symbols: AVAX, UNI, SUI, ETC, ADA, AAVE, ARB,
ATOM, BCH, DOGE, DOT, ETH, FIL, LINK, NEAR, OP, SOL and XRP. This supersedes the
older review's fleet snapshot. The experiment remains the registered BTC/ETH/SOL
subset; adequate development histories for other symbols are **not evaluated**
by this replication. The actual champion is a pinned-combo fleet through the
existing Haskell predictors and deterministic signal/risk/execution gates, not
one interchangeable learned model. Pinned configuration does not provide causal
historical weights or a lawful matched baseline. No production client, database,
credential or runtime snapshot was used.

Direct inspection again confirms `PatchTST` is patch-summary ridge regression,
`TCN` is dilated-lag ridge regression, and `Transformer` is stored-example
similarity attention. Existing identifiers/aliases and artifact semantics are
unchanged. None of those outputs enters this study, so no gain can be attributed
to improved predictors.

## Data and evidence limits

Only the two already authorized Binance USD-M development exports are read.
Panel SHA256: `09a09f0e1065733be623625fa0d67e6a67dad92c53cd8fa92b6d9caa1040674a`.
Settlements SHA256: `2ccba74489f96b9ce0e58842594a524b7c306012086443cbea7305d4011ee899`.
Both immutable byte snapshots are hash-admitted before parsing. Provider terms
apply; no raw data is redistributed. The panel has 4,910 eight-hour bars per
symbol. Training cutoffs are 1600 and 2600; evaluation intervals are
[1606,1671) and [2606,2671), giving 64 outcomes each. A strict 96-bar pre-decision
burn-in fits each episode's immutable scale. Rewards mature inside training.
There is no post-test training region to embargo. The six-bar separation is not
represented as two additive purge/embargo regions.

Availability is **assumed** close+1 ms, not independently witnessed first-seen
time. Funding settles against old inventory before a due target fills at the
next close. Prices and funding are exogenous; fees, fills, units and equity are
action-dependent. Base turnover costs are 5 bp fee + .5 bp spread + 4.5 bp
slippage. Stresses use 1.5×/2× costs, 50 bp severe costs, one added bar of delay,
50% partial fills and doubled signed funding. No calibrated market impact,
historical precision/minimum-notional filters, order queue, margin liquidation
or venue outage model is asserted.

These periods were already consumed by many adaptive trials. The 1,227-return
protected historical holdout and prospective carry outcomes through 2027-01-20
are untouched. No fresh holdout is manufactured by renaming this run. Real-log
propensities, rejected actions and overlap are unavailable; no real IS/DR/FQE or
effective sample size is fabricated. Uniform simulated proposals have probability
1/3, which is not the executed-action distribution after the shield. Replay is
model-based counterfactual evidence only. Block bootstrap intervals average
seeds/symbols before resampling chronological increments; two short dependent
windows do not establish robust significance. DSR/PBO remain insufficient.

## Assurance scope

The existing single assurance ledger is
[`formal/research/proof-ledger.json`](../../formal/research/proof-ledger.json).
Its original 38 obligations have 20 scoped closures and 18 unresolved obligations;
this continuation does not promote any claim. Critical integration gaps remain.

| Required boundary | Evidence and remaining limitation |
|---|---|
| FM-01/02 causality/partitions | Prefix-only immutable burn-in, integer SMT index/order lemmas, future-mutation and cutoff tests; authentic timestamp/model-vintage evidence absent |
| FM-03 accounting | Bounded exact Fraction ledger, reward telescoping, Haskell rational witnesses and trace reconstruction; venue fills/liabilities outside this simplified model |
| FM-04/05 actions/risk | Discrete typed-domain checks, support projection and verified zero-target turnover admission; no complete venue precision/minimum-notional/portfolio proof |
| FM-06/07/11 authority/lifecycle/noninterference | Disconnected default-disabled research interfaces, Haskell capability boundary, source bindings and finite lifecycle checking; not an OS sandbox proof or production compiler refinement |
| FM-08 failure | Invalid observations/artifacts, stale bars, state inconsistency, numerical failure and terminal fixtures fail closed; historical outage and bounded production inference not established |
| FM-09 atomicity | Immutable preview and one pending reservation; finite model with two proposal identities, 8-step bound; not a distributed/shared-production-budget proof |
| FM-10 compatibility | Version/dimension/provenance/hash checks; no authenticated production loader or proof that hashes establish profitability |
| FM-12 recovery | No duplicate modeled pending consumption, terminal clearing and exclusive archive creation; persistent execution recovery/rollback remains outside this isolated path |

The composed checker has seven SMT composition lemmas and two admission lemmas,
a 43-state/107-transition bounded model, and eight exact Python/Haskell ledger
witnesses. Other reused kernel certificates have their own bounds in the ledger.
SMT premises are checked satisfiable and violations unsatisfiable; unknown or
timeout is failure. Exact arithmetic has an 8,192-bit resource cap. CPython,
Fraction, NumPy, GHC, Z3, timestamps and specified disturbance/fill assumptions
remain trusted. Property tests and archived trace checks are not theorem proofs.
No proof guarantees profit, future drawdown, solvency or fills under arbitrary
gaps, outages or venue failure.

The new `compare_rl_replication.py` snapshots and authenticates both full archive
inventories, reconciles all terminal events, compares every policy parameter and
economic path, and requires exact fixed-buffer equality. Only named identity and
resource fields may differ; explicit old-schema defaults are validated. A report
cannot be written into either archive. Regression tests reject tampering,
duplicate JSON, nonfinite values, path traversal, symlinks, incomplete rosters,
changed actions and changed model parameters. This is reproducibility evidence,
not a new safety theorem. The canonical formal/full wrapper runs those tests.

## Search refresh through execution date

The inherited [50-entry paper-to-code matrix](paper-matrix.csv) and
[focused review](review.md) remain the main review, including PPO, SAC/TD3,
Double/Dueling/distributional DQN, BC/CQL/IQL/BCQ/TD3+BC/AWAC, OPE, recurrent and
safe RL, finance baselines and negative evidence. This refresh is not described
as 50 new full-paper readings. Search date: 2026-10-13. Queries:

- `reinforcement learning reliable evaluation trading offline RL 2026 CQL IQL flow Q learning`
- `site.proceedings.mlr.press Agarwal 2021 deep reinforcement learning edge statistical precipice`
- `site.arxiv.org proximal policy optimization conservative q learning implicit q learning safe reinforcement learning shielding`
- `site.proceedings.mlr.press 2025 flow q learning Park`
- `site.arxiv.org 2026 offline reinforcement learning trading evaluation failure costs`
- `site.github.com google-research rliable license`

Primary-source metadata rechecked: [PPO v2](https://arxiv.org/abs/1707.06347v2),
[Double DQN v3](https://arxiv.org/abs/1509.06461v3),
[CQL v3](https://arxiv.org/abs/2006.04779v3),
[IQL](https://arxiv.org/abs/2110.06169), and
[shielding v2](https://arxiv.org/abs/1708.08611v2). CQL section 3 explicitly requires
support and sampling/function-approximation conditions; our fixed α=.1 and small
nonlinear network do not inherit its lower-bound theorem. PPO's HTML endpoint
was unavailable as useful full text; the retained prior technical review and
current abstract metadata are distinguished from a fresh full-paper read.

[Flow Q-Learning](https://proceedings.mlr.press/v267/park25f.html) is ICML 2025;
the [official implementation](https://github.com/seohongpark/fql) is MIT-licensed.
Its expressive continuous policy is not justified for this three-action pilot.
[rliable](https://github.com/google-research/rliable) is Apache-2.0 and supports
reliable RL comparisons, but benchmark-task resampling does not justify counting
correlated trading symbols/seeds as independent markets. Neither library is added.

[MetaTrader](https://ojs.aaai.org/index.php/AAAI/article/view/40027) is published
AAAI 2026, DOI 10.1609/aaai.v40i33.40027. Its bilevel transformation/conservative-TD
approach addresses offline overfitting, but two equity datasets do not establish
crypto efficacy. It is a screened extension, not implemented; independent
replication, implementation license and cost transfer are not verified here.
[Causal FQL](https://arxiv.org/abs/2602.02847) has an
[author repository](https://github.com/CausalAILab/Causal-Flow-Q-Learning-for-Robust-Offline-Reinforcement-Learning)
with MIT license and an author-reported ICML 2026 designation. Its pixel-observation
confounding problem is not evidence that it repairs missing financial support.
It is monitored, not selected; no independent trading replication is established.

[Abbade and Costa, market-impact environments v2](https://arxiv.org/html/2603.29086v2)
is an April 4, 2026 preprint, screened in full HTML. It compares five RL algorithms
on NASDAQ 100 with flat versus nonlinear costs and hyperparameter optimization.
The reported ranking changes support cost-model sensitivity, not adoption here.
Different rewards, equity universe and tuned parameters limit transfer;
independent replication and code license are unverified. Its action-dependent
impact assumptions require calibration unavailable in our close-only panel.
No paper PDFs or third-party implementation code were committed.

Methodological credibility, fit to this decision problem, and demonstrated
improvement remain three separate questions. The search supplies no reason to
change the registered replication or reverse the existing economic rejection.

## Reproduction and publication

Use CPython 3.13.3, NumPy 2.3.5, Z3 4.15.4.0, GHC 9.4.8, Cabal 3.12.1.0,
Node 20.19.0, fourmolu .15.0.0 and hlint 3.8. Execute research with a clean
environment and one BLAS/OMP thread. The exact command and run receipt are
recorded with the final results. Choose a fresh output directory:

```sh
python scripts/research/run_rl_budget_v3.py --run-registered-budget-v3 \
  --panel /path/to/registered-development-panel.csv \
  --settlements /path/to/registered-development-settlements.csv \
  --output /path/to/new-replication-archive
python test/rl_replication_test.py
python test/rl_composition_test.py
bash scripts/verify.sh full
```

The runner requires committed, hash-identified source. For full-archive comparison,
use `python scripts/research/compare_rl_replication.py --help`; supply the frozen
parent index hash above and the new archive's actual index hash. Large archives
and policy parameters remain outside Git. Do not regenerate or inspect protected
full snapshots when the exact development exports are unavailable.

No push or new PR is made. `.github/workflows/autoloop.yml` runs every six hours;
`scripts/autoloop.mjs` sets `LOOP_BRANCH=BASE_BRANCH`, enables review polling by
default, and reads `pulls?state=all`. Its repair path can push the default branch,
whose push triggers deployment. A draft designation is not an isolation boundary.
Existing PR342 is left untouched. Local commits and a patch preserve this work.
No persistent service, scheduled training, real order, live exploration or new
live authorization is introduced. Existing production activity is preserved.
