# Reward and economic accounting — 2026-10-03

**Decision: retain scoped assurance evidence; no candidate adoption.** The
unchanged replay's learning reward is relative to each call's starting equity.
It is not generally additive economic return. The existing economic reporter
correctly reports compounded wealth loss in the preserved counterexample.
No policy, training, financial trial, market-data retrieval, OPE rerun, protected
outcome access or production behavior is introduced.

The [contract](../../formal/research/reward-accounting-contract.md) and
[engineering registration](../registrations/reward-accounting-audit-engineering.json)
were committed as `8fde9176` before implementation/probes. Starting head
`3bb08dac9b6faee15bd2211db16bf4eea9520911` passed
[CI](https://github.com/diegueins680/trader/actions/runs/36807411630).
Latest fetched main remained `dbd45e2691cb37f1421a23306c43b676fa82e6fc`.
The temporary worktree had lost Git metadata and some files; a fresh isolated
checkout restored the committed branch at
`/Users/diegosaa/GitHub/trader-sequential-research-2026-10-03`. The user's dirty
checkout was not modified. The missing temporary Z3 installation was restored
in a new environment using the existing hash-pinned requirements, without
changing dependencies. Work remains on draft PR #284 stacked on #281.

## The witness and its economic meaning

With initial equity one, zero costs/funding/penalty, horizon one and target +0.25,
use a completed prefix at price 100 followed by prices 100, 110, 99. The first
call enters after the unchanged first price. The next call earns +2.5% and
rebalances; the last earns -2.5% and liquidates. Exact arithmetic gives:

- Call-relative returns: 0, +1/40, -1/40.
- Summed learning reward: 100*(0+1/40-1/40) = 0.
- Ending equity: (1+1/40)*(1-1/40) = 1599/1600.
- Whole-episode return: -1/1600 = **-0.0625%**.

Actual binary64 results are recorded in the [fixtures](../../formal/research/reward-accounting-fixtures.json):
reward sum approximately zero and final equity 0.9993749999999999. The economic
report returns approximately -0.000625, correctly retaining the loss. This is a
preregistered interpretation counterexample, not a newly discovered reporter bug,
an exploited market opportunity or evidence about historical prevalence.

The learning objective also discounts rewards and applies a noncash inventory
penalty. Those choices are already registered and are preserved. An accounting
identity does not prove that optimizing that objective optimizes a chosen net
risk-adjusted economic metric. No log-return replacement, reward retuning,
checkpoint selection or new financial trial follows from this audit. Cadences
remain separate hypotheses and cannot be pooled into a favorable result.

As theoretical context, the author-hosted abstract of
[Ng, Harada and Russell (ICML 1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
describes policy-invariance conditions for potential-based reward transformations.
The abstract was checked on 2026-10-03; the full theorem is not applied here.
No potential-based representation or policy-invariance theorem has been supplied
for this replay's inventory penalty. That absence is distinct from proving the
penalty inappropriate: a deliberate risk preference may change the objective.
This citation is theoretical context, not new financial efficacy evidence.

## Formal scope

Full normalized AST hashes bind Replay.step, Replay._trade, economic and
_admit_economic_ledger. Restricted extraction translates the mark increment,
cost debit and merge, row net return, penalty initialization/increment,
row penalty, returned reward and reported net-return expression. Unsupported
syntax and source drift fail closed. Hashes identify source; they do not prove
whole-interpreter semantics.

| Requirement | Status | Scope |
|---|---|---|
| F-RL-ROW-RECONCILE | smt_verified | Exact-real old-position mark, two sequential cost debits, merged row costs and row return. Unused calls contribute zero cost vectors. |
| F-RL-WEALTH-FOLD | smt_verified | Base and induction-step obligations for P=E/E0, plus the actual final-equity report when E0=1. |
| F-RL-REWARD-RECONCILE | smt_verified | Source penalty initialization, nonnegative invariant step and reward-plus-noncash-penalty identity for one call. |
| F-RL-REWARD-ADDITIVE | refuted | Prescribed SAT witness disproves universal equality between summed call rewards and total net return. |

Seven independent SAT-premise/UNSAT-violation pairs cover the three positive
requirements. The separately registered refutation requires SAT. Seed is zero;
each query has a 10-second limit, and UNKNOWN fails. Real scalars have no finite
numeric bound. The wealth and penalty arguments use base/step obligations;
ordinary induction extends them to finite paths under their premises. This is
not a machine-checked Python loop-refinement theorem.

For the wealth fold, require positive starting and intermediate wealth. For the
row identity, costs are supplied nonnegative real vectors; quantity formation,
fee computation and impact square roots are outside the proof. Stable ordinary
state, complete row publication, standard Python/NumPy/AST semantics and returning
helpers are assumptions. Existing ordering and cutoff certificates retain their
previous scope, including their 3,596-state class-seven ordering abstraction.
No new lifecycle, concurrency, authorization or neural-network claim is made.

The environment contract's earlier word “exactly” is now explicitly identified
as an exact-real accounting requirement. The actual reporter uses 1e-10 equity
and 1e-12 row-return tolerances. No universal binary64 rounding bound or finite
output theorem is established. Insolvency violates the positive-wealth fold
premise, and early gate/market failures may publish no row and retain state.
No failed path is reclassified as a completed economic success.

## Implementation conformance and failures

The registered product grid comprises **1,944 complete synthetic episodes** over
three targets, three horizons, two delays, three episode lengths, three cost
multipliers, three funding signs, flat/alternating prices and two penalty
coefficients. Prefix normalization uses only the 25 completed initial bars.
The alternating pattern is 100*(1+0.01*(-1)^i); all generation rules are registered.
Every call's reward is compared with new row penalties and relative equity change;
every row's cash ledger, compounded row returns, final equity and actual economic
report are reconciled. Absolute/relative tolerances are 1e-11. These are bounded
implementation tests, not formal floating-point proofs or market experiments.
Four additional fixtures cover invalid gate, invalid market, solvent risk and
insolvency. The negative-equity insolvency fixture remains outside the positive
fold theorem and retains its failure label.

The first probe exited one before any algebraic conclusion: ast.unparse wrapped
the cost generator in an extra pair of parentheses, defeating a string-prefix
check. Structural AST validation corrected this parser limitation; unsupported
cost keys remain rejected. The next probe passed. Seven initial targeted tests
passed; before freeze the audit additionally extracted the source penalty
initializer and checked nonnegative invariant preservation. All seven tests
passed again after that strengthening. No financial parameter or outcome was used
for that change.

Ten arithmetic mutations exercise wrong reward denominator/sign, an extra cash
debit, missing/double costs, missing liquidation merge, incorrect row return,
missing penalty scaling, wrong economic metric and nonzero penalty initialization.
Four full-source mutations, three actual-ledger tamper cases, unsupported ASTs,
UNKNOWN responses and registration drift also fail as required. Prior certificates
reproduced unchanged. The new checker maps to canonical requirements, assumptions,
implementation, fixtures, tests and CI; broader refinement remains open.

## Reproduction

Use the pinned [formal toolchain](../../formal/research/README.md):

```sh
python scripts/formal/test_integrity.py RewardAccountingTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
python scripts/formal/verify.py --require-complete
```

At source freeze `6ab5423dd75ef45ed5ef74aa69f75990607727c4`, both canonical
wrappers exited **0**. Results: **124 integrity tests**, **41 scoped SMT
requirements**, Haskell format/lint/build/smoke/tests, **241 web tests** and
**185 automation tests** passed. The existing web bundle-size advisory remains;
no test or assertion was disabled. [Implementation CI](https://github.com/diegueins680/trader/actions/runs/37155025632)
passed formal, Haskell, web and automation; Docker/deployment were skipped.
Subsequent changes record this report and its evidence only.

The [receipt](reward-accounting-evidence-receipt-2026-10-03.json) records commands,
exit codes, log hashes, tool versions, environment recovery, probe failure,
conformance results and timings. Targeted tests took 8.420 seconds after the
initializer/invariant strengthening. Standalone formal integrity/certificate
checks took 46.233/18.569 seconds; within full they took 50.592/21.974 seconds.
These shared-host verification timings are not production inference benchmarks.
There are 86 source locks; all earlier certificates reproduced unchanged.

`python scripts/formal/verify.py --require-complete` exited **1** with
`ValueError: research acceptance blocked by open obligations`. This expected
refusal is not a passing research acceptance gate. All 38 broader obligations
remain open or partial, and `RL-OFFLINE-001` remains HIGH/OPEN. No hidden proof
placeholder or ignored counterexample is used to obtain the scoped pass.

General recommendation: **no candidate passed**. RL recommendation: **continue
offline research**, retaining rejection of the frozen configurations for
integration. There is no new out-of-sample, final-holdout, cost-stress, statistical,
drawdown/tail-risk, matched-champion or inference-performance evidence. Protected
periods stay sealed. No live exploration, authenticated trading endpoint, order,
policy promotion, production learning, fleet, ownership, leverage, margin,
exposure-cap, deployment, merge or champion change occurred.
