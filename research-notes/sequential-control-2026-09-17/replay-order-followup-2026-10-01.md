# Replay event ordering — 2026-10-01

**Decision: retain scoped assurance evidence; no candidate adoption.** This audit
checks the unchanged Replay implementation with synthetic inputs. It introduces
no predictor, policy, training run, market-data access, OPE rerun, protected
outcome access or production behavior. Existing rejected financial results and
the champion remain unchanged.

The [contract](../../formal/research/replay-order-contract.md) and
[engineering registration](../registrations/replay-order-audit-engineering.json)
were committed as `c390a7a9` before verification probes. Starting head
`eb2379496f1197a47d794188a3783f02b501bfd2` passed
[CI](https://github.com/diegueins680/trader/actions/runs/36801751295).
Latest fetched main remained `dbd45e2691cb37f1421a23306c43b676fa82e6fc`.
Work continues on `research/sequential-review-2026-09-28`, draft PR #284 stacked
on #281; no merge or deployment.

## Canonical timing clarification

A-SEQUENTIAL-RESEARCH-E1 previously said every simulated fill follows its decision.
That wording conflated two existing operations. Policy-target fills are delayed
until at least the next bar (or two bars with the registered delay stress).
Compulsory terminal liquidation happens at the current endpoint after marking
old inventory and detecting terminal/risk conditions. The canonical statement now
makes both cases explicit and retains all charged costs. This resolves ambiguity
against the unchanged implementation and registered terminal semantics; it does
not relax a production gate or claim liquidation prevents an already incurred loss.

A pending target can fill on the last endpoint and then immediately liquidate.
The preserved fixture has equity 1, price 100, target 0.25 and default costs.
It records mark, target-trade attempt, liquidation attempt and one row; costs are
0.0005 of initial equity, final equity 0.9995 and inventory zero. Both entry and
liquidation costs are present. This is an expected simulator witness, not a new
counterexample, arbitrage opportunity or claim of realistic exchange execution.

Early invalid gate/market/observation failures differ from completed accounting:
they can retain simulated inventory and pending data, without a new bar row.
Insolvent terminal paths can retain inventory because solvent liquidation is not
attempted. Existing transition-admission rules reject incomplete learning samples.
No claim is made that disabling a call erases prior objects or liquidates a real
position. Production rollback/ownership/refinement obligations remain open.

## Formal evidence

| Requirement | Status | Exact scope |
|---|---|---|
| F-RL-REPLAY-DUE | smt_verified | Source-derived due=d+1+delay, delay in {0,1}; admitted policy-target attempts require a present pending target, clear prior risk and endpoint>=due>d. |
| F-RL-REPLAY-ORDER | model_checked | Complete reachable graph of one abstract call: admission, marking, risk, due target, terminal cancellation/liquidation and row publication ordering; conditional finite progress. |

The final model has **2,876 states, 3,474 edges, 84 distinct initial states** and
maximum shortest depth/longest terminating path **48 transitions**. Registered
configurations span remaining bars 1..6, horizons 1/3/6, delays 0/1 and pending
offsets absent/1/2; equivalent end bounds collapse to the 84 initial states.
Observation/data validity and risk outcomes branch nondeterministically. Final
states stutter. Nonterminal deadlocks, stutters and cycles fail the verifier.
The end node means early explicit return or reaching the final reward check;
non-finite reward can still raise there. A helper that never returns is outside
the conditional progress argument; no physical deadline is proved.

States with more than six remaining bars are outside this model, including a
nonterminal six-bar call with a later episode endpoint. No extrapolation to those
states is claimed.

The initial model probe had 2,820 states/3,418 edges. Before code freeze, explicit
ghost observation/data/risk/terminal-handling tags strengthened checks against
bypassed stages; the final counts above apply. This is a verification-model
refinement within the preregistered properties, not a changed simulator, financial
hypothesis, numeric domain or selected market outcome.

The SMT obligation uses separate SAT-premise and UNSAT-violation queries with
pinned Z3 4.15.4, seed 0 and a 10,000 ms limit. Full normalized AST hashes bind
Replay.step/_trade/_risk and Execution.__post_init__. Due and fill-guard expressions
are extracted into the restricted integer/Boolean translator. Source hash binding
alone is not a semantic proof of the entire source.

A-REPLAY-ORDER names stable ordinary fields, Python control flow, source-to-model
abstraction, and returning/exception-free primitive/helper calls before the final
reward check. Risk and solvency are abstract outcomes, not numeric theorems.
Target-fill events mean helper attempts, including zero/partial/missed fills.
Liquidation events mean helper calls, not universally correct cash or quantity
results. No concurrent writers, crash recovery, hidden helper effects, universal
interpreter refinement or real order capabilities are proved by this model.

## Model-to-code conformance

Actual Replay subclasses observe _risk and _trade calls while delegating their
original arithmetic. A list subclass observes row append. No arithmetic is replaced
or exception injected. Normalization is fitted only to the synthetic prefix through
the decision time. At the first risk check of each bar, old inventory and price/
funding accounting are checked within an explicit 1e-13 numeric tolerance. These
numeric comparisons are tests, not a proved binary64 error bound.

Observed mark/target/liquidation/row traces are matched against the model using
closure over unobserved steps and compared with final time, row count and pending
due state. This establishes bounded projected-trace conformance, not full bisimulation,
all-risk-branch coverage in the implementation or universal Python refinement.

The preregistered product grid contains **486 traces**: three targets, three horizons,
two delays, three remaining lengths, three initial-unit values and three next-bar
returns. Eight named cases additionally cover ordinary operation, invalid gate,
invalid market, solvent risk stop, insolvency, inherited pending target, partial
fill and missed fill; one explicit terminal round-trip fixture is retained.
Seven new integrity tests exercise those fixtures plus eight source mutants,
four SMT mutants, six model bypass/stall mutants, three invalid trace orders,
solver-unknown refusal and registration drift. Deliberate mutations are regression
probes, not discovered defects in the frozen simulator.

## Reproduction and verification

Use the existing pinned [formal toolchain](../../formal/research/README.md):

```sh
python scripts/formal/test_integrity.py ReplayOrderTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
python scripts/formal/verify.py --require-complete
```

At source freeze `843e1590404405a374ea09e5a95dd54e3524c308`,
`bash scripts/verify.sh formal` and `bash scripts/verify.sh full` both exited 0.
Results: 111 formal integrity tests, 37 scoped SMT requirements, Haskell checks,
241 web tests and 185 automation tests passed. The existing web bundle-size
advisory remains; no test or assertion was disabled. [Implementation CI](https://github.com/diegueins680/trader/actions/runs/36803840508)
passed formal, Haskell, web and automation; Docker/deployment were skipped.
Subsequent edits record this report and its evidence only.

The [evidence receipt](replay-order-evidence-receipt-2026-10-01.json) contains
commands, exit statuses, log hashes, tool versions, model/conformance results and
verification timings. The acceptance command exited 1 with
`ValueError: research acceptance blocked by open obligations`. This is the expected
refusal, not a passing acceptance check.
The acceptance command must continue to refuse the 38 open/partially verified
broader obligations. New scoped certificates do not authorize an offline, shadow,
paper or live candidate. No new dependency or configuration flag is introduced.

[Python's statement reference](https://docs.python.org/3.13/reference/compound_stmts.html#the-while-statement)
and [Boolean-expression reference](https://docs.python.org/3.13/reference/expressions.html#boolean-operations)
were checked for the assumed sequencing and short-circuit semantics. The runtime
remains pinned to Python 3.13.3. Language documentation is not an implementation
proof, and no new financial efficacy claim follows from these sources.

General recommendation: **no candidate passed**. RL recommendation: **continue
offline research**, while retaining rejection of the frozen tested configurations
for integration. Matched champion evidence, independent historical confirmation,
credible OPE, realistic execution, numerical guarantees and full lifecycle
refinement remain blockers. No protected period was opened and no live permission,
production fleet, ownership, leverage, margin, exposure cap or champion was changed.
