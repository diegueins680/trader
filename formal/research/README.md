# Offline research verification runbook

This is a **scoped assurance gate**, not research acceptance. The rejected
sequential-control screen, champion, protected datasets and production permissions
are unchanged. Start with [the canonical contract and consistency resolutions](contract.md).

## Install and reproduce

Use Python 3.13.3 and GHC 9.4.8. The verifier uses the MIT-licensed
Z3 Python distribution 4.15.4.0 (solver reports 4.15.4) and the existing BSD-licensed
NumPy 2.3.5 replay dependency. Supported CPython 3.13 wheel hashes
are pinned for Linux/macOS x86-64/ARM64. Install once:

```sh
python3 -m venv /private/tmp/trader-formal-tools
/private/tmp/trader-formal-tools/bin/python -m pip install --require-hashes --only-binary=:all: -r scripts/formal/requirements.txt
export TRADER_FORMAL_PYTHON=/private/tmp/trader-formal-tools/bin/python
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

The formal command subsequently needs no network, market data, training artifacts,
exchange credentials or production services. It compiles a base-only Haskell
fixture driver in temporary storage. The existing full wrapper additionally needs
its normal Haskell, web and research dependencies. `verify:formal` is the npm
alias. CI uses the same formal wrapper, exact Python/GHC pins and hash-locked Z3.
The new CI job is a prerequisite for the existing build/deploy jobs; no deployment
command, identity, permission or live setting is changed.

A separate acceptance diagnostic deliberately fails today:

```sh
"${TRADER_FORMAL_PYTHON:-python3}" scripts/formal/verify.py --require-complete
```

It reports all 38 requested whole-system obligations as open/partially verified.
A green scoped gate does not make this draft ready for candidate integration.
There is no switch in this work to authorize a candidate, experiment or order.

## What is checked

- [Fifteen SMT obligations](../../scripts/formal/proofs.py): IEEE binary64 target
  bounds, evidence precedence, malformed/timeout fallback, disabled behavior,
  stable rescreening, constant lack of authority, integer slice/purge lemmas,
  and exact-real accounting identity. Every negated claim must be UNSAT within
  10 seconds. SAT and UNKNOWN are failures. Acceptance satisfiability is also
  checked so a gate rejecting everything cannot satisfy the suite vacuously.
- [Gap-risk contract](gap-risk-contract.md): three additional sufficient-condition
  lemmas over exact reals check jump/debit loss bounds, drawdown composition and
  post-cost exposure. Each premise has a SAT non-vacuity check. Two explicitly
  refuted unconditional-floor claims require their prescribed SAT witnesses.
  The [actual Replay](../../scripts/formal/gap_conformance.py) reproduces these
  failures and checks 180 two-bar exact-accounting fixtures (3 targets, 5 price
  ratios, 4 cost multipliers, 3 funding debits), with absolute tolerance 2e-15.
  No price-path frequency, binary64 universal refinement or real-market risk
  ceiling is established. The finite grid is engineering conformance evidence.
- [Source-linked causal footprint](causal-footprint-contract.md): a restricted
  AST dependency checker derives the actual feature slice and uses SMT to prove
  its bounds for all admitted integer indices. Metadata-only admission helpers
  have exact AST contracts; unknown calls, hidden inputs and unbounded array reads
  fail closed. Leakage mutations and 128 future-corruption cases exercise the
  checker and actual features/observations. The analyzer and NumPy primitive
  semantics remain trusted; publication timing and full replay-state causality
  remain open.
- [Training-prefix contract](training-prefix-contract.md): source-derived price
  and funding prefix stops, Scale.fit range and collector episode bounds satisfy
  six integer SMT implications with satisfiable premises. All nine registered
  fold/horizon cases satisfy the existing six-bar separation rule. CE-RL-005/006
  preserve deliberate fit leakage and episode-overrun mutants. Forty-eight
  normalization corruption cases and 27 paired collector cases supplement the
  certificate. Whole-panel admission, Python/NumPy/constructor semantics and
  complete transition refinement remain outside the proof.
- [Finite protocol](../../scripts/formal/lifecycle.py): all 75 reachable states,
  349 directed transitions, maximum shortest-path depth 6, two callers. Search
  reaches a fixed point, not a depth cutoff. Safety is invariant checking;
  conditional quiescence uses a rank of at most two outstanding calls. No fairness
  means no unconditional termination claim. The model is an abstract client,
  not the server's concurrency implementation.
- [Compiled Haskell conformance](../../scripts/formal/conformance.py): 16,384
  combinations from a declared representative domain plus 4,096 generated cases
  with seed 20260920. Binary64 values are transported as Word64 bits, covering
  signed zero, NaN, infinities, subnormals and adjacent threshold values. All
  240 accepted cases retain exact target bits and return false order authority.
  A negative compilation fixture also checks that the proposal constructor remains
  private. Random/generated testing does not prove universal implementation refinement.
- [Integrity regressions](../../scripts/formal/test_integrity.py): malformed and
  missing traceability, unsupported proof upgrades, source/receipt drift checks,
  duplicate/non-finite JSON, and deliberate unsafe state-model mutations.

The [results](results.json) contain deterministic outputs and source hashes;
wall time is printed per invocation and deliberately excluded from equality.
The [toolchain lock](toolchain.json) pins both manually translated source and
checker inputs. Hash matching binds the review to bytes; it is not a proof that
translation or a trusted compiler is correct. No production module was rewritten.

## Traceability and status

The [proof ledger](proof-ledger.json) is the bidirectional requirement/model/
assumption/artifact/code/test/CI index. Canonical IDs also appear in
[formal/specifications.json](../specifications.json). Every critical introduced
source and materially changed verification entry point has an inverse mapping.
Operational evidence is confined to reproducible offline checks and the prior
rejected research reports, with no order or deployment witness.

The ledger separates `smt_verified`, `model_checked`, `property_tested`, and
`open`; the schema accepts the requested vocabulary but the verifier refuses to
upgrade a claim to an unsupported verification class. The 38-item mission map
is deliberately more demanding than the narrow completed lemmas. It records
full implementation obligations as unresolved, even where a lemma is useful.
No broad “formally verified system” claim is made.

`CE-RL-001` is a preserved [counterexample](counterexamples.json) to an overly
strong specification: enable, start admissible call, disable, finish call. The
caller can retain a non-authorizing immutable proposal. The regression replays
this trace; the contract requires no order authority, not magical object erasure.
No unsafe exchange transition or production bug was demonstrated by this trace.

## Updating evidence

When a pinned file changes, investigate the proof/refinement impact, update the
contract and assumptions if necessary, and deliberately review updated source
hashes. Then use `scripts/formal/verify.py --record` to write results only after
all obligations check, inspect the diff, and rerun the normal wrapper. CI never
records or refreshes expected evidence. Never use pin updates to bypass a failed
property. Runtime timings are diagnostic, not reproducible performance claims.

## Tool choice, security and limitations

Explicit-state enumeration is sufficient for 75 states and avoids another runtime.
Z3 directly represents IEEE binary64; an exact-real theorem alone would miss
NaNs and infinities. A symbolic Haskell/SBV or Liquid Haskell core is a valuable
future way to reduce the translation gap, but adding a production dependency
without a validated challenger is unwarranted here. Lean/Coq/Isabelle can provide
smaller-kernel algebraic proofs; none is installed or claimed. PRISM/Storm need
credible transition probabilities before numerical market-risk claims are useful.
No probabilistic model-checking result is reported. Neural verifiers are not
added: no policy passed evidence gates, and no certified tanh policy domain is
claimed. The independent rejection gate remains the safety boundary.

Z3's [official security page](https://github.com/Z3Prover/z3/security) had no
published advisories when checked on 2026-09-20; this is not a vulnerability-free
guarantee. Its [release and license metadata](https://pypi.org/project/z3-solver/4.15.4.0/)
were checked. The pinned release is older than the latest release, chosen for a
stable reproducible binary64 verification environment; upgrades require rechecking.
Z3 runs on repository-controlled formulas, with a per-obligation solver timeout.
No native solver is exposed to network requests or model-supplied formulas.

Unproved: production ownership, server draining/recovery, real availability and
revisions, full simulator floating-point accounting, exchange rounding/margin,
artifact authenticity, end-to-end capability isolation, universal compiler
refinement, neural robustness, loss bounds through gaps, and future performance.
All empirical acceptance gates remain binding. `RL-OFFLINE-001` stays HIGH/OPEN.

## Offline artifact admission

The [contract](artifact-admission-contract.md), specified at `e12e0bd4` before
implementation, adds `F-RL-ARTIFACT-PATH` (27 states, 40 transitions, depth 13) and
`F-RL-ARTIFACT-METADATA` (two source-derived predicate SMT queries). Run the same
formal/full wrappers; no new tool or dependency is required. Six tests cover
source/model drift, vacuous/false predicates, CE-RL-007/008, actual-loader malformed
inputs, constructor ordering and the 65,536-byte boundary.

Primitive gate outcomes are abstracted, not proved. Source recognition is not a
verified interpreter; supplied provenance is not authenticated. The 75-state
lifecycle model and 27-state loader model are separate, not a composed system
proof. No production loader or policy is added. See the
[scope and receipt](../../research-notes/sequential-control-2026-09-17/artifact-admission-followup-2026-09-28.md).

## Training-transition admission

The [contract](transition-admission-contract.md), specified at `bcc816e0` before
implementation, adds `F-RL-TRANSITION-ADMISSION`. Three source-derived SMT checks
use unbounded integer indices and binary64 reward/equity/units. A collector source
audit binds admission before append. Six regression tests include CE-RL-009,
30 rejected helper fixtures and 36 actual replay cases. Existing formal/full
wrappers run everything; no dependency or research implementation changes.

Scalar/vector helpers and Python semantics remain trusted; terminal flatness
is not proof of complete cash accounting, correct failure classification or
simulator fidelity. See the [scope and receipt](../../research-notes/sequential-control-2026-09-17/transition-admission-followup-2026-09-28.md).

## Terminal learning-target numerics

The [contract](terminal-numerics-contract.md), specified at `409b8a89`, separates
exact-real reconstruction, finite binary64 masked products and two refuted
stronger claims. CE-RL-010/011 reproduce in the **unchanged** advantages function;
they are not guard-removal mutants. Three positive queries require SAT premises
and UNSAT violations; prescribed counterexamples require SAT and actual NumPy
regressions. No learner repair or new financial experiment is delivered.

Run the existing formal/full wrappers. Six additional tests include 27
well-conditioned terminal cases and downstream optimizer rejection. Batch
normalization and full learner/compiler refinement remain outside the proofs.
See the [findings and receipt](../../research-notes/sequential-control-2026-09-17/terminal-numerics-followup-2026-09-28.md).

## Isolated GAE target v2 kernel

See [contract](target-v2-contract.md), [counterexample disposition](target-v2-counterexamples.json) and [report](../../research-notes/sequential-control-2026-09-17/target-v2-followup-2026-09-28.md). Three scoped SMT requirements cover terminal reward-bit preservation, finite/disabled admission and exact-real recurrence. A separate 1..256-row publication model checks atomic output and a progress rank. Source skeletons, Python primitives and scalarization remain trusted; no normalized-advantage, whole-learner, market or production refinement is claimed. The kernel is default-disabled and disconnected from the frozen runner. Old numerical refutations remain valid.

## Independent numerical proof queries

The [query-isolation contract](query-isolation-contract.md) separates non-vacuity
and universal checks in the terminal-numerics and target-v2 proof drivers. Each
solver receives its complete assertions before its only check; witness constraints
are confined to the premise query. The pinned 10-second limit and failure behavior
remain binding. No automatic retry converts unknown/canceled results to success.
Run `bash scripts/verify.sh formal`; new protocol regressions are included. The
[engineering registration](../../research-notes/registrations/numerical-query-isolation-engineering.json)
authorizes three paired synthetic proof timings and zero financial trials.

## PPO surrogate audit

The [contract](ppo-objective-contract.md) is specified before the source-linked
checker. Two exact-real SMT requirements cover ratio/loss and coefficient branches;
[prescribed witnesses](ppo-counterexamples.json) refute unconditional numeric
finiteness and a universal clipping multiplier cap. `bash scripts/verify.sh formal`
runs the checks plus seven new regressions. Full softmax/NumPy/runtime refinement
and learner correction remain open. No trading or training code changes.

## Value-based objective audit

The [contract](value-objective-contract.md) precedes the target-slice and full
loss-function AST audit. Conditional exact-real SMT checks cover Double DQN
selection/evaluation and CQL gradient components. [Numeric witnesses](value-counterexamples.json)
show NaN loss with a zero gradient at alpha zero, and conservative-loss cancellation
under a common shift. Logarithm/softmax calculus and runtime semantics remain
trusted. The formal wrapper runs seven new regressions; no training code changes.

## Optimizer publication audit

[Specification](optimizer-publication-contract.md), [counterexample](optimizer-counterexamples.json)
and `optimizer_publication.py` check unchanged source structure, finite-state
staging and conditional exact-real gradient clipping. The observer extension
preserves partial-publication CE-RL-016; actual opcode-instrumented tests distinguish
failure before publication from interruption between stores. No unconditional
atomicity, floating-norm, Adam convergence or full implementation theorem is claimed.
The existing formal/full wrappers run the new checks without new dependencies.

## Inference admission and liveness audit

The [contract](inference-boundary-contract.md) binds the unchanged inference predicates, action constants and one-call admission model. Two conditional SMT results cover measured-time guards and first-maximum action selection. CE-RL-017 preserves the pending-call lasso; no timeout or cancellation transition exists. Output validation/selection follows the final clock read. Ordinary Exception fallback is distinct from BaseException, clock or predicate failures. No full runtime/neural refinement, end-to-end latency guarantee or production authority is established. Existing formal/full wrappers enforce this scope.
