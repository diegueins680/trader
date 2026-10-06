# Offline research verification runbook

[Quantity rounding](quantity-rounding-contract.md) adds exact integer/real and IEEE binary64 guard lemmas, an independent Fraction oracle (4226 compiled cases), source mutation tests, and production adapter generated properties. No new dependency, temporal state, model integration or financial trial. Run the existing formal/full wrappers; broader rounding remains partial.


Closure correction (2026-10-03): [contract](closure-contract.md), [all-38 audit](obligation-closure-audit.md), and [CE-RL-021](closure-counterexamples.json). Source-derived default paths and [capability isolation](capability-isolation-contract.md) close obligations 11, 24 and 31; [data composition](data-composition-contract.md) additionally closes 3 and 5 for the delivered offline boundary. Same-run certificates are required for closure; 33 other obligations and both research acceptance gates remain blocked.


Reward accounting (2026-10-03): [contract](reward-accounting-contract.md), [registration](../../research-notes/registrations/reward-accounting-audit-engineering.json), [fixtures](reward-accounting-fixtures.json). Seven SAT-premise/UNSAT-violation pairs cover three real-arithmetic requirements; a separate SAT witness refutes additive reward-as-return interpretation. Both wrappers run 1,944 synthetic episode checks. No new dependency.

Replay cutoff (2026-10-01): [contract](replay-cutoff-contract.md), [registration](../../research-notes/registrations/replay-cutoff-audit-engineering.json) and [fixtures](replay-cutoff-fixtures.json) preserve the earlier six-bar certificate while adding an integer branch lemma, a 3,596-state graph and longer-call traces. Both canonical verification wrappers reproduce the scoped evidence offline.

Replay ordering (2026-10-01): [contract](replay-order-contract.md), [registration](../../research-notes/registrations/replay-order-audit-engineering.json) and [fixtures](replay-order-fixtures.json) connect a one-call finite model, due-time SMT and 486 synthetic actual-helper traces. `bash scripts/verify.sh formal` and `full` reproduce the scoped checks. No new dependency or simulator change.

Funding boundary audit (2026-10-01): [contract](funding-boundary-contract.md), [registration](../../research-notes/registrations/funding-boundary-audit-engineering.json), and [CE-RL-019](funding-counterexamples.json) connect conditional timestamp proofs to the unchanged loader and synthetic regressions. `bash scripts/verify.sh formal` checks the SMT/loop scope; `bash scripts/verify.sh full` also exercises the full CSV loader and replay refusal. No new dependency or market data.

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

It computes 33 unresolved obligations and five closed affected-scope obligations. The count is not hardcoded.
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
explicit remaining implementation blockers; sufficient source-bound certificates can close an affected-scope obligation. Supporting lemmas alone do not establish closure.
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

## OPE algebra and underflow audit

The [contract](ope-algebra-contract.md) fixes two-trajectory exact-real ESS/WIS
bounds and scale invariance, plus DR telescoping for horizons 1..6 conditional on
unit weights, q_t=v_t and zero terminal bootstrap. Eight independent SAT-premise
and UNSAT-violation checks cover three requirements. The [prescribed fixture](ope-counterexamples.json)
refutes positive-weight ESS positivity in the public binary64 helper under ignored
underflow; raised underflow produces an explicit error. All 64 deterministic
six-step target patterns and 201 support-count cases check the current restricted
weight domain separately. These are engineering cases, not market OPE evidence.
Eight new integrity tests run in formal/full; no new tool or frozen-estimator change.
Runtime refinement, unbiasedness, confidence coverage and behavior support remain open.

## Isolated exact ESS diagnostic

The [contract](ess-v2-contract.md) specifies native immutable admitted weights,
exact Fraction conversion/accumulation, zero-mass output and bounded positive ESS.
Source-derived SMT base/step/bound lemmas and a 1..256-row publication model
run in formal/full. The model has 66,309 reachable states, 99,718 transitions,
maximum shortest depth 515 and strict rank bound 517 under terminating primitives.
Eight new tests cover extreme weights, invalid admission and exact-reference
conformance. `ess-rational-v2` defaults disabled and is not imported by existing
research consumers. Dynamic import exclusion and full Python/Fraction refinement
are not proved. CE-RL-018 remains in the frozen helper; no statistical or policy
claim is repaired by this independent arithmetic diagnostic.

## Capability source extraction

`ComponentGraph.hs` compiles with `ghc -package ghc-9.4.8 -package Cabal-3.8.1.0`.
Cabal 3.8.1.0 here is the parser library bundled with pinned GHC 9.4.8; the command
line Cabal remains 3.12.1.0. No new package download is required. The formal wrapper
builds the parser in a temporary directory and runs without network access.

The pinned inventory has 116 application source files and 286 local import edges.
All six executable roots are traversed to a fixed point, without a depth cutoff.
Parsed declarations bind the private pure Haskell surface and native JSON decoder;
three Python inference helper ASTs bind a reviewed primitive-effect contract.
The schema SMT query uses 12 Boolean membership variables, checks a satisfiable
premise and an unsatisfiable violation. It does not verify profitability or native
Aeson implementation semantics. Three rejected type forgeries and inference-state
snapshots are conformance tests. Mutation tests exercise missing/transitive imports,
source generation, build branches, new writes/calls and schema overlap.

Re-review is required on source/inventory/schema/helper/packaging drift. Runtime
binary/PATH integrity and no injected code are named assumptions, not checked OS
isolation. Existing production authorization and existing champion learning are
outside these closures. The legacy GHC 8.10.4 Docker recipe remains unmodified and
is not certified buildable; actual deployment images were not inspected or changed.

## Offline snapshot optimizer v2

[Specification](optimizer-snapshot-v2-contract.md) and
[engineering registration](../../research-notes/registrations/optimizer-snapshot-v2-engineering.json)
precede this standalone research repair. `scripts/research/optimizer_snapshot_v2.py`
provides `create_v2`, `update_v2` and `forward_v2`; each defaults `enabled=False`.
Every enabled operation requires CPython 3.13.3 with the GIL and NumPy 2.3.5.
Creation uses seed 0..2^32-1 and output width 1 or 3. Updates accept finite native
float64 batches of 1..256 rows and lr in (0,1]; callers inspect None on refusal.
The public snapshot contains immutable bytes, not writable NumPy buffers.

Forward and update capture exactly one state. A writer stages independently and
publishes one immutable object only if its expected state still matches under the
nonblocking publication lock. An exceptional call after the store can have committed;
there is no exactly-once or durable retry promise. A stranded lock fails later writes
closed; this is not a recovery or hard timeout implementation. No frozen runner calls
this API and no candidate artifact/production interface is added.

The finite model checks 3,970 states and 9,981 transitions (two writers, two calls
each, one retained reader; depth 29; progress rank 49). Two SMT requirements check
integer conditional-publication bounds and the actual selected-slot float64 pack
guard. NumPy scan/byte semantics and CPython reference/lock behavior are named
assumptions; no universal numerical-error or compiler refinement theorem follows.
Conformance exercises 144 paired synthetic updates, concurrent schedules, failure
paths, retained reads and defaults. Existing Haskell proposal conformance remains
in the wrapper; no Haskell production path changes.

CE-RL-023 preserves an initial view-backed forward discrepancy of one ULP on the
observed macOS backend. Native private working copies restore the baseline path on
the registered tests. This is bounded empirical parity, not all-backend bitwise
proof. The original CE-RL-016 remains in frozen v1. Obligations 33/34 gain partial
implementation evidence; 35 broader obligations remain unresolved. Formal/full
wrappers reproduce all new scoped checks without network after pinned installation.

## Offline inference process v2

[Contract](inference-process-v2-contract.md), [registration](../../research-notes/registrations/inference-process-v2-engineering.json)
and [source/control lock](inference-process-source.json) describe the separate
`haskell/research/InferenceProcessV2.hs` executable. Its default and unknown modes
return `(Absent,True)` before input or process creation. Explicit
`--offline-inference-v2` initializes one child, then admits one bounded synthetic
request. There is no saved-policy or live interface. The child has an empty
environment, fixed self-binary arguments and no descendants or parameter writes.
The Boolean result reports cleanup success; `(Absent,False)` is an explicit
cleanup/launch failure and must never be treated as an admitted proposal.

Reproduce all evidence, including actual compiled subprocess fault tests:

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

Standalone engineering build (GHC 9.4.8 and its pinned bundled libraries):

```sh
mkdir -p /tmp/trader-inference-v2-build
ghc -O0 -threaded -with-rtsopts=-V0.001 -package process-1.6.18.0 \
  -package unix-2.7.3 -package bytestring-0.11.5.3 -outputdir /tmp/trader-inference-v2-build \
  haskell/research/InferenceProcessV2.hs -o /tmp/trader-inference-v2
/tmp/trader-inference-v2
```

Use only synthetic fixtures through `scripts/formal/inference_process.py` for
this engineering registration. The 12–16–3 tanh evaluator reads 12 observations
and 259 row-major parameters; it is not a claimed NumPy-equivalent policy loader.
The three output codes represent -1/4, 0 and 1/4 exposure proposals. Ties and
invalid/non-finite/out-of-bound input abstain. No order constructor is present.

Initialization has a separate 1 s ready wait; no request is read before Ready.
The unchanged request budget is 20 ms and includes parsing, computation, cleanup
and the final monotonic guard. A failed or late result cannot be reused because
there is no second request. Cleanup sends TERM, polls for 50 ms, then sends KILL
and polls for another 50 ms. Failure to reap or close pipes is explicit. These
budgets depend on bounded OS primitives and scheduler service; they are not an
unconditional wall-clock theorem. The first cold-start-inclusive design failed
its timing goal and is counted as engineering variant 1, not silently discarded.

The source-bound model has 57 states/143 transitions, depth 8, initial progress
rank 11, one child, one request, three request-time buckets and two abstract poll
steps per window. Z3 checks the unbounded-integer admission predicate. Compiled
conformance covers 226 guard cases, 12 synthetic networks (seeds 11/23/47), ten
normal attempts, seven invalid requests, held-open input, three default modes
and eight worker fault fixtures. Actual PID disappearance is checked after each
fault. Neural arithmetic, arbitrary OS schedules and full IO refinement are not
proved. The frozen learner still has CE-RL-017. No financial trial, holdout access,
champion change, deployment or broader obligation closure follows.


Shutdown follow-up: [contract](shutdown-deadline-contract.md),
[source lock](shutdown-source.json), [counterexamples](shutdown-counterexamples.json).
The formal wrapper compiles actual production budget functions, checks 580
integer cases (256 generated, seed 20261004) and eight runtime tests, alongside
integer SMT and a six-stage model. No new dependencies. The machine-readable
ledger distinguishes those three evidence classes and retains all broad blockers.

Worker registry follow-up: [contract](worker-registry-contract.md), [source lock](worker-registry-source.json), [counterexamples](worker-registry-counterexamples.json), and [engineering report](../../research-notes/sequential-control-2026-09-17/worker-registry-followup-2026-10-04.md). The formal gate compiles old/new witnesses plus production tests, checks the finite protocol and atomic SMT predicates, and rejects drift. Full IO refinement and whole-server shutdown remain unresolved.

Async admission follow-up: [contract](async-job-admission-contract.md), [source manifest](async-job-admission-source.json), [counterexamples](async-job-admission-counterexamples.json), and [report](../../research-notes/sequential-control-2026-09-17/async-job-admission-followup-2026-10-04.md). The existing gate reproduces signed-Int SMT, finite ownership/publication transitions and compiled core tests. Main remains responsible for exception-safe publication; durable recovery and HTTP draining are not proved.

Data composition (2026-10-04): F-RL-DATA-COMPOSITION checks every scale use in
the four delivered screen modules (28 sites); F-RL-DATA-KEYS proves six extracted
key equalities over arbitrary strings. The actual loader/fit/collect/replay/OPE
fixture runs in the existing automation suite, which already pins pandas 2.3.3.
The formal-only environment remains NumPy/Z3-only. Causal normalization and
symbol isolation close under A-DATA-COMPOSITION; publication timing, full split
isolation, state/accounting and lifecycle remain separate obligations.


## PPO successor v2

`bash scripts/verify.sh formal` runs the source-bound publication model, five
independent SAT-premise/UNSAT-violation queries and `PPOSuccessorTests` against the
actual new offline training function. Reproduction uses pinned existing Python
and NumPy; no new dependency or dataset is required. Synthetic seed/horizon/step
coverage, generated cases and deliberate failure/source mutations are described
in [the contract](ppo-successor-v2-contract.md). There is no CLI, artifact writer,
production selector or promotion capability. No financial trial is run by CI.
Use `verify.py --require-complete` to observe the still-blocked whole-mission gate;
a scoped `formal` pass is not research completion.

## PPO snapshot/process bridge v3 (2026-10-05)

The [contract](ppo-process-bridge-v3-contract.md) and
[source inventory](ppo-process-bridge-source.json) bind the pure encoder/decoder
and existing supervisor. `formal` and `full` reproduce codec SMT, the 83-state /
189-transition composed process model, source inventory and nine synthetic PPO
fits across seeds 11/23/47 and horizons 1/3/6 (17 steps each). Twenty-seven
observation requests, bit/invalid-input fixtures and actual timeout/cleanup tests
connect the model to code. No historical data is read. Proof ledger entries
F-RL-BRIDGE-V3-* provide bidirectional file/test/CI traceability.

The bridge fixture uses GHC9.4.8/base4.17.2.1 with `-O2`; the original v2 tests
retain `-O0`. The first unoptimized bridge run did not satisfy per-policy success
within the unchanged 20 ms deadline; optimization is an explicit engineering
build choice. Neither optimized code nor tests establish a real-time OS guarantee.
Reproduce the optional timing report separately (temporary files only):

```sh
"${TRADER_FORMAL_PYTHON:-python3}" scripts/formal/ppo_process_bridge.py --benchmark
```

To compile the research executable outside the working tree:

```sh
mkdir -p /tmp/trader-snapshot-v3-build
ghc -O2 -threaded -with-rtsopts=-V0.001 -ihaskell/research   -package process-1.6.18.0 -package unix-2.7.3 -package bytestring-0.11.5.3   -outputdir /tmp/trader-snapshot-v3-build   haskell/research/InferenceProcessV2.hs -o /tmp/trader-snapshot-v3-build/infer
```

An already constructed width-12 float64 observation and a `TrainingResult` may be
encoded with `encode_request_v3(result, observation, enabled=True)`. Send its
bytes to `/tmp/trader-snapshot-v3-build/infer --offline-snapshot-v3`. Unsupported
input encodes to `None`; the Haskell boundary returns absence on invalid input,
timeout or failed cleanup. `--snapshot-contract-v3` is a pure bit-roundtrip probe.
The existing `--offline-inference-v2` contract is preserved. These commands do not
persist or authenticate models, select datasets, construct observations, authorize
orders or register production components. No new configuration/environment flag.

## PPO artifact byte codec v4

The [contract](ppo-artifact-v4-contract.md) and
[registration](../../research-notes/registrations/ppo-artifact-v4-engineering.json)
cover a pure, offline byte boundary. In `scripts/research/ppo_artifact_v4.py`,
`encode_artifact_v4(result, provenance, enabled=True)` yields canonical bytes;
`decode_artifact_v4(raw, expected_sha256, expected_provenance, enabled=True)`
returns immutable state or None. `request_from_artifact_v4` additionally takes an
already normalized observation and returns the existing v3 request or None.
All three return None by default. Supply independently selected expected metadata;
computing an expected digest from an untrusted file is not provenance authentication.
There is no filesystem, optimizer restore, production activation or order API.

Both canonical formal/full wrappers reproduce source checks, four SMT pairs,
the finite gate model and all nine synthetic trained-policy round trips without
network access. Malformed artifacts and source/model bypass mutants are included.
The frozen financial registry and sealed holdout are untouched. Dependencies and
the Haskell inference protocol remain unchanged.

Upward-rounding verification: `upward_rounding.py` runs four SAT-premise/UNSAT-violation pairs, an eight-state local dispatch model, two compiled Main delegates and current/legacy maker branches on 4226 synthetic cases. See [contract](upward-rounding-contract.md); the canonical formal/full wrappers include it. No new dependency or financial trial.

Order-number verification: `order_numbers.py` checks the shared base-only validator, all five source-bound Binance prefixes, five SMT queries and a 40-state local guard model. Both canonical formal/full wrappers include it. No credential, request or HTTP code is linked into the compiled prefix driver. See [scope and assumptions](order-number-contract.md).

The [wire extension](order-wire-contract.md) updates the same verifier to seven SMT queries, an 80-state local model, 5728 conformance rows and unchanged wire-byte checks. It relies explicitly on pinned base parsing/formatting. No financial trial or broader closure.

The wire-aware maker-price preflight also rejects zero-wire prices before constructor exceptions can enter configured market fallback (CE-ORDER-WIRE-002). Other maker fallback reasons and the fallback flag are unchanged; this is a source-bound local guard with compiled conformance, not a whole-caller proof.

Sizing admission: `sizing_inputs.py` verifies the [registered contract](sizing-input-contract.md) with 14 SMT queries, a 56-state retry model and actual pure Main fragments on 6214 rows. Both formal/full wrappers include it; no exchange module or authenticated effect is linked. Existing downward/upward source bindings are refreshed for the reviewed guards without changing their rounding algorithms.

Closed-trade recovery: `closed_trade_recovery.py` checks the [registered contract](closed-trade-recovery-contract.md), seven SMT queries, 22-state local model, 341 finite histories and actual pure helpers; JSON recovery regressions run in the Haskell suite. No whole-recovery or broad-obligation closure.

Inventory readiness: `inventory_readiness.py` checks the [registered contract](inventory-readiness-contract.md), four SMT predicates, two non-vacuity witnesses, a 25-state publication model and 6468 compiled actual-predicate cases. Obligation16 is partial, not a complete owner-reconciliation proof.

Promotion composition: `promotion_boundary.py` checks the [contract](promotion-boundary-contract.md), complete 12-module source/effect inventory, seven metadata queries, two native-schema exclusions,18 states/88 edges and72 actual codec cases. The ledger requires current-run reproduction of every composing certificate before scoped23/26 closure. `--require-complete` still fails for the31 unresolved formal obligations and empirical gates.

Artifact admission composition: `artifact_composition.py` reproduces six source-bound
SMT queries,848 fixed-loader states/9072 edges and the preserved legacy provenance
race, plus actual codec/interleaving and compatibility tests. See the
[contract](artifact-composition-contract.md). Required current-run certificates close
scoped29/30. `--require-complete` still rejects29 unresolved broader obligations and
separate economic gates. No new dependencies or financial trials.

Bot worker publication: `bot_worker_publication.py` checks the complete helper and
actual Main handoff, six SMT queries,158 states/308 edges, and six compiled cases.
The [contract](bot-worker-publication-contract.md) distinguishes gate publication
from ongoing worker ownership and termination. The preserved legacy ordering
counterexample is not a claim of a real exchange incident. Obligation15 now has
partial evidence;29 broader obligations and economic gates remain unresolved.


The [champion archive contract](champion-archive-contract.md) binds every research
writer to exclusive creation and composes preservation with the existing capability
certificates. `bash scripts/verify.sh formal` runs the new source checks, eight SMT
queries, fixed-point collision/retry model and actual exporter/writer regressions;
`full` also runs the automation exporter tests. Model bounds: 110 states/359 edges,
two writers/two keys/four identities/six byte classes. The legacy 242-state model
and source fixture reproduce truncation of colliding data. Obligation 28 requires
all constituent certificates in the same invocation. Current ledger: 10 scoped
closures, 24 partial, 4 open; kernel refinement, durability and mission completion
are not claimed.
