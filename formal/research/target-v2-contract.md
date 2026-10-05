# Offline GAE target kernel v2 — specification before implementation

Version: `gae-targets-v2`; engineering continuation of CE-RL-010/011, 2026-09-28.
Classification: numerical/functional correctness, isolation, configuration safety,
atomic publication and bounded termination. This is independently callable research
infrastructure, not a trained policy, production integration or a new financial trial.
The frozen `sequential_learning.advantages` and all historical results retain their
identity and counterexamples. No existing identifier or saved artifact changes.

## Mathematical behavior

An input row is (reward r, current critic v, next critic n, terminal d). Rows are
ordered oldest to newest. Discount gamma is in [0,1]; lambda is fixed at 0.95.
Starting with a zero later raw advantage, traverse rows backwards. For a terminal:

    raw = r - v; target = r

For a nonterminal:

    bootstrap = gamma * n
    delta = (r + bootstrap) - v
    trace = (gamma * 0.95) * later_raw
    raw = delta + trace
    target = raw + v

Each operation uses separate IEEE binary64 RNE arithmetic. On terminals neither
bootstrap nor later carry is evaluated in the recurrence. All supplied numeric
inputs must still be finite. Accept only native Python float scalars and exact bool
terminal flags; reject implicit coercions, bool-as-number and hostile subclasses.
No normalized advantages, actor/critic gradients or optimizer steps are returned.

Return either immutable (raw,target) pairs or explicit absence (`None`). Require
finite raw/target outputs. Any invalid row, version, control or non-finite result
rejects the complete batch. Keep staging local; never publish a partial batch.
Empty batches and batches above 256 rows reject. Inputs must be immutable native
tuples of native row tuples. No mutation, callbacks, I/O, network or persistence.

## Activation and boundary

Both step and batch entry points default `enabled=False`. Only exact True and the
exact native string version `gae-targets-v2` may compute. This flag grants no model,
policy, exchange, risk or live-order capability. The existing runner does not import
or call the new module. There is no normalization or training adapter in this scope.
Disabling the function returns absence and leaves no persistent state to clean up.
A future training integration needs an explicit successor registration and counts
as an adaptive financial trial; this engineering registration authorizes none.

## Obligations and verification

- F-RL-TARGET-V2-TERMINAL: an accepted terminal step preserves the exact reward
  binary64 bits in its target, including signed zero; failure is explicit absence.
- F-RL-TARGET-V2-FINITE: every published pair contains finite binary64 values;
  invalid activation and non-finite arithmetic cannot publish a pair.
- F-RL-TARGET-V2-REAL: the source step refines the stated exact-real recurrence
  algebraically, without claiming binary64 nonterminal accuracy or convergence.
- F-RL-TARGET-V2-PUBLISH: an audited batch control-flow abstraction publishes only
  after all 1..256 rows succeed; default-disabled, invalid admission or row failure
  ends in absence; no partial publication or order-authorizing transition exists.

Use restricted source-AST translation with satisfiable-premise checks and SMT
negated obligations; exact-real and binary64 claims are distinct. Bind input guards,
branch structure, default arguments, loop order and publication boundary to audited
source skeletons. Reject unsupported source drift. Model-check batch publication
with explicit numeric bound, state/transition/depth counts and no unchecked result.

A-TARGET-V2: trusted Python float/bool/tuple and math.isfinite semantics, separate
binary64 RNE operations, stable primitive/module bindings, no concurrent monkey
patching, normal resource availability, trusted source translator/compiler/solver.
Do not claim complete interpreter/compiler refinement, bounded wall-clock runtime,
normalization correctness, whole-training safety or financial efficacy. No
nonterminal roundoff bound or full learner numerical theorem is supplied.

Conformance must cover CE-RL-010 exact terminal recovery, CE-RL-011 rejection without
partial output, signed zeros, non-finite/wrong-shape/type/version/enable controls,
overflow, deterministic replay, all failure positions in a bounded batch and the
256-row boundary. Differentially compare a finite grid to an exact rational model
using representable dyadic inputs; generated cases supplement, not replace, proofs.
Mutate target assignment, finiteness guards, activation and publication order to
check the verifier fails. Record all limitations and preserve v1 counterexamples.


2026-10-04 composition: `ppo-successor-v2` now uses batch_v2 for actual synthetic
engineering training. This does not modify the frozen learner or repair its
historical counterexamples. The successor checks normalization separately and
rejects non-finite results; it does not claim nonterminal arithmetic accuracy.
