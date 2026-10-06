# Replay shield-consumer composition v1

Engineering preregistration against main 4894a6d813bad18b5881d40b51b7a835848df316.
No financial trial, historical-data access, runtime policy change or deployment.
Obligation 12 keeps its original scope and criterion: "Every consumed action
passes the deterministic shield and cannot race around it."

## Interpretation and concrete surface

The delivered consumers are the serial offline Replay instances in collect,
replay_policy and short_ope, and the separately typed Haskell proposal/process
boundary. A policy-target fill and compulsory terminal liquidation are distinct:
a target must originate from shield acceptance, while liquidation is the fixed
zero target selected by the deterministic terminal/risk branch after call-entry
admission. The latter is not a fresh policy decision; this interpretation already
appears in replay-order-contract.md. Invalid call-entry proposals terminate before
new simulated fills; pending inventory/data can remain in an incomplete failure.
No cancellation, liquidation-success, arithmetic-correctness or rollback theorem
is implied by precedence.

F-RL-SHIELD-CONSUMERS: extract all constructors, receiver uses, state assignments,
step calls and _trade calls from the complete existing reviewed research source
inventory. Check receiver ownership and callback arguments, including OPE's two
separate instances. Reject additional consumers, aliases, escapes, mutation or
concurrency. Check actual step dominators and pending assignment/clear sites;
check every helper's reviewed effects. Source hashes alone are not a proof.

F-RL-SHIELD-ORIGIN: derive symbolic transfer predicates from those sites. Prove
absent-or-shielded pending origin by induction; a fill reads the pending value
only after successful current admission, observation, due and risk checks.
Terminal liquidation reads literal zero after the terminal/solvency guard.
Prove frame preservation for distinct instance identities. Preconditions must
be satisfiable; solver UNKNOWN or a counterexample rejects.

F-RL-SHIELD-FLOW: explore a finite repeated-call state machine, retaining a pending
proposal across calls, replacement rejection, disabled/invalid/timeout rejection,
terminal cancellation, terminal fixed-zero liquidation and independent instances.
Use fixed-point exploration without a retry-depth cutoff. Record exact bounds,
state/transition counts and counterexamples. Inject bypass transitions to confirm
that origin and precedence violations fail the checker.

F-RL-SHIELD-CONSUMER-CONFORMANCE: instrument actual shield/_trade and all replay
callers on small synthetic paths. Test each action, delay and horizon; retain a
pending target, reject later calls, test both ordinary and terminal fills and
interleaved independent instances. Compare emitted event traces to model rules.
Instrumenting observers must not replace arithmetic or shield results. Generated
fixtures and conformance tests supplement the checked source/SMT/model results.

## Refinement, assumptions and closure

Abstract states retain instance identity, current call admission, observation
admission, pending target/origin, due/risk tags and terminal/solvency tags. Numeric
arithmetic is abstracted; an exception cannot resume the call into a later fill.
Model-to-source correspondence is a reviewed structural effect interpretation,
not a verified Python compiler. Restrict executable inputs to the ordinary
registered built-in/NumPy objects and fixed callbacks already admitted by the
complete source boundary. Libraries do not introspect callers, mutate unrelated
instances or inject asynchronous Python callbacks. No hostile monkeypatching,
subclasses, arbitrary direct _trade invocation or debugger writes are admitted.
The serial/ownership claim must be extracted from actual callers, not assumed
merely because Replay is documented as offline. Any new shared/runtime consumer
reopens this certificate. No production executable may acquire the research
module or policy artifact through an unchecked integration.

A-SHIELD-COMPOSITION names these Python/NumPy control-flow and primitive-effect
assumptions and the trusted checker/solver. No claim covers production order
arithmetic, live server races, truthful provider timestamps, market profitability,
crash recovery, asynchronous revocation or every hard risk constraint.

Close 12 only if consumer source, SMT origin, repeated-call model and all existing
proposal/type/process/production-exclusion certificates freshly reproduce together.
Keep all other obligations and every original scope/criterion unchanged. If the
mapping cannot establish the criterion, keep 12 partial with the discovered gap.
No financial evidence or new policy may be promoted by this verification work.
