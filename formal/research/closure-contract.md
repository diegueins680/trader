# Evidence-driven obligation closure v1

Specified 2026-10-03 before changing the acceptance implementation.

## Defect and resolution

The previous validator admitted only `open` and `partially_verified` for each of
38 obligations, then counted those statuses. It therefore could never accept
completion. Its `missionComplete` output was also a literal false. Neither was
an evidence-based assessment. Preserve historical receipts; replace this rule.

## Closure semantics

Each numbered obligation retains its original title and a reviewed affected
scope, closure criteria, concrete remaining work, and implementation paths.
[The canonical contract registry](obligation-contracts.json) fixes each number, title, scope, criterion, implementation path and required certificate set independently of status updates. The validator rejects ledger/contract drift. Semantic sufficiency of this mapping is explicitly a source-review assumption; certificate existence alone cannot establish it.

`requiredCertificates` names sufficient certificates for that exact scope, not
merely related supporting lemmas. An empty list is never sufficient evidence.
`blockers` names outstanding work, including applicable counterexamples.

Let V be {proved, model_checked, probabilistically_model_checked, smt_verified,
refinement_verified, exhaustively_checked}. For an obligation o:

    closed(o) = requiredCertificates(o) is nonempty
                AND blockers(o) is empty
                AND every required certificate exists with status in V
                AND o.status is the reviewed aggregate verification class

Metadata validation rejects inconsistent closure. Acceptance additionally requires
that every required certificate was reproduced in this invocation, with its
matching verification class. Property tests and empirical results alone cannot
close a formal obligation. Missing certificates, unknown results, solver timeout,
source drift and stale receipts fail closed. Adding a certificate requires the
same source review and canonical registration as existing proofs.

The count is the number of obligations not closed; `formalObligationsComplete`
is exactly count=0 after successful reproduction. This is not mission completion:
`missionComplete` additionally requires all separately recorded research acceptance
gates. Those gates remain blocked without fresh economic evidence, validated
champion comparisons and delivery evidence. `--require-complete` checks both.
It cannot grant production authorization even if every gate eventually passes.

## First closure: obligation 31

Affected scope: delivered offline entry points `screenProposal` (Haskell), `infer`,
`step_v2`, `batch_v2`, `effective_sample_size_v2`, and saved/loaded offline policy
metadata. Training/replay command invocations are explicit research opt-ins, not
production defaults. No candidate is installed in a production selector.

F-RL-DEFAULT-PATH checks the actual source default and initial control path:
Haskell default Disabled selects the first Nothing branch; all four Python
functions default enabled to the singleton False and short-circuit to an absent
result before reading model/data inputs. Inference may read the monotonic clock.
Saved artifacts contain enabled=False; the existing source-bound artifact
metadata certificate verifies refusal of non-False flags. Exhaustively enumerate
the two Boolean enabled cases for each initial branch; the enabled case must
remain reachable (this is not an always-reject implementation).

Abstraction: mode/identity-of-enabled and the first branch program counter. Other
inputs are opaque; disabled execution never inspects them. Source admission
rejects decorators, changed signatures/defaults, inserted pre-guard operations,
changed guard prefixes and changed return forms. Haskell source admission uses a
small exact declaration grammar, not an unverified general Haskell parser.

Assumptions: ordinary source execution, trusted Python AST/short-circuit semantics,
GHC algebraic constructors and guard order, terminating monotonic clock, no runtime
monkeypatching or compiler corruption, defined Haskell arguments. These do not
assume that a disabled caller rejects: that property is checked. Tests execute
actual Python functions with poison inputs and compile actual Haskell conformance.
This is a finite control-flow verification, not universal compiler refinement or
proof of all existing production configuration. Any integration introduces a new
scope and must reopen the obligation. No need to prove floating arithmetic on an
unreachable default path. Other obligations keep their own unresolved boundaries.

## Regression requirements

Reject unsupported closure, missing proof, property-test-only evidence, omitted
obligations, retained blockers, stale source, and missing runtime receipts. Test a
fully certified synthetic 38-obligation ledger to establish that closure is now
reachable; it is a validator fixture, never evidence about trading software.
Test positive formal completion with research gates blocked separately. Preserve
the former impossible-closure case as CE-RL-021 in the closure regression registry.
