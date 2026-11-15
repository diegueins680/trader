# Obligation closure correction — 2026-10-03

The validator could not close any of the original 38 obligations: it permitted
only open/partial statuses, counted both as unresolved, and hardcoded mission
completion to false. CE-RL-021 preserves this defect. The correction replaces it
with explicit, pinned [closure contracts](../../formal/research/obligation-contracts.json),
[per-obligation work](../../formal/research/obligation-closure-audit.md), and same-run
certificate reproduction. A synthetic fully certified ledger can now reach zero;
that fixture is not evidence that the trading requirements are all satisfied.

**One obligation is closed in the delivered affected scope: 31, default-disabled.**
The remaining map is 28 partially verified and 9 open. No obligation was removed
as inapplicable. The inherited ownership, reconciliation and server lifecycle
boundaries remain explicit integration blockers, not assumed production proofs.

The source-derived control abstraction checks four Python default paths (eight
Boolean identity cases), the two Haskell modes, and saved artifact False metadata.
Actual Python calls use poison arguments to detect unexpected reads; compiled
Haskell conformance and the existing binary64/default and artifact-metadata SMT
certificates supplement the finite path proof. Ordinary language execution,
compiler/runtime semantics and a returning monotonic clock remain assumptions.
This does not certify arbitrary production configuration or universal compiler
refinement. Any new entry point or production integration requires reopening the
scope. No trained policy, simulator or production code changed.

During review of the first replacement, CE-RL-022 exposed a label-integrity gap:
the validator admitted the class of any component certificate as the aggregate
class. Obligation 31 could therefore be relabeled SMT-verified even though its
source-control certificate is exhaustively checked. The final validator requires
the exact canonical aggregate class and the fixture now fails as required.

Closure sufficiency is a reviewed semantic relationship, explicitly named
A-CLOSURE-REVIEW; code checks do not infer that relationship from lemma counts.
Separate economic and reproducible-delivery gates prevent formal closure alone
from implying mission acceptance. The acceptance diagnostic still exits 1 because
37 obligations and those gates remain unresolved. It cannot authorize an order.

Verification source: `c9ceacc76a4a507f09779c04e9949c8a2b8e24bb`. The final formal wrapper passes 132 integrity tests
and 41 existing scoped SMT requirements. The full wrapper passes Haskell
format/lint/build/smoke/tests, 241 web tests and 185 automation tests. The full
invocation started at `00603560`; the verifier-only class correction was separately
rechecked by the final formal invocation while the unchanged other stages ran.
All four CI jobs passed at the final source commit:
[run 37164077620](https://github.com/diegueins680/trader/actions/runs/37164077620).
Docker and deployment jobs were skipped. See the [receipt](obligation-closure-evidence-2026-10-03.json)
for exact commands, start commits and log hashes. The linker duplicate-library
warning and web bundle-size advisory remain non-failing existing warnings.

Next work is organized around whole requirement slices: verify actual
capability/dependency exclusion; address atomic optimizer publication; add bounded
inference cancellation; then compose numeric/accounting and causal admission
proofs. Isolated corrected kernels do not repair the frozen learner. Market-data
and holdout gaps remain economic blockers, not excuses to leave unrelated software
requirements permanently open.

No market data, protected holdout or historical OPE was accessed or rerun. Existing verification tests include synthetic training-budget and seed checks; no
new financial training campaign, live exploration, policy adoption, live authorization, fleet change,
merge or deployment occurred. Existing financial rejection and champion are
unchanged. Recommendation: continue offline verification; no candidate integration.

2026-10-04 worker follow-up: [three inherited defects repaired](worker-registry-followup-2026-10-04.md), with source-bound model/SMT and compiled conformance evidence. This narrows worker-lifecycle blockers without closing the broader HTTP, ownership, durable recovery or resource-completion obligations. Aggregate remains 3 closed / 28 partial / 7 open.

2026-10-04 async admission: [reservation and publication repair](async-job-admission-followup-2026-10-04.md) adds source-bound numeric/lifecycle/conformance evidence to 10 and 33–38. Scope remains in-memory admission; durable running-record recovery and whole-server drain coordination remain open. Aggregate remains 3 closed / 28 partial / 7 open.
