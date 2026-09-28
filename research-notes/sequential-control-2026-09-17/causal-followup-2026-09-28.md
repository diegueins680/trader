# Source-linked causality follow-up — 2026-09-28

**No adoption; the broader mission remains incomplete.** This continuation narrows
one assurance gap in the existing rejected RL research environment. It changes
verification infrastructure and documentation only. No financial trial, policy,
feature semantics, data, frozen registration, holdout access or live setting changes.

The previous draft revision `68d1ce38` passed all four remote CI jobs in
[run 36377655623](https://github.com/diegueins680/trader/actions/runs/36377655623).
Docker build and deployment were skipped. That result is separate from the new
checks introduced here. Latest main remains `dbd45e26`; the branch remains stacked
on #281 and published in draft #284, without merge or deployment.

## What the prior lemma left open

`F-RL-CAUSAL-SLICE` proved an independently written integer inequality. It did not
verify that the actual Python function continued using that slice. Source hashes
exposed drift, and finite metamorphic tests supplied conformance evidence, but
neither connected arbitrary admitted source expressions to their read footprint.

The new contract was committed first (`65b24bb0`). The restricted AST checker
parses the existing `market_features` body, checks metadata-only admission helper
contracts and the exact rejection guard, tracks local dependencies, and derives
raw-array slice expressions from the source. Unknown calls, imports, closures,
reserved-name reassignment, global value reads and unbounded consumption of the
price array are rejected. Safe source changes can still be admitted: a local
volatility window of three rather than six bars passes the dependency check.
That test variant is not a financial candidate and is not installed.

For every admitted raw read `[l:u)`, Z3 checks under `24 <= t < n`:

`0 <= l <= u <= n`, `t-24 <= l`, and `u <= t+1`.

The premise must be SAT and the violation UNSAT within ten seconds. The actual
source produces one read, `prices[t-24:t+1]`. This is source-linked static/SMT
verification with a trusted analyzer and explicit primitive semantics, not a
machine-proved interpreter or compiler refinement.

## Counterexample and conformance

[CE-RL-004](../../formal/research/causal-counterexamples.json) deliberately mutates
the upper bound to `t+2`. With t=24, length 26 and constant prefix 100, changing
future price[25] from 100 to 200 alters the mutant's current features. The fixture
checks SAT of that exact integer witness, rejection by the source checker, actual
mutant divergence and unchanged original features. It is not an existing bug or
market experiment. Other mutants exercise whole-series reductions, hidden globals,
unknown calls, admission/helper changes and unsupported syntax.

Thirty-two deterministic generated price paths, each with four future corruptions
(NaN, infinity, negative price and a very large value), compare actual feature
vectors and current replay observations. These 128 cases are property tests,
not universal proofs. Existing 180 accounting cases, 20,480 Haskell cases and the
75-state/349-transition lifecycle model remain unchanged.

## Scope and remaining blockers

`F-RL-FEATURE-FOOTPRINT` adds one scoped `smt_verified` entry; total SMT obligations
are 13. Assumption `A-SOURCE-CAUSAL` records ordinary base numeric NumPy arrays,
fixed metadata, no concurrent mutation, stable module bindings and trusted
Python/parser/checker/NumPy primitives. The checker does not establish its own
soundness mechanically or prove runtime semantics. It does not certify hostile
array subclasses, caller symbol selection, training-only normalization provenance,
recursive past-state causality, publication/revision timing or production features.

The proof ledger maps the claim to the actual feature source, checker, mutation and
conformance tests, deterministic results, source/tool hashes and formal CI. Mission
obligations 1, 5 and 17 gain explicit evidence but remain partial; all 38 still have
open or partial scope. The acceptance diagnostic must continue to fail. No economic
result, OPE estimate, seed selection or prior rejection is changed.

README, CHANGELOG, formal documentation, specifications and risk-register evidence
are updated. `RL-OFFLINE-001` remains HIGH/OPEN. No new dependency or configuration
is introduced. Exact commands and final outcomes will be recorded after the scoped
and full wrappers finish; until then no new full-run pass is asserted.
