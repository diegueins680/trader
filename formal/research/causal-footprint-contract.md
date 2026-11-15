# Source-linked causal read footprint — 2026-09-28 continuation

Specified before implementation. This is verification of the existing rejected
research feature core, not a new feature, candidate, financial experiment or
permission to inspect protected outcomes.

## Requirement F-RL-FEATURE-FOOTPRINT

For the admitted source of `market_features`, every market-value read is in the
half-open index interval `[t-24,t+1)` for `24 <= t < n`. The checker must derive
read bounds from the parsed source and reject unsupported syntax, unbounded array
consumption, hidden global reads and a source-derived bound that exceeds `t`.
The checked result is a source-linked static dependency certificate plus SMT
integer inequalities. It is not a proof of the Python interpreter or NumPy.

Let `p ~t q` mean equal representation/length and equal values at indices `0..t`.
With identical integer `t`, runtime and primitive semantics, the checked dependency
closure implies `market_features(p,t) = market_features(q,t)` (including absence
or the same deterministic exception), regardless of later values. Invalid index
admission uses only public metadata and returns absence before value access.
Symbol isolation is relative to a supplied array: no second array/global value
source may enter this function. This does not prove the caller selected the
correct symbol, completed bar, revision, release timestamp or historical vintage.

## Abstract domain and refinement relation

The source interpreter distinguishes the raw price array, integer public index,
public metadata, and prefix-dependent local values. It follows supported Python
AST expressions/statements and overapproximates dependencies of arithmetic,
comparisons, reductions and conditionals by the union of input dependencies.
Every raw-array slice is recorded from its actual AST. For each slice `[l:u)` the
SMT query checks `24 <= t < n => 0 <= l <= u <= n AND t-24 <= l AND u <= t+1` and non-vacuity.
Only unit-stride bounded slices are supported. A read may be rejected even when
safe; unsupported behavior must never be silently treated as independent.

No numeric forecast formula is reimplemented by the checker. Local array indexing
cannot acquire new market data. Trusted NumPy calls are limited to the explicit
pure operations used by this function; arbitrary calls, reflection, imports,
mutation, closures and dynamic attributes are rejected. Metadata helpers and the
admission guard have separately audited exact AST contracts; changing them requires
review rather than silently assuming the old precondition.

## Assumptions A-SOURCE-CAUSAL

- Input arrays are ordinary base NumPy numeric arrays with fixed representation,
  shape and length, not hostile subclasses or concurrent writers; indices are
  ordinary admitted integers. Function/module bindings are not monkey-patched.
- The pinned Python AST parser, checker, Z3 and NumPy primitives are trusted.
  Primitive operations depend only on explicit arguments. Runtime nondeterminism,
  external warning handlers and hardware side channels are outside observation
  equivalence. Floating-point rounding may change values but cannot introduce
  an unpassed future market input.
- Admission helpers' audited ASTs implement their stated metadata-only contracts.
  The feature function's first rejection guard establishes the range used by SMT.
- Same supplied normalization and replay state are required for observation-level
  metamorphic tests; fitting provenance and past-state causality are not proved
  by a feature-footprint certificate.

## Obligations and evidence plan

1. Reject any unsupported AST or hidden input (fail-closed static analysis).
2. All source-derived read bounds satisfy the SMT predicate, with SAT premises
   and UNSAT violations; solver UNKNOWN/timeout is a failure (10 seconds/query).
3. Mutants reading the next bar, an entire series, another global series, or
   changing admission must fail, including mutations retaining a valid slice.
4. Deterministic property fixtures modify later values, including NaN/infinity,
   and compare actual features and current replay observations. These are tests,
   not universal proofs. No real market data is needed.
5. Model, source, test and checker hashes, requirements and assumptions are linked
   in the ledger and formal CI gate. The broader mission obligations remain partial.

No production source, frozen training implementation, artifact schema, reward,
configuration, model identifier or dataset is changed. This preserves the prior
negative experiment's semantics and avoids another adaptive financial trial.

## Preserved counterexample

[CE-RL-004](causal-counterexamples.json) is a deliberate next-bar leakage mutant,
not an existing defect. At t=24, n=26, changing price[25] from 100 to 200 changes
the mutant's current features. The original feature function remains unchanged.
Its fixture must demonstrate SAT of the prescribed violating read, rejection by
the source checker, and actual NumPy divergence for the mutant only.
