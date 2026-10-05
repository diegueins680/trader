# Binance nonzero wire admission v1

Baseline 5cff8eb6379a46a50db6c37ac247212341d35738. This is a preregistered
engineering repair, with no financial experiment or exchange invocation.

Conflict: positive binary64 admission does not imply a positive serialized
quantity/price. The current eight-decimal formatter emits `0` for `1e-9` and
`5e-9`. Resolve by rejecting a nonpositive or unparseable rendered value before
credential access. Preserve base-before-quote selection and all context guards.
Keep the exact existing formatting algorithm, public order signatures, side
semantics, venue filters, limits, permissions and configuration.

Move the existing pure formatter and trailing-zero trimming into OrderNumeric.
Binance's renderDouble must delegate to this same formatter. After finite/positive
binary64 validation, parse its output as base Fixed E12 (Integer units of 1e-12).
Eight decimal places are exactly representable at that resolution. Only
`Just (MkFixed units)` with `units > 0` accepts. Failed parsing rejects.
The guard and wire use the same immutable value and deterministic formatter.
No rounding algorithm is replaced. Values previously producing nonzero valid
decimal text retain their bytes. Report zero-wire failure distinctly.

Abstract state: constructor, numeric validity, parsed wire validity, context
validity, phase. All five constructor paths must pass numeric, wire and context
guards to reach the credential boundary; rejection is terminal. The finite
model has 40 initial states, 40 terminal states, 40 transitions, depth one.
This boundary is not an authorized order capability.

SMT obligations over binary64 and unbounded Integer parser results: accepted
values are finite/positive; accepted parsed units are positive; parse failure
or nonpositive units reject; base priority and futures requirements remain.
Parsing/formatting implementation correctness is a named GHC/base assumption,
not a theorem about Numeric or the compiler. Compiled actual-prefix conformance,
decimal parsing, old/new wire parity and generated Haskell properties connect
the model to code. Do not describe these tests as proofs.

Preserve existing CE-ORDER-NUM-001/002 and add CE-ORDER-WIRE-001 for positive
input rendering to zero. Extend the 13-word boundary roster with 1e-9 and the
binary64 predecessor, exact value and successor of 5e-9. Run every market,
presence flag, mode and type combination, plus 4096 generated rows (seed
20261005). Independently check current/legacy prefix decisions and unchanged
wire bytes; legacy formatting fixtures must be captured before implementation.

Assumptions: pinned GHC/base Fixed E12 Read semantics, deterministic showFFloat
and trimming, Integer arithmetic, ordinary immutable non-bottom inputs,
IO exception sequencing, reviewed source extraction, and sufficient resources.
Tests must link only inert constructor prefixes, never credential/signing/HTTP
effects. Formal/full wrappers and final CI must pass on frozen sources.

No complete wire cap, tick/lot membership, venue acceptance, full caller retry,
authorization, accounting, resource-bound or exchange implementation theorem.
Ordinary positive decimal rounding can still increase a quantity. Other venues
are outside this repair. Broader obligations 9/10/21 remain unresolved; no new
broad closure or economic evidence follows from this repair.
