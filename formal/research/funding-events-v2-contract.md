# Exact funding events v2

Engineering preregistration, 2026-10-06; base fccd49914baf6d44f9949031d484797cded64c76.

## Scope and interpretation

An independent, disabled-by-default successor for the funding-settlement step of
the frozen development loader (`run_sequential_screen.load_development`). It
addresses preserved counterexample CE-RL-019 only inside the new kernel. The
frozen loader, its archived outputs, F-RL-FUNDING-FINITE (refuted) and all
reported sequential-screen evidence are unchanged. No file, network, hash check,
policy, runner composition, production consumer or order interface is added.
Callers remain responsible for verified bytes, symbol scoping and the registered
close grid. Records are already-decoded text triples.

## Input domain

Close grid: tuple of 1..8192 strictly increasing Python ints in [0, 2^63).
Records: tuple of at most 65536 triples (time, rate, mark) of `str` only.
Time is canonical ASCII decimal with at most 19 digits, no sign, no leading zero,
and value < 2^63. Rate and mark are canonical ASCII fixed-point decimals with an
optional leading minus, a 1..20 digit whole part without leading zeros (except
`0`) and an optional 1..20 digit fraction. Exponents, `inf`, `nan`, whitespace,
`+`, Unicode digits and binary64 values are outside the domain. Mark must be
strictly positive; rate is signed. Times must be strictly increasing.

The domain is a numeric admission contract, not a financial plausibility filter.
It is chosen so that every admitted value is exact and every intermediate
arithmetic size is provably small; it does not assert that any provider emits
canonical text, nor that an observed record is causally available.

## Semantics

Decimal text d with whole part w and fraction f maps exactly to
sign * int(w ++ f) / 10^|f|. An event at time t is assigned to bucket
j = min { k : close[k] >= t }. Events at or before close[0] collect in bucket 0
(matching the frozen loader: replay reads funding after its start). An event
exactly at a close belongs to that endpoint. An event later than the final close
rejects the whole load. A bucket holding more than 128 events rejects the whole
load (equal to replay-accounting-v2 event admission). Published `events[j]` is a
tuple of (mark, rate) Fraction pairs in time order, and
`per_unit[j] = sum(mark * rate)` computed exactly with checked primitives.
Empty buckets mean no settlement event, never missing data.

Load is all-or-nothing: any domain, ordering, endpoint, size, arithmetic or
resource error returns None; no partial Buckets value is published. The
published value is an immutable dataclass of tuples. Inactive version or any
`enabled` value other than `True` returns None before inspecting inputs.

## Obligations / methods

F-RL-FUNDING-V2-SOURCE: complete reviewed AST lock, constant/regex/import/export
checks, immutable output, activation-first default rejection and checked
primitives (exhaustively checked over the reviewed source; interpreter trusted).

F-RL-FUNDING-V2-ARITH: SAT-premise/UNSAT-violation SMT queries showing that, on the
admitted domain, decoded numerators are below 10^40 with denominators dividing
10^20, products have numerator below 10^80 and denominators dividing 10^40, a
128-term accumulation stays below 128*10^120 over the common denominator 10^40,
reduction cannot enlarge a numerator, and Fraction primitive intermediates stay
far below 2^8192. Hence the 8192-bit guard never rejects admitted input (no
overflow, no underflow, no rounding) while out-of-domain text rejects before any
arithmetic. The sweep endpoint is checked by quantifier-free SMT loop-exit,
loop-body and carry lemmas using explicit sortedness instances, and exhaustively against the prefix-count
definition for grid (1,3,5,7), times 0..8 and every strictly increasing event
set of size at most 4.

F-RL-FUNDING-V2-FLOW: finite publication model over activation, grid, record,
bucket and arithmetic outcomes: publication iff every stage succeeds; no partial
publication; failure is terminal; a mutant publishing after a failed stage is
caught.

F-RL-FUNDING-V2-CONFORMANCE: 128 seeded synthetic loads plus registered edge and
CE-RL-019 cases are compared with an independent compiled Haskell Rational oracle
(character parser, prefix-count bucketing). Published buckets are composed with
`replay_accounting_v2.advance_v2` to check exact funding = -units * per_unit.
Tests are not proofs.

A-FUNDING-EVENTS-V2 trusts pinned CPython re/int/Fraction/dataclass semantics,
the reviewed AST translator, Z3, GHC Rational for differential testing, ordinary
immutable values, stable bindings and sufficient resources. Bit limits bound
arithmetic size, not wall-clock execution. Actual provider release time,
first-seen time, revision handling, symbol coverage, the frozen loader and runner
composition remain unverified. The 38 original criteria and scopes are
unchanged; obligation 10 does not close on this kernel.
