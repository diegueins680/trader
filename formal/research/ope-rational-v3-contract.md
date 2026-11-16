# Exact OPE v3 engineering contract

Scope recorded before implementation, 2026-11-15; base
0cd0d8971feb0073624ee0ce436b0bd13147b765. Engineering registration:
`research-notes/registrations/ope-rational-v3-engineering.json`.

## Scope and interpretation

A separate, default-disabled successor of ope-rational-v2 with the same
arithmetic and a rational size limit that its whole registered domain fits
under. `scripts/research/ope_rational_v3.py` is the reviewed v2 module with
exactly three tokens changed: the version string `ope-rational-v3`, the entry
name `estimate_v3`, and `MAX_BITS = 1048576` (2^20) instead of 8192. Admission,
recurrences, checked primitives, publication and the immutable envelope are
otherwise identical, so the [v2 contract](ope-rational-contract.md) defines the
estimators.

ope-rational-v2, causal-ope-v8 (which still calls v2), both successor learners,
frozen `sequential_evaluation` and every preserved counterexample are unchanged.
No adapter, data collection, artifact, policy, promotion or order interface is
added. Acceptance is engineering evidence, not a financial experiment.

## Why a new limit

Every binary64 behavior probability contributes its up-to-53-bit numerator to
the common denominator of the summed weights, and twice that to the sum of
squared weights. v2's 8192-bit limit therefore admits only a few long
episodes: six 32-decision fixture runs reject (causal-ope-v8 finding). The
limit was a resource guard chosen without a size analysis.

## Size certificate

For N in 1..256 episodes and T in 1..32 decisions, every value handed to the
size check in an admitted batch has numerator and denominator bit lengths at
most 1,006,337, which is below 2^20. The size check therefore never rejects an
admitted batch; it stays in place as a defensive check.

Method. A certificate for a rational x is a positive integer L with x*L an
integer and a bound |x| <= 2^mag. L is a product of powers of the behavior
numerators (each below 2^53) and a power of two. If x = n/d in lowest terms
then d divides L, so d <= L and |n| <= |x|*L. Certificates compose:

- product: multipliers multiply, magnitudes add;
- sum: both sides are weakened to a common multiple, magnitude at most doubles;
- 1/b for a binary64 b in (0, 1]: the multiplier is b's reduced numerator
  (below 2^53) and the magnitude is at most 2^1074;
- quotient by a nonzero value over a common multiplier L: x/y = (xL)/(yL).

Inputs: a finite binary64 value times 2^1074 is an integer and is below 2^1024
in magnitude; a target probability or discount in [0, 1] has magnitude at most
one. The certificate is evaluated on the source-extracted recurrences in
execution order for all 8192 pairs (N, T); the largest bound is at (256, 32).

## Requirements / methods

F-RL-OPE-V3-SOURCE: the module equals the hash-pinned reviewed v2 module up to
the three registered tokens, each occurring exactly once; v2's own complete
definition lock, activation-first rejection, whole-batch admission, checked
primitives and private publication review then apply verbatim; constants are
256 episodes, 32 decisions and 2^20 bits; only standard-library imports.

F-RL-OPE-V3-ARITH (SMT, quantifier-free, plus exact integer evaluation): the
six certificate rules above as polynomial implications, one satisfiable
control showing an unreduced fraction is not bounded by its multiplier, and
the evaluated bound 1,006,337 <= 2^20 over all 8192 configurations.

F-RL-OPE-V3-FLOW: v2's finite admission / accumulation / publication model
applies through the source identity: no arithmetic before whole-batch
admission, no partial publication, and a partial-publication mutant rejects.

F-RL-OPE-V3-CONFORMANCE: registered synthetic batches are deterministic and
reject by default and under a wrong version; every published field equals an
independent exact-rational oracle, and equals v2 wherever v2 publishes; in
instrumented runs over uniform, full-mantissa, subnormal, largest-finite and
mixed inputs every checked value is within the certified bound of its own
call; a full 256 x 32 batch that v2 rejects is admitted; lowering the limit
back to 8192 makes the same batch reject; invalid and oversized batches
reject. Tests are not proofs.

## Assumptions and limits

A-OPE-RATIONAL-V3 trusts pinned CPython 3.13.3 float, integer, Fraction
(always in lowest terms) and dataclass semantics on IEEE binary64, the reviewed
AST extraction, Z3 and sufficient resources. Integer closure under addition and
multiplication and Bezout's identity for coprime integers are used without
machine proof; the solver checks the polynomial steps only. The bound is on the
arithmetic objects handed to the size check, not on wall-clock time, process
memory or CPython's unreduced intermediates (at most about twice the bound);
allocation failure still rejects the batch. The bound is not tight: the largest
observed call reaches about two thirds of its certified bound and the observed
full-capacity peak is about 676,000 bits.

Authentic propensities, trajectory completeness, causal logging, support and
valid Q/V meanings remain caller assumptions. No unbiasedness, coverage,
variance or economic claim follows and the estimate is marked unreliable.
Obligation 10 stays open: no adapter calls v3 yet, network and optimizer
arithmetic is guarded rather than bounded, and no frozen path is retired.
