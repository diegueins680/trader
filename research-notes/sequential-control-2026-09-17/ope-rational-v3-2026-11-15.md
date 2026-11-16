# Exact OPE v3 — engineering increment, 2026-11-15

Engineering evidence only: no market data was read, no financial trial was run
and no holdout was opened.

## What was added

`scripts/research/ope_rational_v3.py` is the exact estimator ope-rational-v2
under a new version with a rational size limit of 2^20 bits instead of 8192.
It differs from the reviewed v2 module in exactly three tokens (version, entry
name, limit). `scripts/formal/ope_capacity.py` certifies a worst-case size for
every value the estimator checks.

## Findings

1. **The registered domain fits.** For every admitted batch of up to 256
   episodes and 32 decisions, every checked numerator and denominator has at
   most 1,006,337 bits. The 2^20-bit limit is therefore never reached; the
   check stays as a defensive one. The previous increment found that v2
   rejects six 32-decision runs.
2. **The bound holds for every binary64 input**, including subnormal behavior
   probabilities and largest-finite rewards, not only for typical values. It
   needs no probability floor and no dyadic recording scheme.
3. **The bound is loose but sufficient.** A full 256 x 32 batch of typical
   values peaks near 676,000 bits; the tightest single call observed reaches
   about two thirds of its certified bound. The slack under the limit is
   42,239 bits, so the limit cannot be lowered to 2^19 and the domain cannot
   be widened much without a new analysis.
4. **Cost is quadratic in episodes.** A full typical batch takes several
   seconds of exact arithmetic. The certificate bounds object sizes, not time
   or process memory.

## Evidence

- Source: equal to the hash-pinned v2 module up to three tokens.
- SMT: six rule implications and one satisfiable control; the bound evaluated
  over all 8192 (episodes, decisions) pairs.
- Model: v2's 23-state publication model through the source identity.
- Conformance: 65 batches against an independent exact oracle (64 also against
  v2), 25 instrumented runs with 37,845 per-call bound checks over five input
  families, one full-capacity admission, one limit control, 17 rejections.

## What this does not show

No adapter calls v3: causal-ope-v8 still uses v2 and keeps its small capacity
until a separately versioned adapter is added. Nothing here is statistical
evidence; the estimate stays marked unreliable. Obligation 10 stays open.
Ledger counts stay 20 scoped closures / 17 partial / 1 open.
