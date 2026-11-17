# Causal OPE v9 engineering contract

Scope recorded before implementation, 2026-11-16; base
53715fb41a49b215e22ede51767424bcc4d59f01. Engineering registration:
`research-notes/registrations/causal-ope-v9-engineering.json`.

## Scope and interpretation

A separate, default-disabled successor of causal-ope-v8 that calls the exact
estimator ope-rational-v3 instead of ope-rational-v2.
`scripts/research/causal_ope_v9.py` is the reviewed v8 module with six token
occurrences changed: the version string `causal-ope-v9`, the entry name
`evaluate_causal_ope_v9` (twice), the estimator module `ope_rational_v3` and
the estimator entry `estimate_v3` (twice). Admission, the 2^-10 behavior
floor, the 2^99 magnitude guards, prefix-only run access, stream distinctness,
the checked estimate and the single publication are otherwise identical, so the
[v8 contract](causal-ope-v8-contract.md) defines the adapter and the
[v3 contract](ope-rational-v3-contract.md) the estimator.

causal-ope-v8, ope-rational-v2, ope-rational-v3, both successor learners,
frozen `sequential_evaluation` and every preserved counterexample are
unchanged. No training, sampling, data collection, artifact, policy, promotion
or order interface is added. Acceptance is engineering evidence, not a
financial experiment.

## What changes

v8 inherits v2's 8192-bit rational limit: six 32-decision fixture runs reject.
v3's size check is certified unreachable on its whole admitted domain of 256
episodes and 32 decisions (at most 1,006,337 bits under a 2^20-bit limit).
Every episode v9 builds lies in that domain: behavior in [2^-10, 1], target in
[0, 1], finite values of magnitude at most 2^99, zero value beyond the horizon,
at most 256 runs and 32 decisions. An evaluation that passes the adapter's own
admission is therefore rejected only on resource failure, never for size.

## Requirements / methods

F-RL-OPE-V9-SOURCE: the module equals the hash-pinned reviewed v8 module up to
the registered tokens with exact occurrence counts; v8's complete AST lock and
review then apply verbatim; the estimator behind the new import is the
hash-pinned certified ope-rational-v3; the import closure is v8's with v3 in
place of v2 and contains no frozen module.

F-RL-OPE-V9-ARITH (SMT, quantifier-free): v8's four sampling-law implications
and floor control carry over unchanged (same learners, floor and horizon); two
further implications show the adapter's admitted values and counts lie in the
estimator's admitted domain, with one control showing the magnitude guard
alone does not; the estimator's registered size bound is below its limit.

F-RL-OPE-V9-FLOW: v8's finite stage model applies through the source
identity: publication only after every run is converted, streams are distinct
and the estimate is checked; three mutants reject.

F-RL-OPE-V9-CONFORMANCE: registered synthetic evaluations are deterministic
and reject by default and under a wrong version; every published field equals
an independent exact-rational evaluation of the recorded transitions and
equals v8 wherever v8 publishes; a target equal to the recorded behavior gives
unit weights; six and 48 runs at the full 32-decision horizon, which v8
rejects, are published; a model with subnormal targets and largest admitted
values is published; short, forged, reused, foreign-typed and out-of-domain
inputs reject. Tests are not proofs.

## Assumptions and limits

A-CAUSAL-OPE-V9 is A-CAUSAL-OPE-V8 (rounding model, uniform draws, honest
learner results, caller-supplied target models) together with
A-OPE-RATIONAL-V3 (CPython Fraction semantics, unproved elementary number
theory, a tested rather than proved certificate evaluator, a bound on object
sizes and not on time or memory). Cost grows quadratically with runs.

Capacity is not evidence. Independence across seeds and outcomes is not
established, target models are caller-supplied, the estimate is marked
unreliable and no unbiasedness, coverage, variance or economic claim follows.
Obligation 10 stays open: network and optimizer arithmetic is guarded rather
than bounded and no frozen path is retired.
