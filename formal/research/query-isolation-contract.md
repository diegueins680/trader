# Numerical proof-query isolation — specification before implementation

Version `numerical-query-isolation-v1`, 2026-09-28. Scope: the two existing
`terminal_numerics.prove` and `target_v2.certify` proof drivers only. Requirement
`F-RL-INTEGRITY`; classification: verification integrity, failure behavior and
engineering performance. No new mathematical theorem or trading behavior is proposed.

## Existing requirements and resolution

The terminal and target-v2 contracts require a satisfiable premise and an
unsatisfiable negated claim, with witness constraints absent from the universal
query. The existing push/pop implementation satisfies that intended scope, but
recorded checks intermittently return unknown/canceled under the pinned 10-second
limit. Those attempts produced no certificate. Passing retries are separate
receipts, not permission to ignore unsuccessful checks.

Independent solver instances may replace incremental contexts without changing
assertions. This resolves a verification-engineering limitation, not a contradiction
in GAE arithmetic or evidence gates. The old target-v2 contract's witness removal
means semantic absence from the violation query; it does not require push/pop.

## Executable contract and obligations

Let P be the premise, C the conclusion and W the conjunction of optional concrete
witness constraints (True when absent).

1. Query A is exactly P AND W, on a fresh solver. Continue only for SAT.
2. Query B is exactly P AND NOT C, on a different fresh solver. W must not be
   asserted, substituted into P/C or retained through a model from query A.
3. Publish success only for query-A SAT and query-B UNSAT. SAT counterexamples,
   UNSAT premises, unknown/canceled results and exceptions reject the certificate.
4. Both queries retain timeout=10000 ms and random_seed=0. Each solver is checked
   once. No retry, fallback tactic, additional hypothesis, sampled restriction,
   domain change, float-to-real replacement or altered rounding is allowed.
5. Preserve current formulas, requirement IDs, source translations, binary64/real
   distinctions, known counterexamples, solver/tool versions and result schema.

Algebraically query A certifies non-vacuity; query B searches the entire stated
premise domain. Separate solver objects refine the same two formulas. This claim
relies on trusted Z3/Python semantics (A-SOLVER), not a new verified solver or compiler.
A-SOLVER remains an assumption. Z3's default solver can choose different internal
strategies for independent and incremental use; no performance guarantee follows.

## Conformance, failure tests and evidence

Instrument the actual helpers with recording solver doubles. Enumerate all 3x3
SAT/UNSAT/UNKNOWN result pairs for each helper; only SAT/UNSAT succeeds. Assert
one solver for rejected premises and two distinct solvers otherwise, exactly one
check each, exact ordered formulas and unchanged options. Unknown reason strings
must be retained. Inject exceptions at both checks and require propagation.
Actual Z3 regressions must detect a conclusion true only at the premise witness,
false premises and false claims, while proving representative real and binary64
claims. Retain existing source mutants and CE-RL-010/011. These tests are
implementation-conformance evidence, not a universal Python refinement proof.

Benchmark the old and revised helpers with identical existing terminal/v2
obligations, three fixed rounds, alternating old-first/new-first. Count every
attempt, including timeouts. No timing-driven parameter search. Compare result
payloads excluding timing; fail or defer the change on any semantic mismatch.
Timing differences remain shared-host engineering evidence, not a guaranteed
speedup or a causal explanation for previous failures. The broader 38 mission
obligations and candidate gates remain unchanged.

## Primary tool documentation

Reviewed 2026-09-28: [Z3Py solver API](https://z3prover.github.io/api/html/classz3py_1_1_solver.html)
and [official Z3 introduction](https://microsoft.github.io/z3guide/programming/Z3%20Python/Introduction/).
The pinned executable remains Z3 4.15.4 / distribution 4.15.4.0. No dependency added.
