# Terminal learning-target numerical scope — 2026-09-28

Specified before implementing verification artifacts. Preserve the frozen learner,
registrations, policy artifacts, economic results and unopened holdouts. The initial
synthetic probe found a current-source numerical limitation, not a market result:
a constant finite critic of 1e16 and terminal reward 1 yields reconstructed target 0.
No repair or successor financial experiment is preregistered by this contract.

## Claims

**F-RL-TERMINAL-REAL (smt_verified).** Derive the TD residual, backward GAE recurrence
and target reconstruction expressions from actual `advantages` source. Under exact
real arithmetic and a true Boolean terminal mask, the raw terminal advantage is
reward minus current value and the reconstructed target equals reward, independent
of finite next value and subsequent raw carry. This excludes batch normalization.

**F-RL-TERMINAL-FP-MASK (smt_verified).** For IEEE binary64 round-to-nearest/ties-even,
finite discount in [0,1], finite next value and finite later carry, the actual two
terminal bootstrap products compare equal to zero (signed zeros are equivalent).
This does not imply finite residuals/carry, exact reconstructed targets, or bitwise
identity of signed zeros. NaN/infinite operands are outside this claim.

**F-RL-TERMINAL-EXACT (refuted).** Audit the stronger claim that finite binary64
reward/value/discount inputs and a true terminal flag imply exact reward recovery
from the current target reconstruction. Preserve the cancellation counterexample
in actual NumPy and with an exact binary64 SMT witness.

**F-RL-TERMINAL-FINITE (refuted).** Audit the claim that finite learner inputs and
finite critic outputs imply finite target/advantage outputs. Preserve an overflow
witness and demonstrate how zero times an infinite later carry can contaminate an
earlier terminal, despite a correct Boolean mask. No empirical prevalence claim.

## Source and assumptions

Bind the complete advantages-function AST to an audited skeleton, including loop
order and target reconstruction before batch normalization. Extract arithmetic
slots and translate only supported operations. Require SAT premises and UNSAT
violations for positive claims; require prescribed SAT counterexamples for refuted
claims. Mutations removing either terminal mask must fail the corresponding checks.

A-TERMINAL-NUMERICS: ordinary correctly shaped float64 arrays, Boolean done array,
matching row order, finite supplied critic outputs where a claim requires them,
stable primitive bindings and no concurrent mutation. The AST checker, Python/
NumPy vector-to-scalar correspondence and separate binary64 arithmetic operations
are trusted. No fused algebraic reassociation, extended-precision guarantee,
interpreter proof or complete learner refinement is claimed.

The finite-product theorem has a finite-carry premise; forward-network finiteness
does not establish that premise after GAE arithmetic. Batch-wide normalization
couples rows and can itself be ill-conditioned. Do not infer normalized-advantage
noninterference, learning convergence, optimizer safety or economic superiority.

## Conformance, traceability and disposition

Use small deterministic untrained critic networks and synthetic observations;
exercise current-source counterexamples, finite well-conditioned terminal cases,
and episode-boundary perturbations of raw target inputs. No large artifact or
market data is used. Clearly label these regressions, not universal proofs.

Record the current-source counterexamples as unresolved numeric blockers in the
proof ledger and risk report. Keep the current algorithms and their rejected
financial evidence frozen. Any numerical repair that changes training targets
requires a separately identified implementation/version and reviewed registration
before new financial trials; no silent reinterpretation of prior results.
