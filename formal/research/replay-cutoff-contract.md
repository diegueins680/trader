# Replay ordering cutoff — preregistered contract

Engineering audit v1, 2026-10-01. Classification: numerical correctness of integer
indices, lifecycle ordering, conditional liveness and model conformance. The prior
six-bar certificate remains unchanged. No financial experiment or data access.

For remaining bars R >= 1 and horizon H in {1,3,6}, let E = min(R,H),
C(R) = min(R,7). For every integer 0 <= t <= E:

- min(C(R),H) = E and t <= E <= C(R) <= R;
- (t = R) iff (t = C(R)); consequently the extracted ordering-model predicate
  `failed or t == limit` is preserved for either Boolean failure value.

F-RL-REPLAY-CUTOFF checks these implications by integer SMT, with independent
satisfiable premises and unsatisfiable violations. UNKNOWN/timeouts fail. The
model's sole read of `limit` in its successor function must be that terminal
predicate; transitions must preserve the field via ordinary dataclass replacement.
Pin the complete earlier model source and audit this dependency explicitly. No
Python interpreter refinement is inferred from a source hash.

The abstraction alpha changes only limit to C(limit). Equal end indices and equal
terminal predicates make corresponding branches independent of the uncapped
limit. All other fields, event labels and successor updates are unchanged. This
is the stated simulation argument for the ordering model; trusted Python/AST and
dataclass semantics remain explicit assumptions. A cap of six is deliberately
unsound: R=7, H=6, t=6, failed=False is nonterminal in the original but terminal
after that cap. Preserve this prescribed bad-abstraction witness as a regression,
not as a discovered simulator defect.

F-RL-REPLAY-QUOTIENT explores the complete finite graph with remaining classes
1..7 (7 represents any R>=7), horizons 1/3/6, delays 0/1 and pending offsets
absent/1/2. Reuse all existing edge invariants and acyclic-progress checks. Record
state/edge counts, shortest depth and maximum progress rank. For every reachable
class-seven state, compare labeled successors after lifting to R in
{7,8,12,97,2^63}, then applying alpha. These finite comparisons are implementation
conformance evidence; the integer branch lemma has no finite upper bound on R.

Run the actual unchanged synthetic Replay helper over the preregistered 486-case
grid with remaining {7,8,12}, plus eight named scenarios. Check visible event
traces and final time/row/pending/terminal tags against class seven. In particular,
a six-bar call with seven remaining bars must not liquidate solely because the
call ended. The existing helper checks old-inventory marking using tolerance;
this is a test, not an exact floating-point theorem. No market data is used.

**Scope limit:** actual observation time-to-end depends on R, and policy outputs,
rewards and risk outcomes may therefore differ. This is not an observation,
reward, policy or complete simulator bisimulation. The abstraction deliberately
allows observation/data validity and risk outcomes nondeterministically. It
does not prove multi-call state refinement, numeric accounting, exceptions,
concurrency, physical deadlines, Haskell production behavior or authorization.
Early failure may retain simulated inventory. All 38 mission obligations remain
open or partial; no promotion or champion change is authorized by this result.

Use pinned Python 3.13.3 / NumPy 2.3.5 / Z3 4.15.4, seed 0, separate queries and
10-second solver limits. Reproduce offline through `bash scripts/verify.sh formal`
and `bash scripts/verify.sh full`. Tests must reject source/dependency drift,
the six-cap abstraction and unknown solver results. No new dependency or runtime
configuration is introduced. The existing [ordering contract](replay-order-contract.md)
and its named assumptions apply unchanged.
