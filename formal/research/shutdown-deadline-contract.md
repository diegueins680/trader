# Shutdown deadline contract v1

Engineering registration: `research-notes/registrations/shutdown-deadline-engineering.json`.
Affected broader obligations: 10 (timeout arithmetic), 21 (failure reporting),
36 (inherited server shutdown). No closure of those whole-system obligations is
asserted by this component repair.

## Existing conflict and authoritative interpretation

README promises bounded shutdown, while `runServeShutdown` computes deadlines
from `getTimestampMs` (UTC wall clock), multiplies unbounded configured `Int`
seconds in `Int`, and logs completion even after failed stages. Wall time can
move backward and integer multiplication can overflow. The required intended
behavior is bounded waiting with explicit failure, not forced termination of
uninterruptible threads. Wall time remains appropriate only for audit timestamps.
Existing timeout configuration, stage order and final database-close reserve stay
unchanged. The old implementation is preserved by source commit and counterexample
fixtures; no deployment or live configuration change is authorized.

## Mathematical definition

Let s be the sampled monotonic start in integer nanoseconds and q configured Int
seconds. B = max(0,q) * 10^9 and C = min(2*10^9, B div 4).
Work deadline is s+B-C; final deadline is s+B. All arithmetic before conversion
uses Haskell Integer. A sample n < s or s < 0 is invalid and fails closed.
For valid n, remaining microseconds U = min(maxInt, max(0,(deadline-n) div 1000)).
A stage is dispatched only if U > 0. A reported acknowledgement requires both
successful completion and a fresh sample with U > 0. Sub-microsecond remainder
is conservatively rejected, never rounded up. maxInt is the actual compiled Int
bound; proof considers any positive bound.

F-SHUTDOWN-BUDGET: U is in [0,maxInt]; U*1000 <= deadline-n whenever U>0;
invalid or expired samples admit no work; U cannot increase as valid n advances.
These are exact-integer SMT properties, conditional on the source translation.

F-SHUTDOWN-STAGES: five work stages and one final-close stage execute in order.
Each returns acknowledged or failed; failed/expired work does not skip the later
close attempt. `all acknowledged` iff all six stages acknowledge. A six-stage,
five-time-bucket model checks reachable states and a decreasing stage rank.
Atomic step completion represents return from the bounded wait, not resource
quiescence. Temporal eventual report is conditional on each primitive returning.

F-SHUTDOWN-CONFORMANCE: compiled pure functions must match executable integer
semantics over edge cases and 256 fixed-seed generated inputs. Actual step runner
must reject already expired input without running it, reject exceptions, and
return failure for uninterruptible cleanup. Source checks lock the reviewed main
stage sequence, clock selection, timeout routing and aggregate report.

## Refinement and limits

Concrete ShutdownBudget holds (s,workDeadline,finalDeadline). Abstraction is the
identity on those Integers and stage index; concrete successful/failing waits map
to one model transition. Reviewed source bodies, SMT scalarization and compiled
conformance connect implementation to the model; this is not full Haskell IO
refinement. Compiler, runtimes and checker are trusted (A-SHUTDOWN-CLOCK).

No theorem says all threads are gone when cancellation is delivered. Worker
registration races, multiple shutdown signals, route admission races, blocking
logs, parent cancellation, bot/position reconciliation and the frozen research
runner remain separate unresolved work. No financial trial, holdout read,
authorization change or production deployment belongs to this repair.
