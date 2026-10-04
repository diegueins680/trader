# Inference process v2 — engineering contract

This is a separate, default-disabled offline executable, not a replacement for
the frozen learner or a production service. CE-RL-017 remains valid for v1.
No financial trial, artifact activation, holdout access or order is permitted.

## Representation and refinement boundary

One parent invocation owns one child and at most one request. The child is the
same compiled executable, selected by a fixed internal argument; neither an
arbitrary command nor a shell is accepted. It evaluates an explicit synthetic
12–16–3 tanh network supplied as bounded finite numeric data. This is an
engineering inference fixture, not a validated model or artifact loader. The
input is a Haskell pair `(observation, parameters)`; parameters are row-major
input weights, hidden biases, output weights and output biases (259 numbers).
The input frame is capped at 32768 characters. Every number must be finite and
have absolute value at most 1000. Tied scores abstain. Numeric equivalence to
NumPy, economic value and causal availability are separate, unproved claims.

Parent states are Disabled, Launch, Ready, Pending, Reject, Candidate, Term,
Kill, Quiescent and CleanupFailed. Disable produces absence before reading input
or spawning. Launch records a monotonic timestamp before creating the child.
Ready is an exact bounded protocol frame. The request, reply and final deadline
check must finish strictly before 20,000,000 ns from that timestamp. A reply is
one of three exact versioned frames, representing quarter exposures -1, 0, 1.
Those are data proposals, never authorized orders. All other frames abstain.

Every started session executes cleanup, including successful replies. Send TERM;
poll exit for at most 50 ms; if still pending send KILL to the unreaped child;
poll for at most another 50 ms. Cleanup failure is an explicit terminal result
and always rejects. A candidate can be returned only after successful reap and
a fresh deadline check. No retry, worker reuse, persistence or successor launch
exists. Thus late bytes cannot be consumed by another request. Async parent
cancellation invokes the same cleanup and rethrows; repeated cancellation,
process death and uninterruptible OS calls are not guaranteed recoverable.

## Assumption A-INFERENCE-PROCESS

Pinned GHC 9.4.8, process 1.6.18.0 and unix 2.7.3; trusted compiled self executable;
no external reaper, injected code, hostile filesystem replacement or descendant
process creation by the reviewed worker. The OS maintains an unreaped child's
PID identity, delivers signals, services nonblocking pipe operations, and
schedules parent/timer threads. Monotonic time does not wrap during a session.
Process creation and OS primitives terminate within environment-provided bounds.
There is no unconditional wall-clock theorem on a general-purpose OS. The
completion bound is launch/primitive/scheduler overhead plus the remaining
20 ms admission budget and two 50 ms cleanup windows. An unreaped child after
KILL is reported as CleanupFailed, never described as quiescent.

## Obligations and checks (before implementation)

* F-RL-PROCESS-LIFECYCLE: model-check all bounded timeout, reply, exit, signal and
  cleanup-failure interleavings for one child/request. Safety: no admitted output
  after expiry or failed cleanup, no successor, no live authority. Liveness:
  bounded progress to Quiescent or CleanupFailed under the named assumptions.
* F-RL-PROCESS-ADMISSION: source-bound SMT proof over integer timestamp semantics
  and finite reply codes; accepted target is bounded and elapsed is [0,20 ms).
* F-RL-PROCESS-CONFORMANCE: compile the actual Haskell executable and compare
  pure guards against the model; run actual subprocess fixtures for success,
  invalid input/reply, nonreturning computation, late response, EOF, ignored
  TERM and forced cleanup failure. These tests are not an OS correctness proof.
* F-RL-PROCESS-ISOLATION: audit the exact source/import/launch/output boundary,
  default dispatch, empty child environment, no filesystem writes, fixed self
  command and absence from production executable source roots. Source checking
  is not a malicious-code sandbox or a proof of deployed images.

The model-to-code relation maps the single parent control path, reply guard and
cleanup return to these states. Source locks, exact control skeleton checks and
compiled conformance support that relation; there is no full compiler or IO
refinement theorem. Broader obligations 21/36 remain blocked until the repaired
boundary is composed into a separately registered runner and inherited server
shutdown is verified. Existing safety gates must not be weakened to claim closure.
