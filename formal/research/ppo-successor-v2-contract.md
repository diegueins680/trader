# Composed offline PPO successor v2

Specified against main 0b919b389f02d6f293ab3837d1620cf67ca0ad06 before implementation.
This is an engineering repair of the frozen learner, not a new financial trial.

The public training entry is versioned `ppo-successor-v2`, default disabled, and
has no file, network, order, promotion or production interface. Enabled calls
accept ordinary base arrays containing already selected training prefixes only.
They copy and freeze price/funding prefixes, fit a new shared Scale solely on
those prefixes, and run the existing collector and PPO objective. No held-out
input or arbitrary policy callback is accepted. Finite funding admission occurs
before collection. Prefix provenance/availability remain caller obligations.

Bound the admitted call to 1..8 symbols, 121..4096 rows per symbol, 1..4096 training
steps, horizons 1/3/6 and seed 0..2^32-10000. Each rollout has at most 256 transitions.
All numeric computation runs with overflow/invalid/divide/underflow raising; a
caught arithmetic/shape/memory failure returns absent. Reject nonfinite rollout
arrays, logits, probabilities, objective loss, gradient, targets, normalized
advantages and residuals. Do not substitute zeros for failed data or objectives.

Use batch_v2 for raw GAE and targets and optimizer_snapshot_v2 for every actor
and critic forward/update. Four actor/critic updates per rollout preserve the
frozen PPO schedule; the terminal-target repair intentionally differs from v1.
No mutable Network facade or in-place copy_from is permitted. Optimizers are
private to a call. Publish only after all requested updates succeed, as frozen
actor/critic snapshots and an immutable tuple of finite losses. Failure discards
local progress; no prior policy, caller array or champion artifact is mutated.
A final result is a research training result, never a live-capable policy.

Proof obligations:
- F-RL-PPO-V2-FLOW: derive a finite publication transition model from a reviewed
  complete source skeleton. Each stage failure terminates absent; completion
  requires all stages for every batch. Model all one through sixteen batches/four epochs and retain
  the source-derived integer budget lemma for all 1..4096 steps. No partial result.
- F-RL-PPO-V2-BOUNDS: SMT on source-derived configuration predicates and integer
  batch arithmetic; every collected count is 1..256 and sums to exactly steps.
- F-RL-PPO-V2-FINITE: source-derived binary64 loss guard requires finiteness
  before the actor update; snapshot slot finiteness uses the existing certificate.
- F-RL-PPO-V2-BOUNDARY: complete AST call/import inventory and default guard;
  public output contains only immutable snapshots/primitive metadata. No effects
  beyond owned optimizer state and private arrays under named helper assumptions.
- F-RL-PPO-V2-CONFORMANCE: actual collection, fit, target, optimizer and training
  execution with synthetic seeds 11/23/47; repeatability, caller immutability,
  failure injection, numeric invalids and all default/configuration rejections.

A-PPO-SUCCESSOR: trusted pinned CPython 3.13.3/GIL, NumPy 2.3.5, base arrays and
ordinary standard-library/dataclass semantics; stable caller buffers while copying,
no malicious reflection, package replacement, monkeypatching or arbitrary callback.
Existing A-SNAPSHOT-V2 and target-v2 assumptions apply. Library arithmetic and AST
translation are trusted, not fully refined. Local stage atomicity is not crash
recovery, process cancellation, real-time boundedness or durable publication.
Existing collector accounting, fill fidelity, revision timing and causal dataset
admission limitations remain. Finite output checks do not prove accuracy, absence
of all silent underflow inside BLAS, optimality, risk bounds or profitability.

The frozen screen, learner, policy/artifact IDs, reports and counterexamples stay
unchanged. No statistical claim or historical comparison is authorized. Do not
close any broad obligation solely from these certificates. Extend existing
closed default/capability/no-production-learning requirements to cover this new
surface explicitly. OPE/ESS, artifact decoding and Haskell process inference are
subsequent composition work, not claimed here.

Verification refinement before merge: expand the initially proposed two-batch
bound to every admitted batch count (1..16); the arithmetic proof additionally
checks that the source loop count lies in that complete model domain. No financial
experiment, implementation behavior or statistical rule changes.
