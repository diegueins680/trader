# Downward rounding follow-up — 2026-10-05

This engineering repair addresses broader obligation 9 without claiming full
order-adapter correctness. Baseline main 1911aca834df47849d394913e8660d35a892e054;
preregistration b71f9364, dependency-free property clarification d9e93022, both
before implementation. No financial trials, market data, RL retraining or holdout
access. Current champion, fleet, permissions, caps and deployment are unchanged.

CE-ROUND-001: scale=increment=1, input binary64 word 4607182418800017407
(0.9999999999999999). The inherited epsilon floor returns 1, above input.
The repaired exact grid floor returns zero. Invalid/non-finite inputs and
nonpositive Step fields return zero before rational conversion or division.
The public Step/quantizeDown interface remains unchanged. Exact interpretation
of binary64 may reduce decimal boundary quantities by one increment; 0.3 with
step 0.1 becomes 0.2. No claim of final decimal wire-grid/cap correctness.

The [contract](../../formal/research/quantity-rounding-contract.md),
[source binding](../../formal/research/quantity-rounding-source.json),
[counterexample](../../formal/research/quantity-rounding-counterexamples.json),
[ledger](../../formal/research/proof-ledger.json) and
[verifier](../../scripts/formal/quantity_rounding.py) provide both directions of
requirement-to-code/test/CI traceability. The adapter delegates to the actual
compiled base-only core. No new dependency or state machine is introduced.
Existing capability/lifecycle checks remain mandatory.

Five SAT-premise/UNSAT-violation pairs cover mathematical Euclidean division,
rational reconstruction, binary64 publication guards and invalid-grid rejection.
These are source-bound SMT lemmas under named GHC/base/runtime assumptions;
source review and differential tests do not prove the compiler. Compiled oracle
coverage: 130 boundary cases plus 4096 seeded word patterns, seed 20261005,
including invalid inputs, infinities, NaN, signed zero, subnormal/extreme finite
values and Integer grid fields up to 10^400. Production tests independently check
Binance delegation and finite/non-increasing output across 1000 generated words
plus fixed edge cases and eight grids. Property tests are not formal proofs.

Initial targeted tests passed (2 tests, 18.861s locally). Canonical full/formal
results will be recorded before merge; no readiness claim is made while pending.

Broader status: five scoped closures, 27 partial and six open. Obligation 9 moves
from open to partial; closure criteria are unchanged. Its remaining blockers are
both Main quantizeUp implementations, quantity/notional minimum increases,
8-decimal renderDouble output, side-specific price semantics, other venues and
the relation between every serialized order and its authorized exposure cap.
No out-of-sample return, drawdown, tail-risk or RL efficacy result changes.
Recommendation remains no candidate adoption; continue scoped verification.

The preflight refinement was committed at 820c2de2 before Main edits. During
review, CE-ROUND-002 showed that an intermediate zero-only repair could reach
minimum-size promotion for infinity. Both quantity-normalization callers now
perform shared finite/grid validation first; its errors are not eligible for
minimum-size retry. Two optional-grid variants are checked in all 4226 compiled
cases (8452 preflight checks); source-bound caller/classifier fragments and
Either-composition properties back the correspondence. Full Main correctness is
not proved. The initial local receipt attempt failed in the unchanged PPO-v3
snapshot probe after its 3-second timeout; source 6cf47360 CI reproduction was
superseded by this refinement. No timeout was relaxed.

Refined targeted suite: 2 tests passed in 13.752s. During fixture development,
a checker import expectation and generated Haskell layout error were corrected;
neither was suppressed. Initial CI 37268381702 ran 194 tests successfully, then
failed the existing drain-pool source hash for changed trader.cabal. Reviewed
Main/Cabal/TestMain changes were rebound in six inherited source registries;
all affected extractor checks passed unchanged. Certificate recording and full
verification must reproduce these inherited obligations before merging.

Engineering benchmark (exploratory, additional to preregistered property cases):
GHC 9.4.8/base 4.17.2.1, -O2, macOS x86-64, 100000 deterministic inputs,
scale=100000000, increment=1, x=i/1000003 for i=1..100000. Force the sum with
Control.Exception.evaluate and measure start/end with System.CPUTime.getCPUTime.
Checksum 5000.034499900002; CPU time 0.181828 seconds (1.81828 microseconds/call).
This excludes compilation, I/O, caller validation and order serialization; it is
not a wall-clock deadline or worst-case huge-Integer bound. Reproduce the core:

```haskell
evaluate (sum [quantizeDownExact 100000000 1 (fromIntegral i / 1000003)
              | i <- [1..100000 :: Int]])
```

The local haskell wrapper built the initial core successfully and passed format
checks, then was explicitly stopped during its superseded Main HLint run after
the preflight refinement. It is not reported as a passing final wrapper. Only
that verification subprocess was stopped; user services were not touched.

Pinned reproduction d0373398 failed before proof execution because the main
lock still carried pre-refinement hashes for the already-committed contract
and registration. Those two reviewed hashes were corrected and the entire
locked source roster checked before rerunning; no proof result was edited.
