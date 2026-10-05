# Closed-trade recovery — 2026-10-05

Baseline `d57968a66d8f85d08e6bfb430f1472781dcf2fa5`; contract and
registration committed as `c5336195` before implementation. Engineering repair,
no market data, financial experiments, orders, training, holdout or deployment.

## Conflict and repair

H-EXECUTION-I2 requires identity-matched bounded closed-trade recovery. Existing
code admitted non-finite equity, replaced explicitly invalid returns, could
derive infinity from finite equities, and wrapped synthetic Int indices.
G-FINITE/G-FAIL-CLOSED govern the repair: reject malformed records, derive only
missing finite returns, and publish only representable indices. Entry equity is
finite positive; exit equity is finite, including negative historical insolvency.
Supplied finite returns remain authoritative; no reconciliation theorem is claimed.
Holding periods are nonnegative; optional high-volatility probability lies in
[0,1]. Absent optional fields and zero-duration one-index spans remain compatible.

After existing last-N selection, the recurrence uses exact Integer arithmetic:
exit = entry + max(1,holding); next entry = exit+1. Every index is checked before
narrowing. An unrepresentable selected history returns empty recovery, never a
wrapped index or partially restored prefix. Invalid individual records continue
to be omitted independently by the existing mapMaybe parser boundary. Identity,
position/open-trade exclusion, API/JSON signatures and configured limits remain.
There is no model, policy, dependency, configuration or deployment change.

## Evidence

- CE-RECOVERY-001: `1e300 / 1e-300 - 1` derives infinity.
- CE-RECOVERY-002: an explicit infinite return is replaced by a finite fallback.
- CE-RECOVERY-003: `[maxBound,1]` holding periods wrap the second index pair.
- CE-RECOVERY-004: a supplied finite return permits non-finite entry equity.

These are synthetic code-domain counterexamples, not historical incidents.
Pre-change module source is preserved in the fixture and compared through
compiled actual helpers. No effectful venue code is linked into the driver.

Seven SMT SAT-premise/UNSAT-violation queries check finite admission, explicit
invalid-return exclusion, metadata domain, exact index ordering and 32/64-bit
narrowing bounds. The first attempt encoded binary64 division/subtraction inside
the satisfiability query and failed the fixed ten-second proof gate. The final
proof overapproximates that intermediate as any binary64 value: the publication
guard must reject every non-finite result regardless of its origin. This is a
stronger guard-domain check, not an arithmetic precision or error-bound theorem;
actual division behavior is checked by compiled conformance. No timeout is widened.

A local one-record eligibility/selected-history gate model has 22 states, eight
initial states, 22 transitions including stable terminal retries, depth three.
Finite recurrence enumeration covers 341 histories of length0..4, durations0..3,
bound7, with 74 nonempty admitted histories. It models pure publication, not
persistent filesystem recovery or concurrent bot ownership.

Compiled GHC `-O2` helpers agree with independent Python binary64/integer oracles:
2675 numeric rows (624 boundary combinations, 2048 generated, three witnesses),
1031 index histories (seven boundaries and 1024 generated), seed20261005.
Boundary numerics substitute 13 binary64 words into four slots, both optional
presence flags and holding periods -1/0/3. Each driver mode is repeated twice.
All 1446 admitted finite-return cases retain legacy bits; representable accepted
index histories retain legacy indices and duration fields. The real Haskell
suite exercises Aeson numeric overflow, invalid optional probabilities, independent
record omission, identity/position exclusion, last-N-before-indexing and Int limits.

## Assumptions and limits

A-RECOVERY-NUMERIC names GHC/base/Aeson, binary64, exact Integer, checked narrowing,
ordinary immutable non-bottom inputs, source-to-model mapping and resources.
The base-only driver projects Trade's three index fields. Actual JSON tests do
not constitute complete parser/compiler refinement. Missing provenance/freshness,
resource bounds, snapshot authenticity, durable crash recovery, concurrent
ownership and venue reconciliation remain open. Loss of malformed memory is
observable as omitted history; this patch adds no new runtime telemetry channel.

Broad status stays **5 scoped closures / 27 partial / 6 open**. Current champion
is preserved. Frozen research stays 108 fits, 19440 replays, 19548 registry rows,
contaminated development, 108 invalid OPE batches and 1227 sealed final returns.
No new OOS/holdout, costs, drawdown, tail-risk, RL seeds, OPE or economic evidence.
Recommendation: adopt no research candidate; continue offline work separately.

## Verification

Targeted ClosedTradeRecoveryTests passed (two tests, 12.674 seconds). Targeted
HLint reported no hints on the recovery module. The combined ledger/recovery
targeted run passed 12 tests in 7.994 seconds; formal registry validation passed. Canonical formal/full reproduction and final-head CI
must pass before this scoped repair is ready. No broad completion is asserted.
