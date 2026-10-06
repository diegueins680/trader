# Delivered candidate stage preservation — 2026-10-06

Base main `ac9288ee8c7d61fa6af283f81a4bdf7b47846966`; preregistration commit
`4df1f9bc` preceded implementation. This audit changes verification infrastructure
and documentation only. No runtime source, configuration, model, artifact format,
dataset, training protocol, deployment or financial result changes.

## Consistency finding and authoritative interpretation

Original obligation 25 requires candidate promotion to follow mandatory evidence
stages without skipping or self-promotion. The prior blocker requested a concrete
persistent promotion service, although every delivered artifact remains research
only and no production executable imports its proposal boundary. The negative
research decision has not authorized any candidate to advance.

The [preregistered contract](../../formal/research/lifecycle-stage-contract.md)
resolves this mismatch before the proof work: stage safety requires verification
of the complete existing path, not invention of an unrequested service. The
implementation already enforces a stronger refusal: no eligibility advancement.
A future promotion implementation must reopen this obligation and prove its
history/evidence gates. Its existence, correctness and liveness are not inferred.
All original 38 titles, scope strings and closure criteria remain unchanged.

Eligibility is distinct from computation. A researcher may construct, simulate,
serialize, load and infer with an explicitly enabled offline model without making
that model eligible for an operational replay, shadow or paper stage. In particular,
`OfflineReplayV1` produces a research proposal, not an eligibility certificate.
No existing semantic identifier is renamed or reinterpreted at runtime.

This safety/liveness distinction follows Alpern and Schneider, *Defining Liveness*,
Information Processing Letters 21(4),181–185 (1985),
[DOI](https://doi.org/10.1016/0020-0190(85)90056-0),
[author institution record](https://ecommons.cornell.edu/entities/publication/2ed32f4f-cc5c-413b-ba16-5498641f1939),
verified 2026-10-06. The paper distinguishes invariant restrictions from eventual
progress. Applying that distinction to obligation 25 is our specification judgment;
the paper does not certify this repository. No paper PDF or copied abstract is
committed, and no predictive-efficacy inference follows from this reference.

## Implementation-linked evidence

[Checker](../../scripts/formal/lifecycle_stage.py),
[ledger](../../formal/research/proof-ledger.json),
[closure audit](../../formal/research/obligation-closure-audit.md).

- Source correspondence reuses the complete 15-module/1,949-call-site reviewed
  effect inventory. Actual writer tags and decoder predicates are extracted.
  Existing artifact/Haskell protocol/component/native-schema/default/shield and
  added-boundary certificates remain mandatory in the same verification invocation.
- Eight SMT queries, each with a satisfiable premise and UNSAT violation, cover
  arbitrary string metadata, non-string JSON classes, exact-False identity and
  arbitrary evidence/review booleans. Accepted v1/v4 metadata always projects to
  Research. Unknown metadata is never silently mapped to a valid research state.
- The finite model explores 28 states and 168 transitions to a reachable fixed point,
  maximum shortest depth 6. It includes both formats, seven eligibility stages,
  save/load/infer/failure/disable/retry/successor events, 84 rejected metadata edges
  and 12 states retaining a proposal. Both ordinary inference paths remain reachable.
  Retry histories are not cut off at a chosen depth.
- Actual-code conformance covers 204 metadata cases with a correct recomputed hash
  on every case: 2 ordinary research cases accepted, 202 rejected. Non-string values,
  advanced-stage labels, wrong versions of research labels and false-like values
  cannot bypass the guards. Nested provenance data cannot replace root controls.
- Removing stage or exact-False guards fails the SMT checks independently of
  source hashes. Missing constituent certificates block closure. A stage-skipping
  model mutation is preserved as
  [CE-RL-STAGE-SKIP-MUTANT](../../formal/research/fixtures/lifecycle-stage-skip.json).
  It is a verifier regression, not a claimed historical runtime defect. In that
  trace, the load suffix `True` means the exact-False identity guard holds; it does
  not mean the artifact enabled field is true.

A-STAGE-PROJECTION names the reviewed mapping from concrete metadata/proposals to
eligibility, ordinary pinned runtime semantics, fixed code and the existing
capability assumptions. Source pinning alone is not a proof; predicate extraction,
SMT, finite checking and actual-code conformance have distinct statuses. The finite
model relaxes unrelated loader rejection guards to conservatively admit more
research inputs. It is not a complete parser, runtime, filesystem, concurrency or
compiler refinement. No future promotion service, progress theorem, cryptographic
provenance truth or deployed-image theorem is claimed.

## Closure and unchanged research decision

Only obligation 25 changes from partial to conditional `exhaustively_checked` after
the required certificates reproduce. Totals become **12 scoped closures /24 partial /
2 open**, with 26 unresolved obligations overall. No original criterion is weakened.
Every constituent must reproduce; any new consumer, callback, format, eligibility
transition, promotion service or production integration reopens this closure.

Numeric/accounting, historical availability, rollback, production authority,
ownership/readiness, broader concurrency, shutdown, recovery and progress remain
unresolved as recorded individually. The economic and reproducible-delivery
acceptance gates remain open. `missionComplete` and `formalObligationsComplete`
remain false; no production authorization is implied by a scoped closure.

No candidate passed; continue offline research. The frozen 108 fits, 19,440 replays
and 108 invalid OPE batches remain contaminated development evidence. No new OOS,
transaction-cost/stress, DSR/PBO/SPA, seed, drawdown, tail-risk, training, inference
or simulator-to-history result is reported. The 1,227-return holdout remains sealed;
the prospective embargo stays 2027-01-20T13:00Z. Champion, fleet, live flags,
leverage, risk limits, ownership and production deployment are unchanged.

## Reproduction

Existing pinned Python 3.13.3, NumPy 2.3.5, Z3 distribution 4.15.4.0/solver 4.15.4,
GHC 9.4.8 and Cabal 3.12.1.0; no new dependency or tool. After installing the existing
locked dependencies, proof reproduction requires no market data or network.

```sh
PYTHONPATH=scripts/formal:scripts/research python3 -m unittest test_integrity.LifecycleStageTests test_integrity.ObligationClosureTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

The optional `--require-complete` gate must still reject 26 unresolved obligations
and two research acceptance gates. It is not weakened to enable this scoped merge.
Targeted local tests: 14 passed in 3.280 seconds. All 274 local integrity tests
passed in 288.752 seconds. The first local `verify.py --record` attempt failed in
the unchanged PPO process-bridge conformance with
`ValueError: PPO process bridge: no actual inference for trained policy`.
The bridge retains its existing 20 ms guard; no timeout, assertion or gate is
weakened. That attempt is not a pass. The clean local retry failed at the same unchanged check. Neither attempt is a
pass. Pinned canonical formal/full CI reproduction remains mandatory and pending;
no local service or process is stopped to influence timing.
