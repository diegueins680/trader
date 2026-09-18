# Normalization admission audit — 2026-09-17

Baseline: merged main `6745fefe2b800ee607d48f7ca0e372dd2ad3fc27`.
This is an offline engineering audit using deterministic synthetic fixtures.
No market dataset, protected holdout or new financial experiment is used.

## Reproduced defects

- A fitted scaler's arrays could be made writable with `setflags(write=True)`.
  The frozen dataclass prevented ordinary field replacement but did not make its
  NumPy-owned data immutable.
- A directly constructed scaler retained caller-owned arrays: changing the
  caller's mean array changed normalization afterward.
- Negative deviations were accepted and reversed a positive feature's sign.
  Scalar and boolean inputs could pass the coordinate-support predicate through
  broadcasting or numeric coercion.
- Fitting a valid prefix alongside a shorter-than-25-bar prefix silently ignored
  the short prefix because it generated no feature rows.
- Integer subtraction could wrap before division. Subtracting the maximum int64
  mean from the negative maximum int64 observation yielded positive 2 instead of
  a large negative normalized value.

Four new methods brought the suite to 54 tests and initially reproduced 52
failures and four errors. A later integer-wraparound regression independently
failed before its arithmetic repair. All 54 methods pass after both repairs.

## Repair and semantics

The constructor admits only six-component real unmasked parameter arrays, checks
finite values after float64 conversion, requires positive deviations and ordered
support bounds, then retains immutable backing bytes. Caller mutations and
write-flag reactivation cannot modify these snapshots through ordinary array
access. This is not a sandbox against arbitrary Python code or reflection.

The existing fit path already computed float64 parameters. Direct construction
now explicitly normalizes supported numeric arrays to float64, so integer
arithmetic cannot wrap. Unsupported wider values that become non-finite are
rejected. This is floating-point normalization, not exact arbitrary-precision
integer arithmetic. Existing fitted values, formulas, deviation floor, feature
ordering and saved JSON provenance fields remain unchanged.

Fit admission requires nonempty list/tuple inputs with at least 25 real unmasked
prices per supplied prefix. Feature extraction still validates all training rows
and uses only the supplied prefixes; this adds no validation/test-data access.
Transform and support require finite real unmasked six-component feature vectors.
Invalid transform input or a non-finite result raises; malformed support queries
return false. Replay converts normalization errors into absent observations and
rejects the proposal before creating a fill.

The scaler's support predicate remains a coordinate-wise training-range check.
Well-formed parameter values alone do not prove authentic provenance, correct
training boundaries or joint state-action support. Those requirements remain
external to this object and remain promotion blockers where unproven.

## Verification and parity

Tests cover parameter shape/type/domain, immutable fitted and direct snapshots,
caller mutation, invalid support/transform inputs, integer wraparound, numerical
overflow, replay no-fill behavior, incomplete prefix rejection and the valid
25-bar minimum. Existing training-prefix causality and multi-seed determinism
tests remain in the same suite.

All 108 synthetic scenarios from the [replay recipe](proposal-types-audit.md#reproduction)
remain byte-identical: SHA-256
`22276e1b07f503e97c2bb57315aac1ef3079ed1c25dcbb0a07f6cb7e10f3325a`.
All 18 scenarios from the [collection recipe](episode-admission-audit.md#reproduction)
also retain exact bytes: SHA-256
`81f811bae0802aba61cfba6674be0f90afb65ad828158cda281492a0ba9d1926`.
These are same-host engineering comparisons, not market evidence or a
cross-platform floating-point guarantee. Generated outputs stay outside Git.

Run from the repository root:

```bash
python3 test/sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The PR records the final source revision, actual full-wrapper result and log hash,
and final-head GitHub CI. `A-SEQUENTIAL-RESEARCH-R9` links four executable
witnesses. Haskell and Markdown mitigations remain synchronized;
`RL-OFFLINE-001` remains HIGH/OPEN.

## Decision

No historical result or artifact is rewritten. This does not establish that
historical scaler arrays were previously mutated or that reported outcomes change.
No dependency, configuration, production caller, live authorization, exposure,
risk setting, deployment, champion or protected holdout changes. No live
exploration or order authorization is introduced. No candidate passed; preserve
the champion and continue offline research.
