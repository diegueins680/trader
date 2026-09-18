# Policy-save admission audit — 2026-09-17

Baseline: merged main `8b374b796788aa42e0d6d3623da217424ad74f3e`.
All evidence below uses untrained networks and deterministic synthetic fixtures.
No market input, protected holdout or financial experiment is used.

## Reproduced mismatch

The writer successfully produced an artifact and hash for policies with a missing
parameter, wrong output shape, boolean weights or masked weights. Loading those
same bytes with the returned hash and matching provenance rejected them. The
runner could consequently record a successful fit and artifact even though the
artifact could not be consumed as a v1 policy.

Four new test methods brought the suite to 58 tests. After giving each malformed
case its own temporary path, the pre-repair suite reproduced 19 failures and 14
errors. An earlier fixture reused its path, which mixed subsequent exclusive-open
errors into its result; that fixture was corrected before assessing the repair.
The runner fixture specifically showed a one-output critic written as a policy.

## Repair

One helper defines the exact v1 parameter names and shapes: `w1` 12x16, `b1` 16,
`w2` 16x3, `b2` 3. Saving requires NumPy real unmasked arrays, copies the values,
and checks finite float64 portability. Missing/extra fields, malformed shapes,
boolean, complex, object, masked and non-finite parameters fail before opening
the destination. Floating arrays use portable float64; integer JSON weights keep
their prior numeric representation. The network's own parameters are not changed.

Loading continues to reject nonnumeric JSON leaves, including booleans and nulls,
before converting arrays and applying the same shape/value contract. Invalid
numeric conversion raises an explicit error. The existing digest, provenance,
schema, duplicate-field, 65,536-byte and disabled-state checks remain intact.

The writer retains exclusive creation. Validation failure cannot create an
artifact or return a success hash, and an existing artifact cannot be overwritten.
This is not an atomic-filesystem-write guarantee: I/O failure after creation can
still leave an incomplete file, which the loader must reject. Snapshots prevent
ordinary aliasing during serialization; they are not a defense against arbitrary
code or concurrent mutation while the initial copy is being captured.

## Verification

All 58 Python methods pass after repair. The runner-to-export fixture supplies an
invalid one-output network without actual training, then confirms:

- No policy JSON is created.
- Training records a failure without an artifact digest.
- Every planned fit/replay has a failed terminal event; no trial disappears.
- The compact exporter accepts the failed evidence and prohibits promotion.

Existing valid artifact bytes and loader inference outputs match across 36 cases:
four policy-family labels, seeds 11/23/47, and float64/float32/integer-bias weights.
These labels do not imply any learning or algorithm comparison in this fixture.
The complete sorted-JSON fixture digest is
`020dbfdf167008b238a7d1f414493eb9677686bf2c8d9113fd87ffd7f76474ee`.
The test `test_policy_save_retains_valid_v1_bytes_and_loader_parity` contains the
reproduction recipe and expected digest. The original baseline payload remains
outside Git; no generated model artifact is committed. This is an engineering
compatibility check, not economic evidence or a general cross-platform guarantee.

From the repository root:

```bash
python3 test/sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The PR records the final source revision, actual full-wrapper result and log hash,
and final-head GitHub CI. `A-SEQUENTIAL-RESEARCH-R10` links four executable
witnesses. Haskell and Markdown mitigations remain synchronized;
`RL-OFFLINE-001` remains HIGH/OPEN.

## Limits and recommendation

No historical artifact or report is rewritten. This does not establish that an
original research artifact had invalid parameters or change the original seed,
cost, stress, OPE or rejection results. A loadable policy is not evidence of
causality, statistical merit, adequate support or operational readiness.

No dependency, configuration, production caller, order interface, live exploration,
authorization, risk setting, champion, deployment or protected holdout changes.
No candidate passed; preserve the champion and continue offline research.
