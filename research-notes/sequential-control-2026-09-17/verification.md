# Verification receipt — 2026-09-17

`bash scripts/verify.sh full` ran from the isolated repository root and exited
**0**. All implementation source at completion is in commit `795ccff0` and its
ancestors. Subsequent changes are research documentation/receipts only.

| Check | Actual result |
|---|---|
| Haskell build, Fourmolu, HLint | Passed; HLint no hints |
| Haskell smoke and test suite | Passed, including 2,688 research proposal guard cases |
| Web typecheck/tests/build | Passed; 241 tests, no failures or skips |
| Root formal/automation wrapper | Passed; 185 tests, no failures or skips |
| Python sequential contracts through automation | Passed; 25 tests including 3-seed determinism, gradients, causality, artifacts, failed-training/export integration and all-trial evidence completeness |
| Formal registry at full run | Valid: 39 specs, 349 named features, 291 clauses, 300 implementation files, 128 evidence links, 33 canonical risks |
| Evidence export reproduction | All seven compact exports match committed bytes; all 116 indexed archive members match hashes after relocation |
| Financial experiment | Exit 0; all 108 fits and 19,440 replay paths recorded; this does not mean policies passed |

Targeted `bash scripts/verify.sh haskell` also passed. The initial sandboxed
`bash scripts/verify.sh automation` failed three local HTTP fixture tests with
`listen EPERM: operation not permitted 127.0.0.1`. The same command outside the
sandbox passed all 185 tests, as did the later full wrapper. No check was skipped
or weakened. Web dependency installation initially could not reach the registry
inside the sandbox; the approved outside-sandbox install succeeded.

The attempt to duplicate the large archive encountered `Errno 28: No space left
on device`. Only this task's incomplete duplicate was removed; the intact
original was atomically moved into ignored `.tmp/research/sequential-control-screen-v1`
storage and all indexed hashes checked. The final full verification exited 0.
No unrelated user files or production state were removed or altered.

Full log remains local at `/private/tmp/trader-sequential-verify-full.log`.
Its SHA-256 is `0c906f580aaa304745985dd95583250bbec6aa7d84b13829fbedc10cc25b6437` (logs are not a reproducible output hash across runs).
The CI run is separate from this local receipt; no unobserved remote CI result
is claimed. A passing test suite establishes engineering checks, not statistical
confirmation, exchange fidelity, an economically valid policy or live authority.
