# Sequential input-snapshot audit

This engineering follow-up closes a reproduced data-admission race. It adds no
financial trial, candidate, data collection, holdout access or promotion.
PR #256 was merged at `9aaec9e611f7cd5adee63b8293a068868ff85d1f` after its local
full wrapper and remote Haskell/web/automation jobs passed. The merge commit's
skip marker prevented the main-push workflow from deploying Fly and subsequently
Hetzner, preserving the user's separate no-deployment instruction. No required
PR check was bypassed or repository workflow disabled.

## Defect and reproduced evidence

The prior loader hashed both CSV paths and then reopened those paths in pandas.
A deterministic synthetic fixture replaced both files just before parsing. The
loader admitted a first close of 900 instead of the verified 100 and a funding
amount of 90 instead of the verified 0.1. The regression suite failed before the
repair (29 tests, one failure). No real-market file was involved.

The runner also recomputed data hashes from those mutable paths for its manifest
and every policy artifact. Replacement after admission could therefore attribute
the already loaded arrays to different, unparsed source bytes.

## Contract and repair

Each file is read once into an immutable byte snapshot. Both SHA-256 values must
match the registration before either snapshot is parsed. Pandas consumes those
exact byte buffers. A replacement during the read is rejected unless the captured
bytes still exactly match the registered hash. Replacement after the read cannot
change the parser's source. This relies on the existing SHA-256 integrity
assumption, not a filesystem lock or a claim that files are globally immutable.

Run manifests and policy metadata retain the registered hashes proven at
admission. They never reopen input paths merely to regenerate provenance. A
later invocation still reads and verifies its own snapshots and rejects changed
bytes. Output arrays remain read-only and all schema/grid/funding checks remain.

A second integration fixture uses the real loader on synthetic CSVs, replaces
both paths after admission, then runs the coordinator with a fixed fixture
network and checks both the run manifest and saved policy provenance. Training
is stubbed; replay uses synthetic prices and OPE is explicitly invalid. These are
engineering tests, not additions to the financial experiment registry.

Valid unchanged input bytes have the same price/funding interpretation and
provenance values. No existing model identifier, artifact schema, registered
period, seed roster, cost assumption or production interface changes. The old
frozen experiment source and compact evidence retain their original identity.
Holding both CSV buffers is consistent with the prior whole-file hashing
approach, though the buffers now remain alive together through parsing; this is
not a claim of streaming support for arbitrary-size datasets.

## Verification and limits

The repaired Python suite passes 30 tests, including the two new regressions.
`bash scripts/verify.sh automation` passed all 185 tests. The repo-root
`bash scripts/verify.sh full` exited 0 on implementation commit `354e4c34`:
Haskell build, Fourmolu, HLint, smoke and tests; web typecheck, 241 tests and
build; and 185 automation tests including the 30 Python research contracts.
No check failed or was skipped in this follow-up. The initial deliberately
failing regression is retained as defect evidence, not a verification pass.

The full log is `/private/tmp/trader-input-snapshot-full.log`.
SHA-256: `4796cfbed1a714d224d50c43824d7d58e2573b486117bb35eea020b4e7517130`.
The reproduction notes and this verification receipt are documentation-only
additions after the implementation freeze.

The formal A-SEQUENTIAL-RESEARCH admission and provenance contracts now name
snapshot identity and replacement behavior. RL-OFFLINE-001 remains HIGH/OPEN;
this fix supplies no independent market evidence, matched champion comparison,
credible OPE or actual execution validation. No original financial evidence is
rerun or reclassified. Recommendation remains no adoption.
