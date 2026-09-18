# Source and registration snapshot audit — 2026-09-18

Baseline: merged/deployed main `d689682df282f59ae27c3a284c47b3529e38badf`.
This is engineering evidence from temporary files and synthetic replay fixtures.
No historical archive, protected holdout or new financial experiment is involved.

## Reproduced defect

The runner compared source files with Git, then reopened the registration for
parsing, recomputed registration/source hashes for the manifest and reread the
registration hash for every saved policy. These operations could consume different
bytes during a long run. Replacing a pathname after admission could change parsed
settings, mislabel manifest sources or bind a policy to a different registration.

One regression method failed in all three baseline subcases: replacement after
source validation, after data admission and during training. The fixtures replace
a temporary registration and source file, retaining valid JSON to show the
provenance disagreement rather than merely a parse error. This does not establish
that any previously recorded research run encountered such a replacement.

## Repair and scope

Capture the registered source/registration files once per distinct path, then
compare those captured bytes with blobs from one resolved Git commit. Parse the
registration from its admitted bytes. Derive source and registration SHA-256
values from the same snapshots and retain them for the manifest and all policy
artifacts. Dataset admission still consumes the unchanged registered data hashes.

A mismatch rejects before dataset reads or archive creation. Later replacement
cannot modify settings already parsed from admitted bytes or relabel provenance.
A fresh invocation captures and validates fresh files. The existing no-argument
`source_commit()` checker remains compatible; the runner supplies snapshots to it.
No CLI flag, artifact field, model identifier, policy behavior, registration,
experiment budget or promotion gate changes.

This is a file-byte identity guarantee. It does not attest already loaded Python
modules, dependency binaries, interpreter/environment identity, Git repository
trust or the authenticity of market data. It does not authorize rerunning a
historical campaign, reopen confirmation evidence or make development data fresh.

## Executable evidence

Four added methods bring the research suite to 118 tests:

- Replacement at all three stages preserves original parsed campaign, manifest
  source/registration hashes and the saved policy registration hash.
- Captured admitted bytes validate even if the path is replaced; captured mismatches
  reject without reopening it. The no-argument checker still inspects current bytes.
- An uncommitted source or registration rejects before data admission or output
  directory creation.
- A valid run reads each registered source path exactly once and preserves expected
  commit/source/registration provenance in the manifest and policy artifact.

Existing dataset snapshot, archive export, strict JSON, registry reconciliation,
publication failure, optimizer, inference and multi-seed synthetic tests remain
in verification. The initial full research run passed all 118 tests in 28.239 s.

```bash
python3 test/sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

A 300-iteration local capture-and-hash microbenchmark on the shared macOS Intel
host retained 83,797 source bytes across five files. Median/p99/maximum were
1.36358/3.25508/5.83567 ms. This excludes Git subprocess validation, imports,
datasets and training; it is not an end-to-end timing or memory bound. The change
adds no inference work. No performance budget or timeout gate is relaxed.

Actual wrapper/CI/deployment results and external log hashes are recorded in the
PR. No model or downloaded dataset is committed; fixtures are generated in tests.

## Decision and contracts

`A-SEQUENTIAL-RESEARCH-R26` links four executable witnesses. Haskell/Markdown risk
mitigations remain synchronized and canonical `RL-OFFLINE-001` remains HIGH/OPEN.
README, CHANGELOG and reproduction guidance document snapshot identity.

No candidate passed. Preserve the champion and all existing economic, seed,
OPE/statistical and holdout conclusions. No new algorithm, live exploration,
policy integration, production order authority or automatic promotion is added.
The authorized software deployment must preserve existing production settings;
production is already live and is not described as globally disabled.
