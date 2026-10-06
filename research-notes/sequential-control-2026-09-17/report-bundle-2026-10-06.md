# Recoverable verified report publication

Base main `d1fe16869b5b475e9d953e7ffdd7432ac0e4fbd3`. Registration commit
`f3b37a80` preceded implementation. This is engineering: zero financial trials,
zero historical market-data reads, no training or holdout access.

## Gap, interpretation and delivered behavior

Legacy export safely refuses collisions but can leave a partial new report after
an I/O failure; retry then refuses the existing directory. The preserved
CE-REPORT-V1-RETRY fixture refutes the generalization that exclusive creation alone
implies recoverability. It does not refute the existing champion-preservation
contract, which explicitly allowed partial new output and excluded durability.

The optional `--report-bundle-v2` path runs the same source-index hash admission,
registry reconciliation and report rendering before invoking the new publisher.
Its default is off, and legacy seven-file export retains its exact bytes and
failure semantics. No archive is reclassified, no failed trial erased, and no
training run restarted. The bundle embeds those same seven UTF-8 reports with
fixed research-only metadata and a versioned schema. It cannot be loaded as a
native trading model. No production consumer or deployment integration is added.

The publisher requires an existing directory outside the source archive. It
opens a directory descriptor, compares an existing regular target byte-for-byte,
or exclusively creates a private staging file. It handles short writes, fsyncs
the complete file, publishes via an exclusive hard link, and fsyncs the parent
before reporting success. Matching retries preserve the target inode. Conflicts,
symlinks, directories, FIFOs, malformed reports and oversized bundles fail closed.
Only staging created by the current call may be cleaned up; old files are never
opened for writing or unlinked. Concurrent matching writers can return the same
result; conflicting writers cannot replace the winner.

A process death or cleanup failure can leave private staging files. Retries
ignore those files and create fresh staging. They are not completed evidence.
No unsafe generic scavenger is introduced. Disk capacity and finitely many
crashes are explicit assumptions; this is not a promise of progress under
unbounded failure or storage exhaustion. The new output format is a single
bundle; existing consumers of a seven-file directory must keep the default mode.

## Canonical requirements and proof scope

[Contract](../../formal/research/report-bundle-contract.md),
[registration](../registrations/report-bundle-engineering.json),
[source model](../../scripts/formal/report_bundle.py),
[ledger](../../formal/research/proof-ledger.json).

- F-RL-BUNDLE-SOURCE: full reviewed publisher and exporter composition; 4 publisher
  definitions, 15 filesystem call sites, one explicit exporter call. The source
  checker fixes descriptor modes, same-directory exclusive link, owned cleanup,
  metadata, size/version/default gates and fsync-before-success order.
- F-RL-BUNDLE-ARITH: 4 SAT-premise / UNSAT-violation queries for bounded strict
  partial-write progress, complete cursor, preserved existing content and exact
  identity-based acknowledgement. Hash equality never substitutes for byte equality.
- F-RL-BUNDLE-FLOW: 446 states, 2,127 edges, maximum shortest depth 15; two writers,
  two same/different content cases; crashes/retries explored to a reachable fixed
  point with no retry-depth cutoff. All 254 same-intent states admit a healthy
  completion path. All 508 healthy edges preserve or reduce a recovery rank;
  steps by an unacknowledged writer strictly reduce it (maximum 18). Eventual
  crash-free healthy I/O and weak scheduler fairness are environmental premises.
- F-RL-BUNDLE-CONFORMANCE: 14 actual process-death cases before/after seven boundaries,
  12 injected I/O failures, matching/conflicting concurrent processes, short writes,
  collision protection, strict admission, size limits, mutation regressions, and
  exact parity for all seven old report byte strings.

A-BUNDLE-FS explicitly trusts pinned Python, exclusive POSIX open/link, descriptor
identity and fsync behavior honored by the local filesystem and storage device.
It excludes hostile external mutation, unsupported remote/object filesystems,
media failure and runtime/kernel/compiler defects. Tests kill processes, not
power to hardware. The abstract crash persistence relation is an assumption about
the storage stack; source extraction and conformance are not a kernel proof.

Each critical new/changed source maps to the four requirements, source/model,
ledger, test suite and canonical CI commands. The research effect inventory grows
from 14 to 15 modules, with OS calls confined to the reviewed publisher. All 11
existing scoped closures require fresh bundle source/arithmetic/model certificates.
No claim about full production authorization or concurrency is added.

## Usage and reproduction

Default export remains unchanged. For the opt-in format, first choose an existing
output directory outside an already-approved, complete evidence archive:

```sh
python3 scripts/research/summarize_sequential_screen.py \
  --source /path/to/approved-archive \
  --output /path/to/existing-report-directory \
  --rss-unit bytes --platform recorded-run-platform \
  --expected-index-sha256 RECORDED_INDEX_SHA256 \
  --report-bundle-v2
```

The final file is `report-bundle-v2.json`. Repeating the same command revalidates
the source and confirms matching bytes before returning success. Conflicting
content requires a separately chosen directory; no overwrite option is supplied.
Use the original run host's RSS unit and platform, as before. Do not point this
at a sealed holdout or rerun a financial experiment to test publication.

Synthetic-only checks:

```sh
PYTHONPATH=scripts/formal:scripts/research python3 -m unittest test_integrity.ReportBundleTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

The existing pinned Python 3.13.3 / Z3 4.15.4 / GHC 9.4.8 toolchain applies.
No new dependency or proof tool is introduced. Filesystem tests need local POSIX
hard-link, no-follow, nonblocking-open and directory-fsync support. Failure of
those primitives rejects publication; no fallback weakens the durability contract.

## Decision and remaining obligations

Recovery 37 moves from open to **partially verified**, not closed. Counts are
**11 scoped closures / 25 partial / 2 open**, with 27 unresolved overall.
All 38 original titles, scopes and closure criteria remain unchanged. Training-run
archive checkpoints, inherited bot snapshot persistence, ownership and broader
recovery/progress still need implementation-level verification. Original numeric
obligation 10 and no-permanent-lockout obligation 38 remain open.

No candidate passed; continue offline research. No new OOS, final-holdout,
transaction-cost, stress, drawdown, tail-risk, DSR/PBO/SPA, OPE, seed or inference
result is claimed. Existing 108 fits, 19,440 replays and 108 invalid OPE batches
remain contaminated development evidence. The 1,227-return holdout stays sealed;
the prospective embargo remains 2027-01-20T13:00Z. Champion, live flags, fleet,
leverage, risk limits, ownership and production deployment remain unchanged.

Targeted local checks passed: 10 tests in 2.593 seconds. Complete local integrity checks passed: 265 tests in 237.263 seconds.
The strengthened reconciliation fixture passed the subsequent 10-test recheck in
4.024 seconds. Local proof reproduction passed in 177.784 seconds.
Pinned formal/full reproduction remains pending.


Publication benchmark: seven synthetic 1 MiB ASCII reports (7,340,338 encoded
bytes), pinned CPython 3.13.3 on shared macOS x86_64/local filesystem. Three fresh
publications took 0.471/0.511/0.621 seconds; matching retries took
0.152/0.134/0.133 seconds and retained the target inode. Peak process RSS was
71,487,488 bytes, including interpreter and input buffers. This is an engineering
measurement, not a worst-case SLA, financial metric or hardware durability test.


Filesystem assumptions are grounded in the primary interface documentation:
[Python os](https://docs.python.org/3.13/library/os.html),
[Linux link(2)](https://man7.org/linux/man-pages/man2/link.2.html), and
[Linux fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html), accessed
2026-10-06. The link interface refuses an existing destination; directory
persistence requires a separate directory fsync. These interface contracts
motivate A-BUNDLE-FS; they do not establish hardware durability on every host.
The Linux documentation specifically warns about ambiguous NFS outcomes, which
are outside this local-filesystem contract. Runtime execution remains pinned to
CPython 3.13.3; the rolling Python 3.13 documentation is a reference, not a proof
of the pinned interpreter.
