# Reproduction and verification

Use the [data/license manifest](data-source-license-manifest.json) and
[experiment manifest](experiment-manifest.json). The experiment used Python
3.13.3, NumPy 2.3.5, pandas 2.3.3 on macOS Intel. No GPU or exchange credentials
are needed. The Haskell/Node toolchains remain the root repository pins.

```sh
python3 -m venv /private/tmp/trader-sequential-venv
/private/tmp/trader-sequential-venv/bin/pip install -r scripts/research/sequential-requirements.txt
python3 -m unittest discover -s test -p sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```

Use the venv Python for the test command if dependencies are not already installed
in the active Python. CI explicitly installs the two top-level pinned versions;
transitive wheel/BLAS hashes are not locked, so this is not a bit-identical
cross-platform supply-chain image. No deployment dependency was introduced.

To repeat financial mechanics, use a separate clean checkout of frozen commit
`177936552e358b0736442a463703bb1fba60884b` (registration is already in its history).
Obtain only the previously exported development files; do not inspect full
snapshots or rerun their protected-data generator. Paths below are placeholders
for those existing authorized exports, not downloadable endpoints.

```sh
python3 scripts/research/run_sequential_screen.py \
  --run-registered-development-v1 \
  --panel /path/to/registered-development-panel.csv \
  --settlements /path/to/registered-development-settlements.csv \
  --output /private/tmp/trader-sequential-new-run
```

Both hashes must match before parsing. Output must not already exist. The runner
checks its four source modules and registration against its recorded Git commit;
uncommitted experimental changes are rejected. Return paths and policy parameters
stay outside Git. The original output is `/Users/diegosaa/GitHub/trader/.tmp/research/sequential-control-screen-v1`
in ignored local research storage (about 1.1 GiB); it is not a public download
or off-host backup. Loss of this archive requires rerunning from the exact allowed
inputs. Policy/economic determinism is tested within a fixed runtime, but runtime
timing, timestamps, elapsed durations and BLAS differences prevent promising
whole-run output hash identity. Two original timeout failures are retained even
if a less-loaded repeat does not time out. A rerun is not a new independent trial.

The current branch hardens artifact types and bounded reads, and closes every
planned replay as failed after a training exception. These post-run fixes do not
rewrite the original source identity or financial evidence. For
small review artifacts, run the exporter from the current branch on the completed
archive. It verifies every indexed external file, including raw replay paths,
before exporting any evidence and requires a new output directory:

```sh
python3 scripts/research/summarize_sequential_screen.py \
  --source /Users/diegosaa/GitHub/trader/.tmp/research/sequential-control-screen-v1 \
  --output /private/tmp/trader-sequential-review-copy \
  --rss-unit bytes \
  --platform macOS-14.7.7-x86_64-i386-64bit-Mach-O \
  --expected-index-sha256 764fd123a1570c6b31ecc7e0729ef5dcc1fe19a39c29aee48efc1614c289974b
```

Linux `ru_maxrss` needs `--rss-unit kib` and the original run's platform label;
never infer units from the export host. The committed compact result files are
exact exports of the original archive; latency and memory are host observations,
not acceptance guarantees. No return path, downloaded dataset, paper PDF,
credential or policy parameter file is committed.

The archive index binds the full per-path economic metrics. Committed CSVs retain
all terminal registry events, every algorithm/horizon/seed/stress summary and
baseline-cost per-symbol/fold outcomes. Metrics not implemented (including
holding-period distribution, calibrated Q overestimation, regime/liquidity slices,
full forecast reliability diagnostics and matched champion statistics) remain
unavailable; the registration's requested list is not a claim they were produced.
See [deliverables-index.md](deliverables-index.md) for the exact scope limits.

The current runner also closes a CSV replacement race: it reads each admitted
input into immutable bytes once, verifies both hashes before parsing, and parses
those same byte buffers. All later data hashes in manifests/policy provenance
come from that successful admission, not a reopened pathname. The frozen original
source predates this hardening; its evidence remains unchanged. See
[input-snapshot-audit.md](input-snapshot-audit.md) for synthetic regression evidence.

The compact exporter now captures and verifies the index and seven report-input
files as byte snapshots, parsing those same bytes and retaining the admitted index
hash. Other indexed files, including the 1.14 GB return CSV, are still hash-checked
with 1 MiB streaming reads. Archive replacement after verification cannot alter
report inputs or relabel their provenance. The command above reproduces all seven
committed reports byte-for-byte; see [export-snapshot-audit.md](export-snapshot-audit.md).

Export also reconciles the supplied roster, terminal ledger, fit/replay rows,
successful-fit artifact identities, OPE fit coverage and descriptive group metrics
before creating output. Internally contradictory archives are rejected even with
matching hashes. See [registry-reconciliation-audit.md](registry-reconciliation-audit.md)
for compatibility, rounding tolerance and the distinction from independent
registration completeness or per-bar economic reconstruction.

Terminal-ledger write/flush failures and return-path publication failures now abort
the runner before final summary/index creation. Do not infer run completion from a
terminal line alone or manually index an interrupted directory. Preserve partial
files for investigation and use a new output directory for an explicitly authorized
repeat; there is no automatic resume or recovery. Computational trial failures still
remain in the registry. See the [publication-boundary audit](publication-boundary-audit.md).

The exporter also rejects duplicate object keys (including escaped-key collisions)
and non-finite numeric tokens before reconciliation, even in fields compact reports
omit. This covers the admitted index, every report snapshot and each JSONL event.
Do not rewrite ambiguous archives automatically to obtain acceptance; investigate
the source. Valid legacy report bytes are unchanged. See the
[JSON-admission audit](evidence-json-admission-audit.md).

Choose a new report directory outside the source archive. Exporting below the
archive would add unindexed files and invalidate its inventory, so same/descendant
destinations now fail before evidence reads or directory creation. The exporter
resolves and pins source/output aliases at admission and keeps existing external
destinations exclusive. See the [destination-isolation audit](export-destination-audit.md).

Resource metadata is validated before publication: supplied training durations
must be finite nonnegative numbers and agree with terminal-ledger values when both
exist; supplied artifact byte counts must be positive integers. Both CLI and Python
export require `bytes` or `kib` RSS units. Legacy missing fields remain optional;
do not interpret absent measurements as evidence of zero cost. See the
[resource-evidence audit](resource-evidence-audit.md).

Offline baselines now reject invalid observations and unknown names before rule
computation or random sampling. Absence is rejected by the replay shield; it is
not an executable cash/flatten instruction. Valid actions and forecasts retain
their existing semantics. See the [baseline-admission audit](baseline-admission-audit.md).

OPE requires unmasked trajectory/value inputs. The estimator rejects masked arrays
before conversion can discard missingness, including all-false masks, consistently
with the other research boundaries. Do not fill missing observations or drop
trajectories to force admission. Runner/export records retain explicit failures.
See the [OPE mask-admission audit](ope-mask-admission-audit.md).

Baseline numerical admission validates the fitted parameters needed by the selected
rule and rejects non-finite intermediate scores before clipping or action selection.
Failures return absence, not a zero-position request. Valid formulas, finite-logit
clipping and RNG semantics remain unchanged. See the
[baseline-numerics audit](baseline-numerics-audit.md).

Short OPE policy calls require finite real unmasked observations and three-value
outputs. Malformed evidence aborts before the affected transition, on both logged
and direct paths. Do not impute values or discard episodes to obtain an estimate.
See the [OPE policy-admission audit](ope-policy-admission-audit.md).

Short OPE admits complete windows and aligned price/funding symbol sets before
sampling. Use genuine integer horizons, boundaries, seeds and positive episode
budgets. Arrays must cover the entire requested exclusive stop; admission does not
scan future values. See the [OPE window-admission audit](ope-window-admission-audit.md).

Call economic reporting only after replay stops. A successful report requires
complete terminal accounting state; a failed stopped path keeps its failure and
recorded losses. Unfinished paths cannot be labeled complete. See the
[economic-completion audit](economic-completion-audit.md).

Economic reporting also validates contiguous ledger rows and reconciles every bar
return with its equity change and every equity change with P&L and costs. See the
[ledger-admission audit](ledger-admission-audit.md); failed paths are retained when
their recorded ledger is internally consistent.
