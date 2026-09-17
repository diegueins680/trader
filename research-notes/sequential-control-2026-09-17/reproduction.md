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
