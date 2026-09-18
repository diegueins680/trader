# Resource evidence audit — 2026-09-18

Baseline: merged main `a45e442fb46424c213170ff4ffffeed916780210`.
Synthetic engineering fixtures only; no market archive, policy training experiment,
protected holdout or production action is used.

## Reproduced defect

The exporter validated peak memory and inference latency, but used training
resource metadata without equivalent domain checks. Hash-valid archives could
publish negative or boolean training durations and zero, negative, fractional,
boolean, string or null artifact sizes. A fit and its terminal ledger could also
report different durations. Such figures cannot support resource feasibility.

The new regression exercises 15 malformed or contradictory cases with both failed
and successful synthetic fits. On baseline, 26 of 30 cases exported instead of
rejecting; four already failed through rendering. The unit-admission regression
also failed all six cases: the Python API read evidence before rejecting invalid
units and treated all values other than `bytes` as KiB. The CLI already restricted
its choices. These probes do not demonstrate corruption of historical reports.

## Repair and compatibility

Reconciliation checks every supplied fit/terminal-event `seconds` value using the
existing finite-number validator and a nonnegative bound. Booleans are rejected.
When both records contain a duration, exact equality is required: the runner
writes the same value to both records, so no rounding tolerance is necessary.
A supplied `artifactBytes` must be a positive JSON integer, excluding booleans.
These checks apply to failed fits as well as completed fits.

Export admits only `bytes` and `kib` RSS units before evidence reads. Valid units
retain the existing MiB conversion and the explicit run-host convention. No new
configuration, dependency, artifact schema, model identifier or production path
is added. Legacy omitted fields retain their existing handling; missing timing
must not be interpreted as measured zero cost. All reports still render before
output creation. Filesystem failure behavior is unchanged.

This checks metadata domains and cross-record agreement. It does not establish
that durations were measured correctly, that optional measurements are complete,
or that a positive byte count equals the artifact's actual size. Hash admission
and these checks do not authenticate benchmark claims or establish CPU feasibility.

## Executable evidence

Three integration methods cover:

- Thirty invalid metadata cases across failed/completed fits, each rejected before
  output-directory creation.
- Six unsupported API units, rejected before any evidence read.
- Zero and fractional-second durations, positive artifact sizes and both RSS unit
  conversions, with expected report values.

The suite contains 78 tests. Its existing exact legacy-report regression retains
sorted filename/content SHA-256
`d9cdec2fa64d52f3f12d4d41ec1a8d9a3c3c7aca902dddc9b30f021d5c89661b`.
No generated report or artifact is committed.

```bash
python3 test/sequential_screen_test.py
python3 -m unittest discover -s test -p sequential_screen_test.py -k resource
python3 -m unittest discover -s test -p sequential_screen_test.py -k unknown_rss
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The PR records actual verification results and final-head CI. The executable
contract is `A-SEQUENTIAL-RESEARCH-R16`; Haskell/Markdown risk mitigations remain
synchronized. `RL-OFFLINE-001` remains HIGH/OPEN in the canonical risk register.

## Decision

No financial experiment, seed, statistical inference, OPE/support finding, cost
assumption or historical result changes. No candidate passed. Preserve the current
champion, protected holdouts, production settings and deployed release. Continue
offline research without adoption. No live exploration or order authority is added.
