# Export destination isolation audit — 2026-09-18

Baseline: merged main `dc9642eec71068ed447741f8fb3fe9361cf21827`.
Synthetic engineering fixtures only; no market data, training experiment,
protected holdout or production action is used.

## Reproduced defect

The exporter verified its source inventory before creating reports, but allowed
the output directory to be a descendant of that same archive. A synthetic export
to `archive/review` succeeded and created seven unindexed files. A subsequent
export of that archive failed with `external evidence file inventory mismatch`.
The existing source files were not overwritten, but the bound inventory was no
longer valid. This is not evidence that the historical archive was modified.

Six containment cases were accepted on the baseline: a direct child, a new nested
subtree, an aliased source, an aliased destination, both aliases, and a path using
`..` to enter the archive. An equal source/output path already failed with
`FileExistsError`, after reading evidence; it is now rejected at admission.
The initial containment regression reported six assertion failures and one error.

A separate fixture retargeted an initially external destination symlink to the
source after streaming verification. The baseline followed that alias when
publishing, writing inside the archive instead of the original destination.
Its regression failed because the originally intended destination remained absent.

## Repair and contract

At entry, the exporter resolves source and output paths once. It rejects an output
equal to or beneath the resolved source before reading evidence or creating any
directory. All later reads and writes use the admitted resolved paths.

This handles ordinary static aliases and `..` normalization, and retargeting the
original destination alias cannot redirect publication. A legitimate sibling such
as `archive-review`, a normalized path leaving the archive, or an alias to an
external destination remains valid. Existing external output directories still
raise `FileExistsError` and their files are preserved.

The change is four lines in the existing exporter. There is no new configuration,
dependency, model identifier, artifact schema or separate production service.
Report bytes, hash admission, registry reconciliation and large-file streaming are
unchanged. Source/output CLI flags remain the same; overlapping destinations now
fail earlier with an explicit `ValueError`.

Resolved directory ancestors must remain stable. This is path admission and alias
pinning, not directory-inode locking, detection of mount aliases or atomic
multi-file publication. Retargeting an original symlink is covered; replacing a
resolved directory ancestor is not promised safe by this contract. Filesystem
write failures retain the existing explicit failure behavior.

## Executable evidence

Three new integration methods bring the suite to 75 passing tests:

- Same, direct, nested, aliased and normalized containment cases reject before
  evidence reads. Source file bytes and the complete directory inventory remain
  unchanged, including the would-be new parent directories.
- Retargeting the destination alias during streaming verification leaves reports
  at the admitted external destination and leaves the archive unchanged.
- Three permitted destination forms yield identical seven-file reports. Repeating
  each export refuses to overwrite existing output, preserving its bytes.

The existing exact legacy-report fixture still has sorted filename/content
SHA-256 `d9cdec2fa64d52f3f12d4d41ec1a8d9a3c3c7aca902dddc9b30f021d5c89661b`.
Snapshot-replacement, JSON-admission, registry and runner-publication tests also
remain in the suite. No generated output or policy artifact is committed.

From the repository root:

```bash
python3 test/sequential_screen_test.py
python3 -m unittest discover -s test -p sequential_screen_test.py -k export_rejects_destinations
python3 -m unittest discover -s test -p sequential_screen_test.py -k export_pins_admitted
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

To reproduce the baseline behavior, use the new synthetic fixtures with the
baseline exporter in an isolated checkout. No original market archive is needed.
The PR records frozen source, actual full-wrapper result and log hash, and
final-head GitHub CI. `A-SEQUENTIAL-RESEARCH-R15` links the three witnesses. Haskell
and Markdown risk mitigations remain synchronized; `RL-OFFLINE-001` stays HIGH/OPEN.

## Decision

No financial experiment, cost assumption, seed, OPE/support finding, statistical
inference, historical result or no-adoption conclusion changes. No candidate is
trained, integrated or promoted. Preserve the current champion, protected holdouts,
production settings and deployed release. No live exploration or order authority
is introduced. No candidate passed; continue offline research without adoption.
