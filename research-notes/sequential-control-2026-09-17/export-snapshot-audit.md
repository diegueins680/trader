# Evidence-export snapshot audit — 2026-09-17

This follow-up to PRs #256 and #257 repairs evidence identity in the offline
report exporter. It adds no candidate, financial trial, data collection, holdout
access or production integration. The base is main
`47b42cb205fb6f000c60d3177f0b2e6ed090d41c`.

## Reproduced defects

The exporter verified the index with a streaming hash, reopened it to parse,
verified indexed paths, then reopened report files to parse them. It also
recomputed the index hash when writing provenance. Replacement of a path could
therefore separate verification from consumption or relabel verified evidence.

Two synthetic regressions reproduced the defects before the repair:

- Replacement of the seven report inputs after their hash checks reached the
  parser, raising `JSONDecodeError` on unverified replacement content.
- Replacement of the index after parsing changed the exported provenance digest
  from the admitted index hash to the replacement file's hash.

The before-repair suite ran 33 tests: one failure and one error. A third regression
confirms that tampered stream-verified return paths fail before output creation.
The fixtures contain a single failed training entry, no market observations and
no policy parameters. They neither train nor evaluate a trading policy.

## Repair and compatibility

The index is read once, hash-verified and parsed from those same bytes. Each of
the seven report inputs is retained as an immutable byte snapshot only after its
digest matches the verified index. JSON and event parsing consumes those buffers;
the exported index identity is the proven caller-supplied digest. Buffers are
released as their inputs are parsed.

All indexed files are still checked before output creation. Policy artifacts and
the large return CSV are verified in 1 MiB streaming chunks without retaining
their contents. Replacing report paths after successful verification cannot alter
this invocation's report; a later invocation rejects changed bytes. This does not
lock paths or promise that the external archive stays immutable after export.

CLI arguments, report schemas, original evidence, registrations, policy semantics
and promotion rules are unchanged. No new dependency is needed. The canonical
`RL-OFFLINE-001` remains HIGH/OPEN; its mitigation now explicitly includes verified
evidence snapshots. Formal contracts `A-SEQUENTIAL-RESEARCH-R5/E4` connect the
repair to executable regression evidence.

## Original-archive reproduction and memory

The reproduction command in [reproduction.md](reproduction.md) was run against
the existing permitted results archive, whose index SHA-256 is
`764fd123a1570c6b31ecc7e0729ef5dcc1fe19a39c29aee48efc1614c289974b`.
Every indexed file was checked, including the 1,138,940,436-byte return CSV. All
seven emitted reports matched the committed reports byte-for-byte:

- `experiment-manifest.json`
- `experiment-registry.csv`
- `all-seed-results.csv`
- `symbol-fold-base-results.csv`
- `multi-seed-training.json`
- `ope-report.json`
- `evaluation-summary.json`

A second export, measured with macOS `/usr/bin/time -l`, took 8.26 seconds wall
time, 6.28 seconds user CPU and 1.40 seconds system CPU, with 185,888,768 bytes
(177.28 MiB) maximum RSS. This is a single local observation, not a production
inference benchmark. Report inputs occupy about 41 MiB before JSON decoding;
arbitrarily large report JSON is not promised constant-memory support.

The 33 deterministic Python tests pass after the repair. Full-wrapper results
are recorded below after verification. The unchanged financial decision is
**no candidate passed**. No independent confirmation or new statistical evidence
is claimed.
