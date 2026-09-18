# Registry reconciliation audit — 2026-09-17

This audit starts from merged main `5e502721b2331a2c3081b044513e5a734066bf96`.
It checks internal evidence consistency without running a financial experiment,
reading market inputs, opening holdouts or changing production behavior.

## Defect and scope

The prior exporter checked indexed file hashes and compared the set of planned
IDs with terminal-event IDs. It did not compare those events with training or
evaluation rows, reject duplicate planned IDs, or validate supplied summary
counts and descriptive metrics. Thus a correctly hashed archive could claim
complete registry coverage while omitting a failed replay from its reports.

Sixteen synthetic mutations reproduced this gap: duplicate planned IDs;
missing/duplicate replay or training rows; changed seed metadata; contradictory
status, reason or observation count; an event after termination; missing or
duplicate summary groups; incorrect total/failure counts; invented mean return;
and an OPE row attributed to a failed fit. All sixteen were accepted before the
repair. The before-repair suite ran 34 tests with 16 failing subtests.

An additional schema fixture rejects a non-text terminal reason before output
creation, so malformed reason metadata cannot fail later while writing CSVs.

## Reconciliation contract

A small standard-library module validates admitted snapshots before the exporter
creates its output directory:

- Roster IDs are unique, nonempty and explicitly typed as training or replay.
  Events reference that roster, have a valid lifecycle, and contain exactly one
  terminal outcome per planned ID. Every training attempt and evaluated replay
  requires a start event; only explicit failed-fit cascades may omit one.
- Training and replay rows correspond one-to-one to their roster entries. Their
  algorithm, horizon, fold, seed, stress and symbol agree with their serialized ID.
- Training outcomes agree with terminal events. Successful fits bind the same
  artifact digest in training metadata, the ledger and the verified file index.
  Original v1 successful rows omit `status`; this remains supported with those
  independent completion witnesses.
- Replay status, reason and observation count agree with terminal events.
  Failed fits have only unevaluated `training_failed` replay outcomes. Completed
  replays require observations; metric coverage agrees with observation presence.
  Evaluated RL rows also require finite nonnegative latency p99 and an OOD rate
  in `[0,1]` before output creation. Validation and reporting share the RL family set.
- OPE has exactly one row per successful fit, and none for a failed fit. This
  validates attribution and coverage, not the estimator's statistical validity.
- Total and per-group path, completion and failure counts reconcile exactly.
  Failure rate, mean/worst terminal-or-stopped return, worst drawdown and ES95,
  median available Sharpe, mean fees and mean funding are independently calculated
  from the corresponding replay summaries. Finite numeric comparisons allow only
  `1e-12` absolute/relative rounding tolerance; undefined quantities stay absent.

The checker validates rather than overwrites the archived summaries, preserving
their original bytes. Failed and stopped paths remain included. This does not
repair unequal endpoints or turn their averages into valid superiority tests.

## Verification and compatibility

The deterministic suite passes 36 tests, including all contradictory-archive
fixtures, a hand-calculated mixed complete/failed outcome example, successful-fit
export with original v1 metadata, and rejection of missing successful-fit OPE.
The fixtures contain no actual market observations or live order interface.

Review follow-up fixtures reproduced two missing-start acceptances and two missing
report-metric `KeyError`s before the follow-up fix (36 tests: two failures, two
errors). They now reject before output creation; the successful-fit fixture uses
a valid, untrained temporary policy artifact and proves the positive export path.

A further fixture reproduced a nested `result.seed` overriding the validated
outer seed in CSV output. Result objects now reject reserved trial-identity fields,
so the verified ID metadata cannot be shadowed during report construction.

The existing results archive was re-exported using the command in
[reproduction.md](reproduction.md). Its index SHA-256 remains
`764fd123a1570c6b31ecc7e0729ef5dcc1fe19a39c29aee48efc1614c289974b`.
All 108 fits, 19,440 replay paths and 19,548 planned entries reconcile, and all
seven compact output files match the committed reports byte-for-byte. The archive
still contains all registered seeds 11, 23 and 47, and all 108 OPE batches remain
invalid. No fresh evidence or statistical significance is inferred.

One local macOS `/usr/bin/time -l` export measured 16.12 seconds wall time,
8.91 seconds user CPU, 2.22 seconds system CPU and 197,279,744 bytes peak RSS
(188.14 MiB). This is a single observation with uncontrolled cache/load conditions,
not a comparative performance claim or production inference benchmark. Large
return CSVs remain stream-verified; report snapshots and parsed records require
memory proportional to report size.

Run `python3 -m unittest discover -s test -p sequential_screen_test.py`,
`bash scripts/verify.sh automation` and `bash scripts/verify.sh full` from the root.
The associated PR records the final full-wrapper result, log hash and remote CI
checks. Formal contracts `A-SEQUENTIAL-RESEARCH-R6/E5` map to these tests. Typed
Haskell and Markdown risk mitigations are updated together; canonical
`RL-OFFLINE-001` remains HIGH/OPEN.

## Additional disposition review

Further review identified status/reason contradictions that could classify failed
replays as complete, and summary fields that could contradict the non-authorizing
manifest. Eleven hash-valid mutations reproduced false acceptance before repair.
Failures now require nonblank, non-reserved reasons; completed terminal/replay
outcomes require no failure reason. Manifest flags must be boolean false, and the
summary must retain the v1 contaminated-development class, no-candidate decision,
false promotion/holdout flags and null DSR/PBO/SPA/paired-confidence fields. Extra
statistical claims and summary live authorization are rejected. The original
producer's successful and failed paths remain supported.

The in-progress full wrapper encountered the newly added negative fixtures before
the repair was applied and failed its Python contract suite. This is not a passing
verification result. The associated PR records a fresh full run on frozen code,
its final result, log hash and CI evidence before merge.

## Report-preparation review

A later review found that the unconditionally consumed peak-memory field could
fail after partial output creation. Missing, nonnumeric, nonfinite and negative
RSS fixtures plus malformed training-episode and training-seconds fixtures
reproduced two assertion failures and four exceptions before repair. RSS now
requires a finite nonnegative number, and all compact reports are rendered before
creating the output directory. This prevents field and serialization failures
from leaving partial reports without accumulating the large return-path archive.
Filesystem write failures are still explicit and may require cleanup; publication
is not claimed to be atomic against disk failure.

Full verification and CI passed at predecessor `be0ff9f1` before this later
repair. The PR records a new full run and CI on the final commit; predecessor
passes do not substitute for final verification.

## Producer failure compatibility

Review of the exception handler found that an empty exception message could emit
a blank failed-replay reason. The strict exporter would correctly reject the
invalid record but could not report the runner's failed attempt. Training, OPE
and replay exception records now include the exception class and optional trimmed
message. Synthetic producer-to-export fixtures inject empty, whitespace and
reserved `complete` messages into training and replay/OPE paths. Before repair,
the six subcases produced three assertion failures and three export errors; after
repair, all failed paths remain present and none is counted as complete. No real
market input is read, policy trained or archived financial trial rerun by these
fixtures. Final verification is recorded in the PR on the updated commit.

## Training-reason consistency

Four additional hash-valid fixtures reproduced accepted missing, blank or
contradictory failed-fit reasons and a reason attached to a completed fit. Training
rows now obey the same outcome semantics and match their terminal-event reasons.
Status-omitted successful v1 fits remain supported. Older failed-fit records with
conflicting labels require investigation; the exporter does not rewrite archives
to make them consistent. The original archive contains successful fits and remains
compatible, while synthetic failed-producer fixtures exercise the current path.

## Limits and decision

This is consistency against the **supplied roster**, not proof that the roster
matches an independent preregistration. Coordinated rewriting of all evidence
still requires an independently trusted expected index and registration review.
The checker does not reconstruct individual returns from `returns.csv`, validate
all economic metrics, establish behavior-policy support, make OPE reliable, or
satisfy missing statistical and independent-confirmation gates.

No candidate passed. The current champion, sealed holdouts, original source
identity and economic conclusions remain unchanged. No deployment, promotion or
live-money exploration is introduced.
