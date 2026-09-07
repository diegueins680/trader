# Autoloop ownership and cycle-sequence mitigation

Date: 2026-09-06 local / 2026-09-07 UTC

Risk IDs: `AUTOLOOP-SINGLETON-001`, `AUTOLOOP-RESET-2026-05-30`

Disposition: mitigated in code; post-merge operational witnesses required

## Finding

The prior runner checked an existing plain-text PID and then overwrote the PID
file in separate operations. Two simultaneous launchers could both pass that
check before either write. A second process rejected later by the Node check
could also overwrite shared status through the top-level error handler.

Cycle continuity resumed from completed metrics only. If the supervisor died
after a cycle began but before its metric line was appended, the next process
could reuse that in-flight cycle identifier. The existing May 31 repair
prevented a reset after completed work but did not close this crash gap.

## Ownership contract

The runner now creates a schema-1 JSON `runner.pid` with filesystem-exclusive
`wx` semantics and mode `0600` before clearing the stop file or writing shared
status. The record contains a positive safe-integer PID, a random owner token,
and acquisition time.

- Exactly one contender can create the record.
- A live PID, including an `EPERM` existence result, blocks acquisition.
- A recent empty record is treated as an acquisition in progress and blocks
  takeover. An old empty record may be recovered after the bounded grace period.
- A malformed non-empty record always fails closed for operator review; age does
  not make corrupted ownership safe to reclaim.
- A record with a proven-dead valid owner is atomically renamed to a unique
  `.stale-*` audit path before acquisition retries.
- Release requires the same PID and token. A different process cannot unlink
  the owner record merely because it knows the PID.
- Failure before ownership does not update the active runner's shared status or
  runner log.
- The shell start/stop/status surface accepts both the new JSON record and the
  legacy plain integer during migration.

The concurrency regression starts eight acquisitions together and requires
exactly one winner. It also covers wrong-token release, legacy dead-owner
quarantine, recent incomplete-owner rejection, old empty-owner recovery, and
permanent fail-closed handling of malformed non-empty ownership.

## Cycle identity contract

The runner reads the maximum safe integer witnessed by:

1. completed metrics NDJSON;
2. the prior runner status;
3. the current bounded-cycle `runId`, including incomplete work; and
4. the persistent schema-1 `cycle-sequence.json` record.

Before clearing current-cycle status or starting a child, it atomically writes
the next identifier to the sequence file. A crash after that write may skip an
identifier, but no restart may reuse it. Malformed sequence state and safe-
integer exhaustion fail closed before bounded work starts. Malformed historical
metric lines are ignored only when other valid witnesses preserve the maximum;
they cannot lower it.

The deterministic regression fixes completed metric 41, status 42, incomplete
cycle 43, and persistent reservation 44, then requires the next record to be
45. Fractional and unsafe metric counts do not become identity evidence.

## Verification and remaining gate

The implementation is covered by the root automation suite and the canonical
formal registry. These code-level witnesses justify moving both risks from
open to mitigated, not closed.

Closure requires loading the merged code in the launchd-supervised runtime and
recording both:

- a rejected concurrent launch that leaves the active owner/status unchanged;
  and
- a supervisor restart whose first reservation is strictly greater than the
  last ID issued before restart, including a valid schema-1 sequence record.

No predictor, feature, backtest, strategy, position, order, exchange,
credential, deployment, champion, holdout, or live-authorization behavior is
changed by this automation boundary.
