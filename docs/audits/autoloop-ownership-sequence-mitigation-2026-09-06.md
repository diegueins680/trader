# Autoloop ownership and cycle-sequence mitigation

Date: 2026-09-06 local / 2026-09-07 UTC

Risk IDs: `AUTOLOOP-SINGLETON-001`, `AUTOLOOP-RESET-2026-05-30`

Disposition: closed after code-level and post-merge operational witnesses

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

## Verification and operational closure

The implementation is covered by the root automation suite and the canonical
formal registry. Pull request 228 merged the implementation as commit
`be7b69010f8c3092396efcb9b0685b25075c6398`; its hosted Haskell, web, and
automation checks passed before the operational exercise.

The isolated launchd-supervised checkout then synced cleanly to that exact
commit. Immediately before restart, the legacy process had PID 79272, had
issued `cycle-2761`, and had no sequence file. Its legacy PID-file SHA-256 was
`4d94033e6bf2613051252d0c9efe8da173e3afe993288976dfd66f7397ae9945`. The
bounded child was allowed to finish, and the runner was observed sleeping
before the controlled LaunchAgent kickstart.

At `2026-09-07T02:13:48.822Z`, the replacement process acquired a schema-1
owner as PID 5288 with mode `0600` and a valid private token. The token was
neither printed nor recorded here. At `2026-09-07T02:13:52.374Z`, it atomically
reserved cycle 2762, strictly beyond the pre-restart issued identity. The owner
record SHA-256 was
`a26632e68ce5bfb555ce6ba5497e9bf2b4c7d848ab0e168211b9844328b77bf4`.

While PID 5288 remained live, a direct `node scripts/autoloop-forever.mjs`
contender exited non-zero with `EALREADY` and identified the active PID. After
rejection, the owner digest, owner PID/acquisition time, runner PID/start time,
cycle count 2762, and sequence record were unchanged. The runner remained
healthy with no error or block; only its expected heartbeat advanced.

The restart therefore proves durable monotone identity across the supervised
process boundary, and the collision proves exclusive ownership plus
non-owner status isolation on the merged runtime. These independent
operational witnesses satisfy the recorded closure gates for both risks.

No predictor, feature, backtest, strategy, position, order, exchange,
credential, deployment, champion, holdout, or live-authorization behavior is
changed by this automation boundary.
