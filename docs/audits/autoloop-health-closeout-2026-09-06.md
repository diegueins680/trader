# Autoloop health closeout

Date: 2026-09-06 local / 2026-09-07 UTC

Risk ID: `AUTOLOOP-DOWN-003`

Disposition: closed with a fresh supervisor and bounded-cycle witness

## Decision

The persistent autoloop is supervised, heartbeating, synchronized to the exact
merged `main` commit, and completed a clean bounded cycle after operator review
of its saved recovery patch. The stale down report is replaced with current
evidence, so the permanent risk ID is closed.

This decision does not close `AUTOLOOP-RESET-2026-05-30` or
`AUTOLOOP-SINGLETON-001`. Completed-cycle metrics do not yet reserve a cycle ID
before work starts, and the PID-file ownership check is not yet an atomic lock.

## Recovery review

Cycle 2750 saved recovery commit
`5f919f126063d19a3b9a672fa7e5ceda218f5e55`. Review found a real pandas/Haskell
timestamp boundary mismatch, but the saved patch had a Haskell parse error and
converted timestamps through `Double`, which could round a fractional value
into an integer. It was not merged.

The safe replacement was independently implemented, passed local full
verification, passed all hosted PR checks, and merged through PR #226 as commit
`205591000d06ff368d0219ec71260d1c6042f7ca`. The recovery block was then moved
to the audit-preserving filename
`recovery-block.reviewed-pr-226-20559100-2026-09-07.json`; it was not discarded
before review.

## Fresh operational witness

At 2026-09-07T01:10:34.789Z, the runtime status recorded:

- launchd service `gui/501/ai.openclaw.trader.autoloop.forever` in `running`
  state with PID 79272;
- runner state `sleeping`, a current 15-second heartbeat, and no block reason;
- runtime `main` and `origin/main` both at merged commit `20559100` with a clean
  worktree;
- branch reconciliation outcome `rebased`, with no candidate, conflict, or
  recovery branch promoted;
- cycle 2751 completed with exit code 0 and no recovery snapshot;
- the bounded cycle safely reported `skipped_pending_ci` because the post-merge
  `main` workflow was still running, proving the CI promotion gate remained
  fail-closed rather than performing speculative work.

The evidence was obtained with read-only invocations of:

```sh
launchctl print gui/501/ai.openclaw.trader.autoloop.forever
jq '{pid,state,updatedAt,heartbeatAt,cycleCount,nextRunAt,blockReason,lastCycle}' \
  /Users/diegosaa/.openclaw/orgs/trader-firm/runtime/trader-autoloop-live/.tmp/autoloop/status.json
git -C /Users/diegosaa/.openclaw/orgs/trader-firm/runtime/trader-autoloop-live status --short --branch
git -C /Users/diegosaa/.openclaw/orgs/trader-firm/runtime/trader-autoloop-live rev-parse HEAD origin/main
```

## Status-probe correction

The audit initially saw `scripts/autoloop-forever.sh status` label the same PID
dead from a restricted process context even while launchd and the 15-second
heartbeat proved it alive. The cause was treating every `os.kill(pid, 0)`
`OSError` as absence. Under POSIX, `EPERM` means the process exists but the
caller lacks permission to signal it.

The shared start/status PID-existence probe now treats `ESRCH` as absent and
`EPERM` as existing. A regression invokes the status command against both the
live test-process PID and a deliberately absent PID, while binding the explicit
permission-denied branch. This prevents restricted monitoring from fabricating
a down state or using that false state to justify a second start.

## Scope and residual risk

No predictor, feature, backtest, strategy, position, order, exchange, credential,
deployment, champion, or holdout behavior changed. This operational repair does
not authorize live trading or automatic research promotion.

The runner still needs an atomic, stale-owner-recoverable workspace lock and a
durably reserved monotone cycle sequence before the two adjacent open risks can
be closed. A future loss of heartbeat or supervisor ownership should reopen this
permanent risk ID with the new failure evidence.
