# GitHub 502 bounded-retry validation

Date: 2026-09-06 local / 2026-09-07 UTC

Risk: `GITHUB-502-001`

Disposition: fixed and closed

## Historical finding

The original register described GitHub API 502 retries as unbounded. Commit
`5c64c5084baa93c57ec78ba1f69de49edc5902bf` added three retries after the
initial request with exponential 2, 4, and 8 second delays, but the risk stayed
open pending current operational validation. The implementation was embedded
inside the executable and had no direct fault-injection regression.

## Executable evidence

The unchanged production policy now lives in the exported
`runGitHub502WithRetry` helper. `runGhWithRetry` delegates to it. Production
GitHub repository, pull-request review, commit, workflow-run, failed-workflow
log, and check-suite reads all pass through that wrapper; this audit also found
and migrated two newer direct failed-log calls that had bypassed the original
2026 fix.

Deterministic tests inject the failure sequence and replace the blocking wait
with a delay recorder. They prove:

1. `502`, then `Bad Gateway`, then success makes exactly three calls and
   schedules 2 and 4 second waits.
2. A persistent 502 makes exactly four calls, schedules only 2, 4, and 8 second
   waits, then propagates the original failure.
3. A non-502 error makes one call, schedules no wait, and propagates.
4. The production GitHub read call sites remain wired through the tested
   wrapper.

The injected test does not spend wall-clock time sleeping. Production retains
the synchronous waits, so the maximum added delay before a persistent 502 is
14 seconds and retry count cannot grow with outage duration.

## Current operational witness

At `2026-09-07T04:30:36.554Z`, the launchd-supervised checkout reported:

- runner PID `41459`, alive with parent PID 1;
- state `sleeping` with a fresh 15-second heartbeat;
- monotone cycle count `2773`;
- cycle 2773 completed with exit code 0;
- the bounded cycle scanned 20 pull requests and returned no failure context.

That cycle ran from `2026-09-07T04:25:01.269Z` through
`2026-09-07T04:26:16.271Z` on main commit
`575ebb9b3472ac1dbdf86fc98bf6952783bf1952`, which contains the original
bounded policy. Separately, GitHub Actions run
[`34082708070`](https://github.com/diegueins680/trader/actions/runs/34082708070)
was retrieved through the live GitHub API and reported successful automation,
web, and Haskell jobs for the immediately preceding merged change.

The operational witness establishes current reachability and successful use of
the GitHub review path. The deterministic fault injection establishes behavior
during a 502; no real outage was manufactured.

## Scope and residual boundary

This closure is exact to `GITHUB-502-001`: only HTTP 502 or text containing
`Bad Gateway` is retried. Authentication, authorization, validation, and other
GitHub failures still fail immediately. Broader transport or 503/504 handling
would be a separate change and risk assessment.

No credential value was read into the audit or committed. The change does not
modify the autoloop edit allowlist, promotion rules, deployment, market data,
model selection, holdout state, trading configuration, existing positions, or
live-order authorization.

Reproduce from the repository root:

```bash
node --test --test-name-pattern "GitHub|autoloop GitHub reads" test/autoloop.test.mjs
bash scripts/verify.sh automation
bash scripts/verify.sh full
```
