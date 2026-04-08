# Trader Firm Org

## Ownership map

### Owner / operator
- Diego
- Owns machine-level decisions and persistent service enablement.
- Decides when local worktree changes are ready to commit, stash, park, or discard.

### Trader Firm CTO lane owner
- The repo-resident CTO management loop.
- Owns repo-side contract quality: scripts, docs, tests, status truthfulness, and blocker reporting.
- Must not claim the forever lane is healthy when it is merely running but blocked.

## Permanent engineering lane

### `trader-firm-cto.repo-autoloop-forever`
- Scope: keep the trader repo autoloop continuously runnable and operationally honest.
- Repo artifacts in scope: `README.md`, `scripts/autoloop-forever.sh`, `scripts/autoloop-forever.mjs`, `scripts/install-autoloop-launchagent.sh`, `test/autoloop.test.mjs`.
- Runtime artifacts in scope: `.tmp/autoloop/status.json`, `.tmp/autoloop/current-cycle.json`, `.tmp/autoloop/runner.log`, `~/Library/LaunchAgents/ai.openclaw.trader.autoloop.forever.plist`.

## Blocker ownership rules

- **Repo-side blocker**: CTO lane owns the fix if the contract is unclear, scripts are broken, permissions are wrong, tests are missing, or status lies.
- **Operator-side blocker**: Diego owns the unblock if the service is not installed, credentials/backend are absent, or the worktree is intentionally left dirty.
- **Shared blocker**: report both owners explicitly when the repo is ready but runtime enablement still has not happened.
