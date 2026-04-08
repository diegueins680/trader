# Trader Firm Mission

## Standing mission lane

### Lane: `trader-firm-cto.repo-autoloop-forever`

Keep the trader repo's repo-native autoloop running as a permanent engineering lane on this machine using only the contract documented in `README.md`, `scripts/autoloop-forever.sh`, and `scripts/autoloop-forever.mjs`.

## Success conditions

The lane is considered healthy only when all of the following are true:
- the forever supervisor process is alive and reporting fresh status heartbeats
- the supervisor is service-managed across shell exit / login via the repo-native LaunchAgent path
- bounded cycles are not left parked on recoverable repo-hygiene issues such as dirty worktrees, missing planner backend, or stale PID state
- the repo contract remains verified by `npm run test:autoloop`
- operator instructions for install, restart, stop, and status remain documented in the repo

## Non-negotiable contract

- Use the repo-native runner and supervisor artifacts; do not replace them with ad hoc shell loops.
- Treat `scripts/install-autoloop-launchagent.sh` as the canonical macOS keepalive path.
- Treat a dirty worktree as something the forever runner should first preserve onto a dedicated rescue/checkpoint branch before declaring the lane blocked.
- Keep the lane visible in CTO objective/report files so each management pass can answer: is it live, is it healthy, what is blocking it, and who owns the unblock.
