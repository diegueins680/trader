# Trader Monitoring

## Active Monitoring (set up 2026-05-14)

### LaunchAgents
- `ai.openclaw.trader.api` — Runs `trader-hs --serve` with `KeepAlive` (macOS auto-restart)
- `ai.openclaw.trader.monitor` — Health-check every 120s, restarts if API port is down
  - Install: `scripts/trader-monitor.sh install`
  - Uninstall: `scripts/trader-monitor.sh uninstall`
  - Status: `scripts/trader-monitor.sh status`

### OpenClaw Cron
- `trader-health-check` — Runs every 5 minutes, checks health and alerts on Telegram if problems found

### Logs
- `/tmp/trader-api-launchd.log` — LaunchAgent stdout/stderr
- `/tmp/trader-monitor/diagnostics.log` — Monitor health-check history
- `/tmp/trader-monitor/restarts.log` — Restart events with diagnostics

### Continuous Improvement auto-loop hook
After a green CI push to `main`, `scripts/autoloop.mjs` calls
`scripts/restart-local-stack.sh`, which rewrites `haskell/.build-commit` to
`git rev-parse HEAD` (the SHA `/health` reports) and `launchctl kickstart -k`s
both `ai.openclaw.trader.api` and `ai.openclaw.trader.web`, so the locally
running stack always matches the latest green commit. Best-effort; opt out
with `AUTOLOOP_SKIP_LOCAL_REFRESH=1`.

### Manual Commands
```bash
# Restart API + Web to match git HEAD (used by the auto-loop)
scripts/restart-local-stack.sh

# Restart API only
launchctl kickstart -k gui/$(id -u)/ai.openclaw.trader.api

# Install / manage monitor
scripts/trader-monitor.sh install
scripts/trader-monitor.sh status
scripts/trader-monitor.sh uninstall

# Check status
launchctl list | grep ai.openclaw.trader

# View recent logs
tail -f /tmp/trader-api-launchd.log
tail -f /tmp/trader-monitor/diagnostics.log
tail -f /tmp/trader-monitor/restarts.log
```

### Why `TRADER_API_RESTART_ON_EXIT=0`
The LaunchAgent's `KeepAlive` handles restarts more cleanly than the shell loop.
When the API exits, launchd restarts the entire service with throttling.
