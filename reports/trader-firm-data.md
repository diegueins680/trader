## Finished result — 2026-04-10 16:48 America/Guayaquil — `--vol-conf-gate <preset>`
- Proof command ran from `/Users/diegosaa/GitHub/trader/haskell` and returned the requested first-40-line slice.
- Captured output showed no `git status --porcelain=v1` change lines for `app/Trader/App/Args.hs`, `app/Main.hs`, or `test/TestMain.hs`.
- `app/Trader/App/Args.hs:867-872` exposes `--vol-conf-gate` with `parseVolConfGatePreset` and preset choice help.
- `app/Main.hs:1194` emits `vol_conf_gate`; `app/Main.hs:1207` emits `confidence` in runtime state logging.
- No `test/TestMain.hs` match appeared in the proof output, so the callable preset slice shows CLI wiring and observability fields but no visible data-validation seam.
- Next owner/consumer: trader-firm-cto
no-existing-data-validation-seam
FINAL_STATUS: done — reports/trader-firm-data.md proof block for --vol-conf-gate slice
