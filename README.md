- `--open-threshold 0.002` (or legacy `--threshold`) entry/open direction threshold (fractional deadband)
  - Entries must also clear that deadband with at least `1.5x` edge headroom; forecasts that only barely cross the threshold are held as `HOLD (EDGE_HEADROOM)` instead of opening on a boundary-touching move.
  - Entries are also vetoed when the method-selected absolute edge does not clear the active headroom requirement plus an explicit round-trip fee floor (`2 * fee`), which blocks low-margin trades that only look profitable before fees. Non-finite fee or edge inputs fail closed.
  - Entries are also suppressed when the method-selected absolute edge exceeds `4x` the active open-threshold, which blocks stale/outlier prediction spikes before they turn into exhausted-move entries.
  - Entries are now also vetoed as `NON_DIRECTIONAL_CHOP` when 24-bar price efficiency is `<= 0.25`, and as `NON_DIRECTIONAL_MR` when efficiency stays `<= 0.40` while saved HMM regime probabilities are clearly mean-reversion-dominated (using the existing `--regime-bank-hysteresis` gap). This keeps large raw edges from opening new directional trades in chop or range-drift conditions.
  - Directionality efficiency and z-score are derived from the additive per-bar return path, so clean monotonic trend windows remain directional instead of being misclassified as malformed when the compounded endpoint return slightly exceeds the summed simple-return path.

## Autoloop
- The repo autoloop now treats merged local branches that are still attached to Git worktrees as prune skips instead of blocking the forever runner before any bounded repair cycle starts.
- The bounded autoloop now includes `hlint app test bench` in its safe verification set and can apply direct `Found:` to `Perhaps:` HLint replacements for editable Haskell files when CI fails on simple HLint-only suggestions.
- The GitHub-hosted autoloop now requires `AUTOLOOP_PUSH_TOKEN`; it no longer falls back to `github.token`, and it no longer skips post-push CI waiting. That keeps Actions polling and failed-log ingestion mandatory after each autoloop push.
- The GitHub-hosted autoloop now installs a pinned prebuilt `fourmolu` release binary instead of compiling `fourmolu` from source on every run, which removes a slow bootstrap step before the bounded loop can start.
