## Unreleased
- Autoloop: stop blocking the forever runner on merged local branches that are still attached to Git worktrees, and add `hlint` verification plus direct log-driven HLint auto-fixes for editable Haskell files.
- Autoloop/CI: require `AUTOLOOP_PUSH_TOKEN` in the GitHub-hosted autoloop workflow and remove the `github.token` fallback plus `AUTOLOOP_SKIP_CI_WAIT`, so post-push GitHub Actions polling is mandatory there.
- Autoloop/CI: install a pinned prebuilt `fourmolu` binary in the GitHub-hosted autoloop workflow instead of compiling `fourmolu` from source on each run.
- Trading: add a fee-aware fail-closed entry gate for marginal signals, so entries now require the method-selected absolute edge to clear both the existing adjusted headroom requirement and an explicit round-trip fee floor (`2 * fee`); non-finite fee or edge inputs are blocked conservatively.
- Trading/Formal: document the fee-aware entry-gate contract in `FORMAL_METHODS.md` and add bounded Haskell tests covering fee monotonicity, edge monotonicity, and malformed-input fail-closed behavior.
- Trading: derive low-directionality efficiency and z-score from the additive per-bar return path, so clean monotonic trend windows stay directional instead of being misclassified as malformed when the compounded endpoint return slightly exceeds the summed simple-return path.
... 
