## Unreleased
- Trading: add a fee-aware fail-closed entry gate for marginal signals, so entries now require the method-selected absolute edge to clear both the existing adjusted headroom requirement and an explicit round-trip fee floor (`2 * fee`); non-finite fee or edge inputs are blocked conservatively.
- Trading/Formal: document the fee-aware entry-gate contract in `FORMAL_METHODS.md` and add bounded Haskell tests covering fee monotonicity, edge monotonicity, and malformed-input fail-closed behavior.
- Trading: derive low-directionality efficiency and z-score from the additive per-bar return path, so clean monotonic trend windows stay directional instead of being misclassified as malformed when the compounded endpoint return slightly exceeds the summed simple-return path.
...