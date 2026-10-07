# Bounded values closure (obligation 7) — 2026-10-06

**Obligation 7 closes for the delivered offline path under an explicit
assumption.** Counts move to **14 scoped closures / 23 partial / 1 open**. This
is a read-only audit: no source, registration, archived result or data changed.

## Reading and assumption

The criterion allows "explicit gap/solvency assumptions". The obligation's own
recorded next action fixed the reading:
1. Separate proposal bounds from realized exposure.
2. Specify rejection/liquidation under loss.
3. Quantify admissible gaps.

**A-GAP-BOUND** (registered before probing):
- per-bar |return| ≤ 1/2;
- |funding per unit| ≤ 3% of the prior price;
- finite positive prices and finite funding;
- execution within the registered stress maxima (cost ≤ 2.5×, impact ≤ 10 bp,
  funding ≤ 2×);
- monotone `sqrt` on [0, 1];
- the binary64 standard rounding model.

This is an assumption about markets, not a fact. Paths outside it, such as the
preserved −87.5% and +300% witnesses CE-RL-002/003, can still breach any bound,
and the unconditional floor stays refuted.

## Certified

- **(I)** Every non-terminal `Replay` state satisfies equity ≥ 0.8, drawdown
  ≤ 0.15 and |exposure| ≤ 0.35. The bound check runs after every
  mark-to-market and every admitted trade, and any failure is terminal.
- **(II)** Shielded targets are in {−1/4, 0, 1/4}, and a fresh target's
  post-cost exposure is < 0.2505.
- **(III)** Under A-GAP-BOUND:
  - one bar keeps equity above 0.80 × its prior value;
  - exposure at detection is ≤ 0.66;
  - a trade costs < 0.18% and a liquidation < 0.24% of equity;
  - the data early exits are unreachable.

  So every risk- or horizon-triggered termination liquidates **flat** with
  positive equity. A proposal rejection places no order and leaves the last
  in-bounds position unchanged.

Evidence:
- **F-RL-BOUNDS-SOURCE** (exhaustively checked): binding of the risk
  predicate, step order, early exits, rejection branch, cost terms, shield and
  the nine stress configurations.
- **F-RL-BOUNDS-COMPOSE** (SMT verified): seven pairs.
- **F-RL-BOUNDS-CONFORMANCE** (property tested): 288 episodes, 2,037
  non-terminal checks, 224 breach terminations (all flat), 35 rejections.
- **Existing certificates reused:** the shield, gap, post-cost and drawdown
  certificates plus the research-surface set.

## Corrections during the audit

Binding the source revealed terminations that skip liquidation:
- two data early exits;
- proposal rejection.

The contract records them. The assumption gained finite positive prices and
finite funding, which the frozen loader does not guarantee for funding
(CE-RL-019). The liquidation claim was narrowed to risk- and horizon-triggered
terminations. No certificate was recorded before these corrections.
