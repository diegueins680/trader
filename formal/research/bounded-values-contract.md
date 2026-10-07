# Bounded values (obligation 7)

Engineering preregistration, 2026-10-06. Read-only audit of unchanged source.

## Criterion and recorded reading

Obligation 7: "Shielded targets and post-cost positions respect registered hard
bounds under explicit gap/solvency assumptions." The obligation's recorded next
action fixes the intended reading:
1. Separate proposal bounds from realized exposure.
2. Specify rejection/liquidation under loss.
3. Quantify admissible gap assumptions.

The unconditional claim (equity never falls below 0.80) stays **refuted**
(CE-RL-002/003): a gap can carry equity through any threshold before it is
observed.

## Registered hard bounds (frozen `Replay`)

- Capital floor: equity ≥ 4/5.
- Drawdown: ≤ 3/20 of peak.
- Exposure: |units · price / equity| ≤ 7/20.
- Shielded target: one of {−1/4, 0, 1/4}.
- Turnover per trade: ≤ 1/2 (otherwise rejected).

## Explicit gap/solvency assumption A-GAP-BOUND

- Per bar, |p₁/p₀ − 1| ≤ 1/2.
- Per bar, |funding per unit| ≤ (3/100) · p₀.
- Execution parameters are within the registered stress maxima: cost multiplier
  ≤ 5/2, impact ≤ 10 bp, funding multiplier ≤ 2, fill fraction in (0, 1], delay ≤ 1.
- `numpy.sqrt` is monotone on [0, 1] with sqrt(1) = 1.
- Each binary64 operation obeys the standard rounding model with u = 2⁻⁵³.

This is an assumption about market data, not a fact. Extreme crypto events have
exceeded a 50% eight-hour move. Paths that violate it are outside the claim, and
the refuted unconditional floor still records that they can breach any bound.

## Claims

**(I) Realized-state invariant.** Every non-terminal `Replay` state satisfies
all three realized bounds (floor, drawdown, exposure). The step order is
mark-to-market → `_risk` → pending trade → `_risk` → terminal check, and any
failure ends the step in a terminal state. This holds as evaluated in binary64,
because the bound check is the source predicate itself.

**(II) Proposal bounds.** Shielded targets are in {−1/4, 0, 1/4}
(F-RL-SHIELD-BOUNDS, Haskell boundary). A fresh full-fill target has post-cost
exposure below 0.2505 ≤ 7/20.

**(III) Liquidation under loss (solvency).** Under A-GAP-BOUND, from any
non-terminal state:
- one bar keeps equity above 0.80 × its previous value;
- an admitted trade costs at most 0.18% of equity;
- exposure at detection is at most 0.66;
- the terminal liquidation costs at most 0.24% of equity.

So equity stays strictly positive and every terminal path liquidates to **flat**
inventory, never a failed liquidation.

## Obligations / methods

- **F-RL-BOUNDS-SOURCE (exhaustively checked):** AST binding of the `_risk`
  thresholds and order, the step-loop order and terminal break, the terminal
  liquidation guard, the `_trade` cost and turnover formulas, the shield action
  set and the registered `STRESSES` maxima.
- **F-RL-BOUNDS-COMPOSE (SMT, reals with explicit rounding factors):** lemmas
  for (I)–(III).
- **F-RL-BOUNDS-CONFORMANCE (property tested):** 288 seeded episodes on the
  actual `Replay` across all nine registered stress configurations, with random
  gaps up to the assumption bound. Every non-terminal state satisfies the
  realized bounds, every terminal path ends flat with positive equity, and the
  preserved unconditional-floor witnesses still breach the floor and remain
  outside A-GAP-BOUND.

Closure requires F-RL-BOUNDS-SOURCE and F-RL-BOUNDS-COMPOSE plus the existing F-RL-SHIELD-BOUNDS,
F-RL-GAP-BOUND, F-RL-POSTCOST-EXPOSURE, F-RL-DRAWDOWN-COMPOSE and the complete
research-surface certificates. The production order adapter's sizing and
rounding are obligation 9, not this obligation, and are not claimed here.

## Correction during audit (before any certificate was recorded)

Binding the source exposed three terminations that skip liquidation.

1. `invalid_market_transition`, when p₀, p₁ or funding is non-finite or
   non-positive.
2. `invalid_observation`, when `market_features` returns `None`.
3. A rejected proposal: an invalid action, timeout, invalid ownership or
   observation. It places no order and ends the episode, leaving the last
   checked position unchanged ("incomplete path" by design).

A-GAP-BOUND is extended with finite positive prices and finite funding. The
frozen loader enforces finite positive prices. It does **not** guarantee finite
funding (CE-RL-019), so funding finiteness is part of the assumption. Under it,
(1) and (2) are unreachable:
- p₀ and p₁ stay positive because |r| ≤ 1/2;
- every feature window lies in [24, len), as the `Replay` guard ensures;
- every 24-bar price ratio lies in [2⁻²⁴, (3/2)²⁴], so the features are finite.

Claim (III) is restated:
- every **risk- or horizon-triggered** termination liquidates to flat with
  positive equity;
- a **proposal rejection** places no order and leaves the last checked,
  in-bounds position unchanged.

Claim (I) is unaffected, because rejection does not move the position. The
conformance probe adds rejection episodes.

## Closure certificates versus tested evidence (review correction, 2026-10-06)

The conformance requirement is reproduced on every formal run as supporting
tested evidence. It is **not** a closure certificate. The verifier admits only
proof-class statuses (proved, model/SMT/refinement-verified, exhaustively
checked) as closure certificates, because tests are not proofs. Earlier wording
said closure required all three certificates. That is corrected here, and the
ledger was already consistent with this.

## Faithful binary64 model (review correction, before merge)

Review found that the first COMPOSE lemma modeled the equity update with three
rounding factors. It treated `units·(p1 − p0)` and the funding products as
exact, and it ignored underflow. That certified a different expression from the
source. It is replaced as follows.

- **Z3-certified generic lemmas:**
  - R1: one rounding of a bounded value;
  - R2: one rounding with a lower bound;
  - P: products of bounded magnitudes;
  - S: sums of bounded magnitudes;
  - I: the exact numerator bound implied by a computed comparison
    |fl(fl(n)/m)| ≤ t.

  Each rounding is fl(x) = x(1 + δ) + η with |δ| ≤ 2⁻⁵³ and |η| ≤ 2⁻¹⁰⁷⁵, so
  subnormal results are included.
- **Instantiation along the source sequence:** the lemmas are applied, in exact
  rationals, along the AST-bound operation sequence:
  - `p1 − p0`, `units·d`, `(−units)·f`, `·m`, `gross + funding`, `equity + ·`;
  - in `_trade`, `|new − old|·p`, `/e`, each `cash·c·cm`, the impact product,
    the left-to-right `sum`, and `e − total`.

  Source literals use their exact binary64 values (for example
  `0.35 = 3152519739159347/2⁵³`).
- **Range premise added to A-GAP-BOUND:** prices lie in [2⁻⁹⁰⁰, 2⁹⁰⁰] and equity
  ≤ 2⁹⁰⁰, so no operation overflows.
- **Resulting margins:**
  - equity after a bar ≥ 0.8040·E;
  - exposure at detection ≤ 0.6530;
  - equity after a trade ≥ 0.99825·e;
  - equity after a liquidation ≥ 0.99771·e;
  - post-cost entry exposure ≤ 0.2505.

The instantiation is a computation, not a solver query, so F-RL-BOUNDS-COMPOSE
records both. `numpy.sqrt` is assumed monotone and ≤ 1 on [0, 1].
