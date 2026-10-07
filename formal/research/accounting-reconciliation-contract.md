# Accounting reconciliation (obligation 8)

Engineering preregistration, 2026-10-07. Read-only audit of unchanged source.

## Criterion and reading

Obligation 8: "Every wealth, fee, funding and liquidation debit reconciles actual
binary64 execution within a stated error bound or rejects." The obligation's
recorded next action: "prove bounded arithmetic error and failure rules for a
versioned replay accounting core".

**Reading: per-row reconciliation.** For every row the frozen `Replay`
records, each recorded debit is compared with its stated formula evaluated
exactly on the executed binary64 inputs. The recorded wealth roll-forward is
compared with the exact ledger identity on the recorded values. Each must lie
within a stated binary64 error bound. Bars with non-finite or non-positive market
data reject (`invalid_market_transition`) and record no row.

The earlier blocker also named "calibrated fill semantics", meaning whether the
simulated fills match real venue fills. That is an empirical question for the
economic gate, not a property of the binary64 accounting. It is not claimed or
discharged here.

## Stated bounds (final; u = 2⁻⁵³, H = 2⁻¹⁰⁷⁵)

For a row with prior equity E, recorded gross g, funding f and **recorded
merged** costs (fee, spread, slippage, impact), with
M = E + |g| + |f| + Σ recorded costs:

- **Roll-forward:** |E_row − (E + g + f − fee − spread − slippage − impact)|
  ≤ **15·u·M + 15·H**. This covers 14 roundings: gross+funding, equity+, three
  trade-cost additions, equity−, three liquidation-cost additions, equity−, and
  four cost merges. The constant 14 is provably insufficient.
- **Gross:** |g − U·(p₁ − p₀)| ≤ **3·u·|U·(p₁ − p₀)| + 2⁻⁶⁰⁰·E + 2·H**. This
  uses the carried-inventory premise certified by F-RL-BOUNDS-COMPOSE.
- **Funding:** |f − (−U·f_t·m)| ≤ **3·u·|U·f_t·m| + 4·H**.
- **Fee, spread, slippage:** each recorded term is within **4·u relative +
  10·H** of Σ over the bar's trade and liquidation calls of
  `cash·c·cost_multiplier`, with c the exact binary64 literal.
- **Impact:** within **6·u relative + 10·H** of Σ
  `cash·impact_bps·c₄·sqrt(turnover)`, with the exact square root of the
  computed turnover.

The premises are A-GAP-BOUND's range premise (prices and equity in
[2⁻⁴⁰⁰, 2⁴⁰⁰]) and binary64 arrays from the delivered loader. The sections
below record how these final values were reached.

## Obligations / methods

- **F-RL-RECON-SOURCE (exhaustively checked):** AST binding of the
  mark-to-market expressions, both equity updates, the trade cost terms, the
  `sum` order, the terminal cost merge, the row record and the rejecting early
  exit.
- **F-RL-RECON-ERROR (SMT):** Z3 certifies the single-rounding accumulation
  lemma |rnd(x) − x| ≤ u·|x| + H and the composition step. The per-bound
  operation counts are instantiated exactly along the AST-bound sequences.
- **F-RL-RECON-CONFORMANCE (property tested):** 216 seeded episodes over all
  nine stresses on the actual `Replay`. Every row's roll-forward is checked
  exactly with Fractions against its stated bound. On horizon-1 episodes, gross
  and funding are checked against units captured before each bar. Fee, spread
  and slippage are checked against the recorded merged cash.

Closure requires F-RL-RECON-SOURCE and F-RL-RECON-ERROR, the bounded-values
certificates F-RL-BOUNDS-SOURCE and F-RL-BOUNDS-COMPOSE (see below), the
existing exact identities F-RL-ACCOUNT, F-RL-ROW-RECONCILE and F-RL-WEALTH-FOLD,
and the complete research-surface certificates. Conformance is supporting tested
evidence, not a closure certificate.

## Correction before any certificate (constants)

Deriving the preregistered constants exactly showed three were slightly too tight:

- **Roll-forward:** n = 14 roundings with accumulated error give
  (u·M + H)·((1 + u)ⁿ − 1)/u, which exceeds 14·(u·M + H). The stated bound is
  **15·u·M + 15·H**.
- **Gross:** without relying on Sterbenz's lemma, rounding `p₁ − p₀` contributes
  an absolute term up to |U|·H, with |U| ≤ 0.35·E·2⁴⁰⁰. The stated bound is
  **3·u·|U(p₁ − p₀)| + 2⁻⁶⁰⁰·E + 2·H**.
- **Cost terms:** each product's absolute term is scaled by the cost multiplier
  (≤ 5/2), and the trade/liquidation merge adds one more. The stated bounds are
  **fee, spread, slippage: 4·u relative + 10·H** and **impact: 6·u relative +
  10·H**. **Funding** is **3·u relative + 4·H**.

The checker derives the minimal rigorous constants in exact rationals and
requires every stated bound to be at least the derived one.

## Review corrections (before merge)

- **Inventory premise composed.** The gross bound's absolute term |U|·H ≤
  2⁻⁶⁰⁰·E needs the carried inventory to satisfy |U·p₀| ≤ x₀·E. That premise
  is not local to this audit. F-RL-BOUNDS-COMPOSE certifies it for every
  non-terminal state, and every bar starts from a state whose risk check passed.
  Both bounded-values certificates are therefore required for this closure, and
  the F-RL-RECON-ERROR statement names the premise.
- **Probe now tests the stated bounds exactly.** The first probe compared
  fee/spread/slippage against the recorded merged cash with a looser tolerance,
  and it never checked impact. A pass-through spy on the actual `_trade` now
  records each call's executed cash, equity and turnover. Every row's terms are
  checked against the exact stated bounds (4u + 10H), and impact against 6u + 10H
  using rigorous exact-square-root enclosures. Episode-scoped call logs avoid
  object-id reuse across episodes.

## Mechanical certificate (second review correction, before merge)

The first certificate proved generic lemmas, and the per-bound operation counts
and composition lived in hand-written Python. A wrong count would still have
yielded "unsat". The certificate is now generated from the source AST:

- **Gross and funding:** the reviewed expressions are translated to Z3 with
  fresh rounding variables per binary operation, and the final advertised
  inequality is one query each. The operation count (2) is asserted from the
  AST.
- **Fee, spread, slippage and impact:** the left-associated factor chain is
  extracted from the `terms` dictionary AST. Each multiplication is one
  certified step-lemma instance with concrete rational constants (the correctly
  rounded `sqrt` contributes relative error only). The trade/liquidation merge
  is one more certified instance, bound to the merge comprehension. The
  resulting constants must not exceed the stated bounds.
- **Roll-forward:** the 14 roundings come from the AST: the mark update, the
  `sum` order over the cost keys, both debits and the four merges.
  - Stage A certifies each operand's magnitude, one query per rounding.
  - Stage B certifies the final 15·u·M + 15·H bound in one query.

  With the constant 14 the generated query is satisfiable; that regression is
  kept.

## Third review correction (before merge)

- **Recorded M.** The roll-forward claim now uses M built from the
  **recorded merged** costs, the quantity the contract advertises, instead of
  the unmerged operands. The operand budgets still use the unmerged
  magnitudes, and Z3 certifies that the stated constant absorbs the difference.
- **Control flow bound.** The source check now asserts the bar-loop order:
  1. mark update;
  2. per-bar cost reset;
  3. trade block assigning `costs`;
  4. terminal block (liquidation, then merge);
  5. `rows.append`, followed only by `if self.done: break`.

  It also asserts that no equity write or `_trade` call follows the row record.
  Reordering the record before liquidation now fails the source certificate.
