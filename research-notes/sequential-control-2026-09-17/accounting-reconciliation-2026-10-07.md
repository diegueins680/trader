# Accounting reconciliation closure (obligation 8) — 2026-10-07

**Obligation 8 closes for the delivered offline path.** Counts move to
**15 scoped closures / 22 partial / 1 open**. This is a read-only audit: no
source, registration, archived result or data changed.

## Reading

The criterion is "every wealth, fee, funding and liquidation debit reconciles
actual binary64 execution within a stated error bound or rejects". It is read
as **per-row reconciliation**:
- each recorded debit is compared with its stated formula on the executed
  binary64 inputs;
- the recorded wealth roll-forward is compared with the exact ledger identity
  on the recorded values.

Invalid market data rejects before any row. Whether simulated fills match real
venues (calibration) is empirical, belongs to the economic gate and is not
claimed. The bound is per row: global trajectory error growth is not bounded.

## Certified (u = 2⁻⁵³, H = 2⁻¹⁰⁷⁵)

| Quantity | Stated bound |
|---|---|
| Roll-forward | 15·u·M + 15·H, M = sum of magnitudes |
| Gross | 3·u relative + 2⁻⁶⁰⁰·E + 2·H |
| Funding | 3·u relative + 4·H |
| Fee, spread, slippage | 4·u relative + 10·H, on the executed cash |
| Impact | 6·u relative + 10·H, on executed cash and turnover |

- **F-RL-RECON-SOURCE** (exhaustively checked): the only equity writes are
  initialization, the mark-to-market update and the cost debit. The
  mark-to-market, costs, `sum` order, terminal merge, row record and rejecting
  exit are all bound.
- **F-RL-RECON-ERROR** (SMT verified): five lemmas.
  - single rounding;
  - product-chain induction;
  - accumulation recurrence;
  - the exact roll-forward error identity;
  - the gross absolute term.

  Each stated constant is checked against its exactly derived value. The
  roll-forward needs 14.00000000000001, so the preregistered 14 was a hair too
  tight; 15 was recorded before any certificate.
- **F-RL-RECON-CONFORMANCE** (property tested): 216 episodes, all 5,578 rows,
  1,817 gross/funding checks, 16,734 cost-term checks and 5,578 impact checks.
  A pass-through spy captures every `_trade` call's executed inputs (2,872
  calls), so the exact stated bounds are tested. A deliberately unrecorded
  debit is caught.

## Composition

The gross bound relies on the carried-inventory exposure premise certified by
obligation 7 (F-RL-BOUNDS-COMPOSE), so both bounded-values certificates are
required here.

## Self-caught errors

Two errors were caught during the work:
- the preregistered constants were slightly too tight, and were corrected before
  any certificate;
- the Z3 identity lemma rejected a sign error in the first draft (merge errors
  enter with a plus sign).
