"""SMT proofs about the declared ledger and partition arithmetic (specification level).

These are proofs over real and integer arithmetic of the *declared* transition
(the same equations as ``exposure_env_v1``'s docstring and the exact-rational
reference in ``test/exposure_rl_test.py``). They are not a proof about binary64
execution; the implementation is tied to the declared ledger by conformance
tests against exact rationals, which is property testing, not proof.
"""

from __future__ import annotations

import z3 as z

from shield_proofs import certify, z_abs


def step(equity, units, mark, open_, close, f_open, f_inside, weight, intent, rate, at_close):
    """Declared one-step ledger. Returns (new_equity, new_units, gross, cost, funding)."""
    funding_open = units * f_open
    equity_open = equity + units * (open_ - mark) - funding_open
    # Fill at the open (base) ...
    wanted_o = weight * equity_open / open_
    units_o = z.If(intent, wanted_o, units)
    cost_o = z_abs(units_o - units) * open_ * rate
    inside_o = units_o * f_inside
    equity_o = equity_open - cost_o + units_o * (close - open_) - inside_o
    gross_o = units * (open_ - mark) + units_o * (close - open_)
    # ... or at the close (delay stress).
    inside_c = units * f_inside
    equity_fill = equity_open + units * (close - open_) - inside_c
    wanted_c = weight * equity_fill / close
    units_c = z.If(intent, wanted_c, units)
    cost_c = z_abs(units_c - units) * close * rate
    return (
        z.If(at_close, equity_fill - cost_c, equity_o),
        z.If(at_close, units_c, units_o),
        z.If(at_close, units * (close - mark), gross_o),
        z.If(at_close, cost_c, cost_o),
        funding_open + z.If(at_close, inside_c, inside_o),
    )


def ledger_claims() -> dict:
    equity, units, mark, open_, close, f_open, f_inside, weight, rate = z.Reals("equity units mark open close f_open f_inside weight rate")
    intent, at_close = z.Bools("intent at_close")
    new_equity, new_units, gross, cost, funding = step(equity, units, mark, open_, close, f_open, f_inside, weight, intent, rate, at_close)
    shape = z.And(equity > 0, mark > 0, open_ > 0, close > 0, rate >= 0)
    return {
        # Net equity change is exactly gross P&L minus cost minus funding: nothing counted twice or dropped.
        "F-RLX-LEDGER-IDENTITY": (shape, new_equity - equity == gross - cost - funding),
        "F-RLX-LEDGER-COST-NONNEGATIVE": (shape, cost >= 0),
        # A flat account without an intent is unchanged: no free reward from doing nothing.
        "F-RLX-LEDGER-FLAT": (z.And(shape, units == 0, z.Not(intent)), z.And(new_equity == equity, new_units == 0, cost == 0, funding == 0)),
        # Turnover is never free when the cost rate is positive.
        "F-RLX-LEDGER-TURNOVER-COSTS": (z.And(shape, rate > 0, new_units != units), cost > 0),
        # Without an intent the position is unchanged and no cost is charged.
        "F-RLX-LEDGER-NO-INTENT": (z.And(shape, z.Not(intent)), z.And(new_units == units, cost == 0)),
    }


def fold_claims() -> dict:
    """Two consecutive steps telescope: the wealth fold equals the sum of components."""
    e0, u0, p0, o1, c1, o2, c2, w1, w2, rate = z.Reals("e0 u0 p0 o1 c1 o2 c2 w1 w2 rate")
    fo1, fi1, fo2, fi2 = z.Reals("fo1 fi1 fo2 fi2")
    i1, i2, d = z.Bools("i1 i2 d")
    e1, u1, g1, k1, f1 = step(e0, u0, p0, o1, c1, fo1, fi1, w1, i1, rate, d)
    e2, _, g2, k2, f2 = step(e1, u1, c1, o2, c2, fo2, fi2, w2, i2, rate, d)
    shape = z.And(e0 > 0, p0 > 0, o1 > 0, c1 > 0, o2 > 0, c2 > 0, rate >= 0, e1 > 0)
    return {"F-RLX-LEDGER-WEALTH-FOLD": (shape, e2 == e0 + (g1 + g2) - (k1 + k2) - (f1 + f2))}


def partition_claims(purge: int = 2) -> dict:
    """Index arithmetic of ``run_campaign.fold_ranges`` with a decision's outcome one bar later."""
    require_purge = purge >= 1
    if not require_purge:
        raise ValueError("exposure ledger proof: purge must be at least one bar")
    i0, v0, t = z.Ints("i0 v0 t")
    fit_stop, val_stop = v0 - purge, i0 - purge
    shape = z.And(v0 >= purge, i0 >= v0)
    # A fit decision t has t <= fit_stop - 2 and its outcome bar is t + 1.
    fit_decision = z.And(t >= 0, t <= fit_stop - 2)
    val_decision = z.And(t >= v0, t <= val_stop - 2)
    return {
        "F-RLX-PARTITION-FIT-VAL": (z.And(shape, fit_decision), t + 1 < v0),
        "F-RLX-PARTITION-VAL-TEST": (z.And(shape, val_decision), t + 1 < i0),
        "F-RLX-PARTITION-FIT-TEST": (z.And(shape, fit_decision), t + 1 < i0),
    }


def check() -> dict:
    proved = {}
    for name, (premise, conclusion) in {**ledger_claims(), **fold_claims(), **partition_claims()}.items():
        certify(premise, conclusion)
        proved[name] = "unsat"
    return {"smt": proved, "domain": "declared ledger over the reals; fold-index arithmetic over the integers", "implementationLink": "property_tested (exact-rational conformance), not proved"}


if __name__ == "__main__":
    print(check())
