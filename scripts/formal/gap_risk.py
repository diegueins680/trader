"""Exact-real conditional risk lemmas and explicit loss-floor refutations.

These are mathematical market-transition assumptions, not binary64 refinement
or certified bounds on future market moves. See gap-risk-contract.md.
"""
from fractions import Fraction

import z3 as z


def obligations():
    equity, weight, ret, debit, cap, jump, costs, loss = z.Reals(
        'equity weight ret debit cap jump costs loss')
    ending = equity * (1 + weight * ret - debit)
    bounded = z.And(equity > 0, cap >= 0, cap <= z.RealVal('1/4'),
                    weight >= -cap, weight <= cap, jump >= 0,
                    ret >= -jump, ret <= jump, costs >= 0,
                    debit >= 0, debit <= costs, loss >= 0, loss < 1,
                    cap * jump + costs <= loss)

    peak, prior_loss, next_equity = z.Reals('peak prior_loss next_equity')
    next_peak = z.If(next_equity > peak, next_equity, peak)
    composed = z.And(peak >= equity, equity > 0, prior_loss >= 0,
                     prior_loss < 1, loss >= 0, loss < 1,
                     equity >= peak * (1 - prior_loss),
                     next_equity >= equity * (1 - loss), next_equity > 0)

    target, limit = z.Reals('target limit')
    abs_target = z.If(target >= 0, target, -target)
    funded = z.And(debit >= 0, debit <= costs, costs < 1,
                   limit >= 0, abs_target <= limit * (1 - costs))
    return {
        'F-RL-GAP-BOUND': (bounded, z.And(ending >= equity * (1 - loss), ending > 0)),
        'F-RL-DRAWDOWN-COMPOSE': (
            composed, 1 - next_equity / next_peak <= prior_loss + loss - prior_loss * loss),
        'F-RL-POSTCOST-EXPOSURE': (funded, abs_target / (1 - debit) <= limit),
    }


def check_obligations():
    results = {}
    for name, (premise, conclusion) in obligations().items():
        solver = z.Solver()
        solver.set(timeout=10000, random_seed=0)
        solver.add(premise)
        if solver.check() != z.sat:
            raise RuntimeError(name + ': missing satisfiable premise')
        solver.add(z.Not(conclusion))
        status = solver.check()
        if status != z.unsat:
            detail = str(solver.model()) if status == z.sat else solver.reason_unknown()
            raise RuntimeError(f'{name}: {status}: {detail}')
        results[name] = 'unsat'
    return results


def exact_two_bar(target, price_ratio, cost_multiplier, funding_per_unit):
    """Independent rational reference: enter next close, exit following close.

    Initial equity 1, warmup/entry price 100, no impact, full fills, no misses.
    Funding is charged only to old units on the exit bar. All supplied test
    paths remain solvent before liquidation. This models no other fill path.
    """
    target, ratio, multiplier, funding = map(
        Fraction, (target, price_ratio, cost_multiplier, funding_per_unit))
    entry_cost = abs(target) * multiplier / 1000
    gross = target * (ratio - 1)
    funding_cash = -target * funding / 100
    exit_cost = abs(target) * ratio * multiplier / 1000
    return 1 + gross + funding_cash - entry_cost - exit_cost


def check_counterexamples(document):
    if document['schemaVersion'] != 1:
        raise ValueError('unsupported gap counterexample schema')
    entries = document['entries']
    if [e['id'] for e in entries] != ['CE-RL-002', 'CE-RL-003']:
        raise ValueError('gap counterexample roster drift')
    results = {}
    for entry in entries:
        target = Fraction(entry['target'])
        ratio = Fraction(entry['priceRatio'])
        ending = exact_two_bar(target, ratio, 1, 0)
        if (target not in (Fraction(-1, 4), Fraction(1, 4)) or ratio <= 0 or
                ending != Fraction(entry['endingEquity']) or not 0 < ending < Fraction(4, 5)):
            raise ValueError('invalid loss-floor witness: ' + entry['id'])
        q, r, final = z.Reals('witness_target witness_ratio witness_final')
        solver = z.Solver()
        solver.set(timeout=10000, random_seed=0)
        abs_q = z.If(q >= 0, q, -q)
        solver.add(q == z.RealVal(str(target)), r == z.RealVal(str(ratio)),
                   z.Or(q == z.RealVal('-1/4'), q == z.RealVal('1/4')),
                   r > 0, final == 1 + q * (r - 1) - abs_q * (1 + r) / 1000,
                   final > 0, final < z.RealVal('4/5'))
        if solver.check() != z.sat:
            raise RuntimeError('loss-floor refutation no longer checks: ' + entry['id'])
        if not solver.model().eval(final).eq(z.RealVal(str(ending))):
            raise RuntimeError('counterexample accounting mismatch')
        results[entry['id']] = {'status': 'sat', 'endingEquity': str(ending)}
    return results
