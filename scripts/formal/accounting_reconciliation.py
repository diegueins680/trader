"""Obligation 8: per-row binary64 reconciliation of the frozen Replay ledger against its stated formulas."""
import ast
from fractions import Fraction as Q
import json
from pathlib import Path
import sys

import numpy as np
import z3 as z
from ppo_successor import certify
from bounded_values import ENV, H, HIGH, LOW_PRICE, STRESSES, U, _method, _named, dbl

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = 'research-notes/registrations/accounting-reconciliation-engineering.json'
ROUNDINGS = 14   # 10 chain operations (gross+funding, equity+, 3+1 trade, 3+1 liquidation) + 4 cost merges
STATED = {'rollForward': (Q(15), Q(15)), 'funding': (Q(3), Q(4)), 'costTerm': (Q(4), Q(10)), 'impact': (Q(6), Q(10))}
GROSS_E = Q(1, 2**600)


def require(ok, reason):
    if not ok:
        raise ValueError('accounting reconciliation: ' + reason)


def bind(sources=None):
    sources = sources or {}
    env = ast.parse(sources.get(ENV, (ROOT / ENV).read_text()))
    replay = _named(env, 'Replay')
    writes = sorted(ast.unparse(s) for s in ast.walk(replay) if isinstance(s, (ast.Assign, ast.AugAssign)) and
                    'self.equity' in {ast.unparse(t) for t in ast.walk(s.targets[0] if isinstance(s, ast.Assign) else s.target)})
    require(writes == sorted(["self.equity -= sum((terms[k] for k in ('fee', 'spread', 'slippage', 'impact')))",
                       'self.equity += gross + funding', 'self.units, self.equity, self.peak = (0.0, 1.0, 1.0)']), 'every equity write is a reviewed ledger operation')
    step = _method(replay, 'step')
    texts = {ast.unparse(s) for s in ast.walk(step) if isinstance(s, (ast.Assign, ast.AugAssign))}
    require({'gross, funding = (self.units * (p1 - p0), -self.units * f * self.execution.funding_multiplier)',
             "costs = dict.fromkeys(('turnover', 'fee', 'spread', 'slippage', 'impact'), 0.0)",
             'costs = self._trade(self.pending[1])', 'liquidation = self._trade(0.0, terminal=True)',
             'costs = {k: costs[k] + liquidation[k] for k in costs}'} <= texts, 'mark-to-market, cost and merge expressions')
    row = next(n for n in ast.walk(step) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self.rows.append')
    keys = {ast.unparse(k) for k in row.args[0].keys if k is not None}
    values = {ast.unparse(k): ast.unparse(v) for k, v in zip(row.args[0].keys, row.args[0].values) if k is not None}
    require({"'equity'", "'gross'", "'funding'"} <= keys and values["'equity'"] == 'self.equity' and values["'gross'"] == 'gross'
            and values["'funding'"] == 'funding' and any(k is None and ast.unparse(v) == 'costs' for k, v in zip(row.args[0].keys, row.args[0].values)),
            'row records the executed equity, gross, funding and merged costs')
    trade = _method(replay, '_trade')
    terms = next(s for s in ast.walk(trade) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == 'terms')
    require(ast.unparse(terms.value) == "{'turnover': cash, 'fee': cash * 0.0005 * cfg.cost_multiplier, 'spread': cash * 5e-05 * cfg.cost_multiplier, "
            "'slippage': cash * 0.00045 * cfg.cost_multiplier, 'impact': cash * cfg.impact_bps * 0.0001 * np.sqrt(turnover)}", 'cost formulas')
    rejected = [ast.unparse(r) for r in ast.walk(trade) if isinstance(r, ast.Return)]
    require("return {'turnover': 0.0, 'fee': 0.0, 'spread': 0.0, 'slippage': 0.0, 'impact': 0.0}" in rejected, 'turnover rejection debits nothing')
    loop = next(n for n in ast.walk(step) if isinstance(n, ast.While))
    first_exit = next(s for s in loop.body if isinstance(s, ast.If) and any(isinstance(b, ast.Break) for b in s.body))
    require(ast.unparse(first_exit.test) == 'not all((_finite_real(v) for v in (p0, p1, f))) or min(p0, p1) <= 0' and
            loop.body.index(first_exit) < next(i for i, s in enumerate(loop.body) if 'self.rows.append' in ast.unparse(s)),
            'invalid market data rejects before any row')
    return {'status': 'exhaustively_checked', 'equityWrites': len(writes), 'scope': 'unchanged frozen Replay ledger source; interpreter semantics assumed'}


def lemmas():
    u, h = z.RealVal(str(U)), z.RealVal(str(H))
    x, d, e, p, c, a, m, y = z.Reals('ar_x ar_d ar_e ar_p ar_c ar_a ar_m ar_y')
    small = z.And(d >= -u, d <= u, e >= -h, e <= h)
    absx = z.If(x >= 0, x, -x)
    # L1: one rounding.
    err = x * (1 + d) + e - x
    certify(small, z.And(err <= u * absx + h, err >= -(u * absx + h)))
    # L2: product chain step |P-1| <= c  =>  |P(1+d)-1| <= c(1+u)+u.
    certify(z.And(small, c >= 0, p - 1 <= c, 1 - p <= c), z.And(p * (1 + d) - 1 <= c * (1 + u) + u, 1 - p * (1 + d) <= c * (1 + u) + u))
    # L3: accumulation step: prior error <= a, operand magnitude <= m + a, new rounding of it.
    absy = z.If(y >= 0, y, -y)
    certify(z.And(small, a >= 0, m >= 0, absy <= m + a), (u * absy + h) + a <= a * (1 + u) + u * m + h)
    # L4: roll-forward identity on recorded values (chain errors e_i, merge errors m_k).
    E, g, f = z.Reals('ar_E ar_g ar_f')
    t = z.Reals('ar_t0 ar_t1 ar_t2 ar_t3'); l = z.Reals('ar_l0 ar_l1 ar_l2 ar_l3')
    ei = z.Reals(' '.join(f'ar_e{i}' for i in range(10))); mk = z.Reals('ar_m0 ar_m1 ar_m2 ar_m3')
    s1 = g + f + ei[0]; Ea = E + s1 + ei[1]
    q3 = t[0] + t[1] + ei[2] + t[2] + ei[3] + t[3] + ei[4]; Eb = Ea - q3 + ei[5]
    r3 = l[0] + l[1] + ei[6] + l[2] + ei[7] + l[3] + ei[8]; Ec = Eb - r3 + ei[9]
    rows = [t[k] + l[k] + mk[k] for k in range(4)]
    ledger = E + g + f - sum(rows)
    certify(z.BoolVal(True), Ec - ledger == ei[0] + ei[1] - ei[2] - ei[3] - ei[4] + ei[5] - ei[6] - ei[7] - ei[8] + ei[9] + sum(mk))
    # L5: gross algebra with |U * p0| <= x0 * E: absolute part bounded by 2^-600 * E under the range premise.
    U_, dp, d1, d2, e1, e2, Ev, p0 = z.Reals('ar_U ar_dp ar_d1 ar_d2 ar_e1 ar_e2 ar_Ev ar_p0')
    g_c = (U_ * (dp * (1 + d1) + e1)) * (1 + d2) + e2
    x0 = (dbl(0.35) + 3 * H) / ((1 - U) * (1 - U))
    absU = z.If(U_ >= 0, U_, -U_)
    certify(z.And(z.And(d1 >= -u, d1 <= u, d2 >= -u, d2 <= u, e1 >= -h, e1 <= h, e2 >= -h, e2 <= h),
                  Ev >= z.RealVal('4/5'), Ev <= z.RealVal(str(HIGH)), p0 >= z.RealVal(str(LOW_PRICE)), absU * p0 <= z.RealVal(str(x0)) * Ev),
            absU * h * (1 + u) <= z.RealVal(str(GROSS_E)) * Ev)
    return 5


def derive():
    chain = lambda k: (1 + U) ** k - 1            # relative error of k multiplicative roundings (L2 induction)
    accum = ((1 + U) ** ROUNDINGS - 1) / U         # L3 recurrence: error <= (u*M + H) * accum
    roll_rel, roll_abs = accum, accum
    gross_rel = chain(2)
    funding_rel, funding_abs = chain(2), Q(2) * (1 + U) + 1
    cm = Q(5, 2)
    term_rel = chain(3)
    term_abs = 2 * (cm * (1 + U) ** 2 + (1 + U)) + 1
    impact_rel = chain(5)
    small = Q(10) * dbl(0.0001)
    impact_abs = 2 * ((small + 2) * (1 + U) ** 2) + 1
    derived = {'rollForward': (roll_rel, roll_abs), 'funding': (funding_rel / U, funding_abs), 'costTerm': (term_rel / U, term_abs),
               'impact': (impact_rel / U, impact_abs), 'gross': (gross_rel / U, Q(2))}
    stated = dict(STATED, gross=(Q(3), Q(2)))
    for key, (rel, abs_) in derived.items():
        require(rel <= stated[key][0] and abs_ <= stated[key][1], 'stated bound below derived constant: ' + key)
    return {k: [float(v[0]), float(v[1])] for k, v in derived.items()}


def _replay():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    from sequential_env import Execution, Replay, Scale
    return Execution, Replay, Scale


def _sqrt_interval(x, bits=200):
    """Rigorous rational enclosure [lo, hi] of sqrt(x) for a nonnegative rational x."""
    from math import isqrt
    scale = 2 ** bits
    n = isqrt(x.numerator * scale * scale // x.denominator)
    return Q(n, scale), Q(n + 1, scale)


def conformance():
    Execution, Replay, Scale = _replay()
    reg = json.loads((ROOT / REGISTRATION).read_text())
    rng = np.random.default_rng(reg['probeSeed'])
    rows = gross_checks = cost_checks = impact_checks = 0
    c = {'fee': dbl(0.0005), 'spread': dbl(5e-05), 'slippage': dbl(0.00045)}
    (term_rel, term_abs), (imp_rel, imp_abs) = STATED['costTerm'], STATED['impact']
    calls, trade_calls, impact_seen = [], 0, False
    original = Replay._trade
    def spy(self, target, terminal=False):
        # Pass-through instrumentation: record the executed inputs of each actual _trade call.
        equity, index = self.equity, len(self.rows)
        terms = original(self, target, terminal)
        calls.append((id(self), index, equity, terms))
        return terms
    Replay._trade = spy
    try:
        for k in range(reg['probeEpisodes']):
            stress = list(STRESSES)[k % len(STRESSES)]
            cfg = Execution(**STRESSES[stress])
            n = 140
            gaps = np.clip(rng.uniform(-0.5, 0.5, n) * (rng.random(n) < 0.1) + rng.normal(0, 0.01, n), -0.5, 0.5)
            p = 100 * np.cumprod(1 + gaps)
            f = np.r_[0.0, p[:-1] * rng.uniform(-0.03, 0.03, n - 1)]
            horizon = 1 if k % 2 == 0 else (3, 6)[k % 4 // 2]
            env = Replay(p, f, 60, 130, horizon, Scale.fit([p[:50]]), cfg, enabled=True)
            trade_calls += len(calls)
            calls.clear()   # object ids can be reused across episodes; scope the log to this episode
            previous = Q(env.equity)
            while not env.done:
                units, t0, seen, before = env.units, env.t, len(env.rows), previous
                env.step(float(rng.choice([-0.25, 0.0, 0.25])))
                for index in range(seen, len(env.rows)):
                    row = env.rows[index]
                    g, fu = Q(row['gross']), Q(row['funding'])
                    costs = [Q(row[key]) for key in ('fee', 'spread', 'slippage', 'impact')]
                    M = previous + abs(g) + abs(fu) + sum(costs)
                    ledger = previous + g + fu - sum(costs)
                    require(abs(Q(row['equity']) - ledger) <= STATED['rollForward'][0] * U * M + STATED['rollForward'][1] * H,
                            'roll-forward outside the stated bound')
                    executed = [(Q(e), terms) for owner, i, e, terms in calls if owner == id(env) and i == index]
                    for key, lit in c.items():
                        exact = sum((Q(t['turnover']) * lit * Q(cfg.cost_multiplier) for _, t in executed), Q(0))
                        require(abs(Q(row[key]) - exact) <= term_rel * U * exact + term_abs * H, 'cost term outside the stated bound: ' + key)
                        cost_checks += 1
                    lo = hi = Q(0)
                    for e, t in executed:
                        cash = Q(t['turnover'])
                        if cash:
                            s_lo, s_hi = _sqrt_interval(Q(float(t['turnover']) / float(e)))
                            factor = cash * Q(cfg.impact_bps) * dbl(0.0001)
                            lo, hi = lo + factor * s_lo, hi + factor * s_hi
                    require(Q(row['impact']) >= lo * (1 - imp_rel * U) - imp_abs * H and Q(row['impact']) <= hi * (1 + imp_rel * U) + imp_abs * H,
                            'impact outside the stated bound')
                    impact_checks += 1
                    impact_seen = impact_seen or Q(row['impact']) > 0
                    previous = Q(row['equity']); rows += 1
                if horizon == 1 and len(env.rows) > seen:
                    row = env.rows[seen]
                    exact_g = Q(units) * (Q(p[t0 + 1]) - Q(p[t0]))
                    exact_f = -Q(units) * Q(f[t0 + 1]) * Q(cfg.funding_multiplier)
                    require(abs(Q(row['gross']) - exact_g) <= 3 * U * abs(exact_g) + GROSS_E * before + 2 * H, 'gross outside stated bound')
                    require(abs(Q(row['funding']) - exact_f) <= STATED['funding'][0] * U * abs(exact_f) + STATED['funding'][1] * H,
                            'funding outside stated bound')
                    gross_checks += 1
    finally:
        Replay._trade = original
    trade_calls += len(calls)
    require(rows > 1000 and gross_checks > 100 and impact_seen, 'vacuous probe')
    return {'status': 'property_tested', 'episodes': reg['probeEpisodes'], 'rows': rows, 'grossFundingChecks': gross_checks,
            'costTermChecks': cost_checks, 'impactChecks': impact_checks, 'tradeCalls': trade_calls, 'seed': reg['probeSeed']}


def check_reconciliation():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require(reg['sourceChanges'] == 0 and reg['holdoutOpened'] is False, 'registration')
    lemmas()
    return {'source': bind(), 'smt': {'F-RL-RECON-ERROR': 'unsat'}, 'constants': derive(), 'conformance': conformance()}
