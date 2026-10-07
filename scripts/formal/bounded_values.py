"""Obligation 7: realized-state invariant, proposal bounds and liquidation solvency under A-GAP-BOUND."""
import ast
from fractions import Fraction as Q
import json
from pathlib import Path
import sys

import numpy as np
import z3 as z
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = 'research-notes/registrations/bounded-values-engineering.json'
ENV = 'scripts/research/sequential_env.py'
RUNNER = 'scripts/research/run_sequential_screen.py'
GAPS = 'formal/research/gap-counterexamples.json'
RISK = ['if not isfinite(self.equity) or self.equity <= 0:\n    return \'equity_exhausted\'',
        "if self.equity < 0.8:\n    return 'capital_floor'",
        "if 1 - self.equity / self.peak > 0.15:\n    return 'drawdown_limit'",
        "if abs(self.units * self.prices[self.t] / self.equity) > 0.35:\n    return 'endpoint_exposure'",
        'return None']
STEP_ORDER = ['self.equity += gross + funding', 'self.t += 1', 'self.peak = max(self.peak, self.equity)',
              'self.failure = self._risk()']
STRESSES = {'base': {}, 'cost1_5x': {'cost_multiplier': 1.5}, 'cost2x': {'cost_multiplier': 2},
            'extreme25bp': {'cost_multiplier': 2.5}, 'delay1bar': {'extra_delay': 1}, 'partial50pct': {'fill_fraction': 0.5},
            'missed10pct': {'miss_every': 10}, 'impact10bp': {'impact_bps': 10}, 'funding2x': {'funding_multiplier': 2}}


def require(ok, reason):
    if not ok:
        raise ValueError('bounded values: ' + reason)


def _named(tree, name):
    found = [n for n in tree.body if getattr(n, 'name', None) == name]
    require(len(found) == 1, 'missing/duplicate ' + name)
    return found[0]


def _method(cls, name):
    found = [n for n in cls.body if getattr(n, 'name', None) == name]
    require(len(found) == 1, 'missing method ' + name)
    return found[0]


def bind(sources=None):
    sources = sources or {}
    env = ast.parse(sources.get(ENV, (ROOT / ENV).read_text()))
    replay = _named(env, 'Replay')
    risk = _method(replay, '_risk')
    require([ast.unparse(s) for s in risk.body] == RISK, 'registered hard-bound predicate')
    step = _method(replay, 'step')
    loop = [n for n in ast.walk(step) if isinstance(n, ast.While) and ast.unparse(n.test) == 'self.t < end']
    require(len(loop) == 1, 'single bar loop')
    body = [ast.unparse(s) for s in loop[0].body]
    positions = [next((i for i, s in enumerate(body) if s == want), -1) for want in STEP_ORDER]
    require(all(p >= 0 for p in positions) and positions == sorted(positions), 'mark-to-market then risk order')
    trade = next(s for s in loop[0].body if isinstance(s, ast.If) and 'self.pending[0] <= self.t' in ast.unparse(s.test))
    # Data early exits (no liquidation) are guarded by finiteness/positivity and feature admission only.
    exits = [ast.unparse(s.test) for s in loop[0].body if isinstance(s, ast.If) and any(isinstance(b, ast.Break) for b in s.body) and ast.unparse(s.test) != 'self.done']
    require(exits == ['not all((_finite_real(v) for v in (p0, p1, f))) or min(p0, p1) <= 0', 'x is None'], 'reviewed data early exits')
    rejection = next(s for s in step.body if isinstance(s, ast.If) and ast.unparse(s.test) == 'proposal is None')
    require([ast.unparse(b) for b in rejection.body] == ['self.rejections += 1', "self.failure, self.done = (reason, True)", 'return (None, 0.0, True)'],
            'proposal rejection places no order and leaves inventory unchanged')
    features = _named(env, 'market_features')
    require([ast.unparse(s.test) for s in features.body if isinstance(s, ast.If)] ==
            ['not _real_series(prices) or not _integer(t) or (not 24 <= t < len(prices))', 'not np.isfinite(p).all() or np.any(p <= 0)'] and
            ast.unparse(features.body[-1]) == 'return x if np.isfinite(x).all() else None', 'feature admission conditions')
    require(ast.unparse(trade.test) == 'self.failure is None and self.pending is not None and (self.pending[0] <= self.t)' and
            [ast.unparse(s) for s in trade.body][-1] == 'self.failure = self.failure or self._risk()', 'trade only when solvent-and-in-bounds, then re-check')
    terminal = next(s for s in loop[0].body if isinstance(s, ast.If) and ast.unparse(s.test) == 'terminal')
    require('terminal = self.failure is not None or self.t == self.stop - 1' in body and
            ast.unparse(terminal.body[1].test) == 'isfinite(self.equity) and self.equity > 0' and
            ast.unparse(terminal.body[1].body[0]) == 'liquidation = self._trade(0.0, terminal=True)' and
            ast.unparse(terminal.body[2]) == 'self.done = True', 'every failure is terminal with solvent liquidation')
    require(ast.unparse(loop[0].body[-1]) == 'if self.done:\n    break', 'terminal break')
    tr = _method(replay, '_trade')
    texts = {ast.unparse(s) for s in ast.walk(tr) if isinstance(s, (ast.Assign, ast.AugAssign))}
    require({'desired = target * e / p', 'new = desired if terminal else old + cfg.fill_fraction * (desired - old)',
             'turnover = abs(new - old) * p / e', 'cash = abs(new - old) * p',
             "self.equity -= sum((terms[k] for k in ('fee', 'spread', 'slippage', 'impact')))", 'self.units = new'} <= texts,
            'trade cost/turnover formulas')
    terms = next(s for s in ast.walk(tr) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == 'terms')
    require(ast.unparse(terms.value) == "{'turnover': cash, 'fee': cash * 0.0005 * cfg.cost_multiplier, 'spread': cash * 5e-05 * cfg.cost_multiplier, "
            "'slippage': cash * 0.00045 * cfg.cost_multiplier, 'impact': cash * cfg.impact_bps * 0.0001 * np.sqrt(turnover)}", 'registered cost terms')
    require(any(isinstance(s, ast.If) and ast.unparse(s.test) == 'not terminal and turnover > 0.5 + 1e-12' for s in ast.walk(tr)), 'turnover rejection')
    shield = _named(env, 'shield')
    require(any(ast.unparse(n) == 'action not in (-0.25, 0.0, 0.25)' for n in ast.walk(shield)), 'shield action set')
    runner = ast.parse(sources.get(RUNNER, (ROOT / RUNNER).read_text()))
    stresses = next(n for n in runner.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'STRESSES')
    found = {ast.literal_eval(k): {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords} for k, v in zip(stresses.value.keys, stresses.value.values)}
    require(found == STRESSES, 'registered stress maxima')
    return {'status': 'exhaustively_checked', 'riskClauses': len(RISK), 'stresses': len(found),
            'scope': 'unchanged frozen Replay step, risk, trade, shield and runner stress source; interpreter semantics assumed'}


def prove():
    u = z.RealVal('1/9007199254740992')
    d = z.Reals('bv_d1 bv_d2 bv_d3 bv_d4 bv_d5')
    small = z.And(*[z.And(x >= -u, x <= u) for x in d])
    E, x, r, psi, E1 = z.Reals('bv_E bv_x bv_r bv_psi bv_E1')
    # (I) The negated source predicate is exactly the registered realized bounds.
    eq, peak, units_px = z.Reals('bv_eq bv_peak bv_upx')
    clear = z.And(eq > 0, z.Not(eq < z.RealVal('4/5')), z.Not(1 - eq / peak > z.RealVal('3/20')), z.Not(z.Or(units_px / eq > z.RealVal('7/20'), units_px / eq < -z.RealVal('7/20'))), peak >= eq)
    certify(clear, z.And(eq >= z.RealVal('4/5'), eq >= z.RealVal('17/20') * peak, units_px <= z.RealVal('7/20') * eq, units_px >= -z.RealVal('7/20') * eq))
    # (III-a) One bar under A-GAP-BOUND keeps equity above 0.80x (gross and funding each rounded, sum rounded).
    bar = z.And(E > 0, x >= -z.RealVal('7/20'), x <= z.RealVal('7/20'), r >= -z.RealVal('1/2'), r <= z.RealVal('1/2'),
                psi >= -z.RealVal('6/100'), psi <= z.RealVal('6/100'), small,
                E1 == (E + E * x * r * (1 + d[0]) - E * x * psi * (1 + d[1])) * (1 + d[2]))
    certify(bar, E1 > z.RealVal('4/5') * E)
    # Exposure at detection is at most 0.66: |x|(1+|r|) E / E1 with E1 > 0.8 E; post-trade targets are smaller.
    xd = z.Real('bv_xd')
    certify(z.And(bar, xd * E1 == x * (1 + r) * E), z.And(xd <= z.RealVal('66/100'), xd >= -z.RealVal('66/100')))
    # (III-b) An admitted trade (turnover <= 1/2) and (III-c) a terminal liquidation (turnover <= 0.66) cost little.
    tau, s, cost, after = z.Reals('bv_tau bv_s bv_cost bv_after')
    m = z.RealVal('5/2') * (z.RealVal('5/10000') + z.RealVal('5/100000') + z.RealVal('45/100000'))
    trade = lambda limit: z.And(E1 > 0, tau >= 0, tau <= limit, s >= 0, s <= 1, small,
                                cost == (E1 * tau * m * (1 + d[3]) + E1 * tau * z.RealVal('10') / 10000 * s * (1 + d[4])),
                                after == (E1 - cost) * (1 + d[2]))
    certify(trade(z.RealVal('1/2')), after > z.RealVal('9982/10000') * E1)
    certify(trade(z.RealVal('66/100')), after > z.RealVal('9976/10000') * E1)
    # (II) Fresh full-fill target |w| <= 1/4 with admitted entry costs: post-cost exposure < 0.2505 <= 7/20.
    w, e2 = z.Reals('bv_w bv_e2')
    certify(z.And(E1 > 0, w >= -z.RealVal('1/4'), w <= z.RealVal('1/4'), e2 >= z.RealVal('9982/10000') * E1, e2 <= E1),
            z.And(w * E1 <= z.RealVal('2505/10000') * e2, w * E1 >= -z.RealVal('2505/10000') * e2))
    # Feature finiteness under |r| <= 1/2: every ratio over <= 24 bars lies in [2^-24, (3/2)^24].
    lo, hi = Q(1, 2) ** 24, Q(3, 2) ** 24
    require(lo > 0 and hi < Q(10) ** 300, 'bounded feature ratios')
    return {'F-RL-BOUNDS-COMPOSE': 'unsat'}


def _replay():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    from sequential_env import Execution, Replay, Scale
    return Execution, Replay, Scale


def conformance():
    Execution, Replay, Scale = _replay()
    reg = json.loads((ROOT / REGISTRATION).read_text())
    rng = np.random.default_rng(reg['probeSeed'])
    episodes = nonterminal = flat = breaches = 0
    for k in range(reg['probeEpisodes']):
        stress = list(STRESSES)[k % len(STRESSES)]
        cfg = Execution(**STRESSES[stress])
        n = 160
        gaps = rng.uniform(-0.5, 0.5, n) * (rng.random(n) < 0.15) + rng.normal(0, 0.01, n)
        gaps = np.clip(gaps, -0.5, 0.5)
        p = 100 * np.cumprod(1 + gaps)
        f = np.r_[0.0, p[:-1] * rng.uniform(-0.03, 0.03, n - 1)]  # |funding per unit| <= 3/100 of the prior price
        scale = Scale.fit([p[:60]])
        env = Replay(p, f, 70, 150, (1, 3, 6)[k % 3], scale, cfg, enabled=True)
        reject_at = int(rng.integers(3, 12)) if k % 4 == 3 else -1
        steps = 0
        while not env.done:
            steps += 1
            env.step(0.3 if steps == reject_at else float(rng.choice([-0.25, 0.0, 0.25])))
            if not env.done:
                exposure = abs(env.units * env.prices[env.t] / env.equity)
                require(env.equity >= 0.8 and 1 - env.equity / env.peak <= 0.15 and exposure <= 0.35,
                        'non-terminal state outside realized bounds')
                nonterminal += 1
        if env.failure in ('invalid_action', 'timeout', 'invalid_observation_or_position', 'disabled'):
            require(env.equity > 0 and abs(env.units * env.prices[env.t] / env.equity) <= 0.35, 'rejection left an out-of-bounds position')
        else:
            require(env.failure not in ('invalid_market_transition', 'invalid_observation'), 'data early exit reached under A-GAP-BOUND')
            require(env.equity > 0 and env.units == 0, 'terminal path not liquidated flat under A-GAP-BOUND')
            flat += 1
        breaches += env.failure in ('capital_floor', 'drawdown_limit', 'endpoint_exposure', 'turnover_limit')
        episodes += 1
    require(breaches > 0 and flat < episodes, 'vacuous probe: no bound breach or rejection exercised')
    gaps = json.loads((ROOT / GAPS).read_text())['entries']
    require({g['id'] for g in gaps} == {'CE-RL-002', 'CE-RL-003'} and
            all(abs(Q(g['priceRatio']) - 1) > Q(1, 2) for g in gaps), 'preserved floor witnesses lie outside A-GAP-BOUND')
    return {'status': 'property_tested', 'episodes': episodes, 'nonterminalChecks': nonterminal, 'flatTerminals': flat,
            'breachTerminations': breaches, 'seed': reg['probeSeed']}


def check_bounds():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require(reg['sourceChanges'] == 0 and reg['hardBounds']['maxAbsExposure'] == '7/20' and
            reg['gapAssumption']['maxAbsBarReturn'] == '1/2', 'registration')
    return {'source': bind(), 'smt': prove(), 'conformance': conformance()}
