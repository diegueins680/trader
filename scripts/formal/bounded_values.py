"""Obligation 7: realized-state invariant, proposal bounds and liquidation solvency under A-GAP-BOUND."""
import ast
from fractions import Fraction as Q
import json
import math
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


REPLAY_LOCK = 'formal/research/replay-source-lock.json'


def replay_lock(sources=None):
    """Complete AST lock of the reviewed Replay class and its helpers: any source change fails closed."""
    from promotion_boundary import shape
    import hashlib
    sources = sources or {}
    lock = json.loads((ROOT / REPLAY_LOCK).read_text())
    tree = ast.parse(sources.get(ENV, (ROOT / ENV).read_text()))
    for name, digest in lock['definitions'].items():
        found = [n for n in tree.body if getattr(n, 'name', None) == name]
        require(len(found) == 1 and hashlib.sha256(shape(found[0]).encode()).hexdigest() == digest,
                'reviewed source lock drift: ' + name)
    return len(lock['definitions'])


def bind(sources=None):
    sources = sources or {}
    locked = replay_lock(sources)
    env = ast.parse(sources.get(ENV, (ROOT / ENV).read_text()))
    replay = _named(env, 'Replay')
    risk = _method(replay, '_risk')
    require([ast.unparse(s) for s in risk.body] == RISK, 'registered hard-bound predicate')
    step = _method(replay, 'step')
    loop = [n for n in ast.walk(step) if isinstance(n, ast.While) and ast.unparse(n.test) == 'self.t < end']
    require(len(loop) == 1, 'single bar loop')
    body = [ast.unparse(s) for s in loop[0].body]
    # The exact operation sequence modeled by propagate(): one subtraction, three multiplications, two additions.
    require({'p0, p1 = (self.prices[left], self.prices[left + 1])', 'f = self.funding[left + 1]',
             'gross, funding = (self.units * (p1 - p0), -self.units * f * self.execution.funding_multiplier)'} <= set(body),
            'modeled mark-to-market operations')
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
    # The delivered loader builds binary64 market arrays; slices passed to Replay preserve the dtype.
    loader = _named(runner, 'load_development')
    texts = {ast.unparse(s) for s in ast.walk(loader) if isinstance(s, (ast.Assign, ast.AugAssign))}
    require({'p = rows.close.to_numpy(dtype=float)', 'f = np.zeros(len(p))', 'prices[symbol], funding[symbol] = (p, f)',
             'f[j] += event.fundingRate * event.resolvedMarkPrice'} <= texts, 'binary64 market-array provenance')
    stresses = next(n for n in runner.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'STRESSES')
    found = {ast.literal_eval(k): {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords} for k, v in zip(stresses.value.keys, stresses.value.values)}
    require(found == STRESSES, 'registered stress maxima')
    return {'status': 'exhaustively_checked', 'lockedDefinitions': locked, 'riskClauses': len(RISK), 'stresses': len(found),
            'scope': 'unchanged frozen Replay step, risk, trade, shield and runner stress source; interpreter semantics assumed'}


U = Q(1, 2**53)            # unit roundoff, round to nearest
H = Q(1, 2**1075)          # largest absolute error of a rounded (subnormal/zero) result
LOW_PRICE = Q(1, 2**400)   # A-GAP-BOUND range premise: prices and equity lie in [2^-400, 2^400]
HIGH = Q(2**400)


def dbl(x):
    """Exact rational value of a binary64 source literal."""
    return Q(float(x))


def generic_lemmas():
    """Z3 certificates for the only reasoning steps used by the propagation below."""
    u, h = z.RealVal(str(U)), z.RealVal(str(H))
    x, b, d, e, a, c, l = z.Reals('bv_x bv_b bv_d bv_e bv_a bv_c bv_l')
    rounding = z.And(d >= -u, d <= u, e >= -h, e <= h)
    # R1: one rounding of a bounded value.
    certify(z.And(rounding, b >= 0, x >= -b, x <= b), z.And(x * (1 + d) + e <= b * (1 + u) + h, x * (1 + d) + e >= -(b * (1 + u) + h)))
    # R2: one rounding of a value bounded below by l >= 0.
    certify(z.And(rounding, l >= 0, x >= l), x * (1 + d) + e >= l * (1 - u) - h)
    # P: product of bounded magnitudes.
    certify(z.And(a >= 0, b >= 0, x >= -a, x <= a, c >= -b, c <= b), z.And(x * c <= a * b, x * c >= -a * b))
    # S: sum of bounded magnitudes.
    certify(z.And(a >= 0, b >= 0, x >= -a, x <= a, c >= -b, c <= b), z.And(x + c <= a + b, x + c >= -(a + b)))
    # I: a computed comparison |fl(fl(n)/m)| <= t with m > 0 bounds the exact |n| (inversion of R1 twice).
    n, m, t, d2, e2 = z.Reals('bv_n bv_m bv_t bv_d2 bv_e2')
    q = (n * (1 + d) + e) / m * (1 + d2) + e2
    certify(z.And(rounding, d2 >= -u, d2 <= u, e2 >= -h, e2 <= h, m >= z.RealVal('4/5'), t >= 0, q <= t, q >= -t),
            z.And(n * (1 - u) * (1 - u) <= (t + 3 * h) * m, n * (1 - u) * (1 - u) >= -(t + 3 * h) * m))


class Bound:
    """|value| <= rel * E + abs, with E the reference equity (E >= 4/5)."""
    def __init__(self, rel, abs_=Q(0)):
        self.rel, self.abs = Q(rel), Q(abs_)

    def rounded(self):                          # R1
        return Bound(self.rel * (1 + U), self.abs * (1 + U) + H)

    def scaled(self, k):                         # P with a constant bound k
        return Bound(self.rel * k, self.abs * k)

    def plus(self, other):                       # S
        return Bound(self.rel + other.rel, self.abs + other.abs)

    def relative(self, floor=Q(4, 5)):           # abs <= abs/floor * E since E >= floor
        return self.rel + self.abs / floor


def propagate():
    floor, cap, half = dbl(0.8), dbl(0.35), dbl(0.5)
    turnover_cap = Q(float(0.50 + 1e-12))       # the source compares against the rounded sum
    fee, spread, slip, unit_bp = dbl(0.0005), dbl(5e-05), dbl(0.00045), dbl(0.0001)
    cost_mult, impact_bps, fund_mult = Q(5, 2), Q(10), Q(2)
    # Prior non-terminal state: computed exposure <= 0.35 bounds the exact |U*p0| (lemma I).
    x0 = (cap + 3 * H) / ((1 - U) * (1 - U))
    # gross = units * (p1 - p0): d = rnd(p1 - p0) with |p1 - p0| <= p0/2, then rnd(units * d).
    # |units| * H <= x0 * E * H / p0 <= x0 * E * H * 2^400, folded into the relative term.
    tiny = x0 * H / LOW_PRICE
    gross = Bound(x0 * Q(1, 2) * (1 + U) + tiny, Q(0)).rounded()
    # funding = -units * f * m: |units * f| <= x0 * E * 3/100, then two roundings.
    funding = Bound(x0 * Q(3, 100) + tiny).rounded().scaled(fund_mult).rounded()
    change = gross.plus(funding).rounded()      # rnd(gross + funding)
    # E1 = rnd(E + change) >= (E - |change|)(1 - u) - H     (lemma R2)
    e1_ratio = (1 - change.relative()) * (1 - U) - H * Q(5, 4)
    # Exposure at detection: |units * p1| <= 3/2 * x0 * E (exact), over E1 >= e1_ratio * E.
    detect = Q(3, 2) * x0 / e1_ratio
    # Accepted trade: computed turnover <= turnover_cap bounds the exact |new - old| * p (lemma I);
    # the stored cash is that product rounded once more (cost() rounds it).
    def cost(product_rel, sqrt_bound, floor):
        cash = Bound(product_rel).rounded()
        terms = [cash.scaled(c).rounded().scaled(cost_mult).rounded() for c in (fee, spread, slip)]
        impact = cash.scaled(impact_bps).rounded().scaled(unit_bp).rounded().scaled(sqrt_bound).rounded()
        total = terms[0].plus(terms[1]).rounded().plus(terms[2]).rounded().plus(impact).rounded()
        return (1 - total.relative(floor)) * (1 - U) - H / floor   # equity ratio after rnd(e - total), e >= floor
    trade_product = (turnover_cap + 3 * H) / ((1 - U) * (1 - U))
    after_trade = cost(trade_product, Q(1), Q(4, 5))   # trades run only after the mark check passed: e >= dbl(0.8)
    # Post-trade inventory (relative to the equity e at the trade, where the mark check passed: |old*p| <= x0*e).
    # |h * p| / e <= 2^-1075 * 2^400 * 5/4 under the range premise.
    hp = H / LOW_PRICE * Q(5, 4)
    desired = Q(1, 4) * (1 + U) * (1 + U) + hp * 2            # rnd(rnd(w * e) / p) * p
    diff = (desired + x0) * (1 + U) + hp                      # rnd(desired - old)
    filled = diff * (1 + U) + hp                              # rnd(fill_fraction * diff), fill <= 1
    post = ((x0 + filled) * (1 + U) + hp) / after_trade      # rnd(old + filled), over post-trade equity
    # Liquidation closes either the carried inventory (no trade this bar) or the post-trade inventory.
    liquidation_product = max(detect, post)
    # Liquidation equity can be below 4/5: after a breaching bar (>= e1_ratio * 4/5) or after a trade (>= after_trade * 4/5).
    liquidation_floor = Q(4, 5) * min(e1_ratio, after_trade)
    after_liquidation = cost(liquidation_product, Q(1), liquidation_floor)
    # Full-fill target |w| = 1/4 (fresh entry or rebalance): new = rnd(old + rnd(1 * rnd(desired - old))).
    # The multiplication by fill_fraction = 1 is exact; |desired - old| * p <= (desired + x0) * e.
    full_fill = (desired + U * (desired + x0) * (1 + U) + hp) * (1 + U) + hp
    # _risk evaluates the computed exposure fl(fl(new * p) / e2): round the product and the quotient.
    entry_exposure = ((full_fill * (1 + U) + H * Q(5, 4)) / after_trade) * (1 + U) + H
    # Range premise: every intermediate magnitude stays far below 2^1023 (no overflow) for any
    # admitted state, including desired = rnd(rnd(w*e)/p) computed on the missed-fill branch.
    e_max, p_min, p_max = HIGH, LOW_PRICE, HIGH
    magnitudes = {'w*e': Q(1, 4) * e_max, 'desired=w*e/p': Q(1, 4) * e_max / p_min * (1 + U) ** 2,
                  'units<=x0*e/p': x0 * e_max / p_min, 'post-trade units': post * e_max / p_min,
                  'cash=|dU|*p': (post + x0 + 1) * e_max, 'gross/funding': x0 * e_max * Q(3, 2),
                  'turnover': post + 1, 'equity': e_max * 2}
    largest = max(magnitudes.values())
    range_ok = largest < Q(2) ** 1000
    checks = {'equityAfterBar': e1_ratio > Q(4, 5), 'detectionExposure': detect < Q(66, 100), 'turnoverSqrtArgument': liquidation_product * (1 + U) * (1 + U) + 3 * H < 1,
              'afterTrade': after_trade > Q(998, 1000), 'afterLiquidation': after_liquidation > Q(996, 1000),
              'entryExposure': entry_exposure < Q(2506, 10000) < cap, 'postTradeInventory': post < 1, 'floorLiteral': floor > Q(4, 5), 'noOverflow': range_ok, 'drawdownWithinRegistered': drawdown_bound()[1] < Q(3, 20) + Q(1, 2**50), 'halfLiteral': half == Q(1, 2)}
    require(all(checks.values()), 'exact bound propagation failed: ' + str([k for k, v in checks.items() if not v]))
    return {'realizedDrawdownBound': float(drawdown_bound()[1]), 'largestMagnitudeLog2': float(math.log2(largest)), 'liquidationEquityFloor': float(liquidation_floor), 'equityAfterBarRatio': float(e1_ratio), 'detectionExposure': float(detect), 'postTradeInventory': float(post), 'afterTradeRatio': float(after_trade),
            'afterLiquidationRatio': float(after_liquidation), 'entryExposure': float(entry_exposure), 'checks': len(checks)}


def drawdown_bound():
    """Exact bound implied by the computed check not(fl(1 - fl(eq/peak)) > dbl(0.15))."""
    c = dbl(0.15)
    ratio = (1 - (c + H) / (1 - U) - H) / (1 + U)
    return ratio, 1 - ratio


def prove():
    generic_lemmas()
    # (I) Realized bounds implied by a passing _risk() on the stored doubles. The floor compares the stored
    # equity exactly; exposure is the computed value v (its exact meaning is bounded by lemma I as x0);
    # drawdown rounds a division and a subtraction: q = fl(eq/peak), t = fl(1 - q).
    u, h = z.RealVal(str(U)), z.RealVal(str(H))
    eq, peak, v, rho, q, t, d1, d2, e1, e2 = z.Reals('bv_eq bv_peak bv_v bv_rho bv_q bv_t bv_d1 bv_d2 bv_e1 bv_e2')
    floor, cap, dd = (z.RealVal(str(dbl(x))) for x in (0.8, 0.35, 0.15))
    ratio, _ = drawdown_bound()
    rounding = z.And(*[z.And(x >= -u, x <= u) for x in (d1, d2)], *[z.And(x >= -h, x <= h) for x in (e1, e2)])
    computed = z.And(eq > 0, peak >= eq, rho * peak == eq, rounding,
                     q == rho * (1 + d1) + e1, t == (1 - q) * (1 + d2) + e2,
                     z.Not(eq < floor), z.Not(t > dd), z.Not(z.Or(v > cap, v < -cap)))
    certify(computed, z.And(eq >= floor, rho >= z.RealVal(str(ratio)), eq >= z.RealVal(str(ratio)) * peak, v <= cap, v >= -cap))
    return {'F-RL-BOUNDS-COMPOSE': 'unsat'}, propagate()


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
        require(p.dtype == np.float64, 'probe uses the delivered binary64 dtype')
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
    smt, propagation = prove()
    return {'source': bind(), 'smt': smt, 'propagation': propagation, 'conformance': conformance()}
