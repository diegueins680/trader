"""Size certificate for ope-rational-v3: its rational limit is unreachable on every admitted input."""
import ast
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import z3 as z
import ope_rational as predecessor
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/ope_rational_v3.py'
PREDECESSOR = 'scripts/research/ope_rational_v2.py'
REGISTRY = 'formal/research/ope-rational-v3-source.json'
REGISTRATION = 'research-notes/registrations/ope-rational-v3-engineering.json'
EPISODES, DECISIONS, LIMIT = 256, 32, 2 ** 20
# The successor is the reviewed predecessor with exactly these three tokens replaced.
SUBSTITUTIONS = (('"ope-rational-v3"', '"ope-rational-v2"'), ('def estimate_v3(', 'def estimate_v2('),
                 ('MAX_BITS = %d' % LIMIT, 'MAX_BITS = 8192'))
# Every finite binary64 value is an integer over 2^1074 and below 2^1024 in magnitude; one in (0, 1] has a
# reduced numerator below 2^53.
MANTISSA, SCALE, MAGNITUDE = 53, 1074, 1024


def require(ok, reason):
    if not ok: raise ValueError('OPE capacity: ' + reason)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def registry_for(source, previous=None):
    previous = (ROOT / PREDECESSOR).read_text() if previous is None else previous
    return {'schemaVersion': 1, 'astSha256': sha(ast.dump(ast.parse(source), include_attributes=False)),
            'predecessorSha256': sha(previous)}


def extract(source=None, registry=None, previous=None):
    source = (ROOT / SOURCE).read_text() if source is None else source
    previous = (ROOT / PREDECESSOR).read_text() if previous is None else previous
    registry = json.loads((ROOT / REGISTRY).read_text()) if registry is None else registry
    try:
        tree = ast.parse(source)
    except SyntaxError:
        raise ValueError('OPE capacity: unparsable source')
    require(registry == registry_for(source, previous), 'reviewed source or predecessor drift')
    restored = source
    for new, old in SUBSTITUTIONS:
        require(restored.count(new) == 1 and restored.count(old) == 0, 'substitution site ' + new)
        restored = restored.replace(new, old)
    require(ast.dump(ast.parse(restored)) == ast.dump(ast.parse(previous)), 'difference from the reviewed predecessor')
    # The predecessor's own lock, activation, admission, checked-primitive and recurrence review now applies verbatim.
    nodes, surface = predecessor.extract(restored)
    constants = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    require(constants == {'VERSION': "'ope-rational-v3'", 'MAX_EPISODES': str(EPISODES), 'MAX_DECISIONS': str(DECISIONS),
                          'MAX_BITS': str(LIMIT)}, 'bounds/version')
    require(not any(isinstance(n, (ast.Import, ast.ImportFrom)) and
                    (n.module if isinstance(n, ast.ImportFrom) else n.names[0].name) not in
                    ('__future__', 'dataclasses', 'fractions', 'math') for n in ast.walk(tree)), 'import list')
    return nodes, {'status': 'exhaustively_checked', 'definitions': surface['definitions'],
                   'substitutions': len(SUBSTITUTIONS), 'limitBits': LIMIT, 'predecessor': 'ope-rational-v2',
                   'scope': 'complete module equal to the reviewed predecessor up to three tokens; primitive runtime semantics assumed'}


# Certificate: for positive integers m_a < 2^width(a) there is L = prod m_a^count(a) * 2^two with x*L an integer,
# and |x| <= 2^mag. The reduced denominator then divides L, so it has at most lam+1 bits and the numerator mag+lam+1.
@dataclass(frozen=True)
class Cert:
    odd: object = ()
    two: int = 0
    mag: int = 0
    inverse: object = None

    def lam(self):
        return sum(width * count for (_, width), count in dict(self.odd).items()) + self.two

    def bits(self):
        return self.mag + self.lam() + 1, self.lam() + 1


@dataclass(frozen=True)
class Final:
    numerator: int
    denominator: int

    def bits(self):
        return self.numerator, self.denominator


def _merge(a, b, combine):
    out = dict(a)
    for atom, count in dict(b).items():
        out[atom] = combine(out[atom], count) if atom in out else count
    return out


def mul(x, y):
    return Cert(_merge(x.odd, y.odd, int.__add__), x.two + y.two, x.mag + y.mag)


def add(x, y):
    return Cert(_merge(x.odd, y.odd, max), max(x.two, y.two), max(x.mag, y.mag) + 1)


def div(x, y):
    if y.inverse is not None:
        return mul(x, y.inverse)
    # Nonzero divisor over a common multiplier L: x/y = (xL)/(yL), numerator |x|L and denominator |y|L at most.
    lam = add(x, y).lam()
    return Final(x.mag + lam + 1, y.mag + lam + 1)


ONE = Cert()
FLOAT = Cert((), SCALE, MAGNITUDE)
UNIT = Cert((), SCALE, 0)


def lift(value, atom):
    # Weakening: every mantissa of the episode to the largest count, as one atom for their product.
    return Cert({atom: max(value.odd.values())} if value.odd else (), value.two, value.mag)


class Trace:
    def __init__(self):
        self.calls = []

    def evaluate(self, node, atoms):
        label = ast.unparse(node)
        if label in atoms: return atoms[label]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return self.evaluate(node.operand, atoms)
        require(isinstance(node, ast.Call) and ast.unparse(node.func) in ('_add', '_mul', '_div') and
                len(node.args) == 2 and not node.keywords, 'unsupported expression ' + label)
        a, b = (self.evaluate(n, atoms) for n in node.args)
        require(type(a) is Cert and type(b) is Cert, 'arithmetic after a final quotient')
        value = {'_add': add, '_mul': mul, '_div': div}[ast.unparse(node.func)](a, b)
        self.calls.append(value)
        return value


def episode(nodes, horizon):
    assignments, _, result, _ = nodes
    trace = Trace()
    state = {'weight': ONE, 'discount': ONE, 'total': ONE, 'pdis': ONE, 'dr': FLOAT, 'gamma': UNIT}
    for t in range(horizon):
        behavior = Cert((), SCALE, 0, Cert((((('m', t), MANTISSA), 1),), 0, SCALE))
        state.update(reward=FLOAT, behavior=behavior, target=UNIT, q=FLOAT, next_v=FLOAT)
        for name, node in assignments.items():
            state[name] = trace.evaluate(node, state)
    require([ast.unparse(n) for n in result.args] == ['total', 'weight', '_mul(weight, total)', 'pdis', 'dr'], 'episode result')
    values = [trace.evaluate(n, state) for n in result.args]
    return dict(zip(('discounted_return', 'weight', 'ordinary_is', 'per_decision_is', 'doubly_robust'), values)), trace.calls


def walk(nodes, horizon, total):
    """For each batch size up to `total`: the certificates of that episode's checked calls and of the publication quotients."""
    _, aggregates, _, statements = nodes
    values, calls = episode(nodes, horizon)
    publication = statements[-1].value
    finals = [next(n.value for n in statements if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name).body
              for name in ('wis', 'ess')] + [publication.args[i] for i in (1, 2, 4)]
    require([ast.unparse(n) for n in finals] == ['_div(ordinary, weights)', '_div(_mul(weights, weights), squares)',
                                                 '_div(ordinary, count)', '_div(per_decision, count)',
                                                 '_div(robust, count)'], 'publication quotients')
    state = dict.fromkeys(('weights', 'squares', 'ordinary', 'per_decision', 'robust'), ONE)
    for index in range(total):
        atom = (('episode', index), MANTISSA * horizon)
        state.update({'value.' + name: lift(value, atom) for name, value in values.items()})
        trace = Trace()
        for name in ('weights', 'squares', 'ordinary', 'per_decision', 'robust'):
            state[name] = trace.evaluate(aggregates[name], state)
        final = Trace()
        state['count'] = Cert((), 0, (index + 1).bit_length())
        for node in finals: final.evaluate(node, state)
        yield index + 1, calls + trace.calls, final.calls


def batch(nodes, horizon, count):
    """(certificate, episode index) of every checked call of one admitted batch, in execution order."""
    out = []
    for size, calls, final in walk(nodes, horizon, count):
        out += [(value, size - 1) for value in calls]
    return out + [(value, None) for value in final]


def bits(nodes, horizon, count):
    return max(max(value.bits()) for value, _ in batch(nodes, horizon, count))


def holds(value, certificate, index, numerators):
    """The certificate's own claim on a concrete checked value: integral under its multiplier and within its magnitude."""
    n, d = certificate.bits()
    if value.numerator.bit_length() > n or value.denominator.bit_length() > d: return False
    if type(certificate) is Final: return True
    multiplier = 2 ** certificate.two
    for ((kind, key), _), count in dict(certificate.odd).items():
        multiplier *= (numerators[index][key] if kind == 'm' else math.prod(numerators[key])) ** count
    return multiplier % value.denominator == 0 and abs(value.numerator) <= value.denominator << certificate.mag


def lemmas():
    """The rule each certificate step uses, as polynomial facts over the reals; integrality is closure of Z."""
    x, y, a, b, n, d, l, l1, l2, u, w, q, s, t, c, bx, by, m, p = z.Reals(
        'oc_x oc_y oc_a oc_b oc_n oc_d oc_l oc_l1 oc_l2 oc_u oc_w oc_q oc_s oc_t oc_c oc_bx oc_by oc_m oc_p')
    within = lambda value, bound: z.And(-bound <= value, value <= bound)
    # Reduced form: n/d with Bezout coefficients s, t and n*l = d*q (x*l integral) gives l = d*(s*q + t*l).
    certify(z.And(n * l == d * q, s * n + t * d == 1), l == d * (s * q + t * l))
    certify(z.And(d >= 1, c >= 1, l == d * c, bx >= 0, within(x, bx), n == x * d), z.And(d <= l, within(n, bx * l)))
    certify(z.And(a == x * l1, b == y * l2, bx >= 0, by >= 0, within(x, bx), within(y, by)),
            z.And(x * y * (l1 * l2) == a * b, within(x * y, bx * by)))
    certify(z.And(a == x * l1, b == y * l2, l == l1 * u, l == l2 * w, bx >= 0, by >= 0, within(x, bx), within(y, by),
                  c >= bx, c >= by), z.And((x + y) * l == a * u + b * w, within(x + y, 2 * c)))
    certify(z.And(a == x * l, b == y * l, y != 0, l >= 1, bx >= 0, by >= 0, within(x, bx), within(y, by)),
            z.And(x / y * b == a, b != 0, within(a, bx * l), within(b, by * l)))
    # A binary64 value in (0, 1] is m / p with p a power of two and 1 <= m < 2^53, at least 2^-1074.
    low = z.RealVal(Fraction(1, 2 ** SCALE))
    certify(z.And(x * p == m, m >= 1, p >= 1, x >= low, x <= 1), z.And(1 / x * m == p, 1 / x <= 1 / low, 1 / x >= 1))
    # Control: without coprimality the denominator is not bounded by the multiplier.
    solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
    solver.add(n * l == d * q, n == 2, d == 2, l == 1, q == 1, d > l)
    require(solver.check() == z.sat, 'unreduced control')
    return {'certifiedImplications': 6, 'satisfiableControls': 1}


def prove(nodes, limit=LIMIT):
    summary = lemmas()
    peak, worst, peaks = 0, None, {}
    for horizon in range(1, DECISIONS + 1):
        running = 0
        for count, calls, final in walk(nodes, horizon, EPISODES):
            running = max(running, max(max(c.bits()) for c in calls))
            value = max(running, max(max(c.bits()) for c in final))
            peaks[(count, horizon)] = value
            require(value <= limit, 'certified bound exceeds the limit')
            if value > peak: peak, worst = value, (count, horizon)
    require(len(peaks) == EPISODES * DECISIONS and worst == (EPISODES, DECISIONS), 'configuration coverage')
    require(all(peaks[(n, h)] <= peaks[(min(n + 1, EPISODES), h)] and peaks[(n, h)] <= peaks[(n, min(h + 1, DECISIONS))]
                for n, h in peaks), 'monotone bound')
    calls = len(batch(nodes, DECISIONS, EPISODES))
    return {'F-RL-OPE-V3-ARITH': 'unsat'}, dict(summary, limitBits=limit, certifiedPeakBits=peak, slackBits=limit - peak,
                                                  configurations=len(peaks), checkedCallsAtCapacity=calls,
                                                  smallestConfigurationBits=peaks[(1, 1)], episodes=EPISODES, decisions=DECISIONS)


def _instrumented(impl, batch_, gamma):
    seen, inner = [], impl._bounded

    def spy(value):
        seen.append(value)
        return inner(value)
    impl._bounded = spy
    try:
        return impl.estimate_v3(batch_, gamma, enabled=True), seen
    finally:
        impl._bounded = inner


def _dyadic(value):
    exact = Fraction.from_float(value)
    return (exact.denominator & (exact.denominator - 1) == 0 and exact.denominator <= 2 ** SCALE and
            abs(exact) < 2 ** MAGNITUDE and (not 0 < exact <= 1 or exact.numerator < 2 ** MANTISSA))


def families(rng):
    top, tiny, least = math.nextafter(math.inf, 0.0), 5e-324, math.nextafter(1.0, 0.0)
    dense = lambda: (rng.getrandbits(52) | 2 ** 52 | 1) / 2.0 ** 53
    subnormal = lambda: (rng.getrandbits(51) | 2 ** 51 | 1) * tiny
    signed = lambda f: lambda: rng.choice((-1.0, 1.0)) * f()
    return {
        'uniform': (lambda: rng.uniform(2.0 ** -10, 1.0), rng.random, lambda: rng.gauss(0.0, 1.0), 0.99),
        'dense': (dense, dense, signed(dense), least),
        'subnormal': (subnormal, subnormal, signed(subnormal), least),
        'extreme': (lambda: tiny, lambda: 1.0, signed(lambda: top), 1.0),
        'mixed': (lambda: rng.choice((tiny, dense(), subnormal(), 1.0)), lambda: rng.choice((least, dense(), subnormal(), 1.0)),
                  signed(lambda: rng.choice((top, dense(), subnormal(), 2.0 ** 99))), least),
    }


def make(impl, rng, family, count, horizon):
    behavior, target, value, gamma = family
    draw = lambda f, k: tuple(f() for _ in range(k))
    return tuple(impl.Episode(draw(value, horizon), draw(lambda: rng.randrange(3), horizon), draw(behavior, horizon),
                              draw(target, horizon), draw(value, horizon), draw(value, horizon) + (0.0,))
                 for _ in range(count)), gamma


def fields(estimate):
    return (estimate.ordinary_is, estimate.per_decision_is, estimate.weighted_is, estimate.doubly_robust,
            estimate.effective_sample_size,
            tuple((e.discounted_return, e.weight, e.ordinary_is, e.per_decision_is, e.doubly_robust) for e in estimate.episodes))


def conformance(nodes, reg):
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import ope_rational_v2 as old
    import ope_rational_v3 as impl
    rng = random.Random(reg['seed'])
    require(all(_dyadic(x) for x in (5e-324, 2.2250738585072014e-308, math.nextafter(1.0, 0.0), 1.0,
                                     math.nextafter(math.inf, 0.0), -math.nextafter(math.inf, 0.0), 0.0, 0.99)), 'binary64 format')
    oracle = agree = admitted = 0
    for _ in range(reg['oracleCases']):
        count, horizon = rng.randrange(1, 9), rng.randrange(1, 9)
        name = rng.choice(('uniform', 'dense'))
        episodes, gamma = make(impl, rng, families(rng)[name], count, horizon)
        got = impl.estimate_v3(episodes, gamma, enabled=True)
        require(got is not None and got.version == 'ope-rational-v3' and got.reliable is False and
                got == impl.estimate_v3(episodes, gamma, enabled=True), 'deterministic descriptive envelope')
        require(fields(got) == predecessor.oracle(episodes, gamma), 'exact independent oracle'); oracle += 1
        require(impl.estimate_v3(episodes, gamma) is None and
                impl.estimate_v3(episodes, gamma, enabled=True, version='ope-rational-v2') is None, 'disabled or wrong version')
        before = old.estimate_v2(tuple(old.Episode(e.rewards, e.actions, e.behavior, e.target, e.q, e.v) for e in episodes),
                                 gamma, enabled=True)
        if before is not None:
            require(fields(before) == fields(got), 'predecessor agreement'); agree += 1
    require(agree >= reg['oracleCases'] // 8, 'vacuous predecessor agreement')
    # Every checked value of every instrumented run stays within the certified bound of its own call.
    checked = runs = 0
    ratio = Fraction(0)
    for name in reg['families']:
        for count, horizon in reg['boundConfigurations']:
            family = families(rng)[name]
            episodes, gamma = make(impl, rng, family, count, horizon)
            require(all(_dyadic(x) for e in episodes for xs in (e.rewards, e.behavior, e.target, e.q, e.v) for x in xs),
                    'binary64 format')
            got, seen = _instrumented(impl, episodes, gamma)
            bound = batch(nodes, horizon, count)
            require(got is not None and len(seen) == len(bound), 'instrumented call sequence')
            numerators = [[Fraction.from_float(b).numerator for b in e.behavior] for e in episodes]
            for value, (certificate, index) in zip(seen, bound):
                require(holds(value, certificate, index, numerators), 'certified per-call bound')
                ratio = max(ratio, Fraction(max(value.numerator.bit_length(), value.denominator.bit_length()),
                                            max(certificate.bits())))
            checked += len(seen); runs += 1; admitted += 1
    require(ratio > Fraction(1, 2), 'vacuous bound')
    # Capacity: what the predecessor rejected is admitted, and the limit is still evaluated.
    count, horizon = reg['capacityConfiguration']
    episodes, gamma = make(impl, rng, families(rng)['uniform'], count, horizon)
    require(old.estimate_v2(tuple(old.Episode(e.rewards, e.actions, e.behavior, e.target, e.q, e.v) for e in episodes),
                            gamma, enabled=True) is None, 'predecessor capacity rejection')
    got, seen = _instrumented(impl, episodes, gamma)
    peak = max(max(value.numerator.bit_length(), value.denominator.bit_length()) for value in seen)
    require(got is not None and len(got.episodes) == count and 1 <= got.effective_sample_size <= count and
            8192 < peak <= LIMIT, 'full capacity admission')
    count, horizon = reg['oracleCapacityConfiguration']
    episodes, gamma = make(impl, rng, families(rng)['uniform'], count, horizon)
    got = impl.estimate_v3(episodes, gamma, enabled=True)
    require(got is not None and fields(got) == predecessor.oracle(episodes, gamma), 'oracle beyond the predecessor capacity')
    saved = impl.MAX_BITS
    try:
        impl.MAX_BITS = 8192
        require(impl.estimate_v3(episodes, gamma, enabled=True) is None, 'limit is still evaluated')
    finally:
        impl.MAX_BITS = saved
    episodes, gamma = make(impl, rng, families(rng)['uniform'], 2, 3)
    swap = lambda xs, i, x: xs[:i] + (x,) + xs[i + 1:]
    first = episodes[0]
    bad = [
        (episodes * 200, gamma), ((), gamma), (list(episodes), gamma), (episodes, 1), (episodes, 1.5), (episodes, math.nan),
        (swap(episodes, 0, None), gamma),
        (swap(episodes, 0, impl.Episode(first.rewards[:-1], first.actions, first.behavior, first.target, first.q, first.v)), gamma),
        (swap(episodes, 1, impl.Episode(first.rewards, first.actions, swap(first.behavior, 0, 0.0), first.target, first.q, first.v)), gamma),
        (swap(episodes, 1, impl.Episode(first.rewards, first.actions, first.behavior, swap(first.target, 1, 1.5), first.q, first.v)), gamma),
        (swap(episodes, 1, impl.Episode(swap(first.rewards, 0, math.inf), first.actions, first.behavior, first.target, first.q, first.v)), gamma),
        (swap(episodes, 1, impl.Episode(first.rewards, swap(first.actions, 0, 3), first.behavior, first.target, first.q, first.v)), gamma),
        (swap(episodes, 1, impl.Episode(first.rewards, first.actions, first.behavior, first.target, first.q, swap(first.v, 3, 0.5))), gamma),
        (swap(episodes, 1, old.Episode(first.rewards, first.actions, first.behavior, first.target, first.q, first.v)), gamma),
        (make(impl, rng, families(rng)['uniform'], 1, DECISIONS + 1)[0], gamma),
    ]
    for arguments in bad:
        require(impl.estimate_v3(*arguments, enabled=True) is None, 'invalid input must reject')
    zero = tuple(impl.Episode(e.rewards, e.actions, e.behavior, (0.0,) * 3, e.q, e.v) for e in episodes)
    got = impl.estimate_v3(zero, gamma, enabled=True)
    require(got is not None and got.weighted_is is None and got.effective_sample_size == 0 and
            got.nonzero_trajectories == 0, 'no-support publication')
    return {'status': 'property_tested', 'oracleCases': oracle + 1, 'predecessorAgreements': agree, 'instrumentedRuns': runs,
            'checkedCalls': checked, 'tightestCallPercent': int(ratio * 100), 'capacityEpisodes': reg['capacityConfiguration'][0],
            'capacityDecisions': reg['capacityConfiguration'][1], 'observedCapacityBits': peak, 'rejections': len(bad) + 2,
            'limitControls': 1, 'seed': reg['seed'], 'marketDataReads': 0, 'statisticalReliabilityVerified': False}


def check_ope_capacity():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require((reg['version'], reg['enabledByDefault'], reg['productionIntegration'], reg['financialTrials'],
             reg['historicalDataReads'], reg['holdoutOpened']) == ('ope-rational-v3', False, False, 0, 0, False), 'registration')
    require((reg['maximumEpisodes'], reg['maximumDecisions'], reg['maximumRationalBits']) == (EPISODES, DECISIONS, LIMIT),
            'registered bounds')
    nodes, source = extract()
    smt, bounds = prove(nodes)
    require(bounds['certifiedPeakBits'] == reg['certifiedPeakBits'], 'certified bound matches the registration')
    model = predecessor.check_model()
    try:
        predecessor.check_model(mutant=True)
    except ValueError:
        model['mutantsRejected'] = 1
    else:
        raise ValueError('OPE capacity: partial-publication mutant accepted')
    return {'source': source, 'smt': smt, 'bounds': bounds, 'model': model, 'conformance': conformance(nodes, reg)}


if __name__ == '__main__':
    print(json.dumps(check_ope_capacity(), indent=1, sort_keys=True))
