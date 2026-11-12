"""Source-bound PPO objective v2: rounding-model bounds, branch-selected coefficient, model, oracle and witnesses."""
import ast
from collections import deque
import decimal
from decimal import Decimal as D
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import z3 as z
from promotion_boundary import shape
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/ppo_objective_v2.py'
REGISTRY = 'formal/research/ppo-objective-v2-source.json'
REGISTRATION = 'research-notes/registrations/ppo-objective-v2-engineering.json'
CONTRACT = 'formal/research/ppo-objective-v2-contract.md'
COUNTEREXAMPLES = 'formal/research/ppo-counterexamples.json'
CONSTANTS = {'VERSION': "'ppo-objective-v2'", 'ACTIONS': '3', 'MAX_ROWS': '256', 'BOUND': '2.0 ** 100',
             'MIN_PROB': '2.0 ** (-100)', 'TINY': '2.0 ** (-1022)', 'LOW': '0.8', 'HIGH': '1.2',
             '__all__': "['VERSION', 'Objective', 'objective_v2']"}
FINITE_LIMIT = 2 ** 220


def require(ok, reason):
    if not ok:
        raise ValueError('ppo objective v2: ' + reason)


def registry_for(source):
    tree = ast.parse(source)
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    return {'astSha256': hashlib.sha256(shape(tree).encode()).hexdigest(),
            'definitions': {k: shape(v) for k, v in nodes.items()}}


def extract(source=None, registry=None):
    text = (ROOT / SOURCE).read_text() if source is None else source
    tree = ast.parse(text)
    lock = json.loads((ROOT / REGISTRY).read_text()) if registry is None else registry
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    require(len(nodes) == sum(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body), 'duplicate definition')
    current = registry_for(text)
    require(lock.get('schemaVersion') == 1 and current['definitions'] == lock['definitions'], 'reviewed definition drift')
    require(current['astSha256'] == lock['astSha256'], 'module effect drift')
    for path, value in lock['supportHashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value, 'support drift: ' + path)
    constants = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    require(constants == CONSTANTS, 'version/bounds/exports')
    require([ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))] ==
            ['from __future__ import annotations', 'from dataclasses import dataclass', 'import math'],
            'scalar-only import boundary (no NumPy/BLAS)')
    require([ast.unparse(n) for n in nodes['Objective'].decorator_list] == ['dataclass(frozen=True, slots=True)'], 'immutable output')
    entry = nodes['objective_v2']
    defaults = dict(zip((x.arg for x in entry.args.kwonlyargs), map(ast.unparse, entry.args.kw_defaults)))
    require(defaults == {'enabled': 'False', 'version': 'VERSION'}, 'disabled default')
    require(ast.unparse(entry.body[0]) == "if enabled is not True or type(version) is not str or version != VERSION:\n    return None", 'activation first')
    require(ast.unparse(entry.body[1].body[0]) == 'if not _admissible(logits, actions, old_prob, advantage):\n    return None' and
            ast.unparse(entry.body[1].body[1]) == 'return _compute(logits, actions, old_prob, advantage)' and len(entry.body) == 2,
            'admission before computation')
    require(ast.unparse(nodes['_value'].body[0].value) == 'type(x) is float and math.isfinite(x) and (abs(x) <= BOUND)', 'numeric admission')
    require('if type(old) is not float or not MIN_PROB <= old <= 1.0:\n    return False' in
            [ast.unparse(s) for s in ast.walk(nodes['_admissible']) if isinstance(s, ast.If)], 'old-probability floor')
    row = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in nodes['_row'].body if isinstance(n, ast.Assign)}
    require(row == {'m': 'max(row)', 'terms': '[math.exp(x - m) for x in row]',
                    'underflow': 'sum((1 for t in terms if t < TINY))', 'total': 'math.fsum(terms)',
                    'probs': '[t / total for t in terms]', 'ratio': 'probs[action] / old',
                    'clipped': 'min(max(ratio, LOW), HIGH)', 'plain': 'ratio * adv', 'bounded': 'clipped * adv',
                    'active': 'adv >= 0.0 and ratio <= HIGH or (adv < 0.0 and ratio >= LOW)',
                    'coefficient': 'plain / n if active else 0.0', 'grad': '[]'}, 'difference-only softmax and branch-selected coefficient')
    require(ast.unparse(nodes['_row'].body[-1]) == 'return (min(plain, bounded), grad, underflow)', 'clipped surrogate term')
    loop = next(s for s in nodes['_row'].body if isinstance(s, ast.For))
    require([ast.unparse(s) for s in loop.body] == ['delta = p - (1.0 if j == action else 0.0)', 'g = delta * coefficient',
                                                    'underflow += _lost(delta, coefficient, g)', 'grad.append(g)'], 'gradient row')
    compute = [ast.unparse(s) for s in nodes['_compute'].body]
    require('loss = -math.fsum(terms) / n' in compute and 'loss += 0.0' in compute and
            'published = tuple((tuple((g + 0.0 for g in r)) for r in grads))' in compute and
            compute[-2] == 'if not math.isfinite(loss) or not all((math.isfinite(g) for r in published for g in r)):\n    return None' and
            compute[-1] == 'return Objective(VERSION, loss, published, underflow, n)', 'fsum mean, zero canonicalization and finite guard')
    require(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == 'Objective' for n in ast.walk(tree)) == 1, 'single publication')
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    require(not names & {'np', 'numpy', 'eval', 'exec', 'open', 'socket', 'order', 'submit', 'promote'}, 'effect-free source')
    return nodes, {'status': 'exhaustively_checked', 'definitions': len(nodes), 'defaultDisabledEntries': 1,
                   'scope': 'complete reviewed AST, difference-only softmax, branch-selected coefficient and finite guard; runtime semantics assumed'}


def ceil_log2(value):
    """Smallest integer k with value <= 2^k, by exact rational comparison (no float rounding)."""
    require(type(value) is Q and value > 0, 'positive exact bound')
    k = (value.numerator.bit_length() - value.denominator.bit_length())
    while Q(2) ** k < value:
        k += 1
    while Q(2) ** (k - 1) >= value:
        k -= 1
    return k


def chain():
    # Exact rational propagation of the per-operation (1+u) factor along the source expressions.
    u = Q(1, 2**53); g = 1 + u; bound = Q(2**100); floor = Q(1, 2**100); n = 256
    prob = g                                  # t/total with 0 <= t <= 1 <= total, one rounding
    ratio = prob / floor * g                  # probs[action] / old
    plain = ratio * bound * g                 # ratio * adv
    clipped = Q(12, 10) * g * bound * g       # HIGH (nearest double within one rounding) times adv
    term = max(plain, clipped)                # |min(plain, bounded)| <= max of the magnitudes
    loss = n * term * g / n * g               # fsum of n terms (one rounding), then /n
    coefficient = plain * g                   # plain / n with n >= 1
    gradient = (prob + 1) * g * coefficient * g
    worst = max(ratio, plain, loss, coefficient, gradient)
    require(worst < FINITE_LIMIT < Q(int(sys.float_info.max)), 'admitted-domain magnitude chain')
    return {'ratioBoundLog2': ceil_log2(ratio), 'lossBoundLog2': ceil_log2(loss),
            'gradientBoundLog2': ceil_log2(gradient), 'finiteLimitLog2': 220}


def prove(nodes=None):
    u = z.RealVal(str(Q(1, 2**53)))
    x, cap, delta = z.Reals('po_x po_cap po_delta')
    # Standard model: one rounding scales any magnitude bound by at most (1+u).
    certify(z.And(cap >= 0, x <= cap, x >= -cap, delta <= u, delta >= -u),
            z.And(x * (1 + delta) <= cap * (1 + u), x * (1 + delta) >= -cap * (1 + u)))
    # The row maximum makes every exponent argument nonpositive and one of them zero.
    q0, q1, q2, m = z.Reals('po_q0 po_q1 po_q2 po_m')
    is_max = z.And(m >= q0, m >= q1, m >= q2, z.Or(m == q0, m == q1, m == q2))
    certify(is_max, z.And(q0 - m <= 0, q1 - m <= 0, q2 - m <= 0, z.Or(q0 - m == 0, q1 - m == 0, q2 - m == 0)))
    t0, t1, t2 = z.Reals('po_t0 po_t1 po_t2')
    terms = z.And(*[z.And(t >= 0, t <= 1) for t in (t0, t1, t2)], z.Or(t0 == 1, t1 == 1, t2 == 1))
    certify(terms, z.And(t0 + t1 + t2 >= 1, t0 + t1 + t2 <= 3))
    # A probability is in [0,1]; with the old-probability floor the exact ratio is at most 1/floor.
    p, total, old, floor, ratio = z.Reals('po_p po_total po_old po_floor po_ratio')
    certify(z.And(total >= 1, p >= 0, p <= total, floor > 0, old >= floor, old <= 1, ratio * old == p / total),
            z.And(ratio >= 0, ratio * floor <= 1))
    # Clipping lands in [LOW, HIGH]; the surrogate term never exceeds the larger factor magnitude.
    low, high, clipped, adv, term, big = z.Reals('po_low po_high po_clipped po_adv po_term po_big')
    clip = z.And(low > 0, low < high, clipped == z.If(ratio < low, low, z.If(ratio > high, high, ratio)))
    certify(z.And(clip, ratio >= 0), z.And(clipped >= low, clipped <= high))
    certify(z.And(clip, ratio >= 0, ratio <= big, big >= high, adv <= cap, adv >= -cap, cap >= 0,
                  term == z.If(ratio * adv <= clipped * adv, ratio * adv, clipped * adv)),
            z.And(term <= big * cap, term >= -big * cap))
    # Branch selection: an inactive row has coefficient exactly zero, whatever the ratio.
    active = z.Or(z.And(adv >= 0, ratio <= high), z.And(adv < 0, ratio >= low))
    coefficient, n = z.Reals('po_coefficient po_n')
    certify(z.And(clip, n >= 1, coefficient == z.If(active, ratio * adv / n, 0), z.Not(active)), coefficient == 0)
    certify(z.And(clip, ratio >= 0, n >= 1, coefficient == z.If(active, ratio * adv / n, 0), active, adv >= 0),
            z.And(coefficient >= 0, coefficient <= high * adv))
    return {'F-RL-PPOOBJ-V2-ARITH': 'unsat'}, dict(chain(), certifiedImplications=8)


STAGES = ('start', 'active', 'admitted')


def successors(state, mutant=False):
    phase, published = state
    if phase in ('absent', 'published'):
        return []
    out = [('reject', ('absent', False))]
    if phase in STAGES[:-1]:
        out.append(('advance', (STAGES[STAGES.index(phase) + 1], False)))
    elif phase == 'admitted':
        out += [('compute-active', ('computed', False)), ('compute-inactive', ('computed', False))]
    elif phase == 'computed':
        out.append(('guard-finite', ('published', True)))
        if mutant:
            out.append(('publish-unguarded', ('published', True)))
    return out


def model(mutant=False):
    initial = ('start', False)
    paths = {initial: ()}; queue = deque([initial]); edges = 0
    while queue:
        state = queue.popleft()
        nxt = successors(state, mutant)
        require(bool(nxt) or state[0] in ('absent', 'published'), 'nonterminal deadlock')
        for label, s in nxt:
            edges += 1
            trace = paths[state] + (label,)
            if s[1]:
                require(trace[-1] == 'guard-finite' and 'reject' not in trace, 'publication without finite guard')
            if s not in paths:
                paths[s] = trace; queue.append(s)
    require(('published', True) in paths and ('absent', False) in paths, 'vacuous model')
    return {'status': 'model_checked', 'states': len(paths), 'transitions': edges, 'orderTransitions': 0,
            'scope': 'finite stage abstraction; numeric values abstracted'}


def _impl():
    path = str(ROOT / 'scripts/research')
    if path not in sys.path:
        sys.path.insert(0, path)
    import ppo_objective_v2 as impl
    return impl


def oracle(logits, actions, old_prob, advantage, perturb=None):
    """60-digit reference. Returns loss, gradient, branch decisions and the distance to the nearest clip boundary."""
    n = len(logits); loss = D(0); grad = []; active = []; margin = D(1)
    low, high = D(0.8), D(1.2)   # the exact binary64 constants the kernel compares against
    for i, (row, a, old, adv) in enumerate(zip(logits, actions, old_prob, advantage)):
        r = [D(x) for x in row]
        if perturb is not None and perturb[0] == i:
            r[perturb[1]] += perturb[2]
        m = max(r); t = [(x - m).exp() for x in r]; s = sum(t); p = [v / s for v in t]
        ratio = p[a] / D(old); A = D(adv)
        clipped = min(max(ratio, low), high)
        loss -= min(ratio * A, clipped * A) / n
        on = (A >= 0 and ratio <= high) or (A < 0 and ratio >= low)
        k = ratio * A / n if on else D(0)
        grad.append([(p[j] - (1 if j == a else 0)) * k for j in range(3)])
        active.append(on)
        margin = min(margin, abs(ratio - low), abs(ratio - high))
    return loss, grad, active, margin


def close(x, ref, tol):
    return abs(D(x) - ref) <= tol * max(D(1), abs(ref))


def _batch(rng, scale):
    n = rng.randrange(1, 65)
    logits = tuple(tuple(rng.gauss(0, scale) for _ in range(3)) for _ in range(n))
    return (logits, tuple(rng.randrange(3) for _ in range(n)), tuple(rng.uniform(0.02, 1.0) for _ in range(n)),
            tuple(rng.gauss(0, scale) for _ in range(n)))


def conformance():
    impl = _impl()
    import numpy as np
    import sequential_learning as frozen
    reg = json.loads((ROOT / REGISTRATION).read_text())
    tol = D(reg['oracleRelativeTolerance']); rng = random.Random(reg['oracleSeed'])
    checked = differential = finite_diff = inactive = 0
    with decimal.localcontext() as ctx:
        ctx.prec = reg['oracleDecimalDigits']
        for b in range(reg['oracleBatches']):
            scale = (1.0, 5.0, 40.0, 2.0 ** 30)[b % 4]
            logits, a, old, adv = _batch(rng, scale)
            got = impl.objective_v2(logits, a, old, adv, enabled=True)
            require(got is not None and got == impl.objective_v2(logits, a, old, adv, enabled=True), 'deterministic finite result')
            require(impl.objective_v2(logits, a, old, adv) is None, 'default disabled')
            loss, grad, active, margin = oracle(logits, a, old, adv)
            require(margin > D('1e-9'), 'fixture too close to a clip boundary')
            require(close(got.loss, loss, tol), 'decimal oracle loss')
            require(all(close(got.grad[i][j], grad[i][j], tol) for i in range(len(logits)) for j in range(3)), 'decimal oracle gradient')
            require(all(on or got.grad[i] == (0.0, 0.0, 0.0) for i, on in enumerate(active)), 'inactive rows are exactly zero')
            inactive += sum(not on for on in active)
            checked += 1
            if scale == 1.0:
                i, j = rng.randrange(len(logits)), rng.randrange(3); h = D('1e-25')
                fd = (oracle(logits, a, old, adv, (i, j, h))[0] - oracle(logits, a, old, adv, (i, j, -h))[0]) / (2 * h)
                require(close(got.grad[i][j], fd, D('1e-9')), 'finite-difference gradient')
                finite_diff += 1
    require(inactive > 0, 'oracle batches must exercise inactive rows')
    rng = random.Random(reg['oracleSeed'] + 1)
    for b in range(reg['frozenDifferentialBatches']):
        logits, a, old, adv = _batch(rng, 2.0)
        got = impl.objective_v2(logits, a, old, adv, enabled=True)
        fl, fg = frozen.ppo_gradient(np.array(logits), np.array(a), np.array(old), np.array(adv))
        require(close(got.loss, D(fl), tol) and all(close(got.grad[i][j], D(float(fg[i, j])), tol)
                                                     for i in range(len(logits)) for j in range(3)), 'frozen helper differential')
        differential += 1
    return {'status': 'property_tested', 'oracleBatches': checked, 'frozenDifferential': differential,
            'finiteDifferenceChecks': finite_diff, 'inactiveRows': inactive, 'witnesses': witnesses(),
            'seed': reg['oracleSeed'], 'marketDataReads': 0, 'economicEvidence': False}


def witnesses():
    impl = _impl()
    entries = {e['id']: e for e in json.loads((ROOT / COUNTEREXAMPLES).read_text())['entries']}
    w = entries['CE-RL-012']
    logits = (tuple(float.fromhex(x) for x in w['logitsHex']),)
    old, adv = float.fromhex(w['oldProbabilityHex']), float.fromhex(w['advantageHex'])
    require(0.0 < old < 2.0 ** -1022 and impl.objective_v2(logits, (w['action'],), (old,), (adv,), enabled=True) is None,
            'CE-RL-012 witness must reject')
    inside = impl.objective_v2(logits, (w['action'],), (2.0 ** -100,), (adv,), enabled=True)
    require(inside is not None and inside.loss == float.fromhex(w['lossHex']) and inside.grad == ((0.0, 0.0, 0.0),) and
            inside.underflow == 0, 'CE-RL-012 in-domain analogue: finite loss, exactly zero gradient')
    w = entries['CE-RL-013']
    logits = (tuple(float.fromhex(x) for x in w['logitsHex']),)
    got = impl.objective_v2(logits, (w['action'],), (float.fromhex(w['oldProbabilityHex']),),
                            (float.fromhex(w['advantageHex']),), enabled=True)
    coefficient = float.fromhex(w['coefficientHex'])
    require(got is not None and got.loss == float.fromhex(w['lossHex']) and coefficient == -8.0 and
            got.grad[0][w['action']] == (float.fromhex(w['probabilityHex']) - 1.0) * coefficient and
            math.isclose(max(abs(g) for g in got.grad[0]), 16 / 3, rel_tol=1e-15), 'CE-RL-013 clipped-objective behavior reproduced')
    for bad in (2.0 ** -101, 0.0, -0.5, 1.0000000000000002, float('nan'), 1):
        require(impl.objective_v2(logits, (0,), (bad,), (1.0,), enabled=True) is None, 'old-probability domain')
    tiny = impl.objective_v2(((0.0, -800.0, 0.0),), (1,), (1.0,), (1.0,), enabled=True)
    require(tiny is not None and tiny.underflow >= 1 and all(math.isfinite(g) for g in tiny.grad[0]) and
            all(math.copysign(1, g) == 1 for g in tiny.grad[0] if g == 0), 'underflow counted; canonical zeros')
    return {'CE-RL-012': 'rejected; in-domain analogue finite with zero gradient', 'CE-RL-013': 'reproduced (intended)'}


def check_ppo_objective():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require((reg['actions'], reg['maximumRows'], reg['magnitudeBoundLog2'], reg['minimumOldProbabilityLog2'],
             reg['clipLow'], reg['clipHigh'], reg['enabledByDefault']) == (3, 256, 100, -100, 0.8, 1.2, False), 'registration')
    nodes, source = extract()
    smt, bounds = prove(nodes)
    return {'source': source, 'smt': smt, 'bounds': bounds, 'model': model(), 'conformance': conformance()}


if __name__ == '__main__':
    print(json.dumps(check_ppo_objective(), indent=1, sort_keys=True))
