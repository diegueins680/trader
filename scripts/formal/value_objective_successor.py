"""Source-bound value objective v2: binary64 shift lemma, rounding-model bounds, model, oracle and witnesses."""
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
SOURCE = 'scripts/research/value_objective_v2.py'
REGISTRY = 'formal/research/value-objective-v2-source.json'
REGISTRATION = 'research-notes/registrations/value-objective-v2-engineering.json'
CONSTANTS = {'VERSION': "'value-objective-v2'", 'ACTIONS': '3', 'MAX_ROWS': '256', 'BOUND': '2.0 ** 100',
             'TINY': '2.0 ** (-1022)', '__all__': "['VERSION', 'Objective', 'objective_v2']"}
FINITE_LIMIT = 2 ** 220


def require(ok, reason):
    if not ok:
        raise ValueError('value objective v2: ' + reason)


def extract(source=None, registry=None):
    tree = ast.parse((ROOT / SOURCE).read_text() if source is None else source)
    lock = json.loads((ROOT / REGISTRY).read_text()) if registry is None else registry
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    require(len(nodes) == sum(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body), 'duplicate definition')
    require({k: shape(v) for k, v in nodes.items()} == lock['definitions'], 'reviewed definition drift')
    require(hashlib.sha256(shape(tree).encode()).hexdigest() == lock['astSha256'], 'module effect drift')
    for path, value in lock['supportHashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value, 'support drift: ' + path)
    constants = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    require(constants == CONSTANTS, 'version/bounds/exports')
    require([ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))] ==
            ['from __future__ import annotations', 'from dataclasses import dataclass', 'import math'],
            'scalar-only import boundary (no NumPy/BLAS)')
    require([ast.unparse(n) for n in nodes['Objective'].decorator_list] == ['dataclass(frozen=True, slots=True)'], 'immutable output')
    entry = nodes['objective_v2']
    defaults = dict(zip((x.arg for x in entry.args.kwonlyargs), map(ast.unparse, entry.args.kw_defaults)))
    require(defaults == {'enabled': 'False', 'version': 'VERSION'}, 'disabled default')
    require(ast.unparse(entry.body[0]) == "if enabled is not True or type(version) is not str or version != VERSION:\n    return None", 'activation first')
    require(ast.unparse(nodes['_value'].body[0].value) == 'type(x) is float and math.isfinite(x) and (abs(x) <= BOUND)', 'numeric admission')
    compute = nodes['_compute'].body
    require(ast.unparse(compute[1].value) == '[row[a] - y for row, a, y in zip(q, actions, targets)]', 'residual is a difference')
    require(ast.unparse(compute[2].value) == 'math.fsum((e * e for e in residuals)) / (2 * n)', 'fsum squared-residual mean')
    branch = compute[5]
    require(isinstance(branch, ast.If) and ast.unparse(branch.test) == 'alpha > 0.0' and not branch.orelse,
            'penalty evaluated only when alpha > 0')
    require(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == '_penalty' for n in ast.walk(nodes['_compute'])) == 1 and
            any(isinstance(n, ast.Call) and ast.unparse(n.func) == '_penalty' for n in ast.walk(branch)), 'penalty call site')
    penalty = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in nodes['_penalty'].body if isinstance(n, ast.Assign)}
    require(penalty == {'m': 'max(row)', 'terms': '[math.exp(x - m) for x in row]',
                        'underflow': 'sum((1 for t in terms if t < TINY))', 'total': 'math.fsum(terms)',
                        'gap': 'm - row[action] + math.log(total)'}, 'difference-only penalty')
    require(ast.unparse(compute[6]) == 'loss += 0.0' and
            ast.unparse(compute[7].value) == 'tuple((tuple((g + 0.0 for g in r)) for r in grad))', 'zero canonicalization (CE-RL-024)')
    require(ast.unparse(compute[8].test) == 'not math.isfinite(loss) or not all((math.isfinite(g) for r in published for g in r))' and
            ast.unparse(compute[8].body[0]) == 'return None', 'finite publication guard')
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    require(not names & {'np', 'numpy', 'eval', 'exec', 'open', 'socket', 'order', 'submit', 'promote'}, 'effect-free source')
    return nodes, {'status': 'exhaustively_checked', 'definitions': len(nodes), 'defaultDisabledEntries': 1,
                   'scope': 'complete reviewed AST, difference-only penalty, alpha=0 skip and finite guard; runtime semantics assumed'}


def shift_lemma():
    # A-FP-ROUNDING: each basic operation returns rnd(exact real result). With an uninterpreted
    # rounding function, exactly shifted operands have equal rounded differences (value equality).
    rnd = z.Function('vo_rnd', z.RealSort(), z.RealSort())
    a, b, s, sa, sb = z.Reals('vo_ra vo_rb vo_rs vo_rsa vo_rsb')
    certify(z.And(sa == a + s, sb == b + s), rnd(sa - sb) == rnd(a - b))
    # The row maximum commutes with an exact common shift.
    x0, x1, x2, m, ms = z.Reals('vo_x0 vo_x1 vo_x2 vo_m vo_ms')
    is_max = lambda v, xs: z.And(*[v >= x for x in xs], z.Or(*[v == x for x in xs]))
    certify(z.And(is_max(m, (x0, x1, x2)), is_max(ms, (x0 + s, x1 + s, x2 + s))), ms == m + s)
    # CE-RL-024: the bitwise draft claim is refuted; evaluate the preserved binary32 witness exactly.
    f32 = z.Float32(); rne = z.RNE()
    wa, wb = z.fpMinusZero(f32), z.fpPlusZero(f32)
    ws = z.fpNeg(z.FPVal(1.05073225498199462890625 * 2.0 ** 72, f32))
    exact = z.And(z.fpAdd(z.RTN(), wa, ws) == z.fpAdd(z.RTP(), wa, ws), z.fpAdd(z.RTN(), wb, ws) == z.fpAdd(z.RTP(), wb, ws))
    lhs, rhs = z.fpSub(rne, z.fpAdd(rne, wa, ws), z.fpAdd(rne, wb, ws)), z.fpSub(rne, wa, wb)
    verdict = z.simplify(z.And(exact, z.fpEQ(lhs, rhs), z.Not(lhs == rhs), z.fpIsPositive(lhs), z.fpIsNegative(rhs)))
    require(z.is_true(verdict), 'CE-RL-024 binary32 witness no longer reproduces')


def bound_lemmas():
    u = z.RealVal(Q(1, 2**53).__str__())
    x, cap, delta = z.Reals('vo_x vo_cap vo_delta')
    # Standard model: one rounding scales any magnitude bound by at most (1+u).
    certify(z.And(cap >= 0, x <= cap, x >= -cap, delta <= u, delta >= -u),
            z.And(x * (1 + delta) <= cap * (1 + u), x * (1 + delta) >= -cap * (1 + u)))
    # The row maximum makes every exponent argument nonpositive and one of them zero, so 1 <= total <= 3.
    q0, q1, q2, m = z.Reals('vo_q0 vo_q1 vo_q2 vo_m')
    is_max = z.And(m >= q0, m >= q1, m >= q2, z.Or(m == q0, m == q1, m == q2))
    certify(is_max, z.And(q0 - m <= 0, q1 - m <= 0, q2 - m <= 0, z.Or(q0 - m == 0, q1 - m == 0, q2 - m == 0)))
    t0, t1, t2 = z.Reals('vo_t0 vo_t1 vo_t2')
    terms = z.And(*[z.And(t >= 0, t <= 1) for t in (t0, t1, t2)], z.Or(t0 == 1, t1 == 1, t2 == 1))
    certify(terms, z.And(t0 + t1 + t2 >= 1, t0 + t1 + t2 <= 3))
    # Normalized probabilities are in [0,1]; each conservative component is bounded by alpha/n.
    p, total, alpha, n = z.Reals('vo_p vo_total vo_alpha vo_n')
    certify(z.And(total >= 1, p >= 0, p <= 1, alpha >= 0, alpha <= 1, n >= 1),
            z.And(alpha * (p - 1) / n >= -1, alpha * (p - 1) / n <= 0, alpha * p / n <= 1))
    return chain()


def chain():
    # Exact rational propagation of the per-operation (1+u) factor along the source expression.
    u = Q(1, 2**53); g = 1 + u; bound = Q(2**100); n = 256
    residual = 2 * bound * g
    square = residual * residual * g
    mse = n * square * g / (2 * n) * g      # fsum of n squares (one rounding), then /(2n) (one rounding)
    gap = (2 * bound * g + 2) * g            # (m-q[a]) rounded, plus log(total) <= 2, rounded
    penalty = n * gap * g * g / n * g        # fsum of n gaps, times alpha <= 1, then /n
    loss = (mse + penalty) * g
    gradient = residual * g + 2 * g * g
    worst = max(mse, penalty, loss, gradient)
    require(worst < FINITE_LIMIT < Q(int(sys.float_info.max)), 'admitted-domain magnitude chain')
    return {'lossBoundLog2': math.ceil(math.log2(float(loss))), 'gradientBoundLog2': math.ceil(math.log2(float(gradient))),
            'finiteLimitLog2': 220}


def prove(nodes):
    shift_lemma()
    bounds = bound_lemmas()
    return {'F-RL-VALUE-V2-ARITH': 'unsat'}, bounds


STAGES = ('start', 'active', 'admitted')


def successors(state, mutant=False):
    phase, published = state
    if phase in ('absent', 'published'):
        return []
    out = [('reject', ('absent', False))]
    if phase in STAGES[:-1]:
        out.append(('advance', (STAGES[STAGES.index(phase) + 1], False)))
    elif phase == 'admitted':
        out += [('compute-plain', ('computed', False)), ('compute-penalty', ('computed', False))]
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
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import value_objective_v2 as impl
    return impl


def oracle(q, actions, targets, alpha):
    with decimal.localcontext() as ctx:
        ctx.prec = 60
        n = len(q); loss = D(0); grad = []; underflow = 0
        tiny = D(2) ** -1022
        for row, a, y in zip(q, actions, targets):
            r = [D(x) for x in row]; e = r[a] - D(y)
            loss += e * e / (2 * n)
            g = [(e / n if j == a else D(0)) for j in range(3)]
            if alpha > 0:
                m = max(r); t = [(x - m).exp() for x in r]; s = sum(t)
                underflow += sum(1 for v in t if v < tiny)
                loss += D(alpha) * ((m - r[a]) + s.ln()) / n
                g = [g[j] + D(alpha) * (t[j] / s - (1 if j == a else 0)) / n for j in range(3)]
            grad.append(g)
        return loss, grad, underflow


def close(x, ref, tol):
    return abs(D(x) - ref) <= tol * max(D(1), abs(ref))


def _batch(rng, scale):
    n = rng.randrange(1, 65)
    q = tuple(tuple(rng.gauss(0, scale) for _ in range(3)) for _ in range(n))
    return q, tuple(rng.randrange(3) for _ in range(n)), tuple(rng.gauss(0, scale) for _ in range(n))


def conformance():
    impl = _impl()
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import numpy as np
    import sequential_learning as frozen
    reg = json.loads((ROOT / REGISTRATION).read_text())
    tol = D(reg['oracleRelativeTolerance']); rng = random.Random(reg['oracleSeed'])
    checked = differential = finite_diff = 0
    for b in range(reg['oracleBatches']):
        scale = (1.0, 50.0, 2.0 ** 40, 2.0 ** 90)[b % 4]
        alpha = (0.0, 0.1, 1.0)[b % 3]
        q, a, y = _batch(rng, scale)
        got = impl.objective_v2(q, a, y, alpha, enabled=True)
        require(got is not None and got == impl.objective_v2(q, a, y, alpha, enabled=True), 'deterministic finite result')
        require(impl.objective_v2(q, a, y, alpha) is None, 'default disabled')
        loss, grad, underflow = oracle(q, a, y, alpha)
        require(close(got.loss, loss, tol), 'decimal oracle loss')
        require(all(close(got.grad[i][j], grad[i][j], tol) for i in range(len(q)) for j in range(3)), 'decimal oracle gradient')
        require(got.underflow == underflow, 'underflow accounting')
        checked += 1
        if scale == 1.0 and alpha > 0:
            # Central finite difference of the oracle loss at one coordinate.
            i, j = rng.randrange(len(q)), rng.randrange(3)
            with decimal.localcontext() as ctx:
                ctx.prec = 60; h = D('1e-25')
                def at(delta):
                    rows = [list(map(D, r)) for r in q]; rows[i][j] += delta
                    return oracle_d(rows, a, y, alpha)
                fd = (at(h) - at(-h)) / (2 * h)
            require(close(got.grad[i][j], fd, D('1e-9')), 'finite-difference gradient')
            finite_diff += 1
    rng = random.Random(reg['oracleSeed'] + 1)
    for b in range(reg['frozenDifferentialBatches']):
        alpha = (0.0, 0.1)[b % 2]
        q, a, y = _batch(rng, 2.0)
        got = impl.objective_v2(q, a, y, alpha, enabled=True)
        fl, fg = frozen.bellman_gradient(np.array(q), np.array(a), np.array(y), alpha)
        require(close(got.loss, D(fl), tol) and all(close(got.grad[i][j], D(float(fg[i, j])), tol)
                                                     for i in range(len(q)) for j in range(3)), 'frozen helper differential')
        differential += 1
    return {'status': 'property_tested', 'oracleBatches': checked, 'frozenDifferential': differential,
            'finiteDifferenceChecks': finite_diff, 'witnesses': witnesses(), 'seed': reg['oracleSeed'],
            'marketDataReads': 0, 'economicEvidence': False}


def oracle_d(rows, actions, targets, alpha):
    n = len(rows); loss = D(0)
    for r, a, y in zip(rows, actions, targets):
        e = r[a] - D(y); loss += e * e / (2 * n)
        m = max(r); s = sum((x - m).exp() for x in r)
        loss += D(alpha) * ((m - r[a]) + s.ln()) / n
    return loss


def witnesses():
    impl = _impl()
    M = sys.float_info.max; B = 2.0 ** 100; S = 2.0 ** 54
    require(impl.objective_v2(((M, M, -M),), (2,), (-M,), 0.0, enabled=True) is None, 'CE-RL-014 witness must reject')
    zero = impl.objective_v2(((B, B, -B),), (2,), (-B,), 0.0, enabled=True)
    require(zero is not None and zero.loss == 0.0 and zero.grad == ((0.0, 0.0, 0.0),), 'CE-RL-014 in-domain analogue finite')
    base = impl.objective_v2(((0.0, 0.0, 0.0),), (0,), (0.0,), 0.1, enabled=True)
    shifted = impl.objective_v2(((S, S, S),), (0,), (S,), 0.1, enabled=True)
    require(base == shifted and base.loss == 0.1 * math.log(3.0) and bits(base) == bits(shifted), 'CE-RL-015 shift must be bitwise invariant')
    # CE-RL-024 in binary64: raw differences disagree in zero sign; published values do not.
    a, b, s = -0.0, 0.0, -1.05073225498199462890625 * 2.0 ** 72
    require(math.copysign(1, (a + s) - (b + s)) == 1 and math.copysign(1, a - b) == -1, 'CE-RL-024 binary64 witness')
    raw = impl.objective_v2(((a, b, b),), (0,), (b,), 0.0, enabled=True)
    moved = impl.objective_v2(((a + s, b + s, b + s),), (0,), (b + s,), 0.0, enabled=True)
    require(math.copysign(1, a - b) == -1 and (a + s) - (b + s) == 0.0, 'witness exercises a signed-zero residual')
    require(raw is not None and moved is not None and bits(raw) == bits(moved) and
            all(math.copysign(1, g) == 1 for r in raw.grad for g in r if g == 0), 'CE-RL-024 published canonical zero')
    return {'CE-RL-014': 'rejected; in-domain analogue finite', 'CE-RL-015': 'bitwise invariant', 'CE-RL-024': 'canonicalized'}


def bits(objective):
    return (objective.loss.hex(), tuple(g.hex() for r in objective.grad for g in r))


def check_value_objective():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require((reg['actions'], reg['maximumRows'], reg['magnitudeBoundLog2'], reg['enabledByDefault']) == (3, 256, 100, False), 'registration')
    nodes, source = extract()
    smt, bounds = prove(nodes)
    return {'source': source, 'smt': smt, 'bounds': bounds, 'model': model(), 'conformance': conformance()}
