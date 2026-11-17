"""Numeric audit of causal-optimizer-v5: every site detects overflow or underflow before publication or is bounded."""
import ast
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import sys

import z3 as z
from ppo_objective_successor import ceil_log2
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/causal_optimizer_v5.py'
LEARNERS = ('scripts/research/causal_value_v6.py', 'scripts/research/causal_ppo_v7.py')
REGISTRY = 'formal/research/optimizer-numerics-source.json'
REGISTRATION = 'research-notes/registrations/optimizer-numerics-engineering.json'
U = Q(1, 2 ** 53)
C9, C1, C999, C001, EPS, LR = Q(0.9), Q(0.1), Q(0.999), Q(0.001), Q(1e-8), Q(0.0003)
RATIO = 54                       # m^2 <= RATIO * v along every create/update chain
CLIP = 1 + Q(1, 2 ** 44)         # magnitude of every clipped gradient entry
MOMENT = 1 + Q(1, 2 ** 42)
VARIANCE = 1 + Q(1, 2 ** 40)
STEP = Q(15, 2)                  # |parameter change| <= STEP * lr per update
ENTRIES = 9 * 16 + 16 + 16 * 3 + 3
INITIAL = 8                      # assumed magnitude bound on create_v5 parameters
UPDATES = 4096 * 4               # optimizer steps per network in one learner run
PARAMETER = 2 ** 6
OUTPUT = 2 ** 11
UNDERFLOW = Q(256, 2 ** 1075)    # absolute error of silent product underflow in one matrix-product entry
# Every arithmetic expression of the module, by how a range failure is handled.
#   matmul: BLAS/NumPy matrix product, no flag relied on; overflow stays non-finite until an explicit isfinite check,
#           underflow is bounded by UNDERFLOW.
#   ufunc:  NumPy element-wise operation or reduction inside errstate(all="raise"): overflow, underflow and invalid raise.
#   python: binary64 scalar expression that cannot overflow; 0.9**step underflows to zero, which is the correct limit.
#   exact:  integer, shape or type expression.
SITES = {
    '_admit': {'math.prod(shape) * 8': 'exact'},
    '_pack': {'Snapshot | None': 'exact'},
    'create_v5': {'Snapshot | None': 'exact', '2 ** 32': 'exact', '1 / np.sqrt(9)': 'python'},
    '_evaluate': {'x @ w1 + b1': 'ufunc', 'hidden @ w2 + b2': 'ufunc', 'x @ w1': 'matmul', 'hidden @ w2': 'matmul'},
    'forward_v5': {'np.ndarray | None': 'exact'},
    'update_v5': {
        'Snapshot | None': 'exact', 'snapshot.step + 1': 'exact',
        'residuals @ p[2].T * (1 - hidden ** 2)': 'ufunc', 'residuals @ p[2].T': 'matmul', '1 - hidden ** 2': 'ufunc',
        'hidden ** 2': 'ufunc', 'inputs.T @ dh': 'matmul', 'hidden.T @ residuals': 'matmul',
        'grad / max(1.0, float(norm))': 'ufunc', '0.9 * m + 0.1 * g': 'ufunc', '0.9 * m': 'ufunc', '0.1 * g': 'ufunc',
        '0.999 * v + 0.001 * g ** 2': 'ufunc', '0.999 * v': 'ufunc', '0.001 * g ** 2': 'ufunc', 'g ** 2': 'ufunc',
        'lr * (moment / (1 - 0.9 ** step)) / (np.sqrt(variance / (1 - 0.999 ** step)) + 1e-08)': 'ufunc',
        'lr * (moment / (1 - 0.9 ** step))': 'ufunc', 'moment / (1 - 0.9 ** step)': 'ufunc',
        'np.sqrt(variance / (1 - 0.999 ** step)) + 1e-08': 'ufunc', 'variance / (1 - 0.999 ** step)': 'ufunc',
        'old - delta': 'ufunc', '1 - 0.9 ** step': 'python', '0.9 ** step': 'python', '1 - 0.999 ** step': 'python',
        '0.999 ** step': 'python'},
}
NUMERIC_CALLS = {'np.tanh', 'np.sqrt', 'np.sum', 'dh.sum', 'residuals.sum', 'sum', 'rng.normal'}


def require(ok, reason):
    if not ok: raise ValueError('optimizer numerics: ' + reason)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return (ROOT / path).read_text()


def registry_for(source, learners=None):
    learners = [read(path) for path in LEARNERS] if learners is None else learners
    try:
        tree = ast.parse(source)
    except SyntaxError:
        raise ValueError('optimizer numerics: unparsable source')
    return {'schemaVersion': 1, 'astSha256': sha(ast.dump(tree, include_attributes=False)),
            'learnerSha256': [sha(text) for text in learners]}


def _texts(node):
    return [ast.unparse(statement) for statement in node.body]


def extract(source=None, registry=None, learners=None):
    source = read(SOURCE) if source is None else source
    learners = [read(path) for path in LEARNERS] if learners is None else learners
    registry = json.loads(read(REGISTRY)) if registry is None else registry
    require(registry == registry_for(source, learners), 'reviewed optimizer or learner source drift')
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    found = {}
    for name, node in functions.items():
        sites = {ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.BinOp)}
        if sites: found[name] = sites
    require(found == {name: set(sites) for name, sites in SITES.items()}, 'arithmetic site coverage')
    require(not [n for n in ast.walk(tree) if isinstance(n, (ast.AugAssign, ast.UnaryOp)) and
                 not (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not))], 'unclassified operator')
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    require(NUMERIC_CALLS <= calls and not calls & {'np.exp', 'np.log', 'np.dot', 'np.matmul', 'np.einsum', 'np.power',
                                                    'np.seterr', 'np.cumsum', 'np.prod', 'np.mean', 'np.linalg.norm'},
            'numeric call coverage')
    matmul = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult)]
    require(sorted(matmul) == sorted(text for sites in SITES.values() for text, kind in sites.items() if kind == 'matmul'),
            'matrix-product sites')
    # Every arithmetic statement of the three entries runs inside errstate(all="raise") and a guarded try.
    for name in ('create_v5', 'forward_v5', 'update_v5'):
        body = functions[name].body
        require(len(body) == 2 and isinstance(body[1], ast.Try) and len(body[1].body) == 1 and
                isinstance(body[1].body[0], ast.With) and
                ast.unparse(body[1].body[0].items[0].context_expr) == "np.errstate(all='raise')" and
                ast.unparse(body[1].handlers[0].type) == '(ArithmeticError, ValueError, TypeError, MemoryError)' and
                ast.unparse(body[1].handlers[0].body[0]) == 'return None' and
                not any(isinstance(n, ast.BinOp) and not isinstance(n.op, (ast.BitOr, ast.Pow)) for n in ast.walk(body[0])),
                'raising floating-point state around ' + name)
    # Forward: both matrix products are checked for finiteness before use or return.
    require(_texts(functions['_evaluate']) == [
        'w1, b1, w2, b2 = p', 'preactivation = x @ w1 + b1',
        "if not np.isfinite(preactivation).all():\n    raise ValueError('non-finite hidden state')",
        'hidden = np.tanh(preactivation)', 'output = hidden @ w2 + b2',
        "if not np.isfinite(output).all():\n    raise ValueError('non-finite output')", 'return (hidden, output)'],
        'checked forward products')
    # Update: the three remaining products reach the gradient finiteness check; dh reaches it through its column sum.
    update = _texts(functions['update_v5'].body[1].body[0])
    start = update.index('hidden, _ = _evaluate(p, inputs)')
    require(update[start + 1:start + 6] == [
        'dh = residuals @ p[2].T * (1 - hidden ** 2)',
        'grads = (inputs.T @ dh, dh.sum(0), hidden.T @ residuals, residuals.sum(0))',
        'if not all((np.isfinite(g).all() for g in grads)):\n    return None',
        'norm = np.sqrt(sum((np.sum(g ** 2) for g in grads)))', 'if not np.isfinite(norm):\n    return None'],
        'checked gradient products and norm')
    require(update[-1] == 'return _pack(snapshot.outputs, step, tuple(candidates), tuple(moments), tuple(variances))' and
            _texts(functions['_pack'])[-1] == 'return snapshot if _admit(snapshot) else None' and
            'if not np.isfinite(values).all() or (index == 2 and np.any(values < 0)):\n    return False' in
            [ast.unparse(n) for n in ast.walk(functions['_admit']) if isinstance(n, ast.If)], 'finite, admitted publication')
    constants = sorted({ast.unparse(n) for n in ast.walk(functions['update_v5']) if isinstance(n, ast.Constant)
                        and type(n.value) is float})
    require(constants == ['0.0003', '0.001', '0.1', '0.9', '0.999', '1.0', '1e-08'], 'optimizer constants')
    # The learners start every chain at create_v5 and advance it only with update_v5 at the default learning rate.
    steps = set()
    for text in learners:
        module = ast.parse(text)
        used = [ast.unparse(n) for n in ast.walk(module) if isinstance(n, ast.Call) and
                ast.unparse(n.func) in ('update_v5', 'create_v5', 'Snapshot', '_pack', 'replace')]
        updates = [n for n in ast.walk(module) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'update_v5']
        require(updates and all(len(n.args) == 3 and [k.arg for k in n.keywords] == ['enabled'] for n in updates) and
                any(c.startswith('create_v5(') for c in used) and
                not any(c.startswith(('Snapshot(', '_pack(', 'replace(')) for c in used), 'learner optimizer chain')
        values = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in module.body if isinstance(n, ast.Assign)}
        require('not 1 <= len(outcomes) <= 4096' in text, 'learner step bound')
        steps.add(4096 * int(values['UPDATES']))
    require(steps == {UPDATES}, 'registered update bound')
    return functions, {'status': 'exhaustively_checked', 'sites': sum(len(s) for s in SITES.values()),
                       'matrixProducts': len(matmul),
                       'elementwise': sum(kind == 'ufunc' for s in SITES.values() for kind in s.values()),
                       'scalar': sum(kind == 'python' for s in SITES.values() for kind in s.values()),
                       'learners': len(learners), 'updatesPerRun': UPDATES,
                       'scope': 'complete module AST with every arithmetic expression classified; NumPy, BLAS and '
                                'libm semantics assumed'}


def real(value):
    return z.RealVal(str(Q(value)))


def magnitude(value):
    return z.If(value >= 0, value, -value)


def rounded(value, exact):
    return magnitude(value - exact) <= real(U) * magnitude(exact)


def run_bounds(step=STEP, initial=INITIAL, updates=UPDATES):
    """Exact rational consequences of the per-update lemmas over a whole learner run."""
    # With the binary64 values the source computes, 1 - 0.999^t <= (1 - 0.9^t)^2 for every step of a run, so the
    # corrected ratio is no larger than the uncorrected one.
    for t in range(1, updates + 1):
        d1, d2 = Q(1 - 0.9 ** t), Q(1 - 0.999 ** t)
        require(0 < d2 <= d1 * d1 and d1 <= 1, 'bias correction ordering')
    require(0.9 ** (2 ** 31 - 1) == 0.0 and 0.999 ** (2 ** 31 - 1) == 0.0 and 1 - 0.999 ** 1 > 0, 'scalar powers')
    parameter = (initial + updates * step * LR) * (1 + U) ** updates
    output = 17 * PARAMETER * (1 + U) ** 18
    require(parameter <= PARAMETER and output <= OUTPUT and OUTPUT < 2 ** 99, 'run bounds')
    require(256 * Q(1, 2 ** 1075) == UNDERFLOW, 'underflow accumulation')
    return parameter


def prove(ratio=RATIO, step=STEP, initial=INITIAL, updates=UPDATES):
    parameter = run_bounds(step, initial, updates)
    # (a) Sticky non-finiteness in binary64: once an operand is infinite or NaN, sums, products and fused
    #     multiply-adds stay so, in any evaluation order. A finite checked result had no overflowed intermediate.
    rm = z.RNE(); a, b, c = z.FPs('on_a on_b on_c', z.Float64())
    nonfinite = lambda x: z.Or(z.fpIsInf(x), z.fpIsNaN(x))
    for premise, value in ((nonfinite(a), z.fpAdd(rm, a, b)), (nonfinite(b), z.fpAdd(rm, a, b)),
                           (nonfinite(a), z.fpSub(rm, a, b)), (nonfinite(a), z.fpMul(rm, a, b)),
                           (nonfinite(b), z.fpMul(rm, a, b)), (nonfinite(c), z.fpFMA(rm, a, b, c)),
                           (nonfinite(a), z.fpFMA(rm, a, b, c))):
        certify(premise, nonfinite(value))
    solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
    solver.add(z.fpIsInf(a), z.fpIsInf(b), z.fpIsNaN(z.fpAdd(rm, a, b)))
    require(solver.check() == z.sat, 'cancellation of infinities is NaN, so the check must reject NaN as well')
    # (b) Clipping: every entry divided by max(1, norm) has magnitude at most CLIP.
    require((1 - U) ** (ENTRIES + 2) >= 1 - (ENTRIES + 2) * U, 'sum-of-squares rounding')
    g, s, n, d, q = z.Reals('on_g on_s on_n on_d on_q')
    certify(z.And(g * g * real(1 - (ENTRIES + 2) * U) <= s, n >= 0, magnitude(n * n - s) <= 3 * real(U) * s, d >= 1, d >= n,
                  z.Or(d == 1, d == n), rounded(q, g / d)), magnitude(q) <= real(CLIP))
    # (c) Moments stay bounded and m^2 <= ratio * v is inductive under rounding (Cauchy-Schwarz over the history).
    m, v, x, y, m1, sq, t1, t2, v1 = z.Reals('on_m on_v on_x on_y on_m1 on_sq on_t1 on_t2 on_v1')
    update = z.And(magnitude(q) <= real(CLIP), rounded(x, real(C9) * m), rounded(y, real(C1) * q), rounded(m1, x + y),
                   rounded(sq, q * q), rounded(t1, real(C999) * v), rounded(t2, real(C001) * sq), rounded(v1, t1 + t2))
    certify(z.And(update, magnitude(m) <= real(MOMENT)), magnitude(m1) <= real(MOMENT))
    certify(z.And(update, v >= 0, v <= real(VARIANCE)), z.And(v1 >= 0, v1 <= real(VARIANCE)))
    certify(z.And(update, v >= 0, m * m <= ratio * v), z.And(v1 >= 0, m1 * m1 <= ratio * v1))
    # (d) One update moves a parameter by at most step * lr, given the bias-correction ordering of run_bounds.
    b1, b2, mh, vh, r, den, num, delta, lr = z.Reals('on_b1 on_b2 on_mh on_vh on_r on_den on_num on_delta on_lr')
    certify(z.And(v >= 0, m * m <= ratio * v, b1 > 0, b2 > 0, b2 <= b1 * b1, b1 <= 1, rounded(mh, m / b1), rounded(vh, v / b2),
                  r >= 0, magnitude(r * r - vh) <= 3 * real(U) * vh, den >= r, den >= real(EPS) * real(1 - U), lr > 0, lr <= 1,
                  rounded(num, lr * mh), rounded(delta, num / den)), magnitude(delta) <= real(step) * lr)
    # (e) A rounded accumulation step, used for the parameter drift and for the output layer.
    acc, term, out, bound_a, bound_t = z.Reals('on_acc on_term on_out on_ba on_bt')
    certify(z.And(bound_a >= 0, bound_t >= 0, magnitude(acc) <= bound_a, magnitude(term) <= bound_t, rounded(out, acc + term)),
            magnitude(out) <= (bound_a + bound_t) * real(1 + U))
    # (f) Silent product underflow in a matrix product: each product is within half a subnormal step of exact.
    exact, near, grid = z.Reals('on_exact on_near on_grid')
    k = z.Int('on_k')
    certify(z.And(grid > 0, near == k * grid, exact >= near - grid / 2, exact <= near + grid / 2),
            magnitude(near - exact) <= grid / 2)
    return {'F-RL-OPTNUM-V5-ARITH': 'unsat'}, {
        'certifiedImplications': 14, 'satisfiableControls': 1, 'momentRatio': ratio, 'stepPerLearningRate': float(step),
        'clipExcessLog2': ceil_log2(CLIP - 1), 'initialParameterBound': initial, 'updatesPerRun': updates,
        'parameterBoundLog2': ceil_log2(Q(PARAMETER)), 'outputBoundLog2': ceil_log2(Q(OUTPUT)),
        'certifiedParameterBound': float(parameter), 'underflowLog2': ceil_log2(UNDERFLOW), 'biasCorrectionSteps': updates}


def _impl():
    path = str(ROOT / 'scripts/research')
    if path not in sys.path: sys.path.insert(0, path)
    import causal_optimizer_v5 as impl
    return impl


def forged(impl, outputs, values, step=0, m=0.0, v=0.0):
    """A snapshot with chosen parameter values per array; admitted because every entry is finite."""
    import numpy as np
    shapes = impl._shapes(outputs)
    pack = lambda fill: tuple(np.full(shape, fill[i] if type(fill) is tuple else fill, dtype='<f8').tobytes()
                              for i, shape in enumerate(shapes))
    return impl.Snapshot(impl.VERSION, impl.OBSERVATION, outputs, step, pack(values), pack(m), pack(v))


def arrays(impl, snapshot):
    import numpy as np
    return [[np.frombuffer(raw, dtype='<f8') for raw in group] for group in (snapshot.p, snapshot.m, snapshot.v)]


def conformance():
    import numpy as np
    impl = _impl()
    reg = json.loads(read(REGISTRATION))
    rng = np.random.default_rng(reg['seed'])
    ones = np.ones((2, 9))
    # Overflow in every matrix product is rejected, including cancellation of infinities.
    big, top = 1e200, math.nextafter(math.inf, 0.0)
    rejected = [
        impl.forward_v5(forged(impl, 3, big), ones * big, enabled=True),
        impl.forward_v5(forged(impl, 3, big), np.array([[big, -big] + [0.0] * 7]), enabled=True),
        impl.forward_v5(forged(impl, 3, (1.0, 0.0, top, 0.0)), ones, enabled=True),
        impl.forward_v5(forged(impl, 3, (1.0, 0.0, top, top)), ones, enabled=True),
        impl.update_v5(forged(impl, 3, (0.0, 0.0, big, 0.0)), ones, np.full((2, 3), big), enabled=True),
        impl.update_v5(forged(impl, 3, (0.0, 0.0, 1.0, 0.0)), ones * big, np.full((2, 3), big), enabled=True),
        impl.update_v5(forged(impl, 3, (1.0, 0.0, 0.0, 0.0)), ones, np.full((2, 3), big), enabled=True),
        impl.update_v5(forged(impl, 3, 0.0), ones, np.full((2, 3), top), enabled=True),
        impl.update_v5(forged(impl, 3, 0.0, m=top), ones, np.ones((2, 3)), enabled=True),
        impl.update_v5(forged(impl, 3, 0.0, v=top), ones, np.ones((2, 3)), enabled=True),
        impl.update_v5(forged(impl, 3, 0.0), ones, np.full((2, 3), 1e-200), enabled=True),
        impl.update_v5(forged(impl, 3, top), ones, np.ones((2, 3)), 1.0, enabled=True),
    ]
    require(all(value is None for value in rejected), 'range failure must publish nothing')
    require(impl.update_v5(forged(impl, 3, 0.0, step=impl.STEP_CAP - 1), ones, np.ones((2, 3)), enabled=True) is not None,
            'scalar powers at the step cap')
    require(float(np.abs(np.tanh(np.array([top, -top, 1e-310, 19.0, 0.5]))).max()) <= 1.0, 'tanh range')
    # Silent product underflow in a matrix product stays within the bound of the exact dot product.
    underflow = 0
    with np.errstate(all='ignore'):
        for _ in range(reg['underflowChecks']):
            left = rng.uniform(-1, 1, (1, 9)) * 10.0 ** rng.integers(-170, -150, (1, 9))
            right = rng.uniform(-1, 1, (9, 16)) * 10.0 ** rng.integers(-170, -150, (9, 16))
            got = left @ right
            for j in range(16):
                exact = sum(Q(float(left[0, i])) * Q(float(right[i, j])) for i in range(9))
                largest = sum(abs(Q(float(left[0, i])) * Q(float(right[i, j]))) for i in range(9))
                require(abs(Q(float(got[0, j])) - exact) <= 9 * U * largest + UNDERFLOW, 'matrix-product underflow bound')
                underflow += 1
    # Chains from create_v5: every published update keeps the invariants and moves no parameter beyond the step bound.
    chains = published = refused = 0
    largest_step = largest_parameter = largest_output = largest_initial = Q(0)
    for seed in range(reg['initialSeeds']):
        for outputs in (1, 3):
            largest_initial = max(largest_initial, max(Q(float(np.abs(a).max())) for a in arrays(impl, impl.create_v5(seed, outputs, enabled=True))[0]))
    require(largest_initial <= INITIAL, 'initial parameter bound')
    for seed in reg['chainSeeds']:
        for outputs, lr in ((3, 0.0003), (1, 0.0003), (3, 1.0)):
            snapshot = impl.create_v5(seed, outputs, enabled=True)
            chains += 1
            for index in range(reg['chainSteps']):
                rows = int(rng.integers(1, 17))
                inputs = rng.normal(0, 1, (rows, 9)) * 10.0 ** int(rng.integers(-3, 4))
                residual = rng.normal(0, 1, (rows, outputs)) * 10.0 ** int(rng.integers(-6, 7))
                if index % 11 == 0: residual = np.zeros((rows, outputs))
                following = impl.update_v5(snapshot, inputs, residual, lr, enabled=True)
                if following is None:
                    refused += 1
                    continue
                (p0, _, _), (p1, m1, v1) = arrays(impl, snapshot), arrays(impl, following)
                for old, new, m, v in zip(p0, p1, m1, v1):
                    for a, b_, c_, d_ in zip(old.tolist(), new.tolist(), m.tolist(), v.tolist()):
                        moved = abs(Q(b_) - Q(a))
                        require(moved <= STEP * Q(lr) and abs(Q(c_)) <= MOMENT and 0 <= Q(d_) <= VARIANCE and
                                Q(c_) ** 2 <= RATIO * Q(d_), 'update invariant')
                        largest_step = max(largest_step, moved / Q(lr))
                        largest_parameter = max(largest_parameter, abs(Q(b_)))
                require(following.step == snapshot.step + 1, 'step count')
                out = impl.forward_v5(following, inputs, enabled=True)
                weights = arrays(impl, following)[0]
                limit = 17 * max(Q(float(np.abs(w).max())) for w in weights[2:]) * (1 + U) ** 18
                require(out is not None and Q(float(np.abs(out).max())) <= limit, 'output bound')
                largest_output = max(largest_output, Q(float(np.abs(out).max())))
                snapshot = following; published += 1
    require(published >= chains * reg['chainSteps'] * 3 // 4, 'vacuous chains')
    # Final snapshots of real learner runs satisfy the run bounds.
    import causal_ppo_v7 as ppo
    import causal_replay_v3 as replay
    import causal_value_v6 as value
    from causal_value import fixtures
    learned = 0
    for learner, train, names in ((value, value.train_causal_value_v6, ('online', 'target')),
                                  (ppo, ppo.train_causal_ppo_v7, ('actor', 'critic'))):
        bars, future = fixtures(learner, replay, reg['learnerSteps'])
        for seed in reg['learnerSeeds']:
            result = train(bars, future, seed, enabled=True)
            require(result is not None, 'learner fixture')
            for name in names:
                snapshot = getattr(result, name)
                p, m, v = arrays(impl, snapshot)
                require(snapshot.step <= UPDATES and all(float(np.abs(a).max()) <= PARAMETER for a in p) and
                        all(float(np.abs(a).max()) <= MOMENT for a in m) and
                        all(0 <= float(a.min()) and float(a.max()) <= VARIANCE for a in v) and
                        all(Q(x) ** 2 <= RATIO * Q(y) for a, b_ in zip(m, v) for x, y in zip(a.tolist(), b_.tolist())),
                        'learner snapshot bound')
                learned += 1
    return {'status': 'property_tested', 'rangeRejections': len(rejected), 'underflowChecks': underflow, 'chains': chains,
            'publishedUpdates': published, 'refusedUpdates': refused, 'learnerSnapshots': learned,
            'initialSeeds': reg['initialSeeds'], 'largestInitialParameter': float(largest_initial),
            'largestStepPerLearningRate': float(largest_step), 'largestParameter': float(largest_parameter),
            'largestOutput': float(largest_output), 'seed': reg['seed'], 'marketDataReads': 0, 'economicEvidence': False}


def check_optimizer_numerics():
    reg = json.loads(read(REGISTRATION))
    require((reg['version'], reg['enabledByDefault'], reg['productionIntegration'], reg['financialTrials'],
             reg['historicalDataReads'], reg['holdoutOpened'], reg['runtimeChange']) ==
            ('causal-optimizer-v5', False, False, 0, 0, False, False), 'registration')
    require((reg['momentRatio'], reg['stepPerLearningRate'], reg['initialParameterBound'], reg['updatesPerRun'],
             reg['parameterBoundLog2'], reg['outputBoundLog2']) == (RATIO, float(STEP), INITIAL, UPDATES, 6, 11),
            'registered bounds')
    _, source = extract()
    smt, bounds = prove()
    return {'source': source, 'smt': smt, 'bounds': bounds, 'conformance': conformance()}


if __name__ == '__main__':
    print(json.dumps(check_optimizer_numerics(), indent=1, sort_keys=True))
