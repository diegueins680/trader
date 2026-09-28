"""Source-linked admission predicates; limited by explicit runtime assumptions."""
import ast
import copy
from pathlib import Path
import z3 as z
from causal_footprint import structure, without_doc

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/sequential_env.py'
HELPER_TEMPLATE = r'''
def _admit_training_transition(env: Replay, left: int, nxt: np.ndarray | None, reward: float, done: bool) -> None:
    accounted_risk = ('capital_floor', 'drawdown_limit', 'endpoint_exposure', 'turnover_limit')
    valid = BASE_VALID
    if TERMINAL_BRANCH:
        valid = TERMINAL_VALID
    else:
        valid = NONTERMINAL_VALID
    if not valid:
        raise ValueError(f"incomplete training transition: {env.failure or 'invalid successor state'}")
'''
COLLECT_TEMPLATE = r'''
def collect(prices: dict[str, np.ndarray], funding: dict[str, np.ndarray], scale: Scale, horizon: int, seed: int, count: int, policy=None, execution: Execution=Execution()) -> dict:
    """Training-only prefixes physically bound every replay and episode."""
    if not all((_integer(v) for v in (horizon, seed, count))) or horizon not in (1, 3, 6) or seed < 0 or (count <= 0):
        raise ValueError('invalid collection horizon, seed or transition budget')
    if not isinstance(prices, dict) or not isinstance(funding, dict) or (not prices) or (set(prices) != set(funding)) or any((not isinstance(s, str) or not s for s in prices)):
        raise ValueError('invalid collection symbol coverage')
    for symbol, p in prices.items():
        f = funding[symbol]
        if not _real_series(p) or not _real_series(f) or len(p) != len(f) or (len(p) <= 120):
            raise ValueError('invalid collection series or insufficient episode history')
    horizon, seed, count = (int(horizon), int(seed), int(count))
    rng = np.random.default_rng(seed)
    symbols = sorted(prices)
    rows, episodes, env, starts = ([], [], None, 0)
    while len(rows) < count:
        if env is None or env.done:
            symbol = symbols[int(rng.integers(len(symbols)))]
            p = prices[symbol]
            start = int(rng.integers(24, len(p) - 96))
            env = Replay(p, funding[symbol], start, start + 97, horizon, scale, execution, enabled=True)
            starts += 1
        s = env.observation()
        if s is None:
            raise ValueError('invalid training observation')
        probs = np.full(3, 1 / 3) if policy is None else policy(s)
        if not _real_series(probs) or probs.shape != (3,) or (not np.isfinite(probs).all()) or np.any(probs < 0) or (not np.isclose(probs.sum(), 1)):
            raise ValueError('invalid behavior probabilities')
        a = int(rng.choice(3, p=probs))
        left = env.t
        nxt, reward, done = env.step(float(ACTIONS[a]))
        _admit_training_transition(env, left, nxt, reward, done)
        rows.append((s, a, reward, np.zeros(FEATURE_COUNT) if nxt is None else nxt, done, probs[a]))
        if done:
            episodes.append({'return': env.equity - 1, 'failure': env.failure})
    return {'s': np.array([r[0] for r in rows]), 'a': np.array([r[1] for r in rows]), 'r': np.array([r[2] for r in rows]), 'next': np.array([r[3] for r in rows]), 'done': np.array([r[4] for r in rows]), 'prob': np.array([r[5] for r in rows]), 'episodes': episodes, 'episodeAccountingV2': {'collections': 1, 'started': starts, 'completed': len(episodes), 'truncated': int(not env.done), 'decisions': len(rows)}}
'''


def function(module, name):
    found = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise ValueError('transition admission: missing or repeated function ' + name)
    return copy.deepcopy(found[0])


def extract(source):
    module = ast.parse(source)
    helper = function(module, '_admit_training_transition')
    helper.body = without_doc(helper)
    try:
        expressions = [helper.body[1].value, helper.body[2].test,
                       helper.body[2].body[0].value, helper.body[2].orelse[0].value]
        helper.body[1].value = ast.Name(id='BASE_VALID', ctx=ast.Load())
        helper.body[2].test = ast.Name(id='TERMINAL_BRANCH', ctx=ast.Load())
        helper.body[2].body[0].value = ast.Name(id='TERMINAL_VALID', ctx=ast.Load())
        helper.body[2].orelse[0].value = ast.Name(id='NONTERMINAL_VALID', ctx=ast.Load())
    except (AttributeError, IndexError):
        raise ValueError('transition admission: helper control-flow drift')
    if structure(helper) != structure(ast.parse(HELPER_TEMPLATE).body[0]):
        raise ValueError('transition admission: helper skeleton drift')
    collect = function(module, 'collect')
    if structure(collect) != structure(ast.parse(COLLECT_TEMPLATE).body[0]):
        raise ValueError('transition admission: collector admission/append ordering drift')
    return expressions


def term(node, atoms):
    key = ast.unparse(node)
    if key in atoms:
        return atoms[key]
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return z.BoolVal(node.value)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return z.IntVal(node.value)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        return (z.And if isinstance(node.op, ast.And) else z.Or)(*[term(n, atoms) for n in node.values])
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        a, b = term(node.left, atoms), term(node.right, atoms)
        return a + b if isinstance(node.op, ast.Add) else a - b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'min' and len(node.args) == 2 and not node.keywords:
        a, b = [term(n, atoms) for n in node.args]
        return z.If(a <= b, a, b)
    if isinstance(node, ast.Compare):
        values = [term(n, atoms) for n in [node.left, *node.comparators]]
        comparisons = []
        for a, b, op in zip(values, values[1:], node.ops):
            if z.is_fp(a) or z.is_fp(b):
                a = a if z.is_fp(a) else z.FPVal(a.as_long(), z.Float64())
                b = b if z.is_fp(b) else z.FPVal(b.as_long(), z.Float64())
                operators = {ast.Eq: z.fpEQ, ast.Gt: z.fpGT, ast.GtE: z.fpGEQ, ast.Lt: z.fpLT, ast.LtE: z.fpLEQ}
            else:
                operators = {ast.Eq: lambda x, y: x == y, ast.Gt: lambda x, y: x > y,
                             ast.GtE: lambda x, y: x >= y, ast.Lt: lambda x, y: x < y,
                             ast.LtE: lambda x, y: x <= y}
            if type(op) not in operators:
                raise ValueError('transition admission: unsupported comparison ' + key)
            comparisons.append(operators[type(op)](a, b))
        return z.And(*comparisons)
    raise ValueError('transition admission: unsupported predicate ' + key)


def prove(premise, conclusion):
    solver = z.Solver()
    solver.set(timeout=10000, random_seed=0)
    solver.add(premise)
    if solver.check() != z.sat:
        raise RuntimeError('transition admission: unsatisfied or unknown premise')
    solver.add(z.Not(conclusion))
    result = solver.check()
    if result != z.unsat:
        detail = str(solver.model()) if result == z.sat else solver.reason_unknown()
        raise RuntimeError('transition admission: unsafe admission predicate: ' + detail)


def check_transition(source=None):
    source = (ROOT / SOURCE).read_text() if source is None else source
    base_node, branch_node, terminal_node, nonterminal_node = extract(source)
    left, t, horizon, stop, failure, done = z.Ints('left t horizon stop failure done_tag')
    reward, equity, units = z.FPs('reward equity units', z.Float64())
    nxt_none, pending_none, represented, shaped, finite_vector = z.Bools(
        'nxt_none pending_none represented shaped finite_vector')
    finite = lambda v: z.And(z.Not(z.fpIsNaN(v)), z.Not(z.fpIsInf(v)))
    accounted = z.Or(*[failure == i for i in (1, 2, 3, 4)])
    atoms = {'left': left, 'env.t': t, 'env.horizon': horizon, 'env.stop': stop,
             'env.equity': equity, 'env.units': units,
             'env.failure in (None, *accounted_risk)': z.Or(failure == 0, accounted),
             'env.failure in accounted_risk': accounted, 'env.failure is None': failure == 0,
             '_finite_real(reward)': finite(reward), '_finite_real(env.equity)': finite(equity),
             'done is True': done == 0, 'done is False': done == 1,
             'nxt is None': nxt_none, 'env.pending is None': pending_none,
             '_real_series(nxt)': represented, 'nxt.shape == (FEATURE_COUNT,)': shaped,
             'np.isfinite(nxt).all()': finite_vector}
    base = term(base_node, atoms)
    atoms['valid'] = base
    accepted = z.If(term(branch_node, atoms), term(terminal_node, atoms), term(nonterminal_node, atoms))
    core = z.And(finite(reward), finite(equity), z.fpGT(equity, z.FPVal(0, z.Float64())),
                 z.Or(failure == 0, accounted), left < t, t <= left + horizon, t <= stop - 1)
    terminal = z.And(nxt_none, z.fpEQ(units, z.FPVal(0, z.Float64())), pending_none,
                     z.Or(accounted, t == stop - 1))
    nonterminal = z.And(done == 1, failure == 0, t == left + horizon, t < stop - 1,
                        represented, shaped, finite_vector)
    prove(accepted, core)
    prove(z.And(accepted, done == 0), terminal)
    prove(z.And(accepted, done != 0), nonterminal)
    return {'requirement': 'F-RL-TRANSITION-ADMISSION', 'result': 'unsat',
            'queries': 3, 'premiseChecks': 3, 'integerIndices': 'unbounded',
            'numericDomain': 'IEEE binary64 reward/equity/units',
            'collectorAdmissionBeforeAppend': True, 'universalRuntimeRefinement': False}
