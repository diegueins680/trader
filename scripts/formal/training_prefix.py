"""Source-derived fit/episode bounds; not a universal Python program analysis."""
import ast
import copy
from pathlib import Path
import z3 as z

from causal_footprint import check_source, structure, without_doc

ROOT = Path(__file__).resolve().parents[2]
ENV = 'scripts/research/sequential_env.py'
RUNNER = 'scripts/research/run_sequential_screen.py'
REGISTRATION = 'research-notes/registrations/sequential-control-screen-v1.json'

FIT_TEMPLATE = '''
@classmethod
def fit(cls, prefixes: list[np.ndarray]) -> Scale:
    if (not isinstance(prefixes, (list, tuple)) or not prefixes or
        any(not _real_series(p) or len(p) < 25 for p in prefixes)):
        raise ValueError("incomplete training prefixes")
    rows = [market_features(p, t) for p in prefixes for t in range(FIT_START, FIT_STOP)]
    if not rows or any(x is None for x in rows):
        raise ValueError("incomplete training observations")
    x = np.array(rows)
    values = [x.mean(0), np.maximum(x.std(0), 1e-8), x.min(0), x.max(0)]
    return cls(*values)
'''
SETUP_TEMPLATE = '''
train = {s: p[:PRICE_STOP] for s, p in prices.items()}
funds = {s: p[:FUNDING_STOP] for s, p in funding.items()}
scale = Scale.fit(list(train.values()))
'''
EPISODE_TEMPLATE = '''
if env is None or env.done:
    symbol = symbols[int(rng.integers(len(symbols)))]
    p = prices[symbol]
    start = int(rng.integers(START_LOW, START_HIGH))
    env = Replay(p, funding[symbol], start, EPISODE_STOP, horizon, scale, execution, enabled=True)
    starts += 1
'''
COLLECTION_ADMISSION = '''
for symbol, p in prices.items():
    f = funding[symbol]
    if not _real_series(p) or not _real_series(f) or len(p) != len(f) or len(p) <= 120:
        raise ValueError("invalid collection series or insufficient episode history")
'''


def fail(message):
    raise ValueError('training prefix: ' + message)


def unique(nodes, description):
    if len(nodes) != 1:
        fail('missing or repeated ' + description)
    return nodes[0]


def function(module, name):
    return unique([n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name], name)


def expr(text):
    return ast.parse(text, mode='eval').body


def replace_slots(node, slots):
    """Replace just the extracted arithmetic slots before matching full structure."""
    class Slots(ast.NodeTransformer):
        def visit(self, item):
            for actual, name in slots:
                if item is actual:
                    return ast.Name(id=name, ctx=ast.Load())
            return super().visit(item)
    return Slots().visit(node)


def extract_runner(source):
    run = function(ast.parse(source), 'run')
    loop = unique([n for n in ast.walk(run) if isinstance(n, ast.For) and
                   structure(n.target) == structure(ast.parse('fi, split = x').body[0].targets[0])], 'registered fold loop')
    if structure(loop.iter) != structure(expr('enumerate(registration["validation"]["outerFolds"])')) or loop.orelse:
        fail('fold enumeration drift')
    if len(loop.body) < 3:
        fail('missing fold setup')
    setup = copy.deepcopy(ast.Module(body=loop.body[:3], type_ignores=[]))
    bounds = []
    for statement in setup.body[:2]:
        if not (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.DictComp) and
                isinstance(statement.value.value, ast.Subscript) and isinstance(statement.value.value.slice, ast.Slice)):
            fail('prefix comprehension drift')
        part = statement.value.value.slice
        if part.lower is not None or part.step is not None or part.upper is None:
            fail('prefix must start at zero with a bounded unit stride')
        bounds.append(part.upper)
    replaced = replace_slots(setup, list(zip(bounds, ('PRICE_STOP', 'FUNDING_STOP'))))
    if structure(replaced) != structure(ast.parse(SETUP_TEMPLATE)):
        fail('training setup dataflow drift')
    # No alternate fit, rebinding or aliased class invocation inside this runner.
    if sum(isinstance(n, ast.Name) and n.id == 'Scale' for n in ast.walk(run)) != 1:
        fail('alternate Scale use')
    for name in ('train', 'funds', 'scale', 'split'):
        if sum(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store) for n in ast.walk(run)) != 1:
            fail('training binding drift: ' + name)
    return bounds, loop.body[:3]


def extract_fit(source):
    module = ast.parse(source)
    scale = unique([n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'Scale'], 'Scale')
    fit = copy.deepcopy(function(scale, 'fit'))
    fit.body = without_doc(fit)
    rows = unique([n for n in fit.body if isinstance(n, ast.Assign) and
                   len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'rows'], 'fit rows')
    if not isinstance(rows.value, ast.ListComp) or len(rows.value.generators) != 2:
        fail('fit row comprehension drift')
    loop = rows.value.generators[1].iter
    if not isinstance(loop, ast.Call) or len(loop.args) != 2:
        fail('fit range drift')
    low, high = loop.args
    replaced = replace_slots(fit, [(low, 'FIT_START'), (high, 'FIT_STOP')])
    if structure(replaced) != structure(ast.parse(FIT_TEMPLATE).body[0]):
        fail('fit dependency skeleton drift')
    return low, high


def extract_episode(source):
    collect = function(ast.parse(source), 'collect')
    admission = [n for n in collect.body if isinstance(n, ast.For)]
    if len(admission) != 1 or structure(admission[0]) != structure(ast.parse(COLLECTION_ADMISSION).body[0]):
        fail('collection length admission drift')
    whiles = [n for n in collect.body if isinstance(n, ast.While)]
    loop = unique(whiles, 'collection loop')
    if structure(loop.test) != structure(expr('len(rows) < count')) or not loop.body:
        fail('collection loop drift')
    episode = copy.deepcopy(loop.body[0])
    if not isinstance(episode, ast.If) or len(episode.body) != 5:
        fail('episode setup drift')
    try:
        low, high = episode.body[2].value.args[0].args
        stop = episode.body[3].value.args[3]
    except (AttributeError, IndexError, TypeError, ValueError):
        fail('episode arithmetic slots missing')
    replaced = replace_slots(episode, [(low, 'START_LOW'), (high, 'START_HIGH'), (stop, 'EPISODE_STOP')])
    if structure(replaced) != structure(ast.parse(EPISODE_TEMPLATE).body[0]):
        fail('episode setup dataflow drift')
    return low, high, stop


def arithmetic(node, symbols):
    key = ast.unparse(node)
    if key in symbols:
        return symbols[key]
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return z.IntVal(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = arithmetic(node.left, symbols), arithmetic(node.right, symbols)
        return left + right if isinstance(node.op, ast.Add) else left - right
    fail('unsupported index expression: ' + key)


def prove(name, premise, conclusion):
    solver = z.Solver()
    solver.set(timeout=10000, random_seed=0)
    solver.add(premise)
    if solver.check() != z.sat:
        raise RuntimeError(name + ': unsatisfied or unknown premise')
    solver.add(z.Not(conclusion))
    result = solver.check()
    if result != z.unsat:
        detail = str(solver.model()) if result == z.sat else solver.reason_unknown()
        raise RuntimeError(name + ': violating source bound: ' + detail)


def registered_folds(registration):
    n = registration['data']['rowsPerSymbol']
    horizons = registration['data']['decisionHorizonBars']
    rules = registration['validation']
    if (type(n) is not int or n < 121 or horizons != [1, 3, 6] or
            any(type(h) is not int for h in horizons) or
            type(rules['purgeBars']) is not int or rules['purgeBars'] != 6 or
            type(rules['embargoBars']) is not int or rules['embargoBars'] != 6):
        fail('registered domain drift')
    folds = rules['outerFolds']
    if not isinstance(folds, list) or len(folds) != 3:
        fail('registered fold roster drift')
    for fold in folds:
        if (not isinstance(fold, dict) or set(fold) != {'trainStop', 'testStart', 'testStop'} or
                any(type(v) is not int for v in fold.values()) or
                not 121 <= fold['trainStop'] < fold['testStart'] < fold['testStop'] <= n or
                any(fold['testStart'] < fold['trainStop'] + h for h in horizons)):
            fail('registered split isolation failed')
    return [{'trainStop': f['trainStop'], 'testStart': f['testStart'],
             'testStop': f['testStop'], 'gapBars': f['testStart'] - f['trainStop']} for f in folds]


def check_training(registration, env_source=None, runner_source=None):
    env_source = (ROOT / ENV).read_text() if env_source is None else env_source
    runner_source = (ROOT / RUNNER).read_text() if runner_source is None else runner_source
    check_source(env_source)
    bounds, _ = extract_runner(runner_source)
    fit_low, fit_high = extract_fit(env_source)
    start_low, start_high, episode_stop = extract_episode(env_source)
    T, N, U, t, i, L, start = z.Ints('T N U t i L start')
    symbols = {"split['trainStop']": T, 'len(p)': U}
    upper = [arithmetic(b, symbols) for b in bounds]
    for value in upper:
        prove('F-RL-FIT-PREFIX', z.And(T >= 25, T <= N), z.And(value >= 25, value <= T))
    low, high = arithmetic(fit_low, symbols), arithmetic(fit_high, symbols)
    prove('F-RL-FIT-PREFIX', z.And(U >= 25, U <= T), z.And(low >= 24, low < high, high <= U))
    prove('F-RL-FIT-PREFIX', z.And(U >= 25, U <= T, t >= low, t < high, i >= t-24, i <= t), z.And(i >= 0, i < T))
    lo = arithmetic(start_low, {'len(p)': L})
    hi = arithmetic(start_high, {'len(p)': L})
    stop = arithmetic(episode_stop, {'start': start})
    prove('F-RL-COLLECT-PREFIX', L >= 121, z.And(lo >= 24, lo < hi))
    prove('F-RL-COLLECT-PREFIX', z.And(L >= 121, L <= T, start >= lo, start < hi),
          z.And(start >= 24, start < stop - 1, stop <= L, stop <= T))
    return {'smt': {'F-RL-FIT-PREFIX': 'unsat', 'F-RL-COLLECT-PREFIX': 'unsat'},
            'solverQueries': 6, 'premiseChecks': 6,
            'runnerPrefixStops': [ast.unparse(b) for b in bounds],
            'fitRange': [ast.unparse(fit_low), ast.unparse(fit_high)],
            'episodeStartRange': [ast.unparse(start_low), ast.unparse(start_high)],
            'episodeStop': ast.unparse(episode_stop),
            'registeredFolds': registered_folds(registration), 'foldHorizonCases': 9,
            'universalRuntimeRefinement': False}
