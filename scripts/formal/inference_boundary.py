"""Source-bound inference guards and a non-preemptive call model; scoped only."""
import ast
import copy
from collections import deque
from pathlib import Path
import z3 as z
from causal_footprint import structure
from target_v2 import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/sequential_learning.py'
ENV = 'scripts/research/sequential_env.py'
TEMPLATE = '''
def _finite_real_vector(value, width: int) -> bool:
    return (isinstance(value, np.ndarray) and not np.ma.isMaskedArray(value) and value.shape == (width,) and
            value.dtype.kind in "iuf" and bool(np.isfinite(value).all()))

def infer(net: Network, observation: np.ndarray | None, *, enabled: bool = False) -> tuple[float | None, float]:
    start = time.perf_counter_ns()
    if EARLY:
        return None, (time.perf_counter_ns() - start) / 1e6
    try:
        out = net.forward(observation)
    except Exception:
        out = None
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if LATE:
        return None, elapsed
    return float(ACTIONS[int(np.argmax(out))]), elapsed
'''


def extract(source, env):
    actual = {}
    for name in ('_finite_real_vector', 'infer'):
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(nodes) != 1:
            raise ValueError('inference boundary: missing/duplicate function')
        actual[name] = copy.deepcopy(nodes[0])
    try:
        predicates = [actual['infer'].body[i].test for i in (1, 4)]
        for i, name in ((1, 'EARLY'), (4, 'LATE')):
            actual['infer'].body[i].test = ast.Name(id=name, ctx=ast.Load())
    except (IndexError, AttributeError):
        raise ValueError('inference boundary: control-flow drift')
    if any(structure(actual[n.name]) != structure(n) for n in ast.parse(TEMPLATE).body):
        raise ValueError('inference boundary: source skeleton drift')
    constants = {}
    for n in ast.parse(env).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('ACTIONS', 'FEATURE_COUNT') for t in n.targets):
            if len(n.targets) != 1 or n.targets[0].id in constants:
                raise ValueError('inference boundary: duplicate/aliased constant')
            constants[n.targets[0].id] = ast.unparse(n.value)
    if constants != {'ACTIONS': 'np.array([-0.25, 0.0, 0.25])', 'FEATURE_COUNT': '12'}:
        raise ValueError('inference boundary: action/schema drift')
    return predicates


def predicate(node, atoms):
    key = ast.unparse(node)
    if key in atoms:
        return atoms[key]
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return z.FPVal(node.value, z.Float64())
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return z.Not(predicate(node.operand, atoms))
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        return (z.And if isinstance(node.op, ast.And) else z.Or)(*[predicate(v, atoms) for v in node.values])
    if isinstance(node, ast.Compare) and all(isinstance(op, ast.LtE) for op in node.ops):
        values = [predicate(v, atoms) for v in [node.left, *node.comparators]]
        return z.And(*[z.fpLEQ(a, b) for a, b in zip(values, values[1:])])
    raise ValueError('inference boundary: unsupported predicate ' + key)


def check_predicates(predicates):
    enabled, obs, out = z.Bools('infer_exact_true infer_valid_obs infer_valid_out')
    elapsed = z.FP('infer_elapsed', z.Float64())
    atoms = {'enabled is not True': z.Not(enabled),
             '_finite_real_vector(observation, FEATURE_COUNT)': obs,
             '_finite_real_vector(out, 3)': out, 'elapsed': elapsed}
    early, late = [predicate(n, atoms) for n in predicates]
    finite = z.And(z.Not(z.fpIsNaN(elapsed)), z.Not(z.fpIsInf(elapsed)))
    expected = z.And(enabled, obs, out, finite,
                     z.fpLEQ(z.FPVal(0, z.Float64()), elapsed),
                     z.fpLEQ(elapsed, z.FPVal(20, z.Float64())))
    certify('F-RL-INFER-ADMISSION', z.BoolVal(True), z.Not(z.Or(early, late)) == expected, [])
    q = z.Reals('infer_q0 infer_q1 infer_q2')
    index = z.If(z.And(q[0] >= q[1], q[0] >= q[2]), 0, z.If(q[1] >= q[2], 1, 2))
    score = z.If(index == 0, q[0], z.If(index == 1, q[1], q[2]))
    action = z.If(index == 0, z.RealVal('-1/4'), z.If(index == 1, 0, z.RealVal('1/4')))
    claims = [action >= z.RealVal('-1/4'), action <= z.RealVal('1/4'),
              z.Or(action == z.RealVal('-1/4'), action == 0, action == z.RealVal('1/4'))]
    for i in range(3):
        claims.extend([score >= q[i], z.Implies(score == q[i], index <= i)])
    certify('F-RL-INFER-SELECTION', z.BoolVal(True), z.And(*claims), [])
    return {'F-RL-INFER-ADMISSION': 'unsat', 'F-RL-INFER-SELECTION': 'unsat'}


def transitions(state):
    phase, flags = state
    if phase == 'entry':
        yield 'disabled', ('absent', flags)
        yield 'enable', ('observation', flags | 1)
    elif phase == 'observation':
        yield 'invalid_observation', ('absent', flags)
        yield 'admit_observation', ('running', flags | 2 | 16)
    elif phase == 'running':
        yield 'pending', state
        yield 'ordinary_exception', ('absent', flags)
        yield 'invalid_output', ('absent', flags)
        yield 'valid_output', ('timing', flags | 4)
    elif phase == 'timing':
        yield 'late_or_invalid_time', ('absent', flags)
        yield 'timely', ('proposal', flags | 8)
    else:
        yield 'terminal_stutter', state


def check_model(fixture):
    initial = ('entry', 0)
    queue = deque([initial]); depth = {initial: 0}; edges = 0
    while queue:
        state = queue.popleft(); phase, flags = state
        if (flags & 16 and flags & 3 != 3) or (phase == 'proposal' and flags != 31):
            raise RuntimeError('inference boundary: gate bypass')
        for event, nxt in transitions(state):
            edges += 1
            if phase in ('absent', 'proposal') and nxt != state:
                raise RuntimeError('inference boundary: terminal escaped')
            if nxt not in depth:
                depth[nxt] = depth[state] + 1; queue.append(nxt)
    if ('proposal', 31) not in depth:
        raise RuntimeError('inference boundary: vacuous proposal exclusion')
    entry = fixture['entries'][0]
    if fixture.get('schemaVersion') != 1 or len(fixture['entries']) != 1 or entry['id'] != 'CE-RL-017':
        raise ValueError('inference boundary: witness fixture drift')
    if entry['prefix'] != ['enable', 'admit_observation'] or entry['cycle'] != ['pending']:
        raise ValueError('inference boundary: witness trace drift')
    cursor = initial
    for event in entry['prefix']:
        matches = [n for e, n in transitions(cursor) if e == event]
        if len(matches) != 1: raise RuntimeError('inference boundary: invalid prefix')
        cursor = matches[0]
    if cursor != ('running', 19) or ('pending', cursor) not in list(transitions(cursor)):
        raise RuntimeError('inference boundary: pending lasso missing')
    if entry['elapsedNanoseconds'] != 25000000 or entry['expectedProposal'] is not None:
        raise ValueError('inference boundary: timing witness drift')
    return {'states': len(depth), 'transitions': edges, 'maxShortestDepth': max(depth.values()),
            'calls': 1, 'flagBits': 5, 'search': 'reachable_fixed_point',
            'deadlineCounterexample': {'prefix': entry['prefix'], 'cycle': entry['cycle']},
            'preemptionVerified': False, 'terminationWithoutReturnAssumption': False,
            'liveAuthorizationTransitions': 0}


def check_inference(fixtures, source=None, env=None):
    predicates = extract((ROOT / SOURCE).read_text() if source is None else source,
                         (ROOT / ENV).read_text() if env is None else env)
    return {'smt': check_predicates(predicates), 'queries': 2, 'premiseChecks': 2,
            'model': check_model(fixtures), 'runtimeRefinement': False,
            'clockConversionVerified': False, 'neuralRobustnessVerified': False}
