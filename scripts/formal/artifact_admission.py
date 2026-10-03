"""Audited loader cut points and source-derived predicates; not Python refinement."""
import ast
import copy
from collections import deque
from pathlib import Path
import z3 as z
from causal_footprint import structure

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/sequential_learning.py'
TEMPLATE = r'''
def load_policy(path: Path, expected_sha256: str, expected_provenance: dict) -> Network:
    validate_provenance(expected_provenance)
    with path.open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError('artifact too large')
    if HASH_REJECT:
        raise ValueError('artifact hash mismatch')

    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('duplicate artifact field')
            d[k] = v
        return d
    a = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite artifact')))
    if not isinstance(a, dict) or set(a) != {'schema', 'environment', 'observation', 'actions', 'promotion', 'enabled', 'provenance', 'parameters'}:
        raise ValueError('artifact fields')
    if not isinstance(a['actions'], list) or any((type(v) not in (int, float) for v in a['actions'])) or (not isinstance(a['provenance'], dict)) or (not isinstance(a['parameters'], dict)):
        raise ValueError('artifact types')
    validate_provenance(a['provenance'])
    if METADATA_REJECT:
        raise ValueError('incompatible artifact')

    def numeric(value):
        return all((numeric(v) for v in value)) if isinstance(value, list) else type(value) in (int, float)
    parameters = {}
    for name, value in a['parameters'].items():
        if not numeric(value):
            raise ValueError('non-numeric parameter')
        try:
            parameters[name] = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('invalid numeric parameters') from exc
    snapshots = _parameter_snapshots(parameters)
    net = Network(0)
    net.p = snapshots
    return net
'''

# Every effectful top-level group, in the audited source order. Local function
# definitions and an empty dictionary allocation are not validation gates.
CUTS = ((0, 'expected_provenance'), (1, 'snapshot_read'), (2, 'size'),
        (3, 'digest'), (5, 'json'), (6, 'fields'), (7, 'types'),
        (8, 'artifact_provenance'), (9, 'compatibility'),
        (12, 'parameter_conversion'), (13, 'parameter_validation'),
        (14, 'construction'), (15, 'parameter_binding'))


def extract(source):
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'load_policy']
    if len(functions) != 1:
        raise ValueError('artifact admission: missing/duplicate loader')
    function = copy.deepcopy(functions[0])
    if len(function.body) != 17 or not all(isinstance(function.body[i], ast.If) for i in (3, 9)):
        raise ValueError('artifact admission: control-flow drift')
    predicates = [function.body[i].test for i in (3, 9)]
    for index, name in ((3, 'HASH_REJECT'), (9, 'METADATA_REJECT')):
        function.body[index].test = ast.Name(id=name, ctx=ast.Load())
    if structure(function) != structure(ast.parse(TEMPLATE).body[0]):
        raise ValueError('artifact admission: audited loader skeleton drift')
    return predicates, [name for _, name in CUTS]


def expression(node, atoms):
    key = ast.unparse(node)
    if key in atoms:
        return atoms[key]
    if isinstance(node, ast.Constant) and type(node.value) is str:
        return z.StringVal(node.value)
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return z.BoolVal(node.value)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.Or, ast.And)):
        return (z.Or if isinstance(node.op, ast.Or) else z.And)(*[expression(n, atoms) for n in node.values])
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
        left, right = expression(node.left, atoms), expression(node.comparators[0], atoms)
        return left == right if isinstance(node.ops[0], ast.Eq) else left != right
    raise ValueError('artifact admission: unsupported predicate: ' + key)


def prove(premise, conclusion):
    solver = z.Solver()
    solver.set(timeout=10000, random_seed=0)
    solver.add(premise)
    if solver.check() != z.sat:
        raise RuntimeError('artifact admission: unsatisfied/unknown premise')
    solver.add(z.Not(conclusion))
    result = solver.check()
    if result != z.unsat:
        detail = str(solver.model()) if result == z.sat else solver.reason_unknown()
        raise RuntimeError('artifact admission: unsafe metadata predicate: ' + detail)


def check_predicates(predicates):
    schema, environment, observation, promotion, provenance, expected, digest, expected_digest = z.Strings(
        'schema environment observation promotion provenance expected digest expected_digest')
    actions_match, exact_false = z.Bools('actions_match enabled_exact_false')
    atoms = {"a['schema']": schema, "a['environment']": environment,
             "a['observation']": observation, "a['promotion']": promotion,
             'ENVIRONMENT': z.StringVal('sequential_replay_v1'),
             'OBSERVATION': z.StringVal('close_inventory_v1'),
             "a['actions'] != ACTIONS.tolist()": z.Not(actions_match),
             "a['enabled'] is not False": z.Not(exact_false),
             "json.dumps(a['provenance'], sort_keys=True)": provenance,
             'json.dumps(expected_provenance, sort_keys=True)': expected,
             'hashlib.sha256(raw).hexdigest()': digest, 'expected_sha256': expected_digest}
    hash_reject, metadata_reject = [expression(n, atoms) for n in predicates]
    metadata_ok = z.And(schema == 'offline_policy_v1', environment == 'sequential_replay_v1',
                        observation == 'close_inventory_v1', promotion == 'rejected_research_only',
                        actions_match, exact_false, provenance == expected)
    prove(z.Not(metadata_reject), metadata_ok)
    prove(z.Not(hash_reject), digest == expected_digest)
    return {'requirement': 'F-RL-ARTIFACT-METADATA', 'queries': 2, 'premiseChecks': 2, 'result': 'unsat'}


def successors(state, count):
    index, passed, failed = state
    if failed or index == count:
        yield state  # terminal stuttering; liveness is eventual terminality.
    else:
        yield (index + 1, passed | (1 << index), False)
        yield (index, passed, True)


def check_model(gates):
    count = len(gates)
    initial = (0, 0, False)
    depths = {initial: 0}
    queue = deque([initial])
    edges = 0
    while queue:
        state = queue.popleft()
        index, passed, failed = state
        returned = index == count and not failed
        if passed != (1 << index) - 1 or (returned and passed != (1 << count) - 1):
            raise RuntimeError('artifact admission: gate bypass')
        for next_state in successors(state, count):
            edges += 1
            if failed or index == count:
                if next_state != state:
                    raise RuntimeError('artifact admission: terminal escaped')
            elif next_state not in ((index + 1, passed | (1 << index), False), (index, passed, True)):
                raise RuntimeError('artifact admission: gate bypass or nonterminating gate')
            if next_state not in depths:
                depths[next_state] = depths[state] + 1
                queue.append(next_state)
    if not any(s[0] == count and not s[2] for s in depths):
        raise RuntimeError('artifact admission: vacuous return exclusion')
    return {'requirement': 'F-RL-ARTIFACT-PATH', 'gates': gates, 'states': len(depths),
            'transitions': edges, 'maxDepth': max(depths.values()),
            'terminationGateBound': count, 'terminalStuttering': True,
            'liveAuthorizationTransitions': 0, 'universalRuntimeRefinement': False}


def check_artifact(source=None):
    source = (ROOT / SOURCE).read_text() if source is None else source
    predicates, gates = extract(source)
    return {'metadata': check_predicates(predicates), 'model': check_model(gates)}
