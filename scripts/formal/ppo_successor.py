"""Source-bound offline PPO composition; conditional control model, not runtime proof."""
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import z3 as z

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/ppo_successor_v2.py'
REGISTRY = 'formal/research/ppo-successor-source.json'
DEFINITIONS = {'TrainingResult', '_configuration', '_array', '_prefixes', '_forward',
               '_rollout', '_targets', '_update', 'train_ppo_v2'}
HELPERS = {'scripts/research/gae_targets_v2.py', 'scripts/research/optimizer_snapshot_v2.py',
           'scripts/research/sequential_env.py', 'scripts/research/sequential_learning.py'}


def require(ok, reason):
    if not ok:
        raise ValueError('PPO successor: ' + reason)


def shape(node):
    return ast.dump(node, include_attributes=False)


def extract(source=None, registry=None):
    tree = ast.parse((ROOT / SOURCE).read_text() if source is None else source)
    registry = json.loads((ROOT / REGISTRY).read_text()) if registry is None else registry
    require(type(registry['schemaVersion']) is int and registry['schemaVersion'] == 1, 'schema version')
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    require(set(nodes) == DEFINITIONS and len([n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]) == len(DEFINITIONS), 'definition coverage')
    require(set(registry['definitions']) == DEFINITIONS and set(registry['helperHashes']) == HELPERS, 'mandatory coverage omitted')
    # Every full reviewed body is required: unknown or omitted constructs fail closed.
    for name, node in nodes.items():
        require(shape(node) == registry['definitions'][name], 'reviewed body drift: ' + name)
    imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    calls = sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)})
    require(imports == registry['imports'] and calls == registry['calls'], 'unknown effect boundary')
    require(hashlib.sha256(shape(tree).encode()).hexdigest() == registry['astSha256'], 'module initialization drift')
    for path, expected in registry['helperHashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, 'helper semantics drift: ' + path)
    train = nodes['train_ppo_v2']
    default = ast.parse('if enabled is not True or type(version) is not str or version != VERSION:\n return None').body[0]
    require(shape(train.body[0]) == shape(default), 'default-enabled or unsupported-version path')
    require([ast.unparse(v) for v in train.args.kw_defaults] == ['False', 'VERSION'], 'public defaults')
    publish = [n for n in ast.walk(train) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'TrainingResult']
    require(len(publish) == 1, 'result publication count')
    batch_loop = next(n for n in ast.walk(train) if isinstance(n, ast.For) and ast.unparse(n.target) == 'batch')
    require(isinstance(batch_loop.iter, ast.Call) and ast.unparse(batch_loop.iter.func) == 'range', 'batch loop')
    count = batch_loop.body[0].value
    require(ast.unparse(count.func) == 'min' and len(count.args) == 2, 'count cap')
    return nodes, batch_loop.iter.args[0], count


def expression(node, bindings):
    label = ast.unparse(node)
    if label in bindings:
        return bindings[label]
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return z.IntVal(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return z.Not(expression(node.operand, bindings))
    if isinstance(node, ast.BoolOp):
        fn = z.And if isinstance(node.op, ast.And) else z.Or
        return fn(*[expression(v, bindings) for v in node.values])
    if isinstance(node, ast.BinOp):
        a, b = expression(node.left, bindings), expression(node.right, bindings)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.FloorDiv): return a / b
        if isinstance(node.op, ast.Pow) and isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            return z.IntVal(node.left.value ** node.right.value)
    if isinstance(node, ast.Call) and label.startswith('min(') and len(node.args) == 2:
        a, b = (expression(n, bindings) for n in node.args)
        return z.If(a < b, a, b)
    if isinstance(node, ast.Compare):
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.In) and isinstance(node.comparators[0], ast.Tuple):
            return z.Or(*[expression(node.left, bindings) == expression(v, bindings) for v in node.comparators[0].elts])
        terms = []
        ns = [node.left, *node.comparators]
        for a, op, b in zip(ns, node.ops, ns[1:]):
            x, y = expression(a, bindings), expression(b, bindings)
            if isinstance(op, ast.LtE): terms.append(x <= y)
            elif isinstance(op, ast.Lt): terms.append(x < y)
            elif isinstance(op, ast.Eq): terms.append(x == y)
            else: raise ValueError('unsupported comparison: ' + label)
        return z.And(*terms)
    raise ValueError('unsupported expression: ' + label)


def certify(premise, conclusion):
    for formula, wanted in ((premise, z.sat), (z.And(premise, z.Not(conclusion)), z.unsat)):
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0); solver.add(formula)
        require(solver.check() == wanted, 'SAT/UNKNOWN/vacuous proof')


def prove_bounds(extracted):
    nodes, batches_node, count_node = extracted
    steps, seed, horizon, batch = z.Ints('ppo_steps ppo_seed ppo_horizon ppo_batch')
    types = {f'type({key}) is int': z.Bool('base_int_' + key) for key in ('steps', 'seed', 'horizon')}
    bindings = {'steps': steps, 'seed': seed, 'horizon': horizon, 'batch': batch, **types}
    config = expression(nodes['_configuration'].body[0].value, bindings)
    batches = expression(batches_node, bindings)
    count = expression(count_node, bindings)
    certify(config, z.And(*types.values(), steps >= 1, steps <= 4096, seed >= 0,
                          seed + 1000 + batches - 1 < 2**32, batches >= 1, batches <= 16, z.Or(horizon == 1, horizon == 3, horizon == 6)))
    premise = z.And(config, batch >= 0, batch < batches)
    certify(premise, z.And(count >= 1, count <= 256, batch * 256 + count <= steps))
    certify(z.And(premise, batch < batches - 1), z.And(count == 256, (batch + 1) * 256 <= steps))
    certify(z.And(premise, batch == batches - 1), batch * 256 + count == steps)
    loss = z.FP('ppo_loss', z.Float64())
    finite = z.And(z.Not(z.fpIsNaN(loss)), z.Not(z.fpIsInf(loss)))
    loss_guard = nodes['_update'].body[1].test
    rejected = expression(loss_guard, {
        'type(loss) is not float': z.Bool('wrong_loss_type'),
        'np.isfinite(loss)': finite,
        '_array(gradient, (count, 3))': z.Bool('valid_gradient'),
    })
    certify(z.Not(rejected), finite)
    return {'F-RL-PPO-V2-BOUNDS': 'unsat', 'F-RL-PPO-V2-FINITE': 'unsat'}


# (phase,batch,epoch,actor_updates,critic_updates,losses); private optimizer progress
# is not publication. Helper failure is terminal absent, including critic failure
# after an actor update. No scheduler, arithmetic or library theorem is implied.
def transitions(s, batches=2, mutant=False):
    phase, b, e, a, c, losses = s
    if phase in ('absent', 'published'):
        return []
    failed = ('absent', b, e, a, c, losses)
    result = [failed]
    if phase == 'start': nxt = ('prefixes', 0, 0, 0, 0, 0)
    elif phase == 'prefixes': nxt = ('fit', b, e, a, c, losses)
    elif phase == 'fit': nxt = ('create', b, e, a, c, losses)
    elif phase == 'create': nxt = ('collect', b, e, a, c, losses)
    elif phase == 'collect': nxt = ('rollout', b, e, a, c, losses)
    elif phase == 'rollout': nxt = ('targets', b, e, a, c, losses)
    elif phase == 'targets': nxt = ('actor', b, e, a, c, losses)
    elif phase == 'actor': nxt = ('critic', b, e, a + 1, c, losses)
    elif phase == 'critic': nxt = ('loss', b, e, a, c + 1, losses)
    elif phase == 'loss':
        nxt = (('actor', b, e + 1, a, c, losses + 1) if e < 3 else
               ('collect', b + 1, 0, a, c, losses + 1) if b < batches - 1 else
               ('seal', b, e, a, c, losses + 1))
    elif phase == 'seal': nxt = ('published', b, e, a, c, losses)
    else: raise ValueError('unknown phase')
    result.append(nxt)
    if mutant and phase == 'critic':
        result.append(('published', b, e, a, c, losses))
    return result


def rank(state, batches):
    phase, b, e, _, _, _ = state
    if phase in ('absent', 'published'):
        return 0
    position = {'start': 0, 'prefixes': 1, 'fit': 2, 'create': 3,
                'collect': 4 + b * 15, 'rollout': 5 + b * 15,
                'targets': 6 + b * 15, 'actor': 7 + b * 15 + e * 3,
                'critic': 8 + b * 15 + e * 3, 'loss': 9 + b * 15 + e * 3,
                'seal': 4 + batches * 15}[phase]
    return 5 + batches * 15 - position


def check_model(mutant=False):
    summaries = []
    for batches in range(1, 17):
        initial = ('start', 0, 0, 0, 0, 0)
        depth = {initial: 0}; queue = deque([initial]); edges = terminal = published = 0
        while queue:
            state = queue.popleft()
            phase, b, e, a, c, losses = state
            require(0 <= c <= a <= batches * 4 and a - c <= 1 and losses <= c, 'private update invariant')
            if phase == 'published':
                require(a == c == losses == batches * 4, 'partial result published')
                published += 1
            targets = transitions(state, batches, mutant)
            require(bool(targets) == (phase not in ('absent', 'published')), 'unexpected deadlock')
            terminal += not targets
            for target in targets:
                edges += 1
                require(rank(target, batches) < rank(state, batches), 'nonterminating stage transition')
                if target not in depth:
                    depth[target] = depth[state] + 1
                    queue.append(target)
        require(published == 1 and terminal > 1, 'missing success/failure')
        summaries.append({'batches': batches, 'epochs': 4, 'states': len(depth), 'transitions': edges,
                          'maxDepth': max(depth.values()), 'publishedStates': published, 'terminalStates': terminal})
    return {'states': sum(s['states'] for s in summaries), 'transitions': sum(s['transitions'] for s in summaries),
            'configurations': summaries, 'sourceRelation': 'complete reviewed stage skeleton with conditional helper summaries',
            'scope': 'private-call publication; no OS, neural arithmetic, persistence or whole-runtime refinement'}


def check_successor():
    extracted = extract()
    return {'smt': prove_bounds(extracted), 'model': check_model(),
            'boundary': {'requirement': 'F-RL-PPO-V2-BOUNDARY', 'status': 'exhaustively_checked',
                         'definitions': sorted(DEFINITIONS), 'helperSources': sorted(HELPERS),
                         'publicTrainingEntries': 1, 'defaultEnabled': False, 'resultPublicationSites': 1,
                         'orderFileNetworkPromotionOperations': 0,
                         'scope': 'reviewed source and named helper/runtime assumptions; no production reachability'},
            'sourceSha256': hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest()}
