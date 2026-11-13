"""Source-bound causal value v6: frozen-free closure, target bound lemmas, publication model and exact target oracle."""
import ast
from collections import deque
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import sys

import z3 as z
from default_paths import check_python
from promotion_boundary import shape
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'causal_value_v6'
SOURCE = 'scripts/research/causal_value_v6.py'
REGISTRY = 'formal/research/causal-value-v6-source.json'
REGISTRATION = 'research-notes/registrations/causal-value-v6-engineering.json'
CONTRACT = 'formal/research/causal-value-v6-contract.md'
CONSTANTS = {'VERSION': "'causal-value-v6'", 'BOUND': '2.0 ** 99', 'TINY': '2.0 ** (-1022)', 'GAMMA': '0.99',
             'ALPHA': '0.1', 'EXPLORE': 'F(1, 8)', 'OTHER': 'EXPLORE / 3', 'GREEDY': '1 - EXPLORE + OTHER',
             'ROLLOUT': '256', 'UPDATES': '4',
             '__all__': "['VERSION', 'Outcome', 'TrainingResult', 'train_causal_value_v6']"}
IMPORTS = ['from __future__ import annotations', 'from dataclasses import dataclass',
           'from fractions import Fraction as F', 'import math', 'import numpy as np',
           'import causal_replay_v3 as replay', 'import replay_accounting_v2 as accounting',
           'from causal_transition_v4 import REWARD, Transition, transition_v4',
           'from causal_optimizer_v5 import VERSION as OPTIMIZER, Snapshot, create_v5, forward_v5, update_v5',
           'from value_objective_v2 import VERSION as OBJECTIVE, objective_v2']
CLOSURE = {MODULE, 'causal_replay_v3', 'replay_accounting_v2', 'causal_transition_v4', 'causal_optimizer_v5',
           'value_objective_v2'}
EXTERNAL = {'__future__', 'dataclasses', 'fractions', 'math', 'numpy'}
FROZEN = {'sequential_learning', 'sequential_env', 'sequential_evaluation', 'sequential_registry',
          'run_sequential_screen', 'summarize_sequential_screen', 'causal_ppo_v5', 'ppo_successor_v2',
          'gae_targets_v2', 'optimizer_snapshot_v2', 'ppo_inference_v3', 'ppo_artifact_v4'}
OBJECTIVE_BOUND = Q(2) ** 100
STEP_CAP = 2 ** 31 - 1


def require(ok, reason):
    if not ok:
        raise ValueError('causal value v6: ' + reason)


def read_module(name):
    path = ROOT / 'scripts/research' / (name + '.py')
    return path.read_text() if path.is_file() else None


def closure(read=read_module, root=MODULE):
    """Transitive repository-local imports; a name with no research source is external."""
    local, external, queue = set(), set(), deque([root])
    while queue:
        name = queue.popleft()
        if name in local:
            continue
        source = read(name)
        require(source is not None, 'missing closure module: ' + name)
        local.add(name)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                require(node.level == 0 and node.module is not None, 'relative import')
                names = [node.module]
            else:
                continue
            for imported in names:
                top = imported.split('.')[0]
                if read(top) is None:
                    external.add(top)
                else:
                    queue.append(top)
    return local, external


def registry_for(source):
    tree = ast.parse(source)
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    return {'astSha256': hashlib.sha256(shape(tree).encode()).hexdigest(),
            'definitions': {k: shape(v) for k, v in nodes.items()}}


def _texts(function):
    return [ast.unparse(statement) for statement in function.body]


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def extract(source=None, registry=None, read=read_module):
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
    require([ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))] == IMPORTS,
            'exact import boundary')
    local, external = closure(lambda name: text if name == MODULE else read(name))
    require(local == CLOSURE and not local & FROZEN, 'frozen-free import closure')
    require(external == EXTERNAL, 'external import closure')
    for name in ('Outcome', 'TrainingResult'):
        require([ast.unparse(n) for n in nodes[name].decorator_list] == ['dataclass(frozen=True, slots=True)'], 'immutable value')
    for node in ast.walk(tree):
        require(not isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal, ast.Yield,
                                      ast.YieldFrom, ast.Lambda, ast.Delete)), 'control effect')
        if isinstance(node, ast.Attribute):
            require(not node.attr.startswith('__') and isinstance(node.ctx, ast.Load), 'reflection/attribute write')
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    require(not names & {'eval', 'exec', 'open', 'print', 'compile', 'getattr', 'setattr', 'globals', 'locals', 'socket',
                         'order', 'submit', 'promote', 'save_policy', 'load_policy', 'write', 'read', 'system'},
            'effect-free source')

    entry = nodes['train_causal_value_v6']
    check_python(text, 'train_causal_value_v6')
    defaults = dict(zip((x.arg for x in entry.args.kwonlyargs), map(ast.unparse, entry.args.kw_defaults)))
    require(defaults == {'enabled': 'False', 'version': 'VERSION'}, 'disabled default')
    gate = entry.body[0]
    require(ast.unparse(gate.test) == 'enabled is not True or type(version) is not str or version != VERSION or '
            '(type(seed) is not int) or (not 0 <= seed < 2 ** 32) or (type(outcomes) is not tuple) or '
            '(not 1 <= len(outcomes) <= 4096)', 'activation and count admission first')
    require(len(entry.body) == 2 and isinstance(entry.body[1], ast.Try), 'single guarded body')
    handlers = entry.body[1].handlers
    require(len(handlers) == 1 and ast.unparse(handlers[0].type) == '(ArithmeticError, ValueError, TypeError, MemoryError)'
            and ast.unparse(handlers[0].body[0]) == 'return None', 'failure publishes nothing')
    block = entry.body[1].body
    require(len(block) == 1 and isinstance(block[0], ast.With) and
            ast.unparse(block[0].items[0].context_expr) == "np.errstate(all='raise')", 'raising NumPy error state')
    body = block[0].body
    loop = next((s for s in body if isinstance(s, ast.While)), None)
    require(loop is not None and ast.unparse(loop.test) == 'len(transitions) < len(outcomes)' and not loop.orelse, 'bounded loop')
    publications = _calls(tree, 'TrainingResult')
    require(len(publications) == 1 and isinstance(body[-1], ast.Return) and body[-1].value is publications[0] and
            body.index(loop) < len(body) - 1, 'single publication after the loop')
    require(not any(isinstance(n, ast.Return) and n.value is not None and ast.unparse(n.value) != 'None'
                    for s in body[:-1] for n in ast.walk(s)), 'no early publication')
    steps = _texts(loop)
    updates = next((s for s in loop.body if isinstance(s, ast.For)), None)
    require(updates is not None and ast.unparse(updates.iter) == 'range(UPDATES)' and
            len(_calls(updates, '_update')) == 1 and len(_calls(tree, '_update')) == 1, 'fixed update count')
    syncs = [s for s in ast.walk(entry) if isinstance(s, ast.Assign) and ast.unparse(s) == 'target = online']
    require(len(syncs) == 2 and syncs[0] in body and syncs[1] in loop.body and
            loop.body.index(syncs[1]) == loop.body.index(updates) + 1 and body.index(syncs[0]) < body.index(loop) and
            not any(isinstance(t, ast.Name) and t.id == 'target' for s in ast.walk(entry) if isinstance(s, (ast.Assign, ast.AugAssign))
                    and s not in syncs for t in ast.walk(s) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)),
            'target replaced only after the final update')
    require(steps.index('targets, lost = _targets(rows, rewards, online, target)') < loop.body.index(updates) and
            ast.unparse(loop.body[-1]) == 'if rows[-1].done:\n    break', 'targets fixed before updates; terminal stop')

    collect = nodes['_collect']
    scan = next((s for s in collect.body if isinstance(s, ast.For)), None)
    require(scan is not None and ast.unparse(scan.iter) == 'outcomes[offset:offset + ROLLOUT]' and
            ast.unparse(scan.target) == 'outcome', 'bounded contiguous rollout')
    order = _texts(scan)
    wanted = ['x = _vector(replay.observe_v3(session, enabled=True))', 'q = _values(online, x[None, :])[0]',
              'greedy = q.index(max(q))', 'explore = float(rng.random()) < EXPLORE', 'index = int(rng.integers(3))',
              'action = index if explore else greedy', 'if type(outcome) is not Outcome:\n    raise ValueError(\'invalid outcome\')']
    require(order[:7] == wanted, 'observe, evaluate and draw before the outcome is admitted')
    first_read = min(i for i, s in enumerate(scan.body) for n in ast.walk(s)
                     if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'outcome')
    require(first_read == 7 and len(_calls(tree, 'rng.random')) == 1 and len(_calls(tree, 'rng.integers')) == 1,
            'action draw precedes every outcome field; exactly two draws per step')
    require("if advanced is None:\n    raise ValueError('rejected causal transition')" in order and
            'behavior.append(GREEDY if action == greedy else OTHER)' in order and
            'rewards.append(_bounded(_number(row.reward)))' in order and
            order[-1] == 'if row.done:\n    break', 'checked transition, exact probability and bounded reward')

    require(_texts(nodes['_bounded'])[0] == "if type(value) is not float or not math.isfinite(value) or abs(value) > BOUND:\n"
            "    raise ValueError('value outside the admitted magnitude')", 'magnitude guard')
    values = nodes['_values']
    require(len(_calls(tree, 'forward_v5')) == 1 and len(_calls(values, 'forward_v5')) == 1 and
            ast.unparse(values.body[-1]) == 'return tuple((tuple((_bounded(float(x)) for x in row)) for row in result))' and
            ast.unparse(values.body[1].test) == 'result is None or result.shape != (len(inputs), 3)',
            'every network value is checked and bounded')
    targets = nodes['_targets']
    fold = next((s for s in targets.body if isinstance(s, ast.For)), None)
    require(fold is not None and _texts(fold) == [
        'if row.done:\n    targets.append(reward)\n    continue',
        'selected = evaluate[i][choose[i].index(max(choose[i]))]',
        'bootstrap = GAMMA * selected',
        'if selected != 0.0 and abs(bootstrap) < TINY:\n    underflow += 1',
        'targets.append(reward + bootstrap)'], 'terminal rows take no bootstrap; two scalar operations; counted underflow')
    require('active = [i for i, row in enumerate(rows) if not row.done]' in _texts(targets) and
            ast.unparse(_calls(targets, '_vector')[0].args[0]) == 'rows[i].next_observation' and
            len(_calls(targets, '_vector')) == 1, 'no terminal next observation is evaluated')
    update = _texts(nodes['_update'])
    require(update == ['objective = objective_v2(_values(online, inputs), actions, targets, ALPHA, enabled=True)',
                       "if objective is None:\n    raise ValueError('rejected value objective')",
                       "candidate = update_v5(online, inputs, np.asarray(objective.grad, dtype='float64'), enabled=True)",
                       "if candidate is None:\n    raise ValueError('rejected optimizer update')",
                       'return (candidate, objective.loss, objective.underflow)'], 'checked objective and optimizer stages')
    return nodes, {'status': 'exhaustively_checked', 'definitions': len(nodes), 'defaultDisabledEntries': 1,
                   'closure': sorted(local), 'externalImports': sorted(external), 'frozenModulesInClosure': 0,
                   'scope': 'complete reviewed AST, frozen-free import closure, draw-before-outcome order, magnitude '
                            'guards, checked stages and single publication; runtime semantics assumed'}


def number(text):
    """Evaluate a reviewed numeric constant expression exactly."""
    def go(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Q(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -go(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exponent = go(node.left), go(node.right)
            require(exponent.denominator == 1, 'integer exponent')
            return base ** int(exponent)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Div)):
            a, b = go(node.left), go(node.right)
            return a + b if isinstance(node.op, ast.Add) else a - b if isinstance(node.op, ast.Sub) else a / b
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'F' and len(node.args) == 2:
            return go(node.args[0]) / go(node.args[1])
        if isinstance(node, ast.Name) and node.id in CONSTANTS:
            return go(ast.parse(CONSTANTS[node.id], mode='eval').body)
        raise ValueError('causal value v6: unreviewed constant expression')
    return go(ast.parse(text, mode='eval').body)


def real(value):
    return z.RealVal(str(Q(value)))


def prove(nodes=None, constants=None):
    constants = CONSTANTS if constants is None else constants
    bound, gamma, explore = (number(constants[k]) for k in ('BOUND', 'GAMMA', 'EXPLORE'))
    rollout, updates = int(number(constants['ROLLOUT'])), int(number(constants['UPDATES']))
    u = Q(1, 2 ** 53)
    # One correctly rounded operation scales any magnitude bound by at most (1+u) (standard model).
    x, cap, delta = z.Reals('cv_x cv_cap cv_delta')
    certify(z.And(cap >= 0, x <= cap, x >= -cap, delta <= real(u), delta >= -real(u)),
            z.And(x * (1 + delta) <= cap * (1 + real(u)), x * (1 + delta) >= -cap * (1 + real(u))))
    # Exact chain along `reward + GAMMA * selected`: product bound, then the rounded sum.
    product = gamma * bound * (1 + u)
    total = (bound + product) * (1 + u)
    require(0 < gamma < 1 and total <= OBJECTIVE_BOUND, 'admitted targets stay inside the objective domain')
    r, p, s, t = z.Reals('cv_r cv_p cv_s cv_t')
    inside = lambda v, limit: z.And(v <= real(limit), v >= -real(limit))
    certify(z.And(inside(t, bound), p == real(gamma) * t), inside(p, gamma * bound))
    certify(z.And(inside(r, bound), inside(p, product), s == r + p), inside(s, bound + product))
    # Negative controls: without either guard the 2^100 objective bound is reachable.
    for premise in (z.And(inside(p, product), s == r + p), z.And(inside(r, bound), s == r + p)):
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
        solver.add(premise, s > real(OBJECTIVE_BOUND))
        require(solver.check() == z.sat, 'magnitude guard is not vacuous')
    # Exact epsilon-greedy probabilities for any exploration rate, then the registered rate.
    e, other, greedy = z.Reals('cv_e cv_other cv_greedy')
    certify(z.And(e > 0, e < 1, other == e / 3, greedy == 1 - e + other),
            z.And(other > 0, other < 1, greedy > 0, greedy < 1, greedy + 2 * other == 1, greedy > other - 1))
    require((number(constants['GREEDY']), number(constants['OTHER'])) == (Q(11, 12), Q(1, 24)) and
            number(constants['GREEDY']) + 2 * number(constants['OTHER']) == 1 and explore == Q(1, 8), 'registered behavior probabilities')
    # The first maximal online value is a maximum; its index selects the target-network value.
    o = z.Reals('cv_o0 cv_o1 cv_o2'); tv = z.Reals('cv_t0 cv_t1 cv_t2'); m, chosen = z.Reals('cv_m cv_chosen')
    index = z.If(o[0] == m, 0, z.If(o[1] == m, 1, 2))
    is_max = z.And(*[m >= v for v in o], z.Or(*[m == v for v in o]))
    pick = lambda vs: z.If(index == 0, vs[0], z.If(index == 1, vs[1], vs[2]))
    certify(z.And(is_max, chosen == pick(tv), *[inside(v, bound) for v in tv]),
            z.And(pick(o) == m, *[pick(o) >= v for v in o], inside(chosen, bound), z.Or(*[chosen == v for v in tv])))
    # Each rollout consumes at least one transition, so the optimizer step cap is unreachable.
    steps, rollouts, count, done = z.Ints('cv_steps cv_rollouts cv_count cv_updates')
    certify(z.And(steps >= 1, steps <= 4096, rollouts >= 1, rollouts <= steps, done == updates * rollouts,
                  count >= 1, count <= rollout),
            z.And(done <= updates * 4096, done < STEP_CAP, count <= 256))
    require(rollout == 256 and updates * 4096 == 16384 < STEP_CAP, 'registered rollout and update bounds')
    return {'F-RL-VALUE-V6-ARITH': 'unsat'}, {'targetBoundLog2': 100, 'magnitudeGuardLog2': 99,
                                             'maximumOptimizerSteps': updates * 4096,
                                             'certifiedImplications': 6, 'satisfiableControls': 2}


def successors(state, requested, rollout, updates, mutant=None):
    """Finite stage abstraction: (phase, consumed, rows, updates, done, target_age)."""
    phase, consumed, rows, done_updates, terminal, stale = state
    if phase in ('absent', 'published'):
        return []
    out = [('reject', ('absent', consumed, rows, done_updates, terminal, stale))]
    if phase == 'start':
        out.append(('activate', ('observe', 0, 0, 0, False, 0)))
    elif phase == 'observe':
        out.append(('draw', ('drawn', consumed, rows, 0, False, stale)))
        if mutant == 'consume-before-draw':
            out.append(('consume', ('consumed', consumed + 1, rows + 1, 0, False, stale)))
    elif phase == 'drawn':
        for ends in (False, True):
            out.append(('consume-terminal' if ends else 'consume', ('consumed', consumed + 1, rows + 1, 0, ends, stale)))
    elif phase == 'consumed':
        if terminal or rows == rollout or consumed == requested:
            out.append(('targets', ('update', consumed, rows, 0, terminal, stale)))
        else:
            out.append(('continue', ('observe', consumed, rows, 0, False, stale)))
    elif phase == 'update':
        if done_updates < updates:
            out.append(('update', ('update', consumed, rows, done_updates + 1, terminal, stale + 1)))
            if mutant == 'early-sync' and done_updates:
                out.append(('sync', ('synced', consumed, rows, done_updates, terminal, 0)))
        else:
            out.append(('sync', ('synced', consumed, rows, done_updates, terminal, 0)))
        if mutant == 'publish-partial':
            out.append(('publish', ('published', consumed, rows, done_updates, terminal, stale)))
    elif phase == 'synced':
        if terminal or consumed == requested:
            out.append(('publish', ('published', consumed, rows, done_updates, terminal, stale)))
        else:
            out.append(('next-rollout', ('observe', consumed, 0, 0, False, stale)))
    return out


def model(mutant=None, requests=(1, 2, 3, 4, 5), rollout=2, updates=2):
    states = edges = published = 0
    for requested in requests:
        initial = ('start', 0, 0, 0, False, 0)
        paths = {initial: ()}; queue = deque([initial])
        while queue:
            state = queue.popleft()
            nxt = successors(state, requested, rollout, updates, mutant)
            require(bool(nxt) or state[0] in ('absent', 'published'), 'nonterminal deadlock')
            for label, following in nxt:
                edges += 1
                trace = paths[state] + (label,)
                phase, consumed, rows, done_updates, terminal, stale = following
                require(consumed <= requested and rows <= rollout, 'consumption beyond the request or rollout')
                if label.startswith('consume'):
                    require(len(trace) > 1 and trace[-2] == 'draw', 'outcome consumed before its action draw')
                if label == 'sync':
                    require(done_updates == updates, 'target replaced before the final update')
                if phase == 'published':
                    require('reject' not in trace and trace[-2] == 'sync' and stale == 0 and done_updates == updates and
                            (terminal or consumed == requested) and consumed >= 1, 'publication before every stage completed')
                    published += 1
                if following not in paths:
                    paths[following] = trace; queue.append(following)
        require(any(s[0] == 'published' for s in paths) and any(s[0] == 'absent' for s in paths), 'vacuous model')
        require(any(s[0] == 'published' and s[1] == requested and not s[4] for s in paths), 'input exhaustion unreachable')
        require(requested == 1 or any(s[0] == 'published' and s[1] < requested and s[4] for s in paths), 'terminal stop unreachable')
        states += len(paths)
    return {'status': 'model_checked', 'states': states, 'transitions': edges, 'publishingTransitions': published,
            'requests': list(requests), 'rolloutBound': rollout, 'updatesPerRollout': updates, 'orderTransitions': 0,
            'scope': 'finite stage abstraction over small request, rollout and update bounds; numeric values abstracted'}


def _impl():
    path = str(ROOT / 'scripts/research')
    if path not in sys.path:
        sys.path.insert(0, path)
    import causal_optimizer_v5 as optimizer
    import causal_replay_v3 as replay
    import causal_value_v6 as impl
    return impl, optimizer, replay


def fixtures(impl, replay, count, shift=0, start=0):
    bars = tuple(replay.Bar('SYNTH', i * 10, i * 10 + 1, Q(100)) for i in range(26))
    future = tuple(impl.Outcome(replay.Bar('SYNTH', i * 10, i * 10 + 1,
                                           Q(100) + Q(i % 7 - 3, 8) + (shift if i - 26 >= start else 0)), (), fill=Q(1, 2))
                   for i in range(26, 26 + count))
    return bars, future


def exact_target(gamma, reward, selected):
    # Two correctly rounded binary64 operations evaluated with exact rationals.
    return float(Q(reward) + Q(float(Q(gamma) * Q(selected))))


def conformance():
    import numpy as np
    impl, optimizer, replay = _impl()
    reg = json.loads((ROOT / REGISTRATION).read_text())
    runs = oracle = rewrites = 0
    for seed in reg['seeds']:
        for count in reg['fixtureSteps']:
            bars, future = fixtures(impl, replay, count)
            got = impl.train_causal_value_v6(bars, future, seed, enabled=True)
            require(got is not None and got == impl.train_causal_value_v6(bars, future, seed, enabled=True),
                    'deterministic finite result')
            require(impl.train_causal_value_v6(bars, future, seed) is None and
                    impl.train_causal_value_v6(bars, future, seed, enabled=True, version='causal-ppo-v5') is None,
                    'default and wrong-version rejection')
            rollouts = -(-count // reg['rolloutBound'])
            require((got.steps, got.unused_outcomes, got.stop_reason) == (count, 0, 'input_exhausted') and
                    got.online.step == reg['updatesPerRollout'] * rollouts == len(got.losses) and
                    got.target is got.online and len(got.targets) == count, 'counts, updates and final target replacement')
            require(all(type(y) is float and abs(Q(y)) <= OBJECTIVE_BOUND for y in got.targets) and
                    all(type(v) is float and v == v and abs(v) != float('inf') for v in got.losses) and
                    type(got.underflow) is int and got.underflow >= 0, 'bounded published numbers')
            first = got.transitions[:reg['rolloutBound']]
            snapshot = optimizer.create_v5(seed, 3, enabled=True)
            for row, probability in zip(first, got.behavior_probabilities):
                q = optimizer.forward_v5(snapshot, np.asarray([[float(v) for v in row.observation.values]]), enabled=True)[0].tolist()
                require(probability == (Q(11, 12) if row.action == q.index(max(q)) else Q(1, 24)), 'exact behavior probability')
            require(set(got.behavior_probabilities) <= {Q(11, 12), Q(1, 24)}, 'behavior probability domain')
            # Same stacked batch as the learner: network values depend on batch shape at the last bit.
            batch = np.asarray([[float(v) for v in row.next_observation.values] for row in first])
            for row, q, target in zip(first, optimizer.forward_v5(snapshot, batch, enabled=True).tolist(), got.targets):
                require(not row.done and target.hex() == exact_target(impl.GAMMA, float(row.reward), max(q)).hex(),
                        'exact Double-DQN target oracle')
                oracle += 1
            runs += 1
    bars, future = fixtures(impl, replay, 40)
    for seed in reg['seeds']:
        base = impl.train_causal_value_v6(bars, future, seed, enabled=True)
        for k in (0, 1, 16, 39):
            _, changed = fixtures(impl, replay, 40, shift=3, start=k)
            other = impl.train_causal_value_v6(bars, changed, seed, enabled=True)
            require(other is not None and changed[:k] == future[:k] and changed[k] != future[k] and
                    base.transitions[:k] == other.transitions[:k] and base.targets[:k] == other.targets[:k] and
                    base.behavior_probabilities[:k + 1] == other.behavior_probabilities[:k + 1] and
                    base.transitions[k].action == other.transitions[k].action and
                    base.transitions[k].receipt != other.transitions[k].receipt, 'future rewrite changed an earlier step')
            rewrites += 1
    require(rewrites == reg['prefixRewriteEpisodes'], 'registered prefix-rewrite budget')
    bars, future = fixtures(impl, replay, 3)
    ended = impl.train_causal_value_v6(bars, (future[0], impl.Outcome(future[1].bar, (), terminal=True, fill=Q(1, 2)), None),
                                       reg['seeds'][0], enabled=True)
    require(ended is not None and (ended.steps, ended.unused_outcomes, ended.stop_reason) == (2, 1, 'terminal') and
            ended.transitions[-1].done and ended.targets[-1] == float(ended.transitions[-1].reward), 'terminal row has no bootstrap')
    rejected = 0
    for bad in ((None,), (impl.Outcome(future[0].bar, [], fill=Q(1, 2)),), ()):
        require(impl.train_causal_value_v6(bars, bad, reg['seeds'][0], enabled=True) is None, 'invalid outcome must reject')
        rejected += 1
    for value in (2.0 ** 99 * (1 + 2.0 ** -52), float('inf'), float('nan'), 1):
        try:
            impl._bounded(value)
        except ValueError:
            rejected += 1
            continue
        require(False, 'out-of-bound value admitted')
    require(impl._bounded(2.0 ** 99) == 2.0 ** 99 and
            Q(exact_target(impl.GAMMA, 2.0 ** 99, 2.0 ** 99)) <= OBJECTIVE_BOUND, 'worst admitted corner')
    return {'status': 'property_tested', 'runs': runs, 'oracleTargets': oracle, 'prefixRewrites': rewrites,
            'rejections': rejected, 'seeds': reg['seeds'], 'marketDataReads': 0, 'economicEvidence': False}


def check_causal_value():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require((reg['version'], reg['enabledByDefault'], reg['productionIntegration'], reg['financialTrials'],
             reg['historicalDataReads'], reg['holdoutOpened']) == ('causal-value-v6', False, False, 0, 0, False), 'registration')
    require((reg['rolloutBound'], reg['updatesPerRollout'], reg['stepBound'], reg['magnitudeBoundLog2'], reg['objectiveBoundLog2'])
            == (int(number(CONSTANTS['ROLLOUT'])), int(number(CONSTANTS['UPDATES'])), 4096, 99, 100) and
            Q(reg['explorationNumerator'], reg['explorationDenominator']) == number(CONSTANTS['EXPLORE']) and
            (reg['discountPerTransition'], reg['conservativeAlpha']) == (float(CONSTANTS['GAMMA']), float(CONSTANTS['ALPHA'])) and
            number(CONSTANTS['BOUND']) == Q(2) ** reg['magnitudeBoundLog2'], 'registered constants')
    nodes, source = extract()
    smt, bounds = prove(nodes)
    return {'source': source, 'smt': smt, 'bounds': bounds, 'model': model(), 'conformance': conformance()}


if __name__ == '__main__':
    print(json.dumps(check_causal_value(), indent=1, sort_keys=True))
