"""Source-bound causal PPO v7: frozen-free closure, sampling and normalization lemmas, joint-update model and exact oracles."""
import ast
from collections import deque
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import sys

import z3 as z
from causal_value import _calls, _texts, closure, read_module, registry_for
from default_paths import check_python
from ppo_objective_successor import ceil_log2
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'causal_ppo_v7'
SOURCE = 'scripts/research/causal_ppo_v7.py'
REGISTRY = 'formal/research/causal-ppo-v7-source.json'
REGISTRATION = 'research-notes/registrations/causal-ppo-v7-engineering.json'
CONTRACT = 'formal/research/causal-ppo-v7-contract.md'
CONSTANTS = {'VERSION': "'causal-ppo-v7'", 'BOUND': '2.0 ** 99', 'TINY': '2.0 ** (-1022)', 'MIN_PROB': '2.0 ** (-100)',
             'GAMMA': '0.99', 'FLOOR': '1e-08', 'ROLLOUT': '256', 'UPDATES': '4',
             '__all__': "['VERSION', 'Outcome', 'TrainingResult', 'train_causal_ppo_v7']"}
IMPORTS = ['from __future__ import annotations', 'from dataclasses import dataclass',
           'from fractions import Fraction as F', 'import math', 'import numpy as np',
           'import causal_replay_v3 as replay', 'import replay_accounting_v2 as accounting',
           'from causal_transition_v4 import REWARD, Transition, transition_v4',
           'from causal_optimizer_v5 import VERSION as OPTIMIZER, Snapshot, create_v5, forward_v5, update_v5',
           'from gae_targets_v2 import VERSION as TARGETS, batch_v2',
           'from ppo_objective_v2 import VERSION as OBJECTIVE, objective_v2']
CLOSURE = {MODULE, 'causal_replay_v3', 'replay_accounting_v2', 'causal_transition_v4', 'causal_optimizer_v5',
           'gae_targets_v2', 'ppo_objective_v2'}
EXTERNAL = {'__future__', 'dataclasses', 'fractions', 'math', 'numpy'}
FROZEN = {'sequential_learning', 'sequential_env', 'sequential_evaluation', 'sequential_registry',
          'run_sequential_screen', 'summarize_sequential_screen', 'causal_ppo_v5', 'ppo_successor_v2',
          'optimizer_snapshot_v2', 'ppo_inference_v3', 'ppo_artifact_v4'}
STEP_CAP = 2 ** 31 - 1


def require(ok, reason):
    if not ok:
        raise ValueError('causal ppo v7: ' + reason)


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
    try:
        local, external = closure(lambda name: text if name == MODULE else read(name), MODULE)
    except ValueError as error:
        raise ValueError('causal ppo v7: ' + str(error)) from None
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
                         'order', 'submit', 'promote', 'save_policy', 'load_policy', 'write', 'read', 'system',
                         'softmax', 'ppo_gradient'}, 'effect-free source without frozen helpers')

    entry = nodes['train_causal_ppo_v7']
    check_python(text, 'train_causal_ppo_v7')
    defaults = dict(zip((x.arg for x in entry.args.kwonlyargs), map(ast.unparse, entry.args.kw_defaults)))
    require(defaults == {'enabled': 'False', 'version': 'VERSION'}, 'disabled default')
    require(ast.unparse(entry.body[0].test) == 'enabled is not True or type(version) is not str or version != VERSION or '
            '(type(seed) is not int) or (not 0 <= seed < 2 ** 32 - 1) or (type(outcomes) is not tuple) or '
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
    updates = next((s for s in loop.body if isinstance(s, ast.For)), None)
    require(updates is not None and ast.unparse(updates.iter) == 'range(UPDATES)' and
            len(_calls(updates, '_update')) == 1 and len(_calls(tree, '_update')) == 1 and
            ast.unparse(updates.body[0].targets[0]) == '(actor, critic, loss, lost)', 'fixed joint update count')
    steps = _texts(loop)
    require(steps.index('targets, advantages, lost = _targets(rows, inputs, rewards, critic)') < loop.body.index(updates) and
            ast.unparse(loop.body[-1]) == 'if rows[-1].done:\n    break', 'targets fixed before updates; terminal stop')
    stores = [ast.unparse(s) for s in ast.walk(entry) if isinstance(s, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id in ('actor', 'critic') for x in s.targets for t in ast.walk(x))]
    require(len(stores) == 2 and stores[0] == 'actor, critic = (create_v5(seed, 3, enabled=True), create_v5(seed + 1, 1, enabled=True))',
            'actor and critic change only through the joint update')

    collect = nodes['_collect']
    scan = next((s for s in collect.body if isinstance(s, ast.For)), None)
    require(scan is not None and ast.unparse(scan.iter) == 'outcomes[offset:offset + ROLLOUT]' and
            ast.unparse(scan.target) == 'outcome', 'bounded contiguous rollout')
    order = _texts(scan)
    wanted = ['x = _vector(replay.observe_v3(session, enabled=True))', 'probs = _policy(_values(actor, x[None, :], 3)[0])',
              'u = float(rng.random())', 'action = 0 if u < probs[0] else 1 if u < probs[0] + probs[1] else 2',
              "if type(outcome) is not Outcome:\n    raise ValueError('invalid outcome')"]
    require(order[:5] == wanted, 'observe, evaluate and draw before the outcome is checked')
    first_read = min(i for i, s in enumerate(scan.body) for n in ast.walk(s)
                     if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'outcome')
    draws = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'rng']
    require(first_read == 5 and len(draws) == 1 and draws[0].attr == 'random',
            'action draw precedes every outcome field; exactly one draw per step')
    require("if advanced is None:\n    raise ValueError('rejected causal transition')" in order and
            'behavior.append(probs[action])' in order and 'rewards.append(_bounded(_number(row.reward)))' in order and
            order[-1] == 'if row.done:\n    break', 'checked transition, recorded probability and bounded reward')
    require(_texts(nodes['_policy']) == [
        'm = max(row)', 'terms = [math.exp(x - m) for x in row]', 'total = math.fsum(terms)',
        'probs = tuple((t / total for t in terms))',
        "if not all((MIN_PROB <= p <= 1.0 for p in probs)):\n    raise ValueError('behavior probability outside the admitted domain')",
        'return probs'], 'difference-only softmax with a probability floor on every action')
    require(_texts(nodes['_bounded'])[0] == "if type(value) is not float or not math.isfinite(value) or abs(value) > BOUND:\n"
            "    raise ValueError('value outside the admitted magnitude')", 'magnitude guard')
    values = nodes['_values']
    require(len(_calls(tree, 'forward_v5')) == 1 and len(_calls(values, 'forward_v5')) == 1 and
            ast.unparse(values.body[-1]) == 'return tuple((tuple((_bounded(float(x)) for x in row)) for row in result))' and
            ast.unparse(values.body[1].test) == 'result is None or result.shape != (len(inputs), width)',
            'every network value is checked and bounded')
    targets = _texts(nodes['_targets'])
    for line in ['active = [i for i, row in enumerate(rows) if not row.done]',
                 "if pairs is None:\n    raise ValueError('rejected GAE targets')",
                 'raw = [_bounded(pair[0]) for pair in pairs]', 'targets = tuple((_bounded(pair[1]) for pair in pairs))',
                 'mean = math.fsum(raw) / n', 'centered = [a - mean for a in raw]', 'squares = [c * c for c in centered]',
                 'underflow = sum((1 for c, s in zip(centered, squares) if c != 0.0 and s < TINY))',
                 'scale = max(math.sqrt(math.fsum(squares) / n), FLOOR)',
                 'return (targets, tuple((_bounded(c / scale) for c in centered)), underflow)']:
        require(line in targets, 'guarded GAE targets and scalar normalization: ' + line[:40])
    gae = _calls(nodes['_targets'], 'batch_v2')
    require(len(gae) == 1 and len(_calls(tree, 'batch_v2')) == 1 and ast.unparse(gae[0].args[1]) == 'GAMMA' and
            'following[i][0] if i in following else 0.0' in ast.unparse(gae[0].args[0]) and
            ast.unparse(_calls(nodes['_targets'], '_vector')[0].args[0]) == 'rows[i].next_observation' and
            len(_calls(nodes['_targets'], '_vector')) == 1, 'terminal rows take next value zero without evaluation')
    require(_texts(nodes['_update']) == [
        'objective = objective_v2(_values(actor, inputs, 3), actions, behavior, advantages, enabled=True)',
        "if objective is None:\n    raise ValueError('rejected PPO objective')",
        'n = len(targets)',
        'residual = [[_bounded((value[0] - target) / n)] for value, target in zip(_values(critic, inputs, 1), targets)]',
        "candidate_actor = update_v5(actor, inputs, np.asarray(objective.grad, dtype='float64'), enabled=True)",
        "candidate_critic = update_v5(critic, inputs, np.asarray(residual, dtype='float64'), enabled=True)",
        "if candidate_actor is None or candidate_critic is None:\n    raise ValueError('rejected optimizer update')",
        'return (candidate_actor, candidate_critic, objective.loss, objective.underflow)'],
        'checked objective, bounded residual and joint candidate acceptance')
    return nodes, {'status': 'exhaustively_checked', 'definitions': len(nodes), 'defaultDisabledEntries': 1,
                   'closure': sorted(local), 'externalImports': sorted(external), 'frozenModulesInClosure': 0,
                   'scope': 'complete reviewed AST, frozen-free import closure, draw-before-outcome order, probability '
                            'floor, magnitude guards, checked stages, joint updates and single publication; runtime '
                            'semantics assumed'}


def real(value):
    return z.RealVal(str(Q(value)))


def prove(nodes=None, bound=Q(2) ** 99, rollout=256, updates=4):
    u = Q(1, 2 ** 53)
    # (a) Softmax simplex: one term is 1, the rest lie in [0,1]; quotients are probabilities that sum to one.
    t = z.Reals('pv_t0 pv_t1 pv_t2'); p = z.Reals('pv_p0 pv_p1 pv_p2'); total = z.Real('pv_total')
    terms = z.And(*[z.And(v >= 0, v <= 1) for v in t], z.Or(*[v == 1 for v in t]), total == t[0] + t[1] + t[2])
    certify(terms, z.And(total >= 1, total <= 3))
    certify(z.And(terms, *[p[i] * total == t[i] for i in range(3)]),
            z.And(*[z.And(p[i] >= 0, p[i] <= 1) for i in range(3)], (p[0] + p[1] + p[2]) * total == total))
    # (b) The cumulative rule partitions [0,1): exactly one action, with interval lengths p0, p1, p2.
    x = z.Real('pv_u')
    simplex = z.And(*[v >= 0 for v in p], p[0] + p[1] + p[2] == 1, x >= 0, x < 1)
    action = z.If(x < p[0], 0, z.If(x < p[0] + p[1], 1, 2))
    certify(simplex, z.And(z.Or(action == 0, action == 1, action == 2),
                           (action == 0) == z.And(x >= 0, x < p[0]),
                           (action == 1) == z.And(x >= p[0], x < p[0] + p[1]),
                           (action == 2) == z.And(x >= p[0] + p[1], x < p[0] + p[1] + p[2])))
    # (c) Exact-real normalization: a centered value is at most sqrt(n) scales, so 16 for n <= 256.
    c, s, n = z.Reals('pv_c pv_s pv_n')
    certify(z.And(c * c <= n * s * s, n >= 1, n <= rollout, s > 0), z.And(c <= 16 * s, c >= -16 * s))
    require(rollout == 256 and math.isqrt(rollout) == 16, 'registered rollout bound')
    # (d) Critic residual: two guarded values differ by at most 2B; two roundings keep it below 2^101.
    v, y, d = z.Reals('pv_v pv_y pv_d')
    inside = lambda value, limit: z.And(value <= real(limit), value >= -real(limit))
    certify(z.And(inside(v, bound), inside(y, bound), d == v - y), inside(d, 2 * bound))
    residual = 2 * bound * (1 + u) * (1 + u)
    require(residual < Q(2) ** 101 and ceil_log2(residual) == 101, 'residual magnitude chain')
    solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
    solver.add(inside(v, bound), d == v - y, d > real(2 * bound))
    require(solver.check() == z.sat, 'target guard is not vacuous')
    # (e) Counts: the optimizer step cap and the critic seed stay in range.
    steps, rollouts, done, seed = z.Ints('pv_steps pv_rollouts pv_updates pv_seed')
    certify(z.And(steps >= 1, steps <= 4096, rollouts >= 1, rollouts <= steps, done == updates * rollouts,
                  seed >= 0, seed < 2 ** 32 - 1),
            z.And(done <= updates * 4096, done < STEP_CAP, seed + 1 < 2 ** 32))
    require(updates * 4096 == 16384, 'registered update bound')
    return {'F-RL-PPO-V7-ARITH': 'unsat'}, {'normalizedAdvantageBound': 16, 'residualBoundLog2': 101, 'magnitudeGuardLog2': 99,
                                            'maximumOptimizerSteps': updates * 4096, 'certifiedImplications': 6,
                                            'satisfiableControls': 1}


def successors(state, requested, rollout, updates, mutant=None):
    """Finite stage abstraction: (phase, consumed, rows, actor steps, critic steps, terminal)."""
    phase, consumed, rows, actor, critic, terminal = state
    if phase in ('absent', 'published'):
        return []
    out = [('reject', ('absent', consumed, rows, actor, critic, terminal))]
    if phase == 'start':
        out.append(('activate', ('observe', 0, 0, 0, 0, False)))
    elif phase == 'observe':
        out.append(('draw', ('drawn', consumed, rows, actor, critic, False)))
        if mutant == 'consume-before-draw':
            out.append(('consume', ('consumed', consumed + 1, rows + 1, actor, critic, False)))
    elif phase == 'drawn':
        for ends in (False, True):
            out.append(('consume-terminal' if ends else 'consume', ('consumed', consumed + 1, rows + 1, actor, critic, ends)))
    elif phase == 'consumed':
        if terminal or rows == rollout or consumed == requested:
            out.append(('targets', ('update', consumed, rows, actor, critic, terminal)))
        else:
            out.append(('continue', ('observe', consumed, rows, actor, critic, False)))
    elif phase == 'update':
        if rows:   # rows > 0 marks a rollout whose updates are still pending
            out.append(('update-both', ('update', consumed, rows if (actor + 1) % updates else 0, actor + 1, critic + 1, terminal)))
            if mutant == 'actor-only':
                out.append(('update-actor', ('update', consumed, rows if (actor + 1) % updates else 0, actor + 1, critic, terminal)))
            if mutant == 'publish-partial':
                out.append(('publish', ('published', consumed, rows, actor, critic, terminal)))
        elif terminal or consumed == requested:
            out.append(('publish', ('published', consumed, 0, actor, critic, terminal)))
        else:
            out.append(('next-rollout', ('observe', consumed, 0, actor, critic, False)))
    return out


def model(mutant=None, requests=(1, 2, 3, 4, 5), rollout=2, updates=2):
    states = edges = published = 0
    for requested in requests:
        initial = ('start', 0, 0, 0, 0, False)
        paths = {initial: ()}; queue = deque([initial])
        while queue:
            state = queue.popleft()
            nxt = successors(state, requested, rollout, updates, mutant)
            require(bool(nxt) or state[0] in ('absent', 'published'), 'nonterminal deadlock')
            for label, following in nxt:
                edges += 1
                trace = paths[state] + (label,)
                phase, consumed, rows, actor, critic, terminal = following
                require(consumed <= requested and rows <= rollout, 'consumption beyond the request or rollout')
                if label.startswith('consume'):
                    require(len(trace) > 1 and trace[-2] == 'draw', 'outcome consumed before its action draw')
                if label in ('next-rollout', 'publish'):
                    require(actor == critic, 'actor and critic step counts diverged')
                if phase == 'published':
                    require('reject' not in trace and trace[-2] == 'update-both' and actor == critic and actor % updates == 0 and
                            actor >= updates and rows == 0 and (terminal or consumed == requested) and consumed >= 1,
                            'publication before every stage completed')
                    published += 1
                if following not in paths:
                    paths[following] = trace; queue.append(following)
        require(any(s[0] == 'published' for s in paths) and any(s[0] == 'absent' for s in paths), 'vacuous model')
        require(any(s[0] == 'published' and s[1] == requested and not s[5] for s in paths), 'input exhaustion unreachable')
        require(requested == 1 or any(s[0] == 'published' and s[1] < requested and s[5] for s in paths), 'terminal stop unreachable')
        states += len(paths)
    return {'status': 'model_checked', 'states': states, 'transitions': edges, 'publishingTransitions': published,
            'requests': list(requests), 'rolloutBound': rollout, 'updatesPerRollout': updates, 'orderTransitions': 0,
            'scope': 'finite stage abstraction over small request, rollout and update bounds; numeric values abstracted'}


def _impl():
    path = str(ROOT / 'scripts/research')
    if path not in sys.path:
        sys.path.insert(0, path)
    import causal_optimizer_v5 as optimizer
    import causal_ppo_v7 as impl
    import causal_replay_v3 as replay
    return impl, optimizer, replay


def softmax(row):
    m = max(row); terms = [math.exp(x - m) for x in row]; total = math.fsum(terms)
    return tuple(t / total for t in terms)


def exact_targets(gamma, rows, values, following):
    """The GAE-v2 recursion with every binary64 operation evaluated exactly and rounded once."""
    later, out = 0.0, []
    for row, value, nxt in reversed(list(zip(rows, values, following))):
        reward = float(row.reward)
        if row.done:
            later = float(Q(reward) - Q(value)); out.append(reward)
            continue
        bootstrap = float(Q(gamma) * Q(nxt))
        delta = float(Q(float(Q(reward) + Q(bootstrap))) - Q(value))
        trace = float(Q(float(Q(gamma) * Q(0.95))) * Q(later))
        later = float(Q(delta) + Q(trace))
        out.append(float(Q(later) + Q(value)))
    return list(reversed(out))


def conformance():
    import numpy as np
    from causal_value import fixtures
    impl, optimizer, replay = _impl()
    reg = json.loads((ROOT / REGISTRATION).read_text())
    runs = sampled = oracle = rewrites = 0
    for seed in reg['seeds']:
        for count in reg['fixtureSteps']:
            bars, future = fixtures(impl, replay, count)
            got = impl.train_causal_ppo_v7(bars, future, seed, enabled=True)
            require(got is not None and got == impl.train_causal_ppo_v7(bars, future, seed, enabled=True),
                    'deterministic finite result')
            require(impl.train_causal_ppo_v7(bars, future, seed) is None and
                    impl.train_causal_ppo_v7(bars, future, seed, enabled=True, version='causal-ppo-v5') is None,
                    'default and wrong-version rejection')
            rollouts = -(-count // reg['rolloutBound'])
            require((got.steps, got.unused_outcomes, got.stop_reason) == (count, 0, 'input_exhausted') and
                    got.actor.step == got.critic.step == reg['updatesPerRollout'] * rollouts == len(got.losses) and
                    len(got.targets) == len(got.advantages) == count, 'counts and joint updates')
            require(all(type(y) is float and abs(y) <= 2.0 ** 99 for y in got.targets + got.advantages) and
                    all(type(v) is float and math.isfinite(v) for v in got.losses) and
                    all(2.0 ** -100 <= q <= 1.0 for q in got.behavior_probabilities) and
                    type(got.underflow) is int and got.underflow >= 0, 'bounded published numbers')
            first = got.transitions[:reg['rolloutBound']]
            actor = optimizer.create_v5(seed, 3, enabled=True)
            critic = optimizer.create_v5(seed + 1, 1, enabled=True)
            rng = np.random.default_rng([seed, 7])
            for row, probability in zip(first, got.behavior_probabilities):
                probs = softmax(optimizer.forward_v5(actor, np.asarray([[float(v) for v in row.observation.values]]), enabled=True)[0].tolist())
                draw = float(rng.random())
                require(row.action == (0 if draw < probs[0] else 1 if draw < probs[0] + probs[1] else 2) and
                        probability == probs[row.action], 'sampling oracle')
                sampled += 1
            # Same stacked batches as the learner: network values depend on batch shape at the last bit.
            inputs = np.asarray([[float(v) for v in row.observation.values] for row in first])
            nexts = np.asarray([[float(v) for v in row.next_observation.values] for row in first])
            values = [v[0] for v in optimizer.forward_v5(critic, inputs, enabled=True).tolist()]
            following = [v[0] for v in optimizer.forward_v5(critic, nexts, enabled=True).tolist()]
            expected = exact_targets(impl.GAMMA, first, values, following)
            require([y.hex() for y in got.targets[:len(first)]] == [y.hex() for y in expected], 'exact GAE target oracle')
            oracle += len(expected)
            for start in range(0, count, reg['rolloutBound']):
                chunk = got.advantages[start:start + reg['rolloutBound']]
                require(abs(math.fsum(chunk)) <= 1e-9 * len(chunk) and
                        max(abs(a) for a in chunk) <= math.sqrt(len(chunk)) * (1 + 1e-12), 'normalized advantage bound')
            runs += 1
    bars, future = fixtures(impl, replay, 40)
    for seed in reg['seeds']:
        base = impl.train_causal_ppo_v7(bars, future, seed, enabled=True)
        for k in (0, 1, 16, 39):
            _, changed = fixtures(impl, replay, 40, shift=3, start=k)
            other = impl.train_causal_ppo_v7(bars, changed, seed, enabled=True)
            require(other is not None and changed[:k] == future[:k] and changed[k] != future[k] and
                    base.transitions[:k] == other.transitions[:k] and
                    base.behavior_probabilities[:k + 1] == other.behavior_probabilities[:k + 1] and
                    base.transitions[k].action == other.transitions[k].action and
                    base.transitions[k].receipt != other.transitions[k].receipt, 'future rewrite changed an earlier step')
            rewrites += 1
    require(rewrites == reg['prefixRewriteEpisodes'], 'registered prefix-rewrite budget')
    bars, future = fixtures(impl, replay, 3)
    ended = impl.train_causal_ppo_v7(bars, (future[0], impl.Outcome(future[1].bar, (), terminal=True, fill=Q(1, 2)), None),
                                     reg['seeds'][0], enabled=True)
    require(ended is not None and (ended.steps, ended.unused_outcomes, ended.stop_reason) == (2, 1, 'terminal') and
            ended.transitions[-1].done and ended.targets[-1] == float(ended.transitions[-1].reward), 'terminal target is the reward')
    rejected = 0
    for bad in ((None,), (impl.Outcome(future[0].bar, [], fill=Q(1, 2)),), ()):
        require(impl.train_causal_ppo_v7(bars, bad, reg['seeds'][0], enabled=True) is None, 'invalid outcome must reject')
        rejected += 1
    for call, value in ((impl._bounded, 2.0 ** 99 * (1 + 2.0 ** -52)), (impl._bounded, float('nan')), (impl._bounded, 1),
                        (impl._policy, (0.0, -70.0, 0.0)), (impl._policy, (0.0, -800.0, 0.0))):
        try:
            call(value)
        except ValueError:
            rejected += 1
            continue
        require(False, 'out-of-domain value passed')
    require(min(impl._policy((0.0, -68.0, 0.0))) >= 2.0 ** -100, 'floor boundary fixture')
    return {'status': 'property_tested', 'runs': runs, 'sampledActions': sampled, 'oracleTargets': oracle,
            'prefixRewrites': rewrites, 'rejections': rejected, 'seeds': reg['seeds'], 'marketDataReads': 0,
            'economicEvidence': False}


def check_causal_ppo():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require((reg['version'], reg['enabledByDefault'], reg['productionIntegration'], reg['financialTrials'],
             reg['historicalDataReads'], reg['holdoutOpened']) == ('causal-ppo-v7', False, False, 0, 0, False), 'registration')
    require((reg['rolloutBound'], reg['updatesPerRollout'], reg['stepBound'], reg['magnitudeBoundLog2'],
             reg['minimumProbabilityLog2'], reg['discountPerTransition'], reg['advantageScaleFloor']) ==
            (int(CONSTANTS['ROLLOUT']), int(CONSTANTS['UPDATES']), 4096, 99, -100, float(CONSTANTS['GAMMA']),
             float(CONSTANTS['FLOOR'])), 'registered constants')
    nodes, source = extract()
    smt, bounds = prove(nodes)
    return {'source': source, 'smt': smt, 'bounds': bounds, 'model': model(), 'conformance': conformance()}


if __name__ == '__main__':
    print(json.dumps(check_causal_ppo(), indent=1, sort_keys=True))
