"""Source-bound causal OPE v8: frozen-free closure, sampling-law and weight gap lemmas, publication model and exact oracles."""
import ast
from collections import deque
from dataclasses import replace
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import sys

import z3 as z
import causal_ppo_learner
import causal_value
from causal_value import _calls, _texts, closure, read_module, registry_for
from default_paths import check_python
from ppo_objective_successor import ceil_log2
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'causal_ope_v8'
SOURCE = 'scripts/research/causal_ope_v8.py'
REGISTRY = 'formal/research/causal-ope-v8-source.json'
REGISTRATION = 'research-notes/registrations/causal-ope-v8-engineering.json'
CONTRACT = 'formal/research/causal-ope-v8-contract.md'
CONSTANTS = {'VERSION': "'causal-ope-v8'", 'FLOOR': '2.0 ** (-10)', 'GAMMA': '0.99', 'HORIZON': '32', 'RUNS': '256',
             'LAW_GAP': 'F(1, 2 ** 50)', 'WEIGHT_GAP': 'F(1, 2 ** 34)',
             'LEARNERS': '{value.TrainingResult: value.VERSION, ppo.TrainingResult: ppo.VERSION}',
             'VALUE_LAW': '(value.GREEDY, value.OTHER)',
             '__all__': "['VERSION', 'Model', 'Evaluation', 'evaluate_causal_ope_v8']"}
IMPORTS = ['from __future__ import annotations', 'from dataclasses import dataclass',
           'from fractions import Fraction as F', 'import causal_ppo_v7 as ppo', 'import causal_value_v6 as value',
           'from causal_transition_v4 import Transition',
           'from ope_rational_v2 import VERSION as ESTIMATOR, Episode, Estimate, estimate_v2']
CLOSURE = {MODULE, 'causal_ppo_v7', 'causal_value_v6', 'causal_transition_v4', 'ope_rational_v2', 'causal_replay_v3',
           'replay_accounting_v2', 'causal_optimizer_v5', 'gae_targets_v2', 'ppo_objective_v2', 'value_objective_v2'}
EXTERNAL = {'__future__', 'dataclasses', 'fractions', 'math', 'numpy'}
FROZEN = causal_ppo_learner.FROZEN
GRID = 2 ** 53
U = Q(1, 2 ** 53)
FLOOR = Q(1, 2 ** 10)
LAW_GAP = Q(1, 2 ** 50)
STEP_GAP = Q(1, 2 ** 40)
WEIGHT_GAP = Q(1, 2 ** 34)
HORIZON = 32


def require(ok, reason):
    if not ok:
        raise ValueError('causal ope v8: ' + reason)


def _constants(tree):
    return {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in tree.body if isinstance(n, ast.Assign)}


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
    require(_constants(tree) == CONSTANTS, 'version/bounds/exports')
    require([ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))] == IMPORTS,
            'exact import boundary')
    try:
        local, external = closure(lambda name: text if name == MODULE else read(name), MODULE)
    except ValueError as error:
        raise ValueError('causal ope v8: ' + str(error)) from None
    require(local == CLOSURE and not local & FROZEN, 'frozen-free import closure')
    require(external == EXTERNAL, 'external import closure')
    for name in ('Model', 'Evaluation'):
        require([ast.unparse(n) for n in nodes[name].decorator_list] == ['dataclass(frozen=True, slots=True)'], 'immutable value')
    require(ast.unparse(nodes['Evaluation'].body[-1]) == 'reliable: bool = False', 'never reliable by default')
    for node in ast.walk(tree):
        require(not isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal, ast.Yield,
                                      ast.YieldFrom, ast.Lambda, ast.Delete, ast.While)), 'control effect')
        if isinstance(node, ast.Attribute):
            require(not node.attr.startswith('__') and isinstance(node.ctx, ast.Load), 'reflection/attribute write')
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    require(not names & {'eval', 'exec', 'open', 'print', 'compile', 'getattr', 'setattr', 'globals', 'locals', 'socket',
                         'order', 'submit', 'promote', 'save_policy', 'load_policy', 'write', 'read', 'system', 'np',
                         'random', 'default_rng', 'train_causal_ppo_v7', 'train_causal_value_v6', 'replace'},
            'effect-free source that neither trains nor samples')

    entry = nodes['evaluate_causal_ope_v8']
    check_python(text, 'evaluate_causal_ope_v8')
    defaults = dict(zip((x.arg for x in entry.args.kwonlyargs), map(ast.unparse, entry.args.kw_defaults)))
    require(defaults == {'enabled': 'False', 'version': 'VERSION'}, 'disabled default')
    require(ast.unparse(entry.body[0].test) == 'enabled is not True or type(version) is not str or version != VERSION or '
            '(type(results) is not tuple) or (type(models) is not tuple) or (not 1 <= len(results) <= RUNS) or '
            '(len(models) != len(results)) or (type(horizon) is not int) or (not 1 <= horizon <= HORIZON)',
            'activation and count admission first')
    require(len(entry.body) == 2 and isinstance(entry.body[1], ast.Try), 'single guarded body')
    handlers = entry.body[1].handlers
    require(len(handlers) == 1 and ast.unparse(handlers[0].type) == '(ArithmeticError, ValueError, TypeError, MemoryError)'
            and ast.unparse(handlers[0].body[0]) == 'return None', 'failure publishes nothing')
    body = entry.body[1].body
    require(_texts(entry.body[1]) == [
        'episodes = tuple((_episode(result, model, horizon) for result, model in zip(results, models)))',
        'sources = tuple(((result.version, result.seed) for result in results))',
        "if len(set(sources)) != len(sources):\n    raise ValueError('reused generator stream')",
        'estimate = estimate_v2(episodes, GAMMA, enabled=True)',
        "if estimate is None or len(estimate.episodes) != len(results):\n    raise ValueError('rejected estimate')",
        'return Evaluation(VERSION, ESTIMATOR, GAMMA, horizon, sources, FLOOR, LAW_GAP, WEIGHT_GAP, estimate)'],
        'every run converted, distinct streams, checked estimate, then publication')
    publications = _calls(tree, 'Evaluation')
    require(len(publications) == 1 and body[-1].value is publications[0] and len(_calls(tree, 'estimate_v2')) == 1 and
            len(_calls(tree, '_episode')) == 1 and len(_calls(tree, 'Episode')) == 1, 'single estimate and publication')
    episode = _texts(nodes['_episode'])
    require(episode == [
        "if type(result) not in LEARNERS or type(result.version) is not str or result.version != LEARNERS[type(result)] or "
        "(type(result.seed) is not int) or (type(result.transitions) is not tuple) or "
        "(type(result.behavior_probabilities) is not tuple) or "
        "(len(result.transitions) != len(result.behavior_probabilities)):\n    raise ValueError('unrecognized learner result')",
        "if len(result.transitions) < horizon:\n    raise ValueError('run shorter than the horizon')",
        'rows = result.transitions[:horizon]',
        "if not all((type(row) is Transition and type(row.action) is int and (row.action in (0, 1, 2)) for row in rows)) or "
        "any((row.done is not False for row in rows[:-1])):\n    raise ValueError('invalid transition')",
        "if type(model) is not Model or type(model.target) is not tuple or type(model.q) is not tuple or "
        "(type(model.v) is not tuple) or ((len(model.target), len(model.q), len(model.v)) != (horizon, horizon, horizon + 1)) or "
        "(not all((type(p) is float and 0.0 <= p <= 1.0 for p in model.target))):\n    raise ValueError('invalid target model')",
        'q = tuple((ppo._bounded(x) for x in model.q))', 'v = tuple((ppo._bounded(x) for x in model.v))',
        "if v[-1] != 0.0:\n    raise ValueError('nonzero value beyond the horizon')",
        'rewards = tuple((ppo._bounded(ppo._number(row.reward)) for row in rows))',
        'return Episode(rewards, tuple((row.action for row in rows)), _behavior(result, horizon), model.target, q, v)'],
        'recognized run, no dropped or padded run, guarded values and prefix-only reads')
    require(_texts(nodes['_behavior']) == [
        'recorded = result.behavior_probabilities[:horizon]',
        "if type(result) is value.TrainingResult:\n    if not all((type(p) is F and p in VALUE_LAW for p in recorded)):\n"
        "        raise ValueError('unrecognized value-learner behavior probability')\n"
        "    recorded = tuple((float(p) for p in recorded))",
        "if not all((type(p) is float and FLOOR <= p <= 1.0 for p in recorded)):\n"
        "    raise ValueError('behavior probability below the evaluation floor')",
        'return recorded'], 'exact value-learner law and a behavior floor on every evaluated step')
    # Every index into a run is the horizon prefix: no transition after the horizon is read.
    slices = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice)]
    require(sorted(slices) == ['result.behavior_probabilities[:horizon]', 'result.transitions[:horizon]', 'rows[:-1]'],
            'prefix-only run access')

    # Facts about the learners and the estimator that the lemmas below rely on.
    learner, _ = causal_ppo_learner.extract()
    policy, collect = _texts(learner['_policy']), _texts(next(s for s in learner['_collect'].body if isinstance(s, ast.For)))
    require('total = math.fsum(terms)' in policy and 'probs = tuple((t / total for t in terms))' in policy and
            'u = float(rng.random())' in collect and 'behavior.append(probs[action])' in collect and
            'action = 0 if u < probs[0] else 1 if u < probs[0] + probs[1] else 2' in collect,
            'policy-gradient sampling rule')
    learned, _ = causal_value.extract()
    scan = _texts(next(s for s in learned['_collect'].body if isinstance(s, ast.For)))
    constants = _constants(ast.parse(read('causal_value_v6')))
    require((constants['EXPLORE'], constants['OTHER'], constants['GREEDY']) == ('F(1, 8)', 'EXPLORE / 3', '1 - EXPLORE + OTHER') and
            'explore = float(rng.random()) < EXPLORE' in scan and 'index = int(rng.integers(3))' in scan and
            'action = index if explore else greedy' in scan and
            'behavior.append(GREEDY if action == greedy else OTHER)' in scan, 'value-learner sampling rule')
    estimator = _constants(ast.parse(read('ope_rational_v2')))
    rollouts = {int(constants['ROLLOUT']), int(causal_ppo_learner.CONSTANTS['ROLLOUT'])}
    require((estimator['MAX_EPISODES'], estimator['MAX_DECISIONS']) == (CONSTANTS['RUNS'], CONSTANTS['HORIZON']) and
            min(rollouts) >= HORIZON, 'estimator limits and a horizon inside the first rollout')
    return nodes, {'status': 'exhaustively_checked', 'definitions': len(nodes), 'defaultDisabledEntries': 1,
                   'closure': sorted(local), 'externalImports': sorted(external), 'frozenModulesInClosure': 0,
                   'scope': 'complete reviewed AST, frozen-free import closure, recognized learner results, no dropped or '
                            'padded run, behavior floor, guarded values, prefix-only reads, distinct streams, checked '
                            'estimate and single publication; runtime semantics assumed'}


def real(value):
    return z.RealVal(str(Q(value)))


def within(value, center, gap):
    return z.And(value - center <= gap, center - value <= gap)


def grid(p, s, low, cut):
    """Counting facts for a uniform draw k/2^53: the share of draws below a threshold x in (0,1] lies in [x, x + 2^-53)."""
    step = real(Q(1, GRID))
    return z.And(low >= p[0], low < p[0] + step, z.If(s <= 1, z.And(cut >= s, cut < s + step), cut == 1))


def prove(nodes=None, floor=FLOOR, law_gap=LAW_GAP, horizon=HORIZON):
    eta = 2 * U / (1 - U)
    # (a) Rounded softmax: each quotient is within one rounding of its share and the total within one rounding of the
    #     exact sum, so the three recorded probabilities sum to one within eta. The exponentials need no accuracy.
    r = z.Reals('ov_r0 ov_r1 ov_r2'); g = z.Reals('ov_g0 ov_g1 ov_g2'); p = z.Reals('ov_p0 ov_p1 ov_p2')
    total = p[0] + p[1] + p[2]
    rounded = z.And(*[x >= 0 for x in r], r[0] + r[1] + r[2] == 1,
                    *[z.And(x >= real((1 - U) / (1 + U)), x <= real((1 + U) / (1 - U))) for x in g],
                    *[p[i] == r[i] * g[i] for i in range(3)])
    certify(rounded, within(total, 1, real(eta)))
    # (b) Sampling law: with one rounded addition and a draw on the 2^-53 grid, each action's probability of being
    #     drawn differs from its recorded probability by at most the law gap.
    s, low, cut = z.Reals('ov_s ov_low ov_cut')
    recorded = z.And(*[z.And(x > 0, x <= 1) for x in p], within(total, 1, real(eta)),
                     within(s, p[0] + p[1], real(U) * (p[0] + p[1])), grid(p, s, low, cut))
    law = [low, cut - low, 1 - cut]
    certify(recorded, z.And(*[within(law[i], p[i], real(law_gap)) for i in range(3)]))
    # (c) Above the evaluation floor that gap is a relative error of at most 2^-40 and the law is positive.
    q, drawn = z.Reals('ov_q ov_drawn')
    floored = z.And(q >= real(floor), q <= 1, within(drawn, q, real(law_gap)))
    certify(floored, z.And(drawn > 0, within(drawn, q, real(STEP_GAP) * q)))
    require(law_gap / floor == STEP_GAP, 'registered step gap')
    solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
    solver.add(recorded, p[0] == real(Q(1, 2 ** 100)), low > 1000 * p[0])
    require(solver.check() == z.sat, 'the floor is necessary: an unfloored probability admits an unbounded relative gap')
    # (d) One more decision multiplies the weight ratio by a factor within the step gap.
    w, f, lo, hi = z.Reals('ov_w ov_f ov_lo ov_hi')
    certify(z.And(lo > 0, w >= lo, w <= hi, within(f, 1, real(STEP_GAP))),
            z.And(w * f >= lo * real(1 - STEP_GAP), w * f <= hi * real(1 + STEP_GAP)))
    upper, lower = (1 + STEP_GAP) ** horizon - 1, 1 - (1 - STEP_GAP) ** horizon
    require(0 < lower <= upper <= WEIGHT_GAP and ceil_log2(upper) == ceil_log2(WEIGHT_GAP), 'trajectory weight gap')
    # (e) The value learner's law is exact; only the binary64 conversion of 11/12 and 1/24 rounds.
    for exact in (Q(11, 12), Q(1, 24)):
        require(abs(Q(float(exact)) - exact) <= U * exact and Q(float(exact)) >= floor and
                U <= STEP_GAP, 'value-learner conversion gap')
    return {'F-RL-OPE-V8-ARITH': 'unsat'}, {'behaviorFloorLog2': ceil_log2(floor), 'lawGapLog2': ceil_log2(law_gap),
                                            'stepGapLog2': ceil_log2(STEP_GAP), 'weightGapLog2': ceil_log2(upper),
                                            'horizonBound': horizon, 'certifiedImplications': 4, 'satisfiableControls': 1}


def successors(state, requested, mutant=None):
    """Finite stage abstraction: (phase, converted, kept, estimated)."""
    phase, converted, kept, estimated = state
    if phase in ('absent', 'published'):
        return []
    out = [('reject', ('absent', converted, kept, estimated))]
    if phase == 'start':
        out.append(('activate', ('convert', 0, 0, False)))
    elif phase == 'convert':
        if converted < requested:
            out.append(('accept', ('convert', converted + 1, kept + 1, False)))
            if mutant == 'drop-run':
                out.append(('drop', ('convert', converted + 1, kept, False)))
            if mutant == 'publish-partial':
                out.append(('publish', ('published', converted, kept, False)))
        else:
            out.append(('distinct', ('estimate', converted, kept, False)))
    elif phase == 'estimate':
        out.append(('estimate', ('ready', converted, kept, True)))
        if mutant == 'unchecked-estimate':
            out.append(('publish', ('published', converted, kept, False)))
    elif phase == 'ready':
        out.append(('publish', ('published', converted, kept, estimated)))
    return out


def model(mutant=None, requests=(1, 2, 3, 4)):
    states = edges = published = 0
    for requested in requests:
        initial = ('start', 0, 0, False)
        seen = {initial: ()}; queue = deque([initial])
        while queue:
            state = queue.popleft()
            nxt = successors(state, requested, mutant)
            require(bool(nxt) or state[0] in ('absent', 'published'), 'nonterminal deadlock')
            for label, following in nxt:
                edges += 1
                trace = seen[state] + (label,)
                phase, converted, kept, estimated = following
                require(kept <= converted <= requested, 'conversion beyond the request')
                if phase == 'published':
                    require('reject' not in trace and converted == kept == requested and estimated and
                            trace[-3:] == ('distinct', 'estimate', 'publish'),
                            'publication before every run was converted and the estimate was checked')
                    published += 1
                if following not in seen:
                    seen[following] = trace; queue.append(following)
        require(any(s[0] == 'published' for s in seen) and any(s[0] == 'absent' for s in seen), 'vacuous model')
        states += len(seen)
    return {'status': 'model_checked', 'states': states, 'transitions': edges, 'publishingTransitions': published,
            'requests': list(requests), 'orderTransitions': 0,
            'scope': 'finite stage abstraction over small run counts; numeric values abstracted'}


def _impl():
    path = str(ROOT / 'scripts/research')
    if path not in sys.path:
        sys.path.insert(0, path)
    import causal_ope_v8 as impl
    import causal_optimizer_v5 as optimizer
    import causal_ppo_v7 as ppo
    import causal_replay_v3 as replay
    import causal_value_v6 as value
    return impl, optimizer, ppo, replay, value


def synthetic(impl, index, horizon):
    """Deterministic caller model: dyadic target probabilities and values; nothing is fitted."""
    return impl.Model(tuple(((7 * index + 3 * t) % 9 + 1) / 16 for t in range(horizon)),
                      tuple(((index + t) % 5 - 2) / 8 for t in range(horizon)),
                      tuple(((index + 2 * t) % 7 - 3) / 16 for t in range(horizon)) + (0.0,))


def oracle(runs, models, horizon, gamma):
    """Independent exact evaluation from the recorded transitions, with explicit discount powers."""
    rows = []
    for run, model in zip(runs, models):
        weight, plain, stepwise, robust = Q(1), Q(0), Q(0), Q(model.v[0])
        for t in range(horizon):
            reward = Q(float(run.transitions[t].reward))
            weight *= Q(model.target[t]) / Q(float(run.behavior_probabilities[t]))
            discount = Q(gamma) ** t
            plain += discount * reward
            stepwise += weight * discount * reward
            robust += weight * discount * (reward + Q(gamma) * Q(model.v[t + 1]) - Q(model.q[t]))
        rows.append((plain, weight, weight * plain, stepwise, robust))
    n, mass = len(rows), sum(row[1] for row in rows)
    squares = sum(row[1] ** 2 for row in rows)
    return (sum(row[2] for row in rows) / n, sum(row[3] for row in rows) / n,
            sum(row[2] for row in rows) / mass if mass > 0 else None, sum(row[4] for row in rows) / n,
            mass * mass / squares if squares > 0 else Q(0), max(row[1] for row in rows), rows)


def sampling_law(probs):
    """Exact probability of each action under one draw k/2^53 against the binary64 cumulative thresholds."""
    low = math.ceil(Q(probs[0]) * GRID)
    cut = min(math.ceil(Q(probs[0] + probs[1]) * GRID), GRID)
    return low, cut, (Q(low, GRID), Q(cut - low, GRID), Q(GRID - cut, GRID))


def conformance():
    import numpy as np
    from causal_ppo_learner import softmax
    from causal_value import fixtures
    impl, optimizer, ppo, replay, value = _impl()
    reg = json.loads((ROOT / REGISTRATION).read_text())
    count, horizons = reg['fixtureSteps'], reg['horizons']
    train = {value: value.train_causal_value_v6, ppo: ppo.train_causal_ppo_v7}
    runs = []
    for learner in (value, ppo):
        bars, future = fixtures(learner, replay, count)
        runs += [train[learner](bars, future, seed, enabled=True) for seed in reg['seeds']]
    require(all(run is not None and run.steps == count for run in runs), 'successor learner fixtures')
    runs = tuple(runs)
    evaluations = checks = 0
    # All runs together at the registered horizons, then every run alone at the full horizon.
    cases = [(runs, horizon) for horizon in horizons] + [((run,), HORIZON) for run in runs]
    for runs, horizon in cases:
        models = tuple(synthetic(impl, i, horizon) for i in range(len(runs)))
        got = impl.evaluate_causal_ope_v8(runs, models, horizon, enabled=True)
        require(got is not None and got == impl.evaluate_causal_ope_v8(runs, models, horizon, enabled=True),
                'deterministic evaluation')
        require(impl.evaluate_causal_ope_v8(runs, models, horizon) is None and
                impl.evaluate_causal_ope_v8(runs, models, horizon, enabled=True, version='ope-rational-v2') is None,
                'default and wrong-version rejection')
        ordinary, stepwise, weighted, robust, ess, largest, rows = oracle(runs, models, horizon, impl.GAMMA)
        estimate = got.estimate
        require((estimate.ordinary_is, estimate.per_decision_is, estimate.weighted_is, estimate.doubly_robust,
                 estimate.effective_sample_size, estimate.max_weight) == (ordinary, stepwise, weighted, robust, ess, largest)
                and [(e.discounted_return, e.weight, e.ordinary_is, e.per_decision_is, e.doubly_robust)
                     for e in estimate.episodes] == rows, 'exact estimator oracle')
        require((got.version, got.horizon, got.reliable, got.law_gap, got.weight_gap, got.behavior_floor, estimate.horizon) ==
                ('causal-ope-v8', horizon, False, LAW_GAP, WEIGHT_GAP, float(FLOOR), horizon) and
                got.sources == tuple((run.version, run.seed) for run in runs) and estimate.reliable is False, 'metadata')
        checks += len(rows)
        # On-policy control: a target equal to the recorded behavior gives unit weights and full sample size.
        same = tuple(replace(m, target=tuple(float(p) for p in run.behavior_probabilities[:horizon]))
                     for m, run in zip(models, runs))
        control = impl.evaluate_causal_ope_v8(runs, same, horizon, enabled=True).estimate
        require(all(e.weight == 1 for e in control.episodes) and control.effective_sample_size == len(runs) and
                control.ordinary_is == control.per_decision_is == sum(row[0] for row in rows) / len(runs), 'on-policy control')
        evaluations += 2
    runs = cases[0][0]
    # Capacity: the estimator's 8192-bit limit rejects all six runs at the full horizon, because every binary64
    # behavior probability contributes its 53-bit numerator to the common denominator. Detected, not silent.
    full = tuple(synthetic(impl, i, HORIZON) for i in range(len(runs)))
    require(impl.evaluate_causal_ope_v8(runs, full, HORIZON, enabled=True) is None and
            impl.evaluate_causal_ope_v8(runs[:4], full[:4], HORIZON, enabled=True) is not None, 'registered capacity boundary')
    # Sampling-law gap on the recorded policy-gradient steps, by exact integer counting on the draw grid.
    law = 0
    for run in runs:
        if type(run) is not ppo.TrainingResult:
            require(all(p in (Q(11, 12), Q(1, 24)) for p in run.behavior_probabilities), 'exact value-learner law')
            continue
        actor = optimizer.create_v5(run.seed, 3, enabled=True)
        rng = np.random.default_rng([run.seed, 7])
        for row, probability in zip(run.transitions[:HORIZON], run.behavior_probabilities):
            probs = softmax(optimizer.forward_v5(actor, np.asarray([[float(v) for v in row.observation.values]]), enabled=True)[0].tolist())
            draw = Q(float(rng.random())) * GRID
            low, cut, exact = sampling_law(probs)
            require(draw.denominator == 1 and 0 <= draw < GRID, 'draw on the 2^-53 grid')
            require(row.action == (0 if draw < low else 1 if draw < cut else 2) and probability == probs[row.action] and
                    sum(exact) == 1 and all(abs(exact[i] - Q(probs[i])) <= LAW_GAP for i in range(3)) and
                    abs(exact[row.action] - Q(probability)) <= STEP_GAP * Q(probability) and probability >= FLOOR,
                    'sampling-law gap oracle')
            law += 1
    # The evaluation reads the horizon prefix only: rewriting later outcomes changes nothing.
    rewrites = 0
    for learner in (value, ppo):
        bars, future = fixtures(learner, replay, count)
        for seed in reg['seeds']:
            base = train[learner](bars, future, seed, enabled=True)
            for horizon in reg['prefixHorizons']:
                _, changed = fixtures(learner, replay, count, shift=3, start=horizon)
                other = train[learner](bars, changed, seed, enabled=True)
                model = (synthetic(impl, seed, horizon),)
                require(other is not None and other.transitions[horizon] != base.transitions[horizon] and
                        impl.evaluate_causal_ope_v8((base,), model, horizon, enabled=True) ==
                        impl.evaluate_causal_ope_v8((other,), model, horizon, enabled=True) is not None,
                        'a rewrite beyond the horizon changed the evaluation')
                rewrites += 1
    require(rewrites == reg['prefixRewriteEpisodes'], 'registered prefix-rewrite budget')
    # A run that ends exactly at the horizon is evaluated; one that ends earlier rejects everything.
    bars, future = fixtures(ppo, replay, 3)
    ended = ppo.train_causal_ppo_v7(bars, (future[0], ppo.Outcome(future[1].bar, (), terminal=True, fill=Q(1, 2)), None),
                                    reg['seeds'][0], enabled=True)
    require(ended is not None and ended.steps == 2 and
            impl.evaluate_causal_ope_v8((ended,), (synthetic(impl, 0, 2),), 2, enabled=True) is not None, 'terminal at the horizon')
    horizon = horizons[1]
    models = tuple(synthetic(impl, i, horizon) for i in range(len(runs)))
    policy, learned = runs[-1], runs[0]
    forged = lambda run, index, p: replace(run, behavior_probabilities=run.behavior_probabilities[:index] + (p,) +
                                           run.behavior_probabilities[index + 1:])
    swap = lambda items, index, item: items[:index] + (item,) + items[index + 1:]
    bad = [
        ((ended,), (synthetic(impl, 0, 3),), 3),                                    # ended before the horizon
        (swap(runs, 2, ended), models, horizon),                                    # one short run rejects all
        (runs, models[:-1], horizon), (list(runs), models, horizon), (runs, models, HORIZON + 1), (runs, models, 0),
        (runs, models, True), ((), (), horizon), (runs * 43, models * 43, horizon),  # 258 runs
        (runs + (runs[0],), models + (models[0],), horizon),                        # reused stream
        (swap(runs, 5, forged(policy, 3, 2.0 ** -11)), models, horizon),            # below the floor
        (swap(runs, 5, forged(policy, 3, Q(1, 2))), models, horizon),               # not a binary64 record
        (swap(runs, 0, forged(learned, 3, Q(1, 3))), models, horizon),              # not the epsilon-greedy law
        (swap(runs, 0, forged(learned, 3, 11 / 12)), models, horizon),              # rounded, not exact
        (swap(runs, 0, replace(learned, version='causal-ppo-v7')), models, horizon),
        (swap(runs, 0, replace(learned, behavior_probabilities=learned.behavior_probabilities[:-1])), models, horizon),
        (swap(runs, 0, None), models, horizon),
        (runs, swap(models, 1, replace(models[1], target=swap(models[1].target, 0, 1.5))), horizon),
        (runs, swap(models, 1, replace(models[1], target=swap(models[1].target, 0, float('nan')))), horizon),
        (runs, swap(models, 1, replace(models[1], target=swap(models[1].target, 0, 1))), horizon),
        (runs, swap(models, 1, replace(models[1], q=swap(models[1].q, 0, float('inf')))), horizon),
        (runs, swap(models, 1, replace(models[1], q=swap(models[1].q, 0, 2.0 ** 100))), horizon),
        (runs, swap(models, 1, replace(models[1], v=swap(models[1].v, horizon, 0.5))), horizon),
        (runs, swap(models, 1, replace(models[1], v=models[1].v[:-1])), horizon),
        (runs, swap(models, 1, None), horizon),
    ]
    for arguments in bad:
        require(impl.evaluate_causal_ope_v8(*arguments, enabled=True) is None, 'invalid evaluation input must reject')
    # The estimator's own size limit is a rejection, not a silent loss.
    tiny = swap(models, 1, replace(models[1], target=(5e-324,) * horizon))
    require(impl.evaluate_causal_ope_v8(runs, tiny, horizon, enabled=True) is None, 'rational size limit must reject')
    return {'status': 'property_tested', 'runs': len(runs), 'evaluations': evaluations, 'oracleEpisodes': checks,
            'samplingLawChecks': law, 'prefixRewrites': rewrites, 'rejections': len(bad) + 1, 'capacityRejections': 1,
            'seeds': reg['seeds'],
            'marketDataReads': 0, 'economicEvidence': False}


def check_causal_ope():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require((reg['version'], reg['enabledByDefault'], reg['productionIntegration'], reg['financialTrials'],
             reg['historicalDataReads'], reg['holdoutOpened']) == ('causal-ope-v8', False, False, 0, 0, False), 'registration')
    require((reg['runBound'], reg['horizonBound'], reg['behaviorFloorLog2'], reg['lawGapLog2'], reg['weightGapLog2'],
             reg['discountPerTransition']) ==
            (int(CONSTANTS['RUNS']), int(CONSTANTS['HORIZON']), ceil_log2(FLOOR), ceil_log2(LAW_GAP), ceil_log2(WEIGHT_GAP),
             float(CONSTANTS['GAMMA'])), 'registered constants')
    nodes, source = extract()
    smt, bounds = prove(nodes)
    require((bounds['behaviorFloorLog2'], bounds['lawGapLog2'], bounds['weightGapLog2']) ==
            (reg['behaviorFloorLog2'], reg['lawGapLog2'], reg['weightGapLog2']), 'certified bounds match the registration')
    return {'source': source, 'smt': smt, 'bounds': bounds, 'model': model(), 'conformance': conformance()}


if __name__ == '__main__':
    print(json.dumps(check_causal_ope(), indent=1, sort_keys=True))
