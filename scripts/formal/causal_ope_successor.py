"""Causal OPE v9: causal-ope-v8 over ope-rational-v3, so no admitted evaluation is rejected for size."""
import ast
from dataclasses import replace
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import sys

import z3 as z
import causal_ope as predecessor
import ope_capacity
from causal_value import closure, read_module
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'causal_ope_v9'
SOURCE = 'scripts/research/causal_ope_v9.py'
PREDECESSOR = predecessor.SOURCE
ESTIMATOR = ope_capacity.SOURCE
REGISTRY = 'formal/research/causal-ope-v9-source.json'
REGISTRATION = 'research-notes/registrations/causal-ope-v9-engineering.json'
# The successor is the reviewed predecessor with exactly these tokens replaced: (new, old, occurrences).
SUBSTITUTIONS = (('causal-ope-v9', 'causal-ope-v8', 1), ('ope_rational_v3', 'ope_rational_v2', 1),
                 ('estimate_v3', 'estimate_v2', 2), ('evaluate_causal_ope_v9', 'evaluate_causal_ope_v8', 2))
CLOSURE = (predecessor.CLOSURE - {predecessor.MODULE, 'ope_rational_v2'}) | {MODULE, 'ope_rational_v3'}
HORIZON = predecessor.HORIZON


def require(ok, reason):
    if not ok: raise ValueError('causal ope v9: ' + reason)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def registry_for(source, previous=None, estimator=None):
    previous = (ROOT / PREDECESSOR).read_text() if previous is None else previous
    estimator = (ROOT / ESTIMATOR).read_text() if estimator is None else estimator
    try:
        tree = ast.parse(source)
    except SyntaxError:
        raise ValueError('causal ope v9: unparsable source')
    return {'schemaVersion': 1, 'astSha256': sha(ast.dump(tree, include_attributes=False)),
            'predecessorSha256': sha(previous), 'estimatorSha256': sha(estimator)}


def extract(source=None, registry=None, previous=None, read=read_module):
    source = (ROOT / SOURCE).read_text() if source is None else source
    previous = (ROOT / PREDECESSOR).read_text() if previous is None else previous
    registry = json.loads((ROOT / REGISTRY).read_text()) if registry is None else registry
    require(registry == registry_for(source, previous, read('ope_rational_v3')), 'reviewed source, predecessor or estimator drift')
    restored = source
    for new, old, count in SUBSTITUTIONS:
        require(restored.count(new) == count and restored.count(old) == 0, 'substitution site ' + new)
        restored = restored.replace(new, old)
    require(ast.dump(ast.parse(restored)) == ast.dump(ast.parse(previous)), 'difference from the reviewed predecessor')
    # The predecessor's lock, admission, floor, prefix-only access and publication review now applies verbatim.
    nodes, surface = predecessor.extract(restored, read=read)
    # The estimator behind the new import is the certified one: same envelope, same bounds, standard library only.
    ope_capacity.extract(read('ope_rational_v3'))
    try:
        local, external = closure(lambda name: source if name == MODULE else read(name), MODULE)
    except ValueError as error:
        raise ValueError('causal ope v9: ' + str(error)) from None
    require(local == CLOSURE and not local & predecessor.FROZEN, 'frozen-free import closure')
    require(external == predecessor.EXTERNAL, 'external import closure')
    require((ope_capacity.EPISODES, ope_capacity.DECISIONS) == (int(predecessor.CONSTANTS['RUNS']), HORIZON),
            'estimator domain covers the adapter domain')
    return nodes, {'status': 'exhaustively_checked', 'definitions': surface['definitions'], 'substitutions': len(SUBSTITUTIONS),
                   'substitutedTokens': sum(count for _, _, count in SUBSTITUTIONS), 'closure': sorted(local),
                   'externalImports': sorted(external), 'frozenModulesInClosure': 0, 'estimator': 'ope-rational-v3',
                   'scope': 'complete module equal to the reviewed predecessor up to six token occurrences, over the '
                            'certified estimator; runtime semantics assumed'}


def prove(nodes=None):
    """The sampling-law lemmas are the predecessor's; what is new is that an admitted evaluation cannot be rejected for size."""
    _, gaps = predecessor.prove(nodes)
    # Every value the adapter passes on lies in the estimator's admitted domain.
    b, p, x, last = z.Reals('on_b on_p on_x on_last')
    floor, bound = z.RealVal(str(predecessor.FLOOR)), z.RealVal(2 ** 99)
    certify(z.And(b >= floor, b <= 1, p >= 0, p <= 1, x >= -bound, x <= bound, last == 0),
            z.And(b > 0, b <= 1, p >= 0, p <= 1, last == 0))
    runs, horizon = z.Ints('on_runs on_horizon')
    certify(z.And(runs >= 1, runs <= int(predecessor.CONSTANTS['RUNS']), horizon >= 1, horizon <= HORIZON),
            z.And(runs >= 1, runs <= ope_capacity.EPISODES, horizon >= 1, horizon <= ope_capacity.DECISIONS))
    # Control: the adapter's magnitude guard alone does not give the behavior domain; the floor does.
    solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
    solver.add(b >= -bound, b <= bound, z.Not(z.And(b > 0, b <= 1)))
    require(solver.check() == z.sat, 'domain control')
    reg = json.loads((ROOT / ope_capacity.REGISTRATION).read_text())
    peak = reg['certifiedPeakBits']
    require(peak <= ope_capacity.LIMIT == reg['maximumRationalBits'], 'certified estimator size bound')
    return {'F-RL-OPE-V9-ARITH': 'unsat'}, dict(gaps, certifiedImplications=gaps['certifiedImplications'] + 2,
                                               satisfiableControls=gaps['satisfiableControls'] + 1,
                                               estimatorPeakBits=peak, estimatorLimitBits=ope_capacity.LIMIT)


def _impl():
    old, optimizer, ppo, replay, value = predecessor._impl()
    import causal_ope_v9 as impl
    return impl, old, ppo, replay, value


def fields(estimate):
    return (estimate.ordinary_is, estimate.per_decision_is, estimate.weighted_is, estimate.doubly_robust,
            estimate.effective_sample_size, estimate.max_weight, estimate.nonzero_trajectories, estimate.horizon,
            tuple((e.discounted_return, e.weight, e.ordinary_is, e.per_decision_is, e.doubly_robust) for e in estimate.episodes))


def conformance():
    from causal_value import fixtures
    impl, old, ppo, replay, value = _impl()
    reg = json.loads((ROOT / REGISTRATION).read_text())
    count = reg['fixtureSteps']
    train = {value: value.train_causal_value_v6, ppo: ppo.train_causal_ppo_v7}
    runs = []
    for learner in (value, ppo):
        bars, future = fixtures(learner, replay, count)
        runs += [train[learner](bars, future, seed, enabled=True) for seed in reg['capacitySeeds']]
    require(all(run is not None and run.steps == count for run in runs), 'successor learner fixtures')
    half = len(reg['capacitySeeds'])
    small = tuple(runs[:3] + runs[half:half + 3])
    evaluations = oracle = agreements = 0
    cases = [(small, horizon) for horizon in reg['horizons']] + [(small, HORIZON), (tuple(runs), HORIZON)]
    for group, horizon in cases:
        models = tuple(predecessor.synthetic(impl, i, horizon) for i in range(len(group)))
        got = impl.evaluate_causal_ope_v9(group, models, horizon, enabled=True)
        require(got is not None and got == impl.evaluate_causal_ope_v9(group, models, horizon, enabled=True),
                'deterministic evaluation')
        require(impl.evaluate_causal_ope_v9(group, models, horizon) is None and
                impl.evaluate_causal_ope_v9(group, models, horizon, enabled=True, version='causal-ope-v8') is None,
                'default and wrong-version rejection')
        ordinary, stepwise, weighted, robust, ess, largest, rows = predecessor.oracle(group, models, horizon, impl.GAMMA)
        estimate = got.estimate
        require((estimate.ordinary_is, estimate.per_decision_is, estimate.weighted_is, estimate.doubly_robust,
                 estimate.effective_sample_size, estimate.max_weight) == (ordinary, stepwise, weighted, robust, ess, largest)
                and [(e.discounted_return, e.weight, e.ordinary_is, e.per_decision_is, e.doubly_robust)
                     for e in estimate.episodes] == rows, 'exact estimator oracle')
        require((got.version, got.estimator_version, estimate.version, got.horizon, got.reliable, estimate.reliable,
                 got.law_gap, got.weight_gap, got.behavior_floor) ==
                ('causal-ope-v9', 'ope-rational-v3', 'ope-rational-v3', horizon, False, False, predecessor.LAW_GAP,
                 predecessor.WEIGHT_GAP, float(predecessor.FLOOR)) and
                got.sources == tuple((run.version, run.seed) for run in group), 'metadata')
        oracle += len(rows); evaluations += 1
        # Wherever the predecessor publishes, the successor publishes the same numbers.
        before = old.evaluate_causal_ope_v8(group, tuple(old.Model(m.target, m.q, m.v) for m in models), horizon, enabled=True)
        if before is not None:
            require(fields(before.estimate) == fields(estimate) and before.sources == got.sources, 'predecessor agreement')
            agreements += 1
        else:
            require(horizon == HORIZON, 'predecessor rejected below the full horizon')
        same = tuple(replace(m, target=tuple(float(p) for p in run.behavior_probabilities[:horizon]))
                     for m, run in zip(models, group))
        control = impl.evaluate_causal_ope_v9(group, same, horizon, enabled=True).estimate
        require(all(e.weight == 1 for e in control.episodes) and control.effective_sample_size == len(group) and
                control.ordinary_is == control.per_decision_is == sum(row[0] for row in rows) / len(group), 'on-policy control')
        evaluations += 1
    require(agreements == len(reg['horizons']), 'the predecessor publishes below the full horizon and rejects at it')
    # Extreme admitted models: subnormal targets and largest admitted values publish instead of rejecting for size.
    horizon = HORIZON
    models = tuple(predecessor.synthetic(impl, i, horizon) for i in range(len(small)))
    swap = lambda items, index, item: items[:index] + (item,) + items[index + 1:]
    extreme = swap(models, 1, replace(models[1], target=(5e-324,) * horizon, q=(2.0 ** 99,) * horizon,
                                      v=(-2.0 ** 99,) * horizon + (0.0,)))
    got = impl.evaluate_causal_ope_v9(small, extreme, horizon, enabled=True)
    require(got is not None and got.estimate.episodes[1].weight > 0 and
            fields(got.estimate)[:6] == predecessor.oracle(small, extreme, horizon, impl.GAMMA)[:6], 'extreme admitted model')
    require(old.evaluate_causal_ope_v8(small, tuple(old.Model(m.target, m.q, m.v) for m in extreme), horizon,
                                       enabled=True) is None, 'predecessor rejects the extreme model')
    horizon = reg['horizons'][1]
    models = tuple(predecessor.synthetic(impl, i, horizon) for i in range(len(small)))
    policy, learned = small[-1], small[0]
    forged = lambda run, index, p: replace(run, behavior_probabilities=run.behavior_probabilities[:index] + (p,) +
                                           run.behavior_probabilities[index + 1:])
    bad = [
        (small, models[:-1], horizon), (list(small), models, horizon), (small, models, HORIZON + 1), (small, models, 0),
        (small, models, True), ((), (), horizon), (small * 43, models * 43, horizon),
        (small + (small[0],), models + (models[0],), horizon),
        (swap(small, 5, forged(policy, 3, 2.0 ** -11)), models, horizon),
        (swap(small, 5, forged(policy, 3, Q(1, 2))), models, horizon),
        (swap(small, 0, forged(learned, 3, Q(1, 3))), models, horizon),
        (swap(small, 0, replace(learned, version='causal-ppo-v7')), models, horizon),
        (swap(small, 0, replace(learned, transitions=learned.transitions[:horizon - 1],
                                behavior_probabilities=learned.behavior_probabilities[:horizon - 1])), models, horizon),
        (swap(small, 0, None), models, horizon),
        (small, swap(models, 1, replace(models[1], target=swap(models[1].target, 0, 1.5))), horizon),
        (small, swap(models, 1, replace(models[1], q=swap(models[1].q, 0, 2.0 ** 100))), horizon),
        (small, swap(models, 1, replace(models[1], v=swap(models[1].v, horizon, 0.5))), horizon),
        (small, swap(models, 1, old.Model(models[1].target, models[1].q, models[1].v)), horizon),
        (small, swap(models, 1, None), horizon),
    ]
    for arguments in bad:
        require(impl.evaluate_causal_ope_v9(*arguments, enabled=True) is None, 'invalid evaluation input must reject')
    return {'status': 'property_tested', 'runs': len(runs), 'evaluations': evaluations, 'oracleEpisodes': oracle,
            'predecessorAgreements': agreements, 'capacityRuns': len(runs), 'capacityHorizon': HORIZON,
            'extremeModels': 1, 'rejections': len(bad), 'seeds': len(reg['capacitySeeds']),
            'marketDataReads': 0, 'economicEvidence': False}


def check_causal_ope_successor():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require((reg['version'], reg['estimatorVersion'], reg['enabledByDefault'], reg['productionIntegration'],
             reg['financialTrials'], reg['historicalDataReads'], reg['holdoutOpened']) ==
            ('causal-ope-v9', 'ope-rational-v3', False, False, 0, 0, False), 'registration')
    require((reg['runBound'], reg['horizonBound']) == (int(predecessor.CONSTANTS['RUNS']), HORIZON), 'registered bounds')
    nodes, source = extract()
    smt, bounds = prove(nodes)
    model = predecessor.model()
    rejected = 0
    for mutant in ('drop-run', 'publish-partial', 'unchecked-estimate'):
        try:
            predecessor.model(mutant)
        except ValueError:
            rejected += 1
    require(rejected == 3, 'publication mutant accepted')
    return {'source': source, 'smt': smt, 'bounds': bounds, 'model': dict(model, mutantsRejected=rejected),
            'conformance': conformance()}


if __name__ == '__main__':
    print(json.dumps(check_causal_ope_successor(), indent=1, sort_keys=True))
