"""Obligation 17 audit: preserved training-scale lookahead witness (CE-RL-025) and fixed-scale Replay causality."""
import ast
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = 'research-notes/registrations/observation-causality-audit-engineering.json'
COUNTEREXAMPLE = 'formal/research/observation-counterexamples.json'
RUNNER = 'scripts/research/run_sequential_screen.py'
ENV = 'scripts/research/sequential_env.py'


def require(ok, reason):
    if not ok:
        raise ValueError('observation causality: ' + reason)


def _named(tree, name):
    found = [n for n in tree.body if getattr(n, 'name', None) == name]
    require(len(found) == 1, 'missing/duplicate ' + name)
    return found[0]


def bind(sources=None):
    sources = sources or {}
    runner = ast.parse(sources.get(RUNNER, (ROOT / RUNNER).read_text()))
    fits = [ast.unparse(n) for n in ast.walk(_named(runner, 'run')) if isinstance(n, ast.Assign) and 'Scale.fit' in ast.unparse(n.value)]
    require(fits == ['scale = Scale.fit(list(train.values()))'], 'single prefix-wide scale fit')
    env = ast.parse(sources.get(ENV, (ROOT / ENV).read_text()))
    collect = _named(env, 'collect')
    statements = {ast.unparse(n) for n in ast.walk(collect) if isinstance(n, ast.Assign)}
    require({'start = int(rng.integers(24, len(p) - 96))',
             'env = Replay(p, funding[symbol], start, start + 97, horizon, scale, execution, enabled=True)'} <= statements,
            'collect episodes start inside the prefix with the passed scale')
    replay = _named(env, 'Replay')
    observation = [n for n in replay.body if getattr(n, 'name', None) == 'observation'][0]
    body = {ast.unparse(n) for n in ast.walk(observation) if isinstance(n, ast.Assign)}
    require({'x = market_features(self.prices, self.t)', 'normalized = self.scale.transform(x)', 'p = self.prices[self.t]'} <= body,
            'observation normalizes current features with the replay scale')
    return {'scaleFits': len(fits), 'scope': 'unchanged frozen runner, collect and Replay.observation source'}


def _modules():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    from sequential_env import Replay, Scale
    return Replay, Scale


def witness():
    Replay, Scale = _modules()
    reg = json.loads((ROOT / REGISTRATION).read_text())['witness']
    rng = np.random.default_rng(reg['seed'])
    n, t = reg['bars'], reg['decision']
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    f = np.zeros(n)
    q = p.copy()
    q[t + 1:] = q[t + 1:] * np.exp(np.cumsum(rng.normal(0, reg['futureLogSigma'], n - t - 1)))
    require(np.array_equal(p[:t + 1], q[:t + 1]), 'witness changes only data after the decision')
    # The frozen runner refits on the whole prefix, so a later-bar change changes the scale.
    a = Replay(p, f, t, reg['episodeStop'], 1, Scale.fit([p]), enabled=True).observation()
    b = Replay(q, f, t, reg['episodeStop'], 1, Scale.fit([q]), enabled=True).observation()
    require(a is not None and b is not None, 'witness observations exist')
    gap = float(np.max(np.abs(a - b)))
    require(gap > 0, 'CE-RL-025 no longer reproduces; re-audit obligation 17')
    recorded = json.loads((ROOT / COUNTEREXAMPLE).read_text())
    require(recorded['id'] == 'CE-RL-025' and recorded['witness']['maxAbsObservationDifference'] == round(gap, 9),
            'recorded CE-RL-025 witness drift')
    return {'status': 'refuted', 'counterexample': 'CE-RL-025', 'maxAbsObservationDifference': round(gap, 9)}


def replay_causal():
    Replay, Scale = _modules()
    reg = json.loads((ROOT / REGISTRATION).read_text())
    rng = np.random.default_rng(reg['probeSeed'])
    checked = 0
    for episode in range(reg['probeEpisodes']):
        n = 260
        p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        f = rng.normal(0, 1e-5, n)
        scale = Scale.fit([p[:120]])
        start, stop, h = 130, 200, (1, 3, 6)[episode % 3]
        actions = rng.choice([-0.25, 0.0, 0.25], size=200)
        base = Replay(p, f, start, stop, h, scale, enabled=True)
        trace = []
        k = 0
        while not base.done:
            trace.append((base.t, base.observation(), base.supported(), len(base.rows)))
            base.step(float(actions[k])); k += 1
        for t, obs, sup, rows in trace:
            q, g = p.copy(), f.copy()
            q[t + 1:] *= rng.uniform(0.5, 2.0, n - t - 1)
            g[t + 1:] = rng.normal(0, 1e-3, n - t - 1)
            env = Replay(q, g, start, stop, h, scale, enabled=True)
            j = 0
            while env.t < t and not env.done:
                env.step(float(actions[j])); j += 1
            require(env.t == t and np.array_equal(env.observation(), obs) and env.supported() == sup and
                    env.rows[:rows] == base.rows[:rows], 'fixed-scale observation depends on data after t')
            checked += 1
    return {'status': 'property_tested', 'episodes': reg['probeEpisodes'], 'decisionChecks': checked, 'seed': reg['probeSeed']}


def check_observation():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require(reg['sourceChanges'] == 0 and reg['holdoutOpened'] is False, 'registration')
    return {'source': bind(), 'trainScale': witness(), 'replayCausal': replay_causal()}
