"""Obligation 2: source-bound split isolation of the frozen offline runner under the standard purge/embargo definition."""
import ast
import json
from pathlib import Path
import sys

import numpy as np
import z3 as z
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = 'research-notes/registrations/split-isolation-engineering.json'
SCREEN = 'research-notes/registrations/sequential-control-screen-v1.json'
SOURCE_CAMPAIGN = 'research-notes/registrations/residual-momentum-funding-only-v1.json'
RUNNER = 'scripts/research/run_sequential_screen.py'
ENV = 'scripts/research/sequential_env.py'
EVAL = 'scripts/research/sequential_evaluation.py'

FOLD_CALLS = {
    "train = {s: p[:split['trainStop']] for s, p in prices.items()}",
    "funds = {s: p[:split['trainStop']] for s, p in funding.items()}",
    'scale = Scale.fit(list(train.values()))',
    'controls = Baselines(train, funds, scale, h)',
    'net, info = train_ppo(train, funds, scale, h, seed)',
    "net, info = train_q(train, funds, scale, h, seed, offline=alg != 'double_dqn', risk_penalty=0 if alg == 'cql_no_inventory_penalty' else 0.01)",
    'net = None',
    "ope_result = short_ope(prices, funding, scale, h, split['testStart'], split['testStop'], net, seed)",
    "env, result = replay_policy(prices[symbol], funding[symbol], split['testStart'], split['testStop'], h, scale, choose, cfg)",
}
# Every load of the full prices/funding mappings inside the fold loop, by enclosing statement or expression.
FULL_DATA_USES = {
    "{s: p[:split['trainStop']] for s, p in prices.items()}",
    "{s: p[:split['trainStop']] for s, p in funding.items()}",
    "short_ope(prices, funding, scale, h, split['testStart'], split['testStop'], net, seed)",
    "replay_policy(prices[symbol], funding[symbol], split['testStart'], split['testStop'], h, scale, choose, cfg)",
    'sorted(prices)',
}
REPLAY_READS = {'self.funding[left + 1]', 'self.prices[left + 1]', 'self.prices[left]', 'self.prices[self.t]',
                'market_features(self.prices, left)', 'market_features(self.prices, self.t)'}


def require(ok, reason):
    if not ok:
        raise ValueError('split isolation: ' + reason)


def _tree(path, source=None):
    return ast.parse((ROOT / path).read_text() if source is None else source)


def _named(tree, name):
    found = [n for n in tree.body if getattr(n, 'name', None) == name]
    require(len(found) == 1, 'missing/duplicate ' + name)
    return found[0]


def _statements(node):
    return {ast.unparse(s) for s in ast.walk(node) if isinstance(s, (ast.Assign, ast.AugAssign))}


def bind(sources=None):
    sources = sources or {}
    runner = _tree(RUNNER, sources.get(RUNNER))
    run = _named(runner, 'run')
    folds = [n for n in ast.walk(run) if isinstance(n, ast.For) and ast.unparse(n.target) == '(fi, split)']
    require(len(folds) == 1 and ast.unparse(folds[0].iter) == "enumerate(registration['validation']['outerFolds'])", 'single registered fold loop')
    loop = folds[0]
    present = _statements(loop)
    require(FOLD_CALLS <= present, 'fold-loop data binding drift: ' + str(sorted(FOLD_CALLS - present)))
    # No other assignment rebinds train/funds/split/net/scale inside the loop.
    for name in ('train', 'funds', 'net', 'scale', 'split', 'controls'):
        binds = [ast.unparse(s) for s in ast.walk(loop) if isinstance(s, ast.Assign)
                 and any(name in {t.id for t in ast.walk(target) if isinstance(t, ast.Name)} for target in s.targets)]
        expected = {'train': 1, 'funds': 1, 'net': 3, 'scale': 1, 'split': 0, 'controls': 1}[name]
        require(len(binds) == expected, 'unexpected rebinding of ' + name)
    # Every load of the full-panel mappings inside the fold loop is reviewed.
    parents = {c: p for p in ast.walk(loop) for c in ast.iter_child_nodes(p)}
    uses = set()
    for node in ast.walk(loop):
        if isinstance(node, ast.Name) and node.id in ('prices', 'funding') and isinstance(node.ctx, ast.Load):
            up = parents[node]
            while not (isinstance(up, ast.DictComp) or
                       (isinstance(up, ast.Call) and not ast.unparse(up.func).endswith('.items'))):
                up = parents[up]
            uses.add(ast.unparse(up))
    require(uses == FULL_DATA_USES, 'unreviewed full-panel access in fold loop: ' + str(sorted(uses - FULL_DATA_USES)))
    # Market data enters only through the hash-gated loader.
    loader = _named(runner, 'load_development')
    require(ast.unparse(loader.body[1]) == 'panel_bytes, settlement_bytes = (panel.read_bytes(), settlements.read_bytes())' and
            'hashlib.sha256(panel_bytes).hexdigest() != spec[\'panelSha256\']' in ast.unparse(loader.body[2].test),
            'hash-gated market-data admission')
    reads = sorted(ast.unparse(c) for c in ast.walk(runner) if isinstance(c, ast.Call) and ast.unparse(c.func).endswith(('.read_bytes', '.read_text')))
    require(reads == ['p.read_bytes()', 'panel.read_bytes()', 'path.read_bytes()', 'path.read_bytes()', 'settlements.read_bytes()'],
            'unreviewed file read in runner: ' + str(reads))
    loads = [ast.unparse(n) for n in ast.walk(runner) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
             and ast.unparse(n.value.func) == 'load_development']
    require(loads == ['prices, funding, times = load_development(panel, settlements, registration)'] and
            sum(isinstance(n, ast.Call) and ast.unparse(n.func) == 'load_development' for n in ast.walk(runner)) == 1, 'single loader call')

    env = _tree(ENV, sources.get(ENV))
    replay = _named(env, 'Replay')
    init = [n for n in replay.body if getattr(n, 'name', None) == '__init__'][0]
    guards = [ast.unparse(n.test) for n in ast.walk(init) if isinstance(n, ast.If)]
    require('horizon not in (1, 3, 6) or not 24 <= start < stop - 1 <= len(prices) - 1' in guards, 'Replay window guard')
    t_writes = sorted(ast.unparse(s) for s in ast.walk(replay) if isinstance(s, (ast.Assign, ast.AugAssign)) and
                      'self.t' in {ast.unparse(x) for x in ast.walk(s.targets[0] if isinstance(s, ast.Assign) else s.target)})
    require(t_writes == ['self.start, self.stop, self.t = (start, stop, start)', 'self.t += 1'], 'Replay clock writes')
    step = [n for n in replay.body if getattr(n, 'name', None) == 'step'][0]
    require({'end = min(self.t + self.horizon, self.stop - 1)', 'left, old_equity = (self.t, self.equity)'} <= _statements(step) and
            any(isinstance(n, ast.While) and ast.unparse(n.test) == 'self.t < end' for n in ast.walk(step)), 'Replay step bound')
    reads = {ast.unparse(n) for n in ast.walk(replay) if
             (isinstance(n, ast.Subscript) and ast.unparse(n.value) in ('self.prices', 'self.funding')) or
             (isinstance(n, ast.Call) and any(ast.unparse(a) in ('self.prices', 'self.funding') for a in n.args))}
    require(reads == REPLAY_READS, 'unreviewed Replay market read: ' + str(sorted(reads ^ REPLAY_READS)))

    ev = _tree(EVAL, sources.get(EVAL))
    base = [n for n in _named(ev, 'Baselines').body if getattr(n, 'name', None) == '__init__'][0]
    loops = [ast.unparse(n.iter) for n in ast.walk(base) if isinstance(n, ast.For) and ast.unparse(n.target) == 't']
    require(loops == ['range(24, len(p) - horizon - 1)'], 'Baselines label range')
    labels = {ast.unparse(n) for n in ast.walk(base) if isinstance(n, ast.Subscript) and ast.unparse(n.value) == 'p'}
    require(labels == {'p[t + 1 + horizon]', 'p[t + 1]'}, 'Baselines label reads')
    ope = _named(ev, 'short_ope')
    require({'t = int(rng.integers(start, stop - 6 * horizon))',
             'env = Replay(prices[sym], funding[sym], t, t + 6 * horizon + 1, horizon, scale, enabled=True)',
             'replay = Replay(prices[sym], funding[sym], t, t + 6 * horizon + 1, horizon, scale, enabled=True)'} <= _statements(ope),
            'short OPE window binding')
    require(ast.unparse(ope.body[0]) == 'horizon, start, stop, seed, episodes = _admit_ope_window(prices, funding, horizon, start, stop, seed, episodes)',
            'short OPE admission first')
    policy = _named(ev, 'replay_policy')
    require(ast.unparse(policy.body[0]) == 'env = Replay(p, funding, start, stop, horizon, scale, execution, enabled=True)', 'replay_policy window')
    return {'status': 'exhaustively_checked', 'foldBindings': len(FOLD_CALLS), 'fullPanelUses': len(FULL_DATA_USES),
            'replayReads': len(REPLAY_READS), 'scope': 'unchanged frozen runner, Replay, Baselines and short OPE source; interpreter semantics assumed'}


def regions():
    h, L, t, start, stop, left, end, train_stop, test_start, test_stop, i, j = z.Ints(
        'si_h si_L si_t si_start si_stop si_left si_end si_train si_tstart si_tstop si_i si_j')
    horizons = z.Or(h == 1, h == 3, h == 6)
    # Baselines labels on a training slice of length L = trainStop.
    certify(z.And(horizons, L >= 32, t >= 24, t <= L - h - 2), z.And(t + 1 + h <= L - 1, t + 1 <= L - 1))
    # Replay: clock starts at start and increments; reads at left, left+1 and t never pass stop-1.
    replay = z.And(horizons, start >= 24, start < stop - 1, left >= start, end == z.If(left + h < stop - 1, left + h, stop - 1), left < end)
    certify(replay, z.And(left + 1 >= start + 1, left + 1 <= stop - 1, left >= start))
    certify(z.And(start >= 24, start < stop - 1, t >= start, t <= stop - 1), t <= stop - 1)
    # short OPE: t in [start, stop-6h) and episode stop t+6h+1 keep outcomes in [start+1, stop-1].
    certify(z.And(horizons, start >= 24, t >= start, t <= stop - 6 * h - 1, left >= t, left + 1 <= t + 6 * h),
            z.And(left + 1 >= start + 1, left + 1 <= stop - 1))
    # Standard purge and vacuous embargo for a forward fold.
    fold = z.And(train_stop >= 121, train_stop <= test_start, test_start + 1 < test_stop)
    certify(z.And(fold, i >= 0, i <= train_stop - 1, j >= test_start + 1, j <= test_stop - 1), i < j)
    certify(z.And(fold, i >= 0, i < train_stop), i < test_start)
    return {'F-RL-SPLIT-REGIONS': 'unsat'}


def registered():
    screen = json.loads((ROOT / SCREEN).read_text())
    campaign = json.loads((ROOT / SOURCE_CAMPAIGN).read_text())
    data, folds = screen['data'], screen['validation']['outerFolds']
    rows, interval = data['rowsPerSymbol'], data['intervalMilliseconds']
    for fold in folds:
        a, b, c = fold['trainStop'], fold['testStart'], fold['testStop']
        require(all(type(v) is int for v in (a, b, c)), 'integer folds')
        require(121 <= a <= b and 24 <= b < c - 1 <= rows - 1 and c - b > 6 * 6, 'fold inside panel with Replay/OPE room')
    require(screen['validation']['purgeBars'] == 6 and screen['validation']['embargoBars'] == 6, 'registered purge/embargo labels')
    last_close = data['endOpenTime'] + interval - 1
    holdout = next(v for v in _walk(campaign) if isinstance(v, dict) and 'holdoutStartOpenTime' in v)
    require(holdout['developmentCutoffOpenTime'] == data['endOpenTime'] and last_close < holdout['holdoutStartOpenTime'],
            'panel precedes sealed holdout')
    require(data['startOpenTime'] + (rows - 1) * interval == data['endOpenTime'], 'panel grid')
    return {'folds': len(folds), 'lastDevelopmentClose': last_close, 'holdoutStartOpenTime': holdout['holdoutStartOpenTime']}


def _walk(value):
    yield value
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def _actions():
    cycle = iter([0.25, -0.25, 0.0] * 2000)
    return lambda obs: (next(cycle), 0.0)


def conformance():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    from sequential_env import Scale
    from sequential_evaluation import replay_policy, short_ope
    from sequential_learning import Network
    reg = json.loads((ROOT / REGISTRATION).read_text())
    rng = np.random.default_rng(reg['probeSeed'])
    bars = reg['probePanelBars']
    prices = {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, bars))) for s in ('ALPHA', 'BETA')}
    funding = {s: rng.normal(0, 1e-5, bars) for s in prices}
    checked = 0
    for train_stop, test_start, test_stop in reg['probeFolds']:
        scale = Scale.fit([p[:train_stop] for p in prices.values()])
        # Finite rewrite of every value at or after testStop (admission checks stay satisfied).
        moved_p = {s: np.r_[p[:test_stop], p[test_stop:] * rng.uniform(0.5, 2.0, bars - test_stop)] for s, p in prices.items()}
        moved_f = {s: np.r_[f[:test_stop], rng.normal(0, 1e-3, bars - test_stop)] for s, f in funding.items()}
        for h in (1, 3, 6):
            for s in prices:
                clean, _ = replay_policy(prices[s], funding[s], test_start, test_stop, h, scale, _actions())
                moved, _ = replay_policy(moved_p[s], moved_f[s], test_start, test_stop, h, scale, _actions())
                require(clean.rows == moved.rows and clean.equity == moved.equity and clean.failure == moved.failure,
                        'evaluation depends on values at or after testStop')
                require(clean.rows and all(test_start + 1 <= row['t'] <= test_stop - 1 for row in clean.rows),
                        'outcome outside validation region')
                checked += 1
            net = Network(7)
            a = short_ope(prices, funding, scale, h, test_start, test_stop, net, 11, episodes=24)
            b = short_ope(moved_p, moved_f, scale, h, test_start, test_stop, net, 11, episodes=24)
            require(json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str),
                    'short OPE depends on values at or after testStop')
            checked += 1
    return {'status': 'property_tested', 'checks': checked, 'folds': len(reg['probeFolds']), 'seed': reg['probeSeed'],
            'marketDataReads': 0}


def check_split():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require(reg['sourceChanges'] == 0 and reg['holdoutOpened'] is False, 'registration')
    return {'source': bind(), 'smt': regions(), 'registered': registered(), 'conformance': conformance()}
