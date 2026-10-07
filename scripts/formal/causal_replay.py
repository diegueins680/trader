"""Causal exact replay: source footprint, SMT indices, finite flow and conformance."""
import ast
from collections import deque
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import z3 as z
from promotion_boundary import shape
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/causal_replay_v3.py'
REGISTRY = 'formal/research/causal-replay-v3-source.json'
REGISTRATION = 'research-notes/registrations/causal-replay-v3-engineering.json'


def require(ok, reason):
    if not ok:
        raise ValueError('causal replay v3: ' + reason)


def extract(source=None):
    tree = ast.parse((ROOT / SOURCE).read_text() if source is None else source)
    lock = json.loads((ROOT / REGISTRY).read_text())
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    require(hashlib.sha256(shape(tree).encode()).hexdigest() == lock['astSha256'], 'complete source drift')
    require({k: shape(v) for k, v in nodes.items()} == lock['definitions'], 'definition drift')
    for path, value in lock['supportHashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value, 'support drift: ' + path)
    require([ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))] ==
            ['from __future__ import annotations', 'from dataclasses import dataclass',
             'from fractions import Fraction as F', 'import replay_accounting_v2 as a'], 'effect boundary')
    for name in ('Bar', 'Session', 'Observation'):
        require([ast.unparse(d) for d in nodes[name].decorator_list] == ['dataclass(frozen=True, slots=True)'], 'mutable publication')
    for name in ('start_v3', 'observe_v3', 'step_v3'):
        node = nodes[name]
        defaults = dict(zip((a.arg for a in node.args.kwonlyargs), map(ast.unparse, node.args.kw_defaults)))
        require(defaults['enabled'] == 'False' and defaults['version'] == 'VERSION', 'disabled default')
        require(ast.unparse(node.body[0]) == 'if enabled is not True or type(version) is not str or version != VERSION:\n    return None', 'activation first')
        require(ast.unparse(node.body[1].handlers[0].type) == '(ArithmeticError, ValueError, TypeError, MemoryError)', 'fail-closed exception boundary')
    calls = {name: [ast.unparse(c) for c in ast.walk(nodes[name]) if isinstance(c, ast.Call)] for name in nodes}
    require('_fit(bars[:-1])' in calls['start_v3'], 'fit includes initial/future decision')
    require('range(25, len(bars) + 1)' in calls['_fit'] and '_features(bars[:stop])' in calls['_fit'], 'scale prefix footprint')
    require('tuple((b.price for b in bars[-25:]))' in calls['_features'], 'trailing feature footprint')
    require('_features(session.history)' in calls['_observe'], 'unconsumed observation data')
    require('_fit(value.history[:value.burn])' in calls['_session'], 'forged scale admission')
    step = nodes['step_v3'].body[1].body
    require(ast.unparse(step[0]) == 'observation = observe_v3(session, enabled=True)', 'next bar seen before decision')
    require(ast.unparse(step[2]) == 'chosen = target if observation.supported else F(0)', 'support bypass')
    require(ast.unparse(step[3].value) == 'a.advance_v2(session.state, bar.price, events, chosen, terminal=terminal, fill=fill, multiplier=multiplier, impact=impact, enabled=True)', 'accounting bypass')
    require(ast.unparse(step[5].value) == 'Session(VERSION, session.history + (bar,), session.burn, session.scale, receipt.after)', 'scale mutation or history reorder')
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    require(not names & {'float', 'eval', 'exec', 'open', 'socket', 'submit', 'promote', 'policy'}, 'unreviewed effect')
    return {'status': 'exhaustively_checked', 'definitions': len(nodes), 'defaultDisabledEntries': 3,
            'scope': 'complete reviewed AST, all entry guards and observation/accounting composition; interpreter trusted'}


def prove():
    n, k, stop, h, i, tick = z.Ints('cv_n cv_k cv_stop cv_h cv_i cv_tick')
    initial = z.And(n >= 26, n <= 4096)
    certify(initial, z.And(n - 1 >= 25, n - 1 <= 4095))
    fit = z.And(initial, stop >= 25, stop <= n - 1, i >= 0, i < stop)
    certify(fit, z.And(i >= 0, i < n - 1))
    current = z.And(k >= 25, k < 8192, z.Or(h == 1, h == 3, h == 6, h == 24))
    certify(current, z.And(k - h >= 0, k - h <= k, k - 24 >= 0))
    certify(z.And(n >= 26, n <= 4096, tick >= 0, tick < 4096), n + tick + 1 <= 8192)
    previous_symbol, next_symbol = z.Strings('cv_previous_symbol cv_next_symbol')
    certify(previous_symbol == next_symbol, z.Not(previous_symbol != next_symbol))
    prev, close, available = z.Ints('cv_prev cv_close cv_available')
    certify(z.And(prev >= 0, prev < close, close <= available, available < 2**63), prev < available)
    lo, x, hi, m, w = z.Reals('cv_lo cv_x cv_hi cv_mean cv_width')
    certify(z.And(lo <= x, x <= hi, w > 0), z.And((lo - m) / w <= (x - m) / w, (x - m) / w <= (hi - m) / w))
    action = z.Real('cv_action'); support = z.Bool('cv_support')
    chosen = z.If(support, action, z.RealVal(0))
    certify(z.Or(action == -z.RealVal(1)/4, action == 0, action == z.RealVal(1)/4),
            z.And(chosen >= -z.RealVal(1)/4, chosen <= z.RealVal(1)/4, z.Implies(z.Not(support), chosen == 0)))
    return {'F-RL-CAUSAL-V3-ARITH': 'unsat'}


def model(mutant=False):
    # Finite control abstraction. Market values and kernel risk predicates abstracted.
    # State: number consumed after initial, scale-generation, phase, authority.
    bound = json.loads((ROOT / REGISTRATION).read_text())['modelStepBound']
    initial = (0, 0, 'ready', False)
    queue = deque([initial]); seen = {initial}; edges = 0
    while queue:
        tick, scale, phase, authority = queue.popleft()
        require(scale == 0 and not authority, 'scale change or live authority')
        successors = []
        if phase == 'ready':
            successors = [(tick, scale, 'rejected', authority), (tick, scale, 'observed', authority)]
        elif phase == 'observed':
            successors = [(tick, scale, 'rejected', authority), (tick + 1, scale, 'terminal', authority)]
            if tick + 1 < bound:
                successors.append((tick + 1, scale + int(mutant), 'ready', authority))
        else:
            require(phase in ('rejected', 'terminal'), 'unexpected deadlock')
        for s in successors:
            edges += 1
            require(s[0] <= bound, 'step bound')
            if s not in seen:
                seen.add(s); queue.append(s)
    require(any(s[0] == bound and s[2] == 'terminal' for s in seen), 'vacuous termination')
    return {'status': 'model_checked', 'states': len(seen), 'transitions': edges, 'maxSteps': bound,
            'orderTransitions': 0, 'scope': 'finite acyclic observation/advance abstraction; numeric guards delegated'}


def modules():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import causal_replay_v3 as impl
    return impl


def conformance():
    c = modules(); reg = json.loads((ROOT / REGISTRATION).read_text()); rng = random.Random(reg['probeSeed'])
    wire, expected = [], []
    episodes = transitions = neutralized = 0
    for episode in range(reg['probeEpisodes']):
        prices = tuple(F(rng.randrange(9900, 10101), 100) for _ in range(96))
        bars = tuple(c.Bar("BTC", i * 1000, i * 1000 + 10, p) for i, p in enumerate(prices))
        state = c.start_v3(bars[:50], enabled=True)
        require(state is not None and c.start_v3(bars[:50]) is None, 'initial/default')
        initial = state
        for j in range(50, 66):
            obs = c.observe_v3(state, enabled=True)
            require(obs is not None and obs.decision == bars[j-1].available, 'observation')
            # Independently regenerate the session from the same consumed prefix; all
            # unconsumed prices may be invalid and cannot be passed to observation.
            changed = bars[:j] + tuple(c.Bar("BTC", b.close, b.available, F(-1)) for b in bars[j:])
            replay = c.start_v3(changed[:50], enabled=True)
            for k in range(50, j):
                replay = c.step_v3(replay, changed[k], (), F(1,4), enabled=True)[0]
            require(c.observe_v3(replay, enabled=True) == obs, 'future rewrite altered observation')
            raw = c._features(state.history)
            # Oracle input includes independent feature history + scale + inventory state.
            values = tuple(b.price for b in state.history[-25:]) + sum(state.scale[:2], ()) + (
                state.state.units, state.state.price, state.state.equity, state.state.peak)
            wire.append(str([(x.numerator, x.denominator) for x in values]))
            expected.append(str([(x.numerator, x.denominator) for x in obs.values]).replace(' ', ''))
            pair = c.step_v3(state, bars[j], (), F(1,4), terminal=j == 65, enabled=True)
            require(pair is not None, 'transition rejected')
            following, receipt = pair
            require(following.scale is initial.scale and initial.history == bars[:50], 'mutation')
            chosen = F(1,4) if obs.supported else F(0)
            direct = c.a.advance_v2(state.state, bars[j].price, (), chosen, terminal=j == 65, enabled=True)
            require(receipt == direct, 'shield/accounting composition')
            costs = receipt.costs
            require(receipt.after.equity == receipt.before.equity + receipt.gross + receipt.funding -
                    costs.fee - costs.spread - costs.slippage - costs.impact, 'wealth identity')
            neutralized += not obs.supported
            transitions += 1; state = following
        require(state.state.terminal and c.observe_v3(state, enabled=True) is None, 'terminal observation')
        episodes += 1
    out = subprocess.run(['runghc', str(ROOT / 'formal/research/CausalReplayV3.hs')], input='\n'.join(wire)+'\n',
                         capture_output=True, text=True, timeout=120, check=True)
    require(out.stdout.splitlines() == expected, 'Haskell exact observation oracle')
    return {'status': 'property_tested', 'episodes': episodes, 'transitions': transitions, 'haskellRows': len(wire),
            'neutralized': neutralized, 'seed': reg['probeSeed'], 'historicalDataReads': 0}


def check_causal():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require(reg['financialTrials'] == 0 and not reg['holdoutOpened'] and not reg['enabledByDefault'], 'registration drift')
    return {'source': extract(), 'smt': prove(), 'model': model(), 'conformance': conformance()}
