"""Source-bound exact episode runner: index/telescoping/causality SMT, bounded model, oracle and metamorphic tests."""
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
SOURCE = 'scripts/research/replay_runner_v2.py'
REGISTRY = 'formal/research/replay-runner-v2-source.json'
REGISTRATION = 'research-notes/registrations/replay-runner-v2-engineering.json'
ADVANCE = ('accounting.advance_v2(state, prices[left + k], buckets.events[left + k], targets[k - 1], '
           'terminal=k == len(targets), fill=fill, multiplier=multiplier, impact=impact, enabled=True)')


def require(ok, reason):
    if not ok:
        raise ValueError('replay runner v2: ' + reason)


def extract(source=None, registry=None):
    tree = ast.parse((ROOT / SOURCE).read_text() if source is None else source)
    lock = json.loads((ROOT / REGISTRY).read_text()) if registry is None else registry
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    require(len(nodes) == sum(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body), 'duplicate definition')
    require({k: shape(v) for k, v in nodes.items()} == lock['definitions'], 'reviewed definition drift')
    require(hashlib.sha256(shape(tree).encode()).hexdigest() == lock['astSha256'], 'module effect drift')
    for path, value in lock['supportHashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value, 'support drift: ' + path)
    constants = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    require(constants == {'VERSION': "'replay-runner-v2'", 'MAX_STEPS': '4096', 'MAX_BARS': '8192',
                          '__all__': "['VERSION', 'Episode', 'run_v2']"}, 'version/bounds/exports')
    require([ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))] ==
            ['from __future__ import annotations', 'from dataclasses import dataclass',
             'from fractions import Fraction as F', 'import funding_events_v2 as funding',
             'import replay_accounting_v2 as accounting'], 'composition-only import boundary')
    require([ast.unparse(n) for n in nodes['Episode'].decorator_list] == ['dataclass(frozen=True, slots=True)'], 'immutable output')
    run = nodes['run_v2']
    defaults = dict(zip((x.arg for x in run.args.kwonlyargs), map(ast.unparse, run.args.kw_defaults)))
    require(defaults['enabled'] == 'False' and defaults['version'] == 'VERSION', 'disabled default')
    require(ast.unparse(run.body[0]) == "if enabled is not True or type(version) is not str or version != VERSION:\n    return None", 'activation first')
    require(ast.unparse(run.body[1].handlers[0].type) == '(ArithmeticError, ValueError, TypeError, MemoryError)', 'all-or-nothing rejection')
    body = nodes['_run'].body
    require(ast.unparse(body[0]) == 'state = accounting.initial_v2(prices[left], enabled=True)', 'initial read')
    loop = body[3]
    require(isinstance(loop, ast.For) and ast.unparse(loop.iter) == 'range(1, len(targets) + 1)', 'step range')
    require(ast.unparse(loop.body[0].value) == ADVANCE, 'only aligned reads and forced final terminal')
    require(ast.unparse(loop.body[4]) == 'if state.terminal:\n    break', 'stop at first terminal')
    require(ast.unparse(body[4]) == 'if not state.terminal:\n    return None', 'unterminated episode rejects')
    # Subscripts anywhere in the runner: only the reviewed aligned reads exist.
    reads = sorted(ast.unparse(n) for n in ast.walk(nodes['_run']) if isinstance(n, ast.Subscript))
    require(reads == ['buckets.events[left + k]', 'prices[left + k]', 'prices[left]', 'targets[k - 1]'], 'unreviewed indexed read')
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    require(not names & {'float', 'eval', 'exec', 'open', 'socket', 'order', 'submit', 'promote', 'policy'}, 'effect-free source')
    return nodes, {'status': 'exhaustively_checked', 'definitions': len(nodes), 'indexedReads': len(reads),
                   'defaultDisabledEntries': 1, 'scope': 'complete reviewed AST and every indexed read; runtime semantics assumed'}


def prove(nodes):
    # (a) Index alignment for H steps from left over n bars, admitted by _admissible.
    left, h, n, k = z.Ints('rr_left rr_h rr_n rr_k')
    admitted = z.And(left >= 0, h >= 1, h <= 4096, n >= 1, n <= 8192, left + h < n)
    certify(admitted, z.And(left >= 0, left < n))
    step = z.And(admitted, k >= 1, k <= h)
    certify(step, z.And(left + k >= 1, left + k < n, k - 1 >= 0, k - 1 < h))
    certify(step, z.And(left + k - 1 >= left, left + k <= left + h))
    certify(step, z.Implies(k == h, z.Not(k < h)))  # forced terminal only at the final step
    # (b) Telescoping from the per-step identity proved for advance_v2.
    e_prev, e_next, g, f, d, gs, fs, ds = z.Reals('rr_e0 rr_e1 rr_g rr_f rr_d rr_gs rr_fs rr_ds')
    certify(z.And(e_prev == 1 + gs + fs - ds, e_next == e_prev + g + f - d),
            e_next == 1 + (gs + g) + (fs + f) - (ds + d))
    # (c) Funding causality on the grid: step k charges bucket j=left+k>=1, never the bucket-0 prehistory exception.
    j, t, c_prev, c_j = z.Ints('rr_j rr_t rr_cprev rr_cj')
    bucket = z.And(j >= 1, c_prev < t, t <= c_j)  # funding-events-v2 endpoint lemma for j>=1
    certify(z.And(step, j == left + k), j >= 1)
    certify(z.And(step, j == left + k, bucket), z.And(t <= c_j, t > c_prev))
    return {'F-RL-RUNNER-V2-ARITH': 'unsat'}


def successors(state, horizon, mutant=False):
    step, terminal, published, rejected = state
    if published or rejected:
        return []
    out = [('reject', (step, terminal, False, True))]
    if terminal:
        out.append(('publish', (step, True, True, False)))
        return out
    if step < horizon:
        nxt = step + 1
        out.append(('advance-final' if nxt == horizon else 'advance', (nxt, nxt == horizon, False, False)))
        if nxt < horizon:
            out.append(('advance-stop', (nxt, True, False, False)))
    if mutant and step >= 1:
        out.append(('publish-unterminated', (step, False, True, False)))
    return out


def model(mutant=False, bound=8):
    states = edges = 0
    for horizon in range(1, bound + 1):
        initial = (0, False, False, False)
        seen = {initial}; queue = deque([initial])
        while queue:
            state = queue.popleft()
            nxt = successors(state, horizon, mutant)
            require(bool(nxt) or state[2] or state[3], 'nonterminal deadlock')
            for _, s in nxt:
                edges += 1
                if s[2]:
                    require(s[1] and 1 <= s[0] <= horizon, 'publication without terminal episode')
                require(s[0] <= horizon, 'step beyond horizon')
                if s not in seen:
                    seen.add(s); queue.append(s)
        require(any(s[2] and s[0] == horizon for s in seen) and any(s[2] and s[0] < horizon for s in seen) or horizon == 1,
                'vacuous early/final termination coverage')
        states += len(seen)
    return {'status': 'model_checked', 'horizons': bound, 'states': states, 'transitions': edges,
            'orderTransitions': 0, 'scope': 'finite step abstraction for horizons 1..8; numeric values abstracted'}


def _modules():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import funding_events_v2 as fe
    import replay_runner_v2 as rr
    return fe, rr


def _episode(rng, fe):
    bars = rng.randrange(4, 14)
    closes = tuple(1000 * (i + 1) - 1 for i in range(bars))
    times = sorted(rng.sample(range(0, closes[-1] + 1), rng.randrange(0, 2 * bars)))
    rows = tuple((str(t), rng.choice(('', '-')) + '0.000' + str(rng.randrange(1, 10)), str(rng.randrange(50, 150)))
                 for t in times)
    prices = tuple(F(rng.randrange(80, 121)) for _ in closes)
    left = rng.randrange(0, bars - 1)
    h = rng.randrange(1, bars - left)
    targets = tuple(rng.choice((F(-1, 4), F(0), F(1, 4))) for _ in range(h))
    return closes, rows, prices, left, targets


def conformance():
    fe, rr = _modules()
    reg = json.loads((ROOT / REGISTRATION).read_text())
    rng = random.Random(reg['oracleSeed'])
    wire, expected, episodes, rows_checked, early = [], [], 0, 0, 0
    for _ in range(reg['oracleEpisodes']):
        closes, rows, prices, left, targets = _episode(rng, fe)
        buckets = fe.load_v2(closes, rows, enabled=True)
        require(buckets is not None, 'synthetic funding load')
        ep = rr.run_v2(prices, buckets, left, targets, impact=F(1, 10000), enabled=True)
        require(ep is not None and ep.final.terminal, 'ordinary episode absent')
        require(ep == rr.run_v2(prices, buckets, left, targets, impact=F(1, 10000), enabled=True), 'deterministic episode')
        require(rr.run_v2(prices, buckets, left, targets) is None, 'default disabled')
        gs = sum((r.gross for r in ep.receipts), F(0)); fs = sum((r.funding for r in ep.receipts), F(0))
        ds = sum((r.costs.fee + r.costs.spread + r.costs.slippage + r.costs.impact for r in ep.receipts), F(0))
        require(ep.final.equity == 1 + gs + fs - ds, 'episode telescoping')
        early += len(ep.receipts) < len(targets)
        for k, r in enumerate(ep.receipts, start=1):
            per_unit = buckets.per_unit[left + k]
            require(r.funding == -r.before.units * per_unit, 'funding bucket alignment')
            require(r.after.price == prices[left + k], 'price alignment')
            c = r.costs
            values = (r.before.equity, r.before.units, r.before.price, r.after.price, per_unit, c.fee, c.spread, c.slippage, c.impact)
            wire.append(str([(v.numerator, v.denominator) for v in values]))
            expected.append(str((r.after.equity.numerator, r.after.equity.denominator)).replace(' ', ''))
            rows_checked += 1
        episodes += 1
    cmd = ['runghc', str(ROOT / 'formal/research/ReplayAccountingV2.hs')]
    out = subprocess.run(cmd, input='\n'.join(wire) + '\n', text=True, capture_output=True, timeout=120, check=True)
    require(out.stdout.splitlines() == expected, 'Haskell exact episode-row conformance')
    return {'status': 'property_tested', 'episodes': episodes, 'haskellRows': rows_checked, 'earlyTerminations': early,
            'seed': reg['oracleSeed'], 'marketDataReads': 0, 'economicEvidence': False}


def metamorphic(run=None):
    fe, rr = _modules()
    run = rr.run_v2 if run is None else run
    reg = json.loads((ROOT / REGISTRATION).read_text())
    rng = random.Random(reg['metamorphicSeed'] + 1)
    checked = 0
    for _ in range(reg['metamorphicEpisodes']):
        closes, rows, prices, left, targets = _episode(rng, fe)
        base = run(prices, fe.load_v2(closes, rows, enabled=True), left, targets, enabled=True)
        k = rng.randrange(1, len(targets) + 1)
        cut = closes[left + k]
        # Rewrite every price and settlement strictly after the step-k decision close.
        future_prices = prices[:left + k + 1] + tuple(F(rng.randrange(1, 1000)) for _ in prices[left + k + 1:])
        kept = tuple(r for r in rows if int(r[0]) <= cut)
        later = range(cut + 1, closes[-1] + 1)
        extra = tuple((str(t), '0.5', '777') for t in sorted(rng.sample(later, min(3, len(later)))))
        changed = run(future_prices, fe.load_v2(closes, kept + extra, enabled=True), left, targets, enabled=True)
        require(base is not None and changed is not None, 'metamorphic episode absent')
        prefix = min(k, len(base.receipts))
        require(base.receipts[:prefix] == changed.receipts[:prefix], 'future data changed an earlier receipt')
        checked += 1
    return {'status': 'property_tested', 'episodes': checked, 'seed': reg['metamorphicSeed'] + 1}


def check_runner():
    reg = json.loads((ROOT / REGISTRATION).read_text())
    require((reg['maximumSteps'], reg['maximumBars'], reg['enabledByDefault']) == (4096, 8192, False), 'registration')
    nodes, source = extract()
    return {'source': source, 'smt': prove(nodes), 'model': model(), 'conformance': conformance(),
            'metamorphic': metamorphic()}
