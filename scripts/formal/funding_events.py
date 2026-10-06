"""Source-bound exact funding bucketing: SMT size/endpoint lemmas, publication model and oracle."""
import ast
from collections import deque
from fractions import Fraction as F
from itertools import combinations
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
SOURCE = 'scripts/research/funding_events_v2.py'
REGISTRY = 'formal/research/funding-events-v2-source.json'
REGISTRATION = 'research-notes/registrations/funding-events-v2-engineering.json'
CONSTANTS = {
    'VERSION': "'funding-events-v2'", 'MAX_BITS': '8192', 'MAX_BARS': '8192', 'MAX_RECORDS': '65536',
    'MAX_EVENTS': '128', 'MAX_TIME': '2 ** 63',
    'DECIMAL': "re.compile('-?(?:0|[1-9][0-9]{0,19})(?:\\\\.[0-9]{1,20})?', re.ASCII)",
    'INTEGER': "re.compile('0|[1-9][0-9]{0,18}', re.ASCII)",
    '__all__': "['VERSION', 'Buckets', 'load_v2']"}


def require(ok, reason):
    if not ok:
        raise ValueError('funding events v2: ' + reason)


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
    require(constants == CONSTANTS, 'version/bounds/domain/exports')
    require([ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))] ==
            ['from __future__ import annotations', 'from dataclasses import dataclass',
             'from fractions import Fraction as F', 'import re'], 'effects import boundary')
    require([ast.unparse(n) for n in nodes['Buckets'].decorator_list] == ['dataclass(frozen=True, slots=True)'], 'immutable output')
    load = nodes['load_v2']
    defaults = dict(zip((x.arg for x in load.args.kwonlyargs), map(ast.unparse, load.args.kw_defaults)))
    require(defaults == {'enabled': 'False', 'version': 'VERSION'}, 'disabled default')
    require(ast.unparse(load.body[0]) == "if enabled is not True or type(version) is not str or version != VERSION:\n    return None", 'activation first')
    require(ast.unparse(load.body[1].handlers[0].type) == '(ArithmeticError, ValueError, TypeError, MemoryError)' and
            ast.unparse(load.body[1].handlers[0].body[0]) == 'return None', 'all-or-nothing rejection')
    for name, op in (('_add', 'a + b'), ('_mul', 'a * b')):
        require(ast.unparse(nodes[name].body[0]) == 'return _bounded(' + op + ')', 'checked primitive')
    require(ast.unparse(nodes['_bounded'].body[0].test) == 'not _valid(x)', 'numeric guard')
    require(ast.unparse(nodes['_valid'].body[0].value) ==
            'type(x) is F and x.numerator.bit_length() <= MAX_BITS and (x.denominator.bit_length() <= MAX_BITS)', 'exact admission bound')
    decimal = nodes['_decimal'].body
    require(ast.unparse(decimal[0].test) == 'type(text) is not str or DECIMAL.fullmatch(text) is None', 'decimal admission precedes arithmetic')
    require(ast.unparse(decimal[-1].value) == '_bounded(F(sign * int(whole + fraction), 10 ** len(fraction)))', 'exact decimal decode')
    require(ast.unparse(nodes['_per_unit'].body[1].body[1].body[0]) == 'total = _add(total, _mul(mark, rate))', 'checked accumulation')
    # No float, eval, file, network or order surface anywhere in the module.
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    require(not names & {'float', 'eval', 'exec', 'open', 'socket', 'urlopen', 'order', 'submit', 'promote'}, 'effect-free source')
    return nodes, {'status': 'exhaustively_checked', 'definitions': len(nodes), 'checkedPrimitives': 2,
                   'defaultDisabledEntries': 1, 'scope': 'complete reviewed AST, regex domain and arithmetic sites; runtime semantics assumed'}


def sweep_lemma(nodes):
    # The sweep is the only bucketing loop; bind its exact guard text before reasoning about it.
    loop = nodes['_bucket'].body[2].body[0]
    require(isinstance(loop, ast.While) and ast.unparse(loop.test) == 'j < len(closes) and closes[j] < time' and
            ast.unparse(loop.body[0]) == 'j += 1', 'sweep guard')
    # Quantifier-free: strict sortedness enters only through the explicit instance
    # c_i <= c_{j-1} for 0 <= i <= j-1; c_j is the close at the exit index.
    t, prev, ci, cj, cjm1, nxt, i, j, n = z.Ints('fe_t fe_prev fe_ci fe_cj fe_cjm1 fe_next fe_i fe_j fe_n')
    exit_ = z.And(n >= 1, 0 <= j, j <= n, z.Implies(j > 0, cjm1 < t), z.Implies(j < n, cj >= t))
    # Every earlier close is before t, so j is the first close no earlier than t (and j == n rejects).
    certify(z.And(exit_, j > 0, 0 <= i, i <= j - 1, ci <= cjm1), ci < t)
    certify(z.And(exit_, j < n, j > 0), z.And(cjm1 < t, t <= cj))
    # Loop body: stepping past a close earlier than t preserves the invariant.
    certify(z.And(cj < t, nxt == cj), nxt < t)
    # Invariant carried from the previous strictly earlier event is sound for the next one.
    certify(z.And(prev < t, cjm1 < prev), cjm1 < t)


def size_lemmas():
    tw, tf, big = z.IntVal(10**20), z.IntVal(10**40), z.IntVal(2**8192)
    x, y, k1, k2, s, acc, term, step = z.Ints('fe_x fe_y fe_k1 fe_k2 fe_s fe_acc fe_term fe_step')
    # Decoded |numerator| < 10^40 (20+20 digits); denominator 10^k, 0<=k<=20.
    certify(z.And(x >= 0, x < tf), z.And(x < big, tw <= big))
    # Product of two decoded values; s=10^(40-k1-k2) rescales to the common denominator 10^40.
    certify(z.And(x >= 0, x < tf, y >= 0, y < tf), x * y < tf * tf)
    certify(z.And(x >= 0, x < tf * tf, s >= 1, s <= tf), x * s < tf * tf * tf)
    # Inductive accumulation over at most 128 terms, each below T=10^120 over 10^40.
    bound = tf * tf * tf
    certify(z.And(step >= 0, step < 128, acc >= -step * bound, acc <= step * bound, term > -bound, term < bound),
            z.And(acc + term >= -(step + 1) * bound, acc + term <= (step + 1) * bound))
    certify(z.And(step >= 0, step <= 128, acc >= -step * bound, acc <= step * bound), z.And(acc < big, -acc < big))
    # Reduction n/d of an exact N/10^40 with 1<=d<=10^40 never enlarges |numerator|.
    num, den, red = z.Ints('fe_num fe_den fe_red')
    certify(z.And(den >= 1, den <= tf, num >= 0, red >= 0, red * tf == num * den), red <= num)
    # Fraction primitive intermediates for reduced operands of this size stay below 2^8192.
    m = z.IntVal(2**407)
    certify(z.And(x >= 0, x < m, y >= 0, y < m, k1 >= 1, k1 <= tf, k2 >= 1, k2 <= tf),
            z.And(x * k2 + y * k1 < big, k1 * k2 < big, x * y < big))
    require(128 * 10**120 < 2**407 and 10**40 < 2**407, 'concrete size constant')


def prove(nodes):
    sweep_lemma(nodes)
    size_lemmas()
    return {'F-RL-FUNDING-V2-ARITH': 'unsat'}


def exhaustive():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import funding_events_v2 as impl
    reg = json.loads((ROOT / REGISTRATION).read_text())
    grid, times, limit = tuple(reg['exhaustiveGrid']), reg['exhaustiveTimes'], reg['exhaustiveMaxEvents']
    cases = 0
    for size in range(limit + 1):
        for chosen in combinations(times, size):
            rows = tuple((str(t), '1', '1') for t in chosen)
            got = impl.load_v2(grid, rows, enabled=True)
            index = [sum(c < t for c in grid) for t in chosen]
            if any(k == len(grid) for k in index):
                require(got is None, 'beyond-final-close publication')
            else:
                require(got is not None and tuple(len(b) for b in got.events) ==
                        tuple(index.count(k) for k in range(len(grid))), 'sweep differs from prefix count')
            cases += 1
    return {'status': 'exhaustively_checked', 'cases': cases, 'grid': list(grid), 'times': times, 'maxEvents': limit}


# Model=(phase, published). Stages may fail at any point and discard the candidate.
STAGES = ('start', 'active', 'grid', 'records', 'buckets', 'arith')


def successors(state, mutant=False):
    phase, published = state
    if phase in ('absent', 'published'):
        return []
    out = [('reject', ('absent', False))]
    following = STAGES.index(phase) + 1
    out.append(('advance', (STAGES[following], False) if following < len(STAGES) else ('published', True)))
    if mutant and phase == 'buckets':
        out.append(('publish-after-failure', ('published', True)))
    return out


def model(mutant=False):
    initial = ('start', False)
    depth = {initial: 0}
    paths = {initial: ()}
    queue = deque([initial])
    edges = 0
    while queue:
        state = queue.popleft()
        following = successors(state, mutant)
        require(bool(following) or state[0] in ('absent', 'published'), 'nonterminal deadlock')
        for label, nxt in following:
            edges += 1
            trace = paths[state] + (label,)
            if nxt[1]:
                require(set(trace) == {'advance'} and len(trace) == len(STAGES), 'publication without every stage')
            if nxt not in depth:
                depth[nxt] = depth[state] + 1
                paths[nxt] = trace
                queue.append(nxt)
    require(('absent', False) in depth and ('published', True) in depth, 'vacuous model')
    return {'status': 'model_checked', 'states': len(depth), 'transitions': edges, 'maxShortestDepth': max(depth.values()),
            'orderTransitions': 0, 'scope': 'finite stage abstraction; no whole-language refinement'}


def wire(grid, rows):
    return '(' + json.dumps(list(grid)) + ',[' + ','.join('(' + ','.join(json.dumps(v) for v in r) + ')' for r in rows) + '])'


def encode(result):
    if result is None:
        return 'invalid'
    return str([(len(b), v.numerator, v.denominator) for b, v in zip(result.events, result.per_unit)]).replace(' ', '')


def edge_cases():
    top = '99999999999999999999.99999999999999999999'
    huge = '%.0f' % 1.7976931348623157e308
    return [
        ((9, 19, 29), (('10', '2', huge),)),                       # CE-RL-019 product witness as decimal text
        ((9, 19, 29), (('10', '0.75', huge), ('11', '0.75', huge))),  # CE-RL-019 sum witness
        ((9, 19, 29), (('10', '2', top),)),
        ((9, 19, 29), tuple((str(10 + k), '-' + top, top) for k in range(9))),
        ((9, 19, 29), (('0', '1', '1'), ('9', '1', '1'), ('19', '1', '1'), ('29', '1', '1'))),
        ((9, 19, 29), (('30', '1', '1'),)),
        ((9, 19, 29), (('10', '1', '0'),)), ((9, 19, 29), (('10', '1', '-1'),)),
        ((9, 19, 29), (('10', '1e3', '1'),)), ((9, 19, 29), (('10', '01', '1'),)), ((9, 19, 29), (('10', '1.', '1'),)),
        ((9, 19, 29), (('10', '0.000000000000000000001', '1'),)),
        ((9, 19, 29), (('10', '1', '1'), ('10', '1', '1'))),
        ((9, 19, 29), (('9223372036854775808', '1', '1'),)),
        ((9, 9, 29), (('1', '1', '1'),)), ((), ()),
        ((1000,), tuple((str(k), '1', '1') for k in range(128))),
        ((1000,), tuple((str(k), '1', '1') for k in range(129))),
    ]


def conformance():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import funding_events_v2 as impl
    import replay_accounting_v2 as replay
    rng = random.Random(json.loads((ROOT / REGISTRATION).read_text())['oracleSeed'])
    cases = edge_cases()
    for _ in range(128):
        grid = tuple(sorted(rng.sample(range(0, 10**6, 1000), rng.randrange(1, 12))))
        times = sorted(rng.sample(range(0, grid[-1] + 2000), rng.randrange(0, 24)))
        rows = tuple((str(t), rng.choice(('', '-')) + str(rng.randrange(0, 10**4)) + '.' + str(rng.randrange(10**8)).zfill(8),
                      str(rng.randrange(1, 10**6)) + '.' + str(rng.randrange(100)).zfill(2)) for t in times)
        cases.append((grid, rows))
    lines, expected, published, composed = [], [], 0, 0
    for grid, rows in cases:
        got = impl.load_v2(grid, rows, enabled=True)
        require(got == impl.load_v2(grid, rows, enabled=True), 'deterministic load')
        require(impl.load_v2(grid, rows) is None, 'default disabled')
        lines.append(wire(grid, rows))
        expected.append(encode(got))
        if got is None:
            continue
        published += 1
        require(all(len(b) <= replay.MAX_EVENTS for b in got.events), 'replay admission size')
        state = replay.initial_v2(F(100), enabled=True)
        entry = replay.advance_v2(state, F(100), (), F(1, 4), multiplier=F(0), enabled=True)
        for bucket, per_unit in zip(got.events[:1], got.per_unit[:1]):
            step = replay.advance_v2(entry.after, F(100), bucket, F(0), terminal=True, multiplier=F(0), enabled=True)
            if step is not None:
                require(step.funding == -entry.after.units * per_unit, 'replay funding composition')
                composed += 1
    cmd = ['runghc', str(ROOT / 'formal/research/FundingEventsV2.hs')]
    out = subprocess.run(cmd, input='\n'.join(lines) + '\n', text=True, capture_output=True, timeout=120, check=True)
    require(out.stdout.splitlines() == expected, 'Haskell exact funding conformance')
    require(expected[0] == expected[1] == 'invalid', 'CE-RL-019 witness text must reject')
    return {'status': 'property_tested', 'cases': len(cases), 'published': published, 'rejected': len(cases) - published,
            'replayCompositions': composed, 'seed': json.loads((ROOT / REGISTRATION).read_text())['oracleSeed'],
            'haskellRows': len(cases), 'marketDataReads': 0, 'economicEvidence': False}


def check_funding_events():
    registered = json.loads((ROOT / REGISTRATION).read_text())
    require((registered['maximumBars'], registered['maximumRecords'], registered['maximumEventsPerBucket'],
             registered['maximumRationalBits'], registered['enabledByDefault']) == (8192, 65536, 128, 8192, False), 'registration')
    nodes, source = extract()
    return {'source': source, 'smt': prove(nodes), 'exhaustive': exhaustive(), 'model': model(), 'conformance': conformance()}
