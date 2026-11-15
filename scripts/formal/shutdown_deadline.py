"""Source-bound shutdown budget proof, finite stage model and Haskell conformance."""
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import tempfile

import z3 as z

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'haskell/app/Trader/App/GracefulShutdown.hs'
MAIN = 'haskell/app/Main.hs'
REGISTRY = 'formal/research/shutdown-source.json'


def require(ok, message):
    if not ok:
        raise ValueError('shutdown: ' + message)


def extract():
    registry = json.loads((ROOT / REGISTRY).read_text())
    for path, expected in registry['hashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected,
                'reviewed source drift: ' + path)
    for path, bodies in registry['bodies'].items():
        source = (ROOT / path).read_text()
        for name, body in bodies.items():
            require(body in source and source.count(name + ' ::') == 1,
                    'reviewed control drift: ' + name)
    core = (ROOT / SOURCE).read_text()
    require('newtype ShutdownBudget' not in core, 'representation drift')
    require('    ShutdownBudget,' in core and 'ShutdownBudget (..)' not in core.split(') where')[0],
            'deadline constructors became public')
    main = registry['bodies'][MAIN]['runServeShutdown']
    require(main.count('getTimestampMs') == 1 and main.count('getMonotonicTimeNSec') == 2,
            'wall time entered timing path')
    require(main.count('runBefore WorkCleanup') == 5 and main.count('runBefore FinalCleanup') == 1,
            'stage roster drift')
    require('shutdownTimeoutSec *' not in main and 'if and outcomes then' in main,
            'overflow or summary regression')
    return registry


def remaining(start, seconds, final, now, bound):
    total = max(0, seconds) * 1000000000
    reserve = min(2000000000, total // 4)
    deadline = start + total - (0 if final else reserve)
    if start < 0 or now < start:
        return 0
    return min(bound, max(0, (deadline - now) // 1000))


def prove_budget():
    start, seconds, now, later, bound = z.Ints('sd_start sd_seconds sd_now sd_later sd_bound')
    final = z.Bool('sd_final')
    total = z.If(seconds > 0, seconds, 0) * 1000000000
    reserve = z.If(total / 4 < 2000000000, total / 4, 2000000000)
    deadline = start + total - z.If(final, 0, reserve)

    def value(sample):
        raw = (deadline - sample) / 1000
        return z.If(z.Or(start < 0, sample < start), 0,
                    z.If(raw < 0, 0, z.If(raw > bound, bound, raw)))

    u = value(now)
    claims = [z.And(u >= 0, u <= bound),
              z.Implies(u > 0, z.And(now >= start, start >= 0, u * 1000 <= deadline - now)),
              z.Implies(z.Or(now >= deadline, now < start, start < 0), u == 0),
              z.Implies(z.And(start >= 0, now >= start, later >= now), value(later) <= u)]
    for claim in claims:
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
        solver.add(bound > 0)
        require(solver.check() == z.sat, 'vacuous integer premise')
        solver.add(z.Not(claim))
        require(solver.check() == z.unsat, 'budget counterexample or unknown')
    return {'F-SHUTDOWN-BUDGET': 'unsat'}


# Abstract nanoseconds scale to five instants. This is a finite model, not a
# universal dense-time abstraction theorem. Each wait returns under A-SHUTDOWN-CLOCK.
TIMES = (0, 16, 18, 19, 20)


def successors(state):
    index, now, outcomes = state
    if index == 6:
        return []
    deadline = 18 if index < 5 else 20
    if now >= deadline:
        return [(index + 1, now, outcomes + (False,))]
    return [(index + 1, end, outcomes + (completed and end < deadline,))
            for end in TIMES if end >= now for completed in (False, True)]


def check_model(next_states=successors):
    initial = (0, 0, ())
    queue = deque([initial]); seen = {initial}; edges = terminal = acknowledged = 0
    while queue:
        state = queue.popleft()
        index, now, outcomes = state
        require(len(outcomes) == index and 0 <= index <= 6, 'stage index/ledger drift')
        following = next_states(state)
        require(bool(following) == (index < 6), 'unexpected deadlock')
        if index == 6:
            terminal += 1
            acknowledged += all(outcomes)
            require(all(outcomes) == (False not in outcomes), 'false completion report')
        deadline = 18 if index < 5 else 20
        for target in following:
            j, end, results = target
            require(j == index + 1 and results[:-1] == outcomes, 'stage skipped or history lost')
            require(end >= now, 'clock regressed in monotonic model')
            require(not results[-1] or (now < deadline and end < deadline), 'late acknowledgement')
            require(now < deadline or (end == now and not results[-1]), 'expired stage dispatched')
            edges += 1
            if target not in seen:
                seen.add(target); queue.append(target)
    require(0 < acknowledged < terminal, 'vacuous failure/success coverage')
    return {'states': len(seen), 'transitions': edges, 'maxShortestDepth': 6,
            'initialRank': 6, 'terminalStates': terminal, 'allAcknowledgedStates': acknowledged,
            'stages': 6, 'timeBuckets': list(TIMES),
            'scope': 'finite sequential control; each bounded wait and logging primitive must return'}


def counterexamples():
    entries = json.loads((ROOT / 'formal/research/shutdown-counterexamples.json').read_text())['entries']
    rollback, overflow, report = entries
    old = (rollback['startMs'] + rollback['seconds'] * 1000 - rollback['sampleMs']) * 1000
    require(old == rollback['oldRemainingUs'] > rollback['budgetUs'], 'missing UTC counterexample')
    require(remaining(1000000000, 20, True, 999999999, 2**63 - 1) == 0, 'regression admitted')
    bits = overflow['bits']; unsigned = overflow['seconds'] * 1000000 % 2**bits
    signed = unsigned if unsigned < 2**(bits - 1) else unsigned - 2**bits
    require(signed == overflow['oldWrappedUs'] < 0, 'missing overflow witness')
    require(remaining(0, overflow['seconds'], True, 0, 2**63 - 1) == 2**63 - 1, 'unsafe conversion')
    require(not all(report['outcomes']) and len(report['outcomes']) == 6, 'missing misleading-log witness')
    return [entry['id'] for entry in entries]


def conformance():
    with tempfile.TemporaryDirectory(prefix='trader-shutdown-proof-') as temp:
        exe = str(Path(temp) / 'shutdown')
        subprocess.run(['ghc', '-v0', '-O0', '-threaded', '-ihaskell/app', '-ihaskell/test',
                        '-outputdir', temp, 'formal/research/ShutdownDeadline.hs', '-o', exe],
                       cwd=ROOT, check=True, capture_output=True, text=True, timeout=90)
        bound = int(subprocess.check_output([exe, '--int-bound'], text=True, timeout=10))
        require(bound in (2**31 - 1, 2**63 - 1), 'unsupported Int width')
        cases = list(itertools.product((-1, 0, 1000000000),
                     (-bound - 1, -1, 0, 1, 20, bound), (False, True),
                     (-1, 0, 999999999, 1000000000, 18999999999, 19000000000, 20999999999, 21000000000, 2**80)))
        rng = random.Random(20261004)
        for _ in range(256):
            start = rng.randrange(0, 2**64)
            seconds = rng.randrange(-bound - 1, bound + 1)
            now = start + rng.randrange(-1000000000, max(1, seconds * 1000000000 + 1))
            cases.append((start, seconds, bool(rng.getrandbits(1)), now))
        payload = ''.join(str(row) + '\n' for row in cases)
        actual = subprocess.check_output([exe, '--cases'], input=payload, text=True, timeout=15)
        require([int(line) for line in actual.splitlines()] == [remaining(*row, bound) for row in cases],
                'compiled budget/model mismatch')
        subprocess.run([exe], check=True, capture_output=True, text=True, timeout=10)
        return {'cases': len(cases), 'generatedCases': 256, 'seed': 20261004,
                'intBits': bound.bit_length() + 1, 'runtimeTests': 8,
                'scope': 'compiled differential, regression and fault tests; not IO refinement'}


def check_shutdown():
    extract()
    return {'smt': prove_budget(), 'model': check_model(),
            'counterexamples': counterexamples(), 'conformance': conformance()}
