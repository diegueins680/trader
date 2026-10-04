"""Shared STM drain ordering: source binding, finite model, SMT, compiled fixtures."""
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import z3 as z
from async_job_admission import compile_haskell

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = 'formal/research/drain-pool-source.json'
SOURCES = ['haskell/app/Trader/App/GracefulShutdown.hs',
           'haskell/app/Trader/App/AsyncJobAdmission.hs',
           'haskell/app/Trader/App/BacktestGate.hs', 'haskell/app/Main.hs']


def require(ok, message):
    if not ok:
        raise ValueError('drain pool: ' + message)


def extract():
    registry = json.loads((ROOT / REGISTRY).read_text())
    for path, digest in registry['hashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, 'source drift: ' + path)
    for path, bodies in registry['bodies'].items():
        source = (ROOT / path).read_text()
        for name, body in bodies.items():
            require(body in source and source.count(name + ' ::') == 1, 'body drift: ' + name)
    main = (ROOT / SOURCES[3]).read_text()
    require(main.count('newBacktestGateWithDrain drain maxBacktestRunning backtestTimeoutSec') == 1, 'backtest shared latch')
    require(main.count('newJobStore drain "') == 3, 'async shared latch')
    require('running <- newJobSlotsWithDrain drain maxRunning' in main, 'store shared latch')
    require('newBacktestGate ' not in main and 'newJobSlots ' not in main, 'standalone constructor bypass')
    version = subprocess.check_output(['ghc-pkg', 'field', 'stm', 'version'], text=True).strip()
    require(version == 'version: 2.5.1.0', 'STM version drift')
    return registry


# Caller: (phase, ingress snapshot, ever reserved after drain). Phases 0..4.
INITIAL = (False, (0, 0), (False, False), ((0, False, False), (0, False, False)), (False, False))


def replace(values, i, value):
    return values[:i] + (value,) + values[i+1:]


def successors(state, capacity, assignment):
    draining, counts, closed, callers, drainers = state
    steps = []
    for i, (phase, snapshot, late) in enumerate(callers):
        pool = assignment[i]
        def move(label, target, delta=0, snap=snapshot, post=late):
            steps.append((label, (draining, replace(counts, pool, counts[pool] + delta), closed,
                                 replace(callers, i, (target, snap, post)), drainers)))
        if phase == 0:
            move('snapshot', 1, snap=draining)
        elif phase == 1:
            if not snapshot and not draining and not closed[pool] and counts[pool] < capacity:
                move('reserve', 2, 1, post=draining)
            else:
                move('reject', 4)
        elif phase == 2:
            move('callback', 3)
        elif phase == 3:
            move('release', 4, -1)
    for i, done in enumerate(drainers):
        if not done:
            steps.append(('drain', (True, counts, closed, callers, replace(drainers, i, True))))
    for i, done in enumerate(closed):
        if not done:
            steps.append(('close-pool', (draining, counts, replace(closed, i, True), callers, drainers)))
    return steps


def rank(state):
    _, _, closed, callers, drainers = state
    return sum(4-p for p, _, _ in callers) + sum(not c for c in closed) + sum(not d for d in drainers)


def check_model(next_states=successors):
    receipts = []
    for capacity in (1, 2):
        for assignment in itertools.product((0, 1), repeat=2):
            queue = deque([(INITIAL, 0)]); seen = {INITIAL}
            edges = depth = terminals = stale = pending = 0
            while queue:
                state, distance = queue.popleft()
                draining, counts, closed, callers, drainers = state
                for pool in (0, 1):
                    owners = sum(phase in (2, 3) and assignment[i] == pool for i, (phase, _, _) in enumerate(callers))
                    require(counts[pool] == owners and 0 <= owners <= capacity, 'count ownership')
                require(not any(late for _, _, late in callers), 'post-drain reservation')
                stale += draining and any(p == 1 and not snap for p, snap, _ in callers)
                pending += draining and sum(counts) > 0
                depth = max(depth, distance)
                following = next_states(state, capacity, assignment)
                if not following:
                    terminals += 1
                    require(draining and all(closed) and all(drainers) and not sum(counts)
                            and all(p == 4 for p, _, _ in callers), 'deadlock / missing termination')
                for label, target in following:
                    require(rank(target) < rank(state), 'nonprogress transition')
                    require(not draining or target[0], 'drain reopened')
                    require(not draining or label != 'reserve', 'post-drain reserve edge')
                    require(not draining or all(after <= before for before, after in zip(counts, target[1])), 'post-drain count increase')
                    require(not draining or all(not (before[0] not in (2, 3) and after[0] in (2, 3))
                                                for before, after in zip(callers, target[3])), 'post-drain new owner')
                    edges += 1
                    if target not in seen:
                        seen.add(target); queue.append((target, distance + 1))
            require(stale and pending and terminals, 'vacuous interleavings')
            receipts.append(dict(capacity=capacity, assignment=list(assignment), states=len(seen),
                                 transitions=edges, maxShortestDepth=depth, terminalStates=terminals,
                                 staleIngressStates=stale, preDrainOwnerStates=pending))
    return dict(states=sum(r['states'] for r in receipts), transitions=sum(r['transitions'] for r in receipts),
                initialRank=rank(INITIAL), configurations=receipts,
                scope='AG reservation safety; finite completion under primitive/callback progress; not whole-server drain')


def prove_order():
    d, c, other_closed, stop = z.Bools('dp_d dp_c dp_other_closed dp_stop')
    n, m, limit, other_limit = z.Ints('dp_n dp_m dp_limit dp_other_limit')
    def reserve(count, closed, capacity):
        return z.If(z.And(z.Not(d), z.Not(closed), count < capacity), count + 1, count)
    after = reserve(n, c, limit)
    released = z.If(n > 0, n - 1, 0)
    def first(pair):
        x, y = pair
        return (reserve(x, c, limit), y)
    def second(pair):
        x, y = pair
        return (x, reserve(y, other_closed, other_limit))
    left = first(second((n, m))); right = second(first((n, m)))
    claims = [z.Implies(d, after == n), z.And(after >= 0, after <= limit),
              z.Implies(d, z.Or(d, stop)), z.And(released >= 0, released <= n),
              z.And(*(x == y for x, y in zip(left, right)))]
    for claim in claims:
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
        solver.add(limit > 0, other_limit > 0, n >= 0, n <= limit, m >= 0, m <= other_limit)
        require(solver.check() == z.sat, 'vacuous SMT premise')
        solver.add(z.Not(claim)); require(solver.check() == z.unsat, 'order counterexample or unknown')
    return {'F-DRAIN-POOL-ORDER': 'unsat'}


def conformance():
    entries = json.loads((ROOT / 'formal/research/drain-pool-counterexamples.json').read_text())['entries']
    with tempfile.TemporaryDirectory(prefix='trader-drain-') as name:
        temp = Path(name)
        for version in ('old', 'new', 'suite'):
            (temp / version).mkdir()
        for module, kind in [('Trader/App/AsyncJobAdmission.hs', 'async'), ('Trader/App/BacktestGate.hs', 'backtest'), ('LegacyDrain.hs', 'drain')]:
            path = temp / 'old' / module; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / f'formal/research/fixtures/drain-pool-before-{kind}.hs').read_bytes())
        for version, includes in [('old', ['-DLEGACY', '-i' + str(temp / 'old')]), ('new', ['-ihaskell/app'])]:
            exe = compile_haskell(ROOT / 'formal/research/fixtures/drain-pool-witness.hs', temp / version, includes)
            output = subprocess.check_output([exe], text=True, timeout=10).splitlines()
            expected = ['("ingress snapshot",False)'] + [e['before' if version == 'old' else 'after'] for e in entries]
            require(output == expected, 'stale ingress counterexample regression')
        exe = compile_haskell(ROOT / 'formal/research/DrainPool.hs', temp / 'suite', ['-ihaskell/app', '-ihaskell/test'])
        subprocess.run([exe], check=True, capture_output=True, text=True, timeout=15)
    return dict(counterexamples=[e['id'] for e in entries], runtimeTests=4, concurrentCases=32, seed=20261004,
                scope='real pool/latch helpers and source-bound Main constructors; not full HTTP/IO refinement')


def check_drain_pool():
    extract()
    return dict(smt=prove_order(), model=check_model(), conformance=conformance())
