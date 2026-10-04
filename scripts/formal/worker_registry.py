"""Finite worker protocol, atomic predicate SMT checks and compiled regressions.

This is source-bound conformance, not a proof of GHC IO refinement.
"""
from collections import deque
import hashlib
import json
import re
from pathlib import Path
import subprocess
import tempfile

import z3 as z

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'haskell/app/Trader/App/GracefulShutdown.hs'
REGISTRY = 'formal/research/worker-registry-source.json'


def require(ok, message):
    if not ok:
        raise ValueError('worker registry: ' + message)


def extract():
    registry = json.loads((ROOT / REGISTRY).read_text())
    for path, expected in registry['hashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected,
                'reviewed source drift: ' + path)
    source = (ROOT / SOURCE).read_text()
    for name, body in registry['bodies'].items():
        require(body in source and source.count(name + ' ::') == 1,
                'reviewed control drift: ' + name)
    exports = source.split(') where')[0]
    require(not re.search(r'\b(?:SupervisedWorker|WorkerRegistryState)\b|WorkerRegistry\s*\(', exports),
            'private cells exported')
    main = (ROOT / 'haskell/app/Main.hs').read_text()
    calls = [line.strip() for line in main.splitlines() if '<- forkSupervisedWorker ' in line]
    require(len(calls) == 5 and all(line.startswith('_ <-') for line in calls),
            'production call contract drift')
    return registry


def prove_predicates():
    closed, stop, start, registered, finished, requested, dispatch, delivered = z.Bools(
        'wr_closed wr_stop wr_start wr_registered wr_finished wr_requested wr_dispatch wr_delivered')
    after = z.Or(closed, stop)
    admitted = z.And(start, z.Not(closed))
    retained = z.And(registered, z.Not(finished))
    helper = z.And(dispatch, z.Not(requested))
    next_request = z.Or(requested, dispatch)
    # Per-entry implication composes over every member of the captured finite list.
    acknowledged = z.And(closed, z.Implies(registered, finished))
    claims = [z.Implies(closed, after), z.Implies(admitted, z.Not(closed)),
              z.Implies(z.And(registered, z.Not(finished)), retained),
              z.Implies(z.And(acknowledged, registered), finished),
              z.Implies(helper, z.Not(requested)),
              z.Implies(dispatch, z.Not(z.And(dispatch, z.Not(next_request)))),
              z.Implies(requested, next_request)]
    for claim in claims:
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
        require(solver.check() == z.sat, 'vacuous Boolean premise')
        solver.add(z.Not(claim))
        require(solver.check() == z.unsat, 'predicate counterexample or unknown')
    solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
    solver.add(closed, registered, delivered, z.Not(finished), z.Not(acknowledged))
    require(solver.check() == z.sat, 'delivery/completion distinction lost')
    return {'F-WORKER-REGISTRY-INVARIANTS': 'unsat'}


# worker: 0 unattempted, 1 unfinished, 2 finished, 3 rejected.
# caller: -1 pending, 0/1 dispatch index, 2 wait, 3 success, 4 failure.
# Requests count helpers (rather than merely masking duplicate creation).
# Captured masks are immutable per caller, including workers finishing before dispatch.
INITIAL = (False, (0, 0), (0, 0), (False, False), (-1, -1), (2, 2), (0, 0))


def replace(values, index, value):
    return values[:index] + (value,) + values[index + 1:]


def successors(state):
    closed, workers, requests, delivered, callers, ticks, captured = state
    steps = []
    for i, worker in enumerate(workers):
        if worker == 0:
            steps.append(('start', (closed, replace(workers, i, 3 if closed else 1), requests, delivered, callers, ticks, captured)))
        if worker == 1:
            steps.append(('finish', (closed, replace(workers, i, 2), requests, delivered, callers, ticks, captured)))
        if requests[i] and not delivered[i]:
            steps.append(('deliver', (closed, workers, requests, replace(delivered, i, True), callers, ticks, captured)))
    for i, pc in enumerate(callers):
        if pc == -1:
            steps.append(('close', (True, workers, requests, delivered, replace(callers, i, 0), ticks, replace(captured, i, sum(1 << j for j, w in enumerate(workers) if w == 1)))))
            # Timeout/interruption before the registry lock need not close it.
            steps.append(('lock-failure', (closed, workers, requests, delivered, replace(callers, i, 4), ticks, captured)))
        if 0 <= pc <= 2:
            steps.append(('timeout', (closed, workers, requests, delivered,
                         replace(callers, i, 4) if ticks[i] == 1 else callers,
                         replace(ticks, i, ticks[i] - 1), captured)))
            steps.append(('interrupt', (closed, workers, requests, delivered, replace(callers, i, 4), ticks, captured)))
        if pc in (0, 1):
            updated = replace(requests, pc, 1) if captured[i] & (1 << pc) else requests
            steps.append(('request', (closed, workers, updated, delivered, replace(callers, i, pc + 1), ticks, captured)))
        if pc == 2 and 1 not in workers:
            steps.append(('acknowledge', (closed, workers, requests, delivered, replace(callers, i, 3), ticks, captured)))
    return steps


def rank(state):
    _, workers, requests, delivered, callers, ticks, _ = state
    return (sum((2, 1, 0, 0)[w] for w in workers) + 4 - sum(requests) - sum(delivered)
            + sum({-1: 6, 0: 5, 1: 4, 2: 3, 3: 0, 4: 0}[pc] for pc in callers) + sum(ticks))


def check_model(next_states=successors):
    queue = deque([(INITIAL, 0)]); seen = {INITIAL}
    edges = terminals = successes = failures = delivered_unfinished = depth = after_finish = 0
    while queue:
        state, distance = queue.popleft()
        closed, workers, requests, delivered, callers, ticks, captured = state
        depth = max(depth, distance)
        require(all(r in (0, 1) for r in requests), 'duplicate cancellation helper')
        require(all(not r or w in (1, 2) for r, w in zip(requests, workers)), 'unregistered cancellation')
        require(all(not d or r for d, r in zip(delivered, requests)), 'unrequested delivery')
        require(all(not r or any(cap & (1 << i) for cap in captured) for i, r in enumerate(requests)), 'uncaptured cancellation')
        require(3 not in callers or (closed and 1 not in workers), 'unfinished success')
        require(all(0 <= t <= 2 for t in ticks), 'invalid timeout budget')
        require(all(not (cap & (1 << i)) or workers[i] in (1, 2) for cap in captured for i in range(2)), 'capture of unregistered worker')
        following = next_states(state)
        if not following:
            terminals += 1
            require(all(pc in (3, 4) for pc in callers) and all(w in (2, 3) for w in workers), 'unexpected deadlock')
        successes += 3 in callers
        failures += 4 in callers
        delivered_unfinished += any(d and w == 1 for d, w in zip(delivered, workers))
        for label, target in following:
            c, ws, rs, ds, pcs, ts, caps = target
            require(not closed or c, 'registry reopened')
            require(all(pc == -1 or old == new for pc, old, new in zip(callers, captured, caps)), 'captured snapshot changed')
            require(all(not (old == 0 and new == 1) or not closed for old, new in zip(workers, ws)), 'closed admission')
            require(all(old != 1 or new in (1, 2) for old, new in zip(workers, ws)), 'unfinished entry forgotten')
            require(all(old != 2 or new == 2 for old, new in zip(workers, ws)), 'completion revoked')
            require(all(a <= b for a, b in zip(requests, rs)), 'request forgotten')
            require(all(not a or b for a, b in zip(delivered, ds)), 'delivery forgotten')
            require(label == 'finish' or ws == workers or label == 'start', 'delivery mistaken for finalization')
            require(rank(target) < rank(state), 'protocol rank failed')
            after_finish += label == 'request' and any(w == 2 and a == 0 and b == 1 for w, a, b in zip(workers, requests, rs))
            edges += 1
            if target not in seen:
                seen.add(target); queue.append((target, distance + 1))
    require(successes and failures and delivered_unfinished and after_finish, 'vacuous success/failure/delivery coverage')
    return {'states': len(seen), 'transitions': edges, 'maxShortestDepth': depth,
            'initialRank': rank(INITIAL), 'terminalStates': terminals,
            'successStates': successes, 'failureStates': failures,
            'deliveryWithoutCompletionStates': delivered_unfinished,
            'requestAfterCompletionTransitions': after_finish,
            'workers': 2, 'stopCallers': 2, 'timeoutTicksPerCaller': 2,
            'scope': 'atomic finite registry protocol; conditional primitive progress, not IO refinement or real-time bound'}


def run_haskell(main, directory, include):
    output = directory / 'build'; output.mkdir()
    exe = str(directory / 'test')
    subprocess.run(['ghc', '-v0', '-O0', '-threaded', *include, '-outputdir', str(output),
                    str(main), '-o', exe], cwd=ROOT, check=True, capture_output=True, text=True, timeout=90)
    return subprocess.check_output([exe], cwd=ROOT, text=True, stderr=subprocess.PIPE, timeout=20).splitlines()


def conformance():
    counterexamples = json.loads((ROOT / 'formal/research/worker-registry-counterexamples.json').read_text())['entries']
    witness = ROOT / 'formal/research/fixtures/worker-registry-witness.hs'
    with tempfile.TemporaryDirectory(prefix='trader-worker-proof-') as temporary:
        temp = Path(temporary)
        for name in ('old', 'new', 'suite'):
            (temp / name).mkdir()
        legacy = temp / 'old/Trader/App/GracefulShutdown.hs'
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes((ROOT / 'formal/research/fixtures/worker-registry-before.hs').read_bytes())
        before = run_haskell(witness, temp / 'old', ['-i' + str(temp / 'old')])
        after = run_haskell(witness, temp / 'new', ['-ihaskell/app'])
        require(before == [entry['before'] for entry in counterexamples], 'legacy counterexample drift')
        require(after == [entry['after'] for entry in counterexamples], 'regression reproduced in repaired source')
        run_haskell(ROOT / 'formal/research/WorkerRegistry.hs', temp / 'suite', ['-ihaskell/app', '-ihaskell/test'])
    return {'counterexamples': [entry['id'] for entry in counterexamples], 'runtimeTests': 5,
            'concurrentCases': 32, 'seed': 20261004,
            'scope': 'compiled barriers and generated schedules; not exhaustive scheduler or implementation proof'}


def check_capture_trace():
    entry = json.loads((ROOT / 'formal/research/worker-registry-counterexamples.json').read_text())['modelRefinements'][0]
    state = INITIAL
    for event, expected in zip(entry['events'], entry['states']):
        expected = (expected[0], *(tuple(part) for part in expected[1:]))
        require((event, expected) in successors(state), 'capture/finalization interleaving lost')
        state = expected
    require(state[1][0] == 2 and state[2][0] == 1, 'post-completion request not represented')
    return entry['id']


def check_worker_registry():
    extract()
    return {'smt': prove_predicates(), 'model': check_model(), 'conformance': conformance(),
            'modelRefinementRegression': check_capture_trace()}
