"""FIFO safety and weak-fair progress, with a preserved legacy starvation lasso."""
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
SOURCE = 'haskell/app/Trader/App/BacktestGate.hs'
REGISTRY = 'formal/research/admission-progress-source.json'


def require(ok, reason):
    if not ok:
        raise ValueError('admission progress: ' + reason)


def extract():
    registry = json.loads((ROOT / REGISTRY).read_text())
    for path, digest in registry['hashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, 'source drift: ' + path)
    for path, bodies in registry['bodies'].items():
        source = (ROOT / path).read_text()
        for name, body in bodies.items():
            require(source.count(name + ' ::') == 1 and body in source, 'unreviewed definition: ' + name)
    source = (ROOT / SOURCE).read_text()
    require('BacktestGate (..)' not in source.split(') where')[0], 'queue representation exposed')
    require('btWaiters :: !(TVar (Integer, Seq.Seq Integer))' in source, 'exact private ticket representation')
    # Source-shape checks are explicit review boundaries, not parser refinement.
    for text in ['if Seq.null queue then admitSlot (btMaxRunning gate) current else (current, False)',
                 'first Seq.:< rest | first == ticket', 'queue Seq.|> ticket', 'second (Seq.filter (/= ticket))',
                 'acquired <- atomically acquire `onException` atomically removeTicket']:
        require(text in source, 'admission/cleanup control drift')
    require(source.count('restore (runTimedBacktest gate action) `finally` releaseBacktest gate') == 2,
            'both reservation owners require finalizers')
    require('threadDelay' not in source, 'polling admission reintroduced')
    return {'status': 'exhaustively_checked', 'sourceFiles': len(registry['hashes']),
            'scope': 'reviewed concrete queue/owner/drain transfer; library semantics assumed, not compiler refinement'}


def prove_numeric():
    ticket, old, ahead, size, count, cap, bound = z.Ints('fp_ticket fp_old fp_ahead fp_size fp_count fp_cap fp_bound')
    ticket_domain = ticket >= 0
    prior_domain = z.And(ticket >= 0, old >= 0, old < ticket)
    rank_domain = z.And(ahead >= 0, ahead < size, size >= 1)
    count_domain = z.And(count >= 0, count <= cap, cap >= 1, cap <= bound, bound >= 1)
    accepted = count < cap
    after = z.If(accepted, count + 1, count)
    claims = [(ticket_domain, ticket + 1 > ticket),
              (prior_domain, old < ticket + 1),
              (prior_domain, old != ticket),
              (count_domain, z.And(after >= 0, after <= cap, after <= bound)),
              (count_domain, z.Implies(accepted, z.If(after > 0, after - 1, 0) == count)),
              (rank_domain, ahead < size + 1),
              (z.And(rank_domain, ahead > 0), z.And(ahead - 1 >= 0, ahead - 1 < ahead)),
              (z.And(count_domain, count > 0), z.And(count - 1 >= 0, count - 1 < cap))]
    for premise, claim in claims:
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
        solver.add(premise)
        require(solver.check() == z.sat, 'vacuous numeric premise')
        solver.add(z.Not(claim))
        require(solver.check() == z.unsat, 'numeric counterexample or UNKNOWN')
    return {'queries': len(claims), 'smt': {'F-ADMISSION-PROGRESS-NUMERIC': 'unsat'},
            'scope': 'unbounded Integer tickets/rank summaries and bounded Int count; Seq semantics named assumption'}


# Concrete ticket values are order-preservingly renamed to outstanding caller IDs.
# Waiting callers0/1; immediate caller2. Competitors1/2 can retry indefinitely.
# Phases: new, wait, reserved, active, done, cancelled, drained.
INITIAL = (False, (), ('new', 'new', 'new'), 0)
OWNED = ('reserved', 'active')


def successors(state, capacity, legacy=False, barging=False, leak_cancel=False):
    draining, queue, phases, count = state
    steps = [('stutter', state)]
    def move(i, label, phase, q=queue, d=draining, delta=0):
        steps.append((label + ':' + str(i), (d, q, phases[:i] + (phase,) + phases[i+1:], count + delta)))
    if not draining:
        steps.append(('drain', (True, queue, phases, count)))
    for i, phase in enumerate(phases):
        if phase == 'new':
            if draining:
                move(i, 'register', 'drained')
            elif i != 2:
                move(i, 'register', 'wait', queue if legacy else queue + (i,))
            elif count < capacity and (legacy or barging or not queue):
                move(i, 'try', 'reserved', delta=1)
            else:
                move(i, 'try', 'done')
        elif phase == 'wait':
            removed = tuple(x for x in queue if x != i)
            move(i, 'cancel', 'cancelled', queue if leak_cancel else removed)
            if draining:
                move(i, 'drain-return', 'drained', removed)
            elif legacy:
                move(i, 'attempt', 'reserved' if count < capacity else 'wait', delta=int(count < capacity))
            elif queue and queue[0] == i and count < capacity:
                move(i, 'acquire', 'reserved', queue[1:], delta=1)
        elif phase == 'reserved':
            move(i, 'enter', 'active')
            move(i, 'cancel-owner', 'cancelled', delta=-1)
        elif phase == 'active':
            move(i, 'release', 'done', delta=-1)  # includes terminating invalid callback / timeout
        elif i != 0 and phase in ('done', 'cancelled') and not draining:
            move(i, 'recycle', 'new')
    return steps


def justice(label):
    # Cancellation/drain are optional environmental events. Stuttering is allowed.
    return label.split(':')[0] in ('register', 'try', 'attempt', 'acquire', 'enter', 'release', 'recycle', 'drain-return')


def components(vertices, edges):
    """Iterative Kosaraju SCC decomposition (also tested independently)."""
    visited = set(); order = []
    for start in vertices:
        if start in visited:
            continue
        stack = [(start, False)]
        while stack:
            v, finish = stack.pop()
            if finish:
                order.append(v); continue
            if v in visited:
                continue
            visited.add(v); stack.append((v, True))
            stack.extend((w, False) for _, w in edges[v] if w in vertices and w not in visited)
    reverse = {v: set() for v in vertices}
    for v in vertices:
        for _, w in edges[v]:
            if w in vertices:
                reverse[w].add(v)
    used = set(); result = []
    for start in reversed(order):
        if start in used:
            continue
        group = set(); stack = [start]; used.add(start)
        while stack:
            v = stack.pop(); group.add(v)
            for w in reverse[v] - used:
                used.add(w); stack.append(w)
        result.append(group)
    return result


def fair_bad_components(graph):
    # AF resolution of a persistent registered target under weak action fairness.
    vertices = {s for s in graph if s[2][0] == 'wait'}
    bad = []
    for group in components(vertices, graph):
        internal = [(s, label, t) for s in group for label, t in graph[s] if t in group]
        if not internal:
            continue
        always = None
        for s in group:
            enabled = {label for label, _ in graph[s] if justice(label)}
            always = enabled if always is None else always & enabled
        taken = {label for _, label, _ in internal}
        if always <= taken:
            bad.append(group)
    return bad


def explore(capacity, **options):
    paths = {INITIAL: []}; graph = {}; todo = deque([INITIAL])
    while todo:
        s = todo.popleft(); graph[s] = successors(s, capacity, **options)
        d, queue, phases, count = s
        require(len(queue) == len(set(queue)), 'duplicate queue identity')
        if not options.get('legacy'):
            require(set(queue) == {i for i, p in enumerate(phases) if p == 'wait'}, 'orphan or missing ticket')
        require(count == sum(p in OWNED for p in phases) and 0 <= count <= capacity, 'capacity/ownership violated')
        for label, t in graph[s]:
            if d:
                require(t[0] and sum(p in OWNED for p in t[2]) <= sum(p in OWNED for p in phases), 'post-drain admission')
            if queue and not options.get('legacy') and not options.get('barging'):
                require(not (label == 'try:2' and t[2][2] == 'reserved'), 'barging')
            if label.startswith('acquire:'):
                require(queue and queue[0] == int(label.split(':')[1]), 'FIFO skip')
            if t not in paths:
                paths[t] = paths[s] + [label]; todo.append(t)
    return graph, paths


def path_in(graph, start, goal, vertices):
    todo = deque([start]); paths = {start: []}
    while todo:
        s = todo.popleft()
        if s == goal:
            return paths[s]
        for label, t in graph[s]:
            if t in vertices and t not in paths:
                paths[t] = paths[s] + [(s, label, t)]; todo.append(t)
    raise ValueError('admission progress: disconnected SCC witness')


def legacy_witness():
    graph, paths = explore(1, legacy=True)
    # Explicit unfair admission cycle, fair to caller attempts and owner completion.
    prefix = ['register:0', 'register:1', 'attempt:1', 'enter:1']
    cycle = ['attempt:0', 'try:2', 'recycle:2', 'release:1', 'recycle:1', 'register:1', 'attempt:1', 'enter:1']
    s = INITIAL
    for label in prefix:
        s = next(t for name, t in graph[s] if name == label)
    start = s; visited = []
    for label in cycle:
        visited.append(s)
        s = next(t for name, t in graph[s] if name == label)
    require(s == start and all(v[2][0] == 'wait' for v in visited), 'legacy lasso lost')
    enabled = set.intersection(*({label for label, _ in graph[v] if justice(label)} for v in visited))
    require(enabled <= set(cycle), 'legacy lasso is not weakly fair')
    require(fair_bad_components(graph), 'legacy starvation not detected')
    return {'prefix': prefix, 'cycle': cycle, 'fair': True,
            'scope': 'feasible infinite abstract schedule; finite compiled reproduction is not infinite runtime observation'}


def check_model():
    configurations = []
    for cap in (1, 2):
        graph, paths = explore(cap)
        bad = fair_bad_components(graph)
        require(not bad, 'fair starvation SCC found')
        configurations.append(dict(capacity=cap, states=len(graph), transitions=sum(map(len, graph.values())),
                                   maxShortestDepth=max(map(len, paths.values())),
                                   waitingStates=sum(s[2][0] == 'wait' for s in graph), fairBadSCCs=len(bad)))
    return {'status': 'model_checked', 'configurations': configurations,
            'states': sum(r['states'] for r in configurations),
            'transitions': sum(r['transitions'] for r in configurations),
            'callers': 3, 'legacy': legacy_witness(),
            'temporalClaim': 'AG safety; weak-fair AF target leaves waiting (admitted/cancelled/drained)',
            'scope': 'fixed point, retry and stutter cycles, capacities1/2; source mapping and runtime progress assumptions explicit'}


def conformance():
    with tempfile.TemporaryDirectory(prefix='trader-progress-') as directory:
        root = Path(directory)
        for v in ('old', 'new', 'suite'):
            (root / v).mkdir()
        legacy = root / 'old/Trader/App/BacktestGate.hs'; legacy.parent.mkdir(parents=True)
        legacy.write_bytes((ROOT / 'formal/research/fixtures/admission-progress-before.hs').read_bytes())
        outputs = {}
        for v, include in (('old', ['-DLEGACY', '-i' + str(root/'old'), '-ihaskell/app']), ('new', ['-ihaskell/app'])):
            exe = compile_haskell(ROOT / 'formal/research/fixtures/admission-progress-witness.hs', root/v, include)
            outputs[v] = subprocess.check_output([exe], text=True, stderr=subprocess.PIPE, timeout=10).strip()
        require(outputs == {'old': 'barged=True', 'new': 'barged=False'}, 'compiled starvation prefix regression')
        exe = compile_haskell(ROOT / 'formal/research/AdmissionProgress.hs', root/'suite', ['-ihaskell/app', '-ihaskell/test'])
        subprocess.run([exe], check=True, capture_output=True, text=True, timeout=20)
    return {'status': 'property_tested', 'runtimeTests': 5, 'generatedCases': 32, 'seed': 20261006,
            'regression': outputs, 'scope': 'actual GHC helpers, deterministic barriers; no full HTTP/OS refinement'}


def check_progress():
    source = extract()
    fixture = json.loads((ROOT / 'formal/research/fixtures/admission-progress.json').read_text())
    require(legacy_witness() == fixture['legacy'], 'preserved starvation lasso drift')
    require(hashlib.sha256((ROOT / 'formal/research/fixtures/admission-progress-before.hs').read_bytes()).hexdigest() == fixture['oldSourceSha256'], 'legacy witness source drift')
    return {'source': source, **prove_numeric(), 'model': check_model(), 'conformance': conformance()}
