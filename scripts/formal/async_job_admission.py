"""Source-bound async admission ownership model, integer proofs and compiled tests."""
from collections import deque
import ast
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import tempfile
import z3 as z

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'haskell/app/Trader/App/AsyncJobAdmission.hs'
REGISTRY = 'formal/research/async-job-admission-source.json'


def require(ok, message):
    if not ok:
        raise ValueError('async admission: ' + message)


def extract():
    registry = json.loads((ROOT / REGISTRY).read_text())
    for path, expected in registry['hashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, 'source drift: ' + path)
    for path, bodies in registry['bodies'].items():
        source = (ROOT / path).read_text()
        for name, body in bodies.items():
            require(body in source and source.count(name + ' ::') == 1, 'control drift: ' + name)
    core = (ROOT / SOURCE).read_text()
    require('JobSlots (..)' not in core.split(') where')[0], 'private counter exposed')
    main = (ROOT / 'haskell/app/Main.hs').read_text()
    require(main.count('startBoundedJob (jsRunning store) prepare execute publish') == 1, 'Main handoff drift')
    require('jsRunning :: !JobSlots' in main and 'running <- newJobSlots maxRunning' in main,
            'private pool representation drift')
    require('readMVar (jsRunning store)' not in main and 'modifyMVar (jsRunning store)' not in main,
            'legacy counter bypass')
    return registry


def admission(limit, count):
    return (count, False) if count < 0 or limit <= 0 or count >= limit else (count + 1, True)


def release(count):
    return count - 1 if count > 0 else 0


def prove_numeric():
    limit, count, bound = z.Ints('aj_limit aj_count aj_bound')
    accepted = z.And(count >= 0, limit > 0, count < limit)
    after = z.If(accepted, count + 1, count)
    freed = z.If(count > 0, count - 1, 0)
    claims = [z.And(after >= -bound - 1, after <= bound),
              z.Implies(accepted, z.And(after == count + 1, after <= limit, after > 0)),
              z.Implies(z.And(count >= 0, count <= limit, limit > 0), z.And(after >= 0, after <= limit)),
              z.And(freed >= 0, freed <= bound),
              z.Implies(count >= 0, freed <= count),
              z.Implies(accepted, z.If(after > 0, after - 1, 0) == count)]
    for claim in claims:
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
        solver.add(bound > 0, count >= -bound - 1, count <= bound, limit >= -bound - 1, limit <= bound)
        require(solver.check() == z.sat, 'vacuous Int domain')
        solver.add(z.Not(claim))
        require(solver.check() == z.unsat, 'numeric counterexample or unknown')
    return {'F-ASYNC-ADMISSION-NUMERIC': 'unsat'}


# Per caller: (parent phase, child phase, gate, published, everExecuted).
# Parent owns a reservation in prepare/ready; child owns it while gated/running.
CALLER = ('new', 'none', None, False, False)
INITIAL = (0, (CALLER, CALLER))


def successors(state, capacity):
    count, callers = state
    steps = []
    for i, (parent, child, gate, published, ran) in enumerate(callers):
        def add(label, delta=0, p=parent, c=child, g=gate, visible=published, executed=ran):
            replacement = (p, c, g, visible, executed)
            steps.append((label, (count + delta, callers[:i] + (replacement,) + callers[i + 1:])))
        if parent == 'new':
            if count < capacity:
                add('reserve', 1, p='prepare')
            else:
                add('reject', p='rejected')
        if parent == 'prepare':
            add('prepared', p='ready')
            add('preparation-failure', -1, p='failed')
        if parent == 'ready':
            add('fork', p='waiting', c='gated')
            add('fork-failure', -1, p='failed')
        if parent == 'waiting':
            add('publish', p='published', visible=True)
            add('publication-failure', p='failed', g=False)
        if parent == 'published':
            add('enable', p='returned', g=True)
        if child == 'gated':
            add('cancel-gated', -1, c='done')
            if gate is False:
                add('abort', -1, c='done')
            if gate is True:
                add('execute', c='running', executed=True)
        if child == 'running':
            add('complete-or-cancel', -1, c='done')
    return steps


def rank(state):
    return sum({'new': 7, 'prepare': 6, 'ready': 5, 'waiting': 4, 'published': 3,
                'returned': 0, 'failed': 0, 'rejected': 0}[p]
               + {'none': 3, 'gated': 2, 'running': 1, 'done': 0}[c]
               for p, c, _, _, _ in state[1])


def check_model(next_states=successors):
    receipts = []
    for capacity in (1, 2):
        queue = deque([(INITIAL, 0)]); seen = {INITIAL}; edges = terminals = executed = rejected = depth = 0
        while queue:
            state, distance = queue.popleft(); count, callers = state
            depth = max(depth, distance)
            owners = sum(p in ('prepare', 'ready') or c in ('gated', 'running') for p, c, _, _, _ in callers)
            require(count == owners and 0 <= count <= capacity, 'reservation ownership/bound violated')
            for p, c, gate, published, ran in callers:
                require(not ran or published, 'callback before publication')
                require(c != 'running' or gate is True, 'callback without enabled gate')
                require(p not in ('prepare', 'ready') or c == 'none', 'double reservation owner')
                require(gate is not True or published, 'unpublished gate enabled')
            following = next_states(state, capacity)
            if not following:
                terminals += 1
                require(count == 0 and all(p in ('failed', 'returned', 'rejected') for p, _, _, _, _ in callers),
                        'stranded reservation or unexpected deadlock')
            executed += any(row[4] for row in callers)
            rejected += any(row[0] == 'rejected' for row in callers)
            for _, target in following:
                require(rank(target) < rank(state), 'nonterminating protocol transition')
                require(all(not old[3] or new[3] for old, new in zip(callers, target[1])), 'publication revoked')
                edges += 1
                if target not in seen:
                    seen.add(target); queue.append((target, distance + 1))
        require(executed > 0 and terminals > 0 and (capacity != 1 or rejected > 0), 'vacuous model coverage')
        receipts.append({'capacity': capacity, 'states': len(seen), 'transitions': edges,
                         'maxShortestDepth': depth, 'terminalStates': terminals,
                         'executedStates': executed, 'rejectedStates': rejected})
    return {'states': sum(r['states'] for r in receipts), 'transitions': sum(r['transitions'] for r in receipts),
            'initialRank': rank(INITIAL), 'callers': 2, 'capacities': receipts,
            'scope': 'finite atomic protocol; liveness conditional on callbacks and primitives returning, not IO refinement'}


def compile_haskell(main, directory, includes):
    build = directory / 'build'; build.mkdir()
    exe = str(directory / 'test')
    subprocess.run(['ghc', '-v0', '-O0', '-threaded', *includes, '-outputdir', str(build), str(main), '-o', exe],
                   cwd=ROOT, check=True, capture_output=True, text=True, timeout=90)
    return exe


def conformance():
    entries = json.loads((ROOT / 'formal/research/async-job-admission-counterexamples.json').read_text())['entries']
    with tempfile.TemporaryDirectory(prefix='trader-async-proof-') as name:
        temp = Path(name)
        for child in ('old', 'new', 'suite'):
            (temp / child).mkdir()
        legacy = temp / 'old/Trader/App/AsyncJobAdmission.hs'; legacy.parent.mkdir(parents=True)
        legacy.write_bytes((ROOT / 'formal/research/fixtures/async-job-before.hs').read_bytes())
        witness = ROOT / 'formal/research/fixtures/async-job-witness.hs'
        for version, includes in [('old', ['-i' + str(temp / 'old')]), ('new', ['-ihaskell/app'])]:
            exe = compile_haskell(witness, temp / version, includes)
            lines = subprocess.check_output([exe], cwd=ROOT, text=True, stderr=subprocess.PIPE, timeout=10).splitlines()
            require(lines == [e['before' if version == 'old' else 'after'] for e in entries], 'control-slice regression drift')
        exe = compile_haskell(ROOT / 'formal/research/AsyncJobAdmission.hs', temp / 'suite', ['-ihaskell/app', '-ihaskell/test'])
        bound = int(subprocess.check_output([exe, '--int-bound'], text=True, timeout=10))
        require(bound in (2**31 - 1, 2**63 - 1), 'unsupported Int width')
        cases = list(itertools.product((-bound-1, -1, 0, 1, 2, bound), repeat=2))
        rng = random.Random(20261004)
        cases += [(rng.randrange(-bound-1, bound+1), rng.randrange(-bound-1, bound+1)) for _ in range(256)]
        output = subprocess.check_output([exe, '--cases'], input=''.join(str(row)+'\n' for row in cases), text=True, timeout=10)
        require([ast.literal_eval(line) for line in output.splitlines()] == [(admission(*row), release(row[1])) for row in cases],
                'compiled numeric mismatch')
        subprocess.run([exe], check=True, capture_output=True, text=True, timeout=20)
    return {'counterexamples': [e['id'] for e in entries], 'intBits': bound.bit_length()+1, 'numericCases': len(cases),
            'generatedNumericCases': 256, 'runtimeTests': 11, 'concurrentCases': 32, 'seed': 20261004,
            'scope': 'compiled helper and source-bound baseline control slice; not full HTTP/runtime refinement'}


def check_async_admission():
    extract()
    return {'smt': prove_numeric(), 'model': check_model(), 'conformance': conformance()}
