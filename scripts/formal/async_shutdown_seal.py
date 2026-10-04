"""Finite composition of reservation ownership with permanent async pool sealing."""
from collections import deque
import ast
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import z3 as z
import async_job_admission as admission

ROOT = admission.ROOT
SOURCE = admission.SOURCE
REGISTRY = 'formal/research/async-shutdown-seal-source.json'
require = admission.require
# Old admission state, closed, completed closers, pending notification bits, signal.
INITIAL = (admission.INITIAL, False, (False, False), 0, False)


def extract():
    registry = json.loads((ROOT / REGISTRY).read_text())
    for path, digest in registry['hashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, 'seal source drift: '+path)
    for path, bodies in registry['bodies'].items():
        source = (ROOT / path).read_text()
        for name, body in bodies.items():
            require(body in source and source.count(name+' ::') == 1, 'seal control drift: '+name)
    return registry


def successors(state, capacity):
    base, closed, closers, pending, signal = state
    count, callers = base
    steps = []
    for label, target in admission.successors(base, capacity):
        if closed and label == 'reserve':
            i = next(i for i in range(2) if target[1][i] != callers[i])
            row = ('rejected', 'none', None, False, False)
            target = (count, callers[:i]+(row,)+callers[i+1:])
            label = 'reject-sealed'
        notify = pending
        if target[0] < count and target[0] == 0 and closed:
            notify |= 4  # Last reservation finalizer owes a nonblocking notification.
        steps.append((label, (target, closed, closers, notify, signal)))
    for i, done in enumerate(closers):
        if not done:
            following = closers[:i]+(True,)+closers[i+1:]
            notify = pending | (1 << i) if count == 0 else pending
            steps.append(('seal', (base, True, following, notify, signal)))
    for bit in (1, 2, 4):
        if pending & bit:
            steps.append(('notify', (base, closed, closers, pending ^ bit, True)))
    return steps


def rank(state):
    base, _, closers, pending, _ = state
    return 10*admission.rank(base)+3*sum(not c for c in closers)+pending.bit_count()


def check_model(next_states=successors):
    receipts = []
    for capacity in (1, 2):
        queue = deque([(INITIAL, 0)]); seen = {INITIAL}; edges = terminals = depth = delayed = rejected = 0
        while queue:
            state, distance = queue.popleft()
            base, closed, closers, pending, signal = state
            count, callers = base
            owners = sum(p in ('prepare','ready') or c in ('gated','running') for p,c,_,_,_ in callers)
            require(count == owners and 0 <= count <= capacity, 'seal count ownership')
            require(not signal or (closed and count == 0), 'premature shutdown acknowledgement')
            require(not pending or (closed and count == 0), 'invalid completion notification')
            require(all(not ran or published for _,_,_,published,ran in callers), 'seal unpublished execution')
            depth = max(depth, distance)
            delayed += bool(pending and not signal)
            following = next_states(state, capacity)
            if not following:
                terminals += 1
                require(closed and all(closers) and signal and count == 0, 'seal deadlock or stranded reservation')
            for label, target in following:
                require(rank(target) < rank(state), 'seal nonterminating transition')
                require(not closed or target[1], 'seal reopened')
                require(not signal or target[4], 'completion consumed')
                require(not closed or target[0][0] <= count, 'post-seal admission')
                rejected += label == 'reject-sealed'
                edges += 1
                if target not in seen:
                    seen.add(target); queue.append((target, distance+1))
        require(delayed and rejected and terminals, 'seal vacuous coverage')
        receipts.append(dict(capacity=capacity,states=len(seen),transitions=edges,maxShortestDepth=depth,
                             terminalStates=terminals,delayedNotificationStates=delayed,sealedRejections=rejected))
    return dict(states=sum(r['states'] for r in receipts),transitions=sum(r['transitions'] for r in receipts),
                callers=2,closers=2,initialRank=rank(INITIAL),capacities=receipts,
                scope='finite admission/seal composition; conditional primitive and callback progress, not full IO refinement')


def prove_invariants():
    n, limit = z.Ints('seal_n seal_limit'); closed = z.Bool('seal_closed')
    accepted = z.And(z.Not(closed), n >= 0, limit > 0, n < limit)
    after = z.If(accepted,n+1,n)
    released = z.If(n>0,n-1,0)
    notified = z.And(closed,released==0)
    claims = [z.Implies(closed,z.And(z.Not(accepted),after==n)),
              z.And(after>=0,after<=limit),z.And(released>=0,released<=n),
              z.Implies(notified,z.And(closed,released==0)),
              z.Implies(z.And(closed,n==0),z.And(after==0,released==0)),
              z.Implies(accepted,z.And(z.Not(closed),after==n+1))]
    for claim in claims:
        solver=z.Solver(); solver.set(timeout=10000,random_seed=0)
        solver.add(limit>0,n>=0,n<=limit)
        require(solver.check()==z.sat,'seal vacuous domain')
        solver.add(z.Not(claim)); require(solver.check()==z.unsat,'seal SMT counterexample or unknown')
    return {'F-ASYNC-SEAL-INVARIANTS':'unsat'}


def conformance():
    entries=json.loads((ROOT/'formal/research/async-shutdown-seal-counterexamples.json').read_text())['entries']
    with tempfile.TemporaryDirectory(prefix='trader-seal-') as name:
        temp=Path(name)
        for version in ('old','new','numeric'):
            (temp/version).mkdir()
        legacy=temp/'old/Trader/App/AsyncJobAdmission.hs';legacy.parent.mkdir(parents=True)
        legacy.write_bytes((ROOT/'formal/research/fixtures/async-seal-before.hs').read_bytes())
        for version,includes in [('old',['-DLEGACY','-i'+str(temp/'old')]),('new',['-ihaskell/app'])]:
            exe=admission.compile_haskell(ROOT/'formal/research/fixtures/async-seal-witness.hs',temp/version,includes)
            output=subprocess.check_output([exe],text=True,timeout=10).splitlines()
            require(output==[e['before' if version=='old' else 'after'] for e in entries],'seal counterexample regression')
        exe=admission.compile_haskell(ROOT/'formal/research/AsyncSeal.hs',temp/'numeric',['-ihaskell/app'])
        bound=2**63-1
        cases=list(itertools.product((-bound-1,-1,0,1,2,bound),repeat=2))
        cases=[(*row,closed) for row in cases for closed in (False,True)]
        output=subprocess.check_output([exe],input=''.join(str(row)+'\n' for row in cases),text=True,timeout=10)
        expected=[]
        for limit,count,closed in cases:
            next_count,accepted=admission.admission(limit,count)
            expected.append((count if closed else next_count,closed,2 if closed else 0 if accepted else 1))
        require([ast.literal_eval(line) for line in output.splitlines()]==expected,'seal numeric conformance')
    return dict(counterexamples=[e['id'] for e in entries],numericCases=len(cases),runtimeTests=4,
                concurrentCases=32,seed=20261004,runtimeSuite='asyncJobAdmission.conformance',
                scope='compiled helper and Main snapshot control slice; source-bound composition, not full HTTP refinement')


def check_async_seal():
    extract()
    return dict(smt=prove_invariants(),model=check_model(),conformance=conformance())
