"""Source-bound backtest gate arithmetic, finite lifecycle and compiled regressions."""
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import tempfile
import z3 as z
from async_job_admission import compile_haskell

ROOT=Path(__file__).resolve().parents[2]
SOURCE='haskell/app/Trader/App/BacktestGate.hs'
REGISTRY='formal/research/backtest-gate-source.json'

def require(ok,message):
    if not ok: raise ValueError('backtest gate: '+message)

def extract():
    registry=json.loads((ROOT/REGISTRY).read_text())
    for path,digest in registry['hashes'].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,'source drift: '+path)
    for path,bodies in registry['bodies'].items():
        source=(ROOT/path).read_text()
        for name,body in bodies.items():
            require(body in source and source.count(name+' ::')==1,'body drift: '+name)
    core=(ROOT/SOURCE).read_text()
    require('BacktestGate (..)' not in core.split(') where')[0],'gate constructor exposed')
    main=(ROOT/'haskell/app/Main.hs').read_text()
    require('data BacktestGate' not in main and 'runBacktestWithGate ::' not in main,'legacy gate bypass')
    require('import Trader.App.BacktestGate' in main,'Main handoff absent')
    return registry

def micros(seconds,bound):return min(bound,max(1,seconds)*1000000)

def prove_numeric():
    seconds,later,bound=z.Ints('bt_seconds bt_later bt_bound')
    def value(s):
        raw=z.If(s>1,s,1)*1000000
        return z.If(raw>bound,bound,raw)
    u=value(seconds); raw=z.If(seconds>1,seconds,1)*1000000
    claims=[z.And(u>=1,u<=bound),u<=raw,z.Implies(raw<=bound,u==raw),
            z.Implies(raw>bound,u==bound),z.Implies(later>=seconds,value(later)>=u)]
    for claim in claims:
        solver=z.Solver();solver.set(timeout=10000,random_seed=0)
        solver.add(bound>0,seconds>=-bound-1,seconds<=bound,later>=-bound-1,later<=bound)
        require(solver.check()==z.sat,'vacuous numeric domain')
        solver.add(z.Not(claim));require(solver.check()==z.unsat,'numeric counterexample or unknown')
    return {'F-BACKTEST-GATE-NUMERIC':'unsat'}

# Caller = (phase, causal event, outcome, callback entered).
CALLER=('new','none','none',False)
INITIAL=(0,(CALLER,CALLER))
OUTCOMES={'success':'success','sync':'error','own-timer':'timeout','cancel':'propagated','outer-timer':'propagated'}

def successors(state,capacity,modes):
    count,callers=state;steps=[]
    for i,(phase,event,outcome,ran) in enumerate(callers):
        def add(label,p,delta=0,e=event,o=outcome,r=ran):
            row=(p,e,o,r)
            steps.append((label,(count+delta,callers[:i]+(row,)+callers[i+1:])))
        if phase=='new':
            add('cancel-before-reserve','done',e='cancel',o='propagated')
            if count<capacity:add('reserve','reserved',1)
            # Queue-erased overapproximation: FIFO contention may defer/reject
            # even with an available slot. Fair ordering is checked separately.
            if modes[i]=='waiting':add('wait','waiting')
            else:add('busy','done',e='busy',o='busy')
        elif phase=='waiting':
            add('retry','new')
            add('cancel-wait','done',e='cancel',o='propagated')
        elif phase=='reserved':
            add('enter','active',r=True)
            add('cancel-after-reserve','cleanup',e='cancel',o='propagated')
        elif phase=='active':
            for cause,result in OUTCOMES.items():add(cause,'cleanup',e=cause,o=result)
        elif phase=='cleanup':add('release','done',-1)
    return steps

def check_model(next_states=successors):
    receipts=[]
    for capacity in (1,2):
        for modes in itertools.product(('immediate','waiting'),repeat=2):
            queue=deque([(INITIAL,0)]);seen={INITIAL};reverse={INITIAL:set()};terminals=set();edges=depth=waits=0
            while queue:
                state,distance=queue.popleft();count,callers=state
                owners=sum(p in ('reserved','active','cleanup') for p,_,_,_ in callers)
                require(count==owners and 0<=count<=capacity,'reservation bound/ownership')
                depth=max(depth,distance)
                for phase,cause,result,ran in callers:
                    require(not ran or phase in ('active','cleanup','done'),'callback before reservation')
                    require(cause not in OUTCOMES or result==OUTCOMES[cause],'exception routing')
                    require(cause!='busy' or (not ran and result=='busy'),'busy callback')
                following=next_states(state,capacity,modes)
                if not following:
                    require(count==0 and all(p=='done' for p,_,_,_ in callers),'stranded terminal owner')
                    terminals.add(state)
                waits+=any(p=='waiting' for p,_,_,_ in callers)
                for label,target in following:
                    edges+=1;reverse.setdefault(target,set()).add(state)
                    if target not in seen:seen.add(target);queue.append((target,distance+1))
            reachable=set(terminals);queue=deque(terminals)
            while queue:
                for parent in reverse[queue.popleft()]:
                    if parent not in reachable:reachable.add(parent);queue.append(parent)
            require(reachable==seen,'state cannot reach quiescence')
            require(terminals and edges,'vacuous lifecycle')
            receipts.append(dict(capacity=capacity,modes=list(modes),states=len(seen),transitions=edges,
                                 maxShortestDepth=depth,terminalStates=len(terminals),waitingStates=waits))
    return dict(states=sum(r['states'] for r in receipts),transitions=sum(r['transitions'] for r in receipts),
                callers=2,configurations=receipts,temporalClaim='AG safety and EF terminal from every reachable state; retry cycles explicit, no unconditional AF/fairness claim')

def conformance():
    registry=json.loads((ROOT/'formal/research/backtest-gate-counterexamples.json').read_text())
    with tempfile.TemporaryDirectory(prefix='trader-backtest-') as name:
        temp=Path(name)
        for version in ('old','new','suite'):(temp/version).mkdir()
        legacy=temp/'old/Trader/App/BacktestGate.hs';legacy.parent.mkdir(parents=True)
        legacy.write_bytes((ROOT/'formal/research/fixtures/backtest-gate-before.hs').read_bytes())
        for version,includes in [('old',['-DLEGACY','-i'+str(temp/'old')]),('new',['-ihaskell/app'])]:
            exe=compile_haskell(ROOT/'formal/research/fixtures/backtest-gate-witness.hs',temp/version,includes)
            output=subprocess.check_output([exe],text=True,stderr=subprocess.PIPE,timeout=15).splitlines()
            require(output==registry['before' if version=='old' else 'after'],'baseline regression drift')
        exe=compile_haskell(ROOT/'formal/research/BacktestGate.hs',temp/'suite',['-ihaskell/app','-ihaskell/test'])
        bound=int(subprocess.check_output([exe,'--int-bound'],text=True,timeout=10))
        require(bound in (2**31-1,2**63-1),'unsupported Int domain')
        cases=[-bound-1,-1,0,1,2,bound//1000000,bound//1000000+1,bound]
        rng=random.Random(20261004);cases += [rng.randrange(-bound-1,bound+1) for _ in range(256)]
        output=subprocess.check_output([exe,'--cases'],input=''.join(str(x)+'\n' for x in cases),text=True,timeout=10)
        require([int(x) for x in output.splitlines()]==[micros(x,bound) for x in cases],'numeric conformance')
        subprocess.run([exe],check=True,capture_output=True,text=True,timeout=20)
    return dict(counterexamples=[r['id'] for r in registry['entries']],intBits=bound.bit_length()+1,
                numericCases=len(cases),runtimeTests=7,concurrentCases=32,seed=20261004,
                scope='compiled extracted module, literal baseline function adapter and labeled reserve-gap schedule; no full HTTP/IO refinement')

def check_backtest_gate():
    extract()
    return dict(smt=prove_numeric(),model=check_model(),conformance=conformance())
