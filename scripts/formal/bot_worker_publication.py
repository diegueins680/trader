"""Source-bound worker publication, finite interleavings and compiled conformance."""
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import z3 as z

ROOT=Path(__file__).resolve().parents[2]
SOURCE='haskell/app/Trader/App/WorkerPublication.hs'
TEST='haskell/test/Trader/Test/WorkerPublication.hs'
REGISTRY='formal/research/bot-worker-publication-source.json'
LEGACY='formal/research/bot-worker-publication-legacy.json'
MAIN='haskell/app/Main.hs'


def require(ok,message):
    if not ok:raise ValueError('bot worker publication: '+message)


def application_body(source):
    start=source.index('botStartSymbolWithSettings allowExisting')
    return source[start:source.index('\nbotStartWorker ::',start)]


def extract(core=None,main=None,registry=None):
    core=(ROOT/SOURCE).read_text() if core is None else core
    main=(ROOT/MAIN).read_text() if main is None else main
    registry=json.loads((ROOT/REGISTRY).read_text()) if registry is None else registry
    require(core==registry['helper'],'complete helper drift')
    body=application_body(main)
    require(body==registry['application'],'application binding drift')
    require(core.count('forkIOWithUnmask $')==1 and core.count('when accepted (unmask action)')==1,'execution gate')
    require('modifyMVarMasked\n' in core and 'plan <- restore (prepare state)' in core,'transaction/preparation masking')
    require(core.index('`onException` void (tryPutMVar gate False)')<core.index('void (tryPutMVar gate True)'),'commit gate order')
    require(core.count('evaluate ')==3,'publication WHNF cuts')
    require(body.count('publishWorker (bcRuntime ctrl)')==1 and 'forkIO' not in body,'unguarded application fork')
    require(body.count('HM.lookup sym tenantMap')==2 and body.count('Retain ')==3,'duplicate check coverage')
    require(body.index('now <- getTimestampMs')<body.index('pure $ Launch '),'prepare metadata before fork')
    require(body.count('bsrThreadId = tid')==1 and body.count('HM.insert sym st tenantMap')==1,'identity/map binding')
    require(main.count('publishWorker (bcRuntime ctrl)')==1 and main.count('botStartWorker')==3,'application call coverage')
    for path,expected in registry['supportHashes'].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,'support source drift: '+path)
    return {'status':'exhaustively_checked','productionCallSites':1,'duplicateChecks':2,
            'privateGates':1,'forkSites':1,'publicationWHNFCuts':3,
            'scope':'complete helper and actual startup binding; reviewed MVar/HashMap/IO correspondence, not GHC refinement'}


def prove(premise,conclusion):
    solver=z.Solver();solver.set(timeout=10000,random_seed=0);solver.add(premise)
    require(solver.check()==z.sat,'unsatisfied or unknown premise')
    solver.add(z.Not(conclusion));require(solver.check()==z.unsat,'counterexample or unknown')


def gates(mutant=False):
    committed,failed=z.Bools('bot_committed bot_failed')
    gate=z.BoolVal(True) if mutant else z.And(committed,z.Not(failed))
    prove(z.BoolVal(True),z.Implies(gate,z.And(committed,z.Not(failed))))
    prove(failed,z.Not(gate))
    # Abstract the actual nested HashMap inserts, with Nothing represented by -1.
    row=z.ArraySort(z.IntSort(),z.IntSort());state=z.Array('bot_map',z.IntSort(),row)
    tenant,symbol,other_tenant,other_symbol,tid=z.Ints('bot_tenant bot_symbol bot_other_tenant bot_other_symbol bot_tid')
    current=z.Select(z.Select(state,tenant),symbol);empty=current==-1
    inserted=z.Store(state,tenant,z.Store(z.Select(state,tenant),symbol,tid))
    after=z.If(empty,inserted,state)
    prove(z.And(tid>=0,empty),z.Select(z.Select(after,tenant),symbol)==tid)
    prove(z.Not(empty),after==state)
    prove(other_tenant!=tenant,z.Select(after,other_tenant)==z.Select(state,other_tenant))
    prove(other_symbol!=symbol,z.Select(z.Select(after,tenant),other_symbol)==z.Select(z.Select(state,tenant),other_symbol))
    return {'F-BOT-PUBLICATION-GATES':'unsat'}


# Parent phase, child phase, gate (-1 pending,0 reject,1 accept), committed, executed.
CALLER=('new','none',-1,False,False)


def successors(state,legacy=False):
    owner,lock,callers=state;out=[]
    for i,(parent,child,gate,committed,ran) in enumerate(callers):
        def add(label,p=parent,c=child,g=gate,visible=committed,executed=ran,o=owner,l=lock):
            entry=(p,c,g,visible,executed)
            out.append((str(i)+':'+label,(o,l,callers[:i]+(entry,)+callers[i+1:])))
        if parent=='new' and lock==-1:add('lock',p='prepare',l=i)
        elif parent=='prepare':
            add('prepare-failure',p='failed',g=0,l=-1)
            if owner==-1:add('prepared',p='ready')
            else:add('retain-commit',p='retained-ready',l=-1)
        elif parent=='ready':
            add('fork',p='publishing',c='gated')
            add('fork-failure',p='failed',g=0,l=-1)
        elif parent=='publishing':
            add('force-publication',p='commit-ready')
            add('publication-failure',p='failed',g=0,l=-1)
        elif parent=='commit-ready':add('commit',p='committed',visible=True,o=i,l=-1)
        elif parent=='committed':add('release',p='returned',g=1)
        elif parent=='retained-ready':add('release-unused',p='retained',g=1)
        if child=='gated':
            # External interruption may terminate a waiting child. It cannot run it.
            add('cancel-child',c='done')
            if gate==0:add('reject-child',c='done')
            elif gate==1 or legacy:add('execute',c='running',executed=True)
    return out


def rank(state):
    parents={'new':7,'prepare':6,'ready':5,'publishing':4,'commit-ready':3,'committed':2,
             'retained-ready':2,'returned':0,'retained':0,'failed':0}
    return sum(parents[p]+{'none':2,'gated':1,'done':0,'running':0}[c] for p,c,_,_,_ in state[2])


def model(legacy=False,next_states=successors):
    results=[];witness=None
    for existing in (-1,2):
        initial=(existing,-1,(CALLER,CALLER));queue=deque([initial]);seen={initial:[]};edges=terminals=executed=aborted=0
        while queue:
            state=queue.popleft();owner,lock,callers=state
            held=[i for i,(p,_,_,_,_) in enumerate(callers) if p in ('prepare','ready','publishing','commit-ready')]
            require(held==([] if lock==-1 else [lock]),'transaction lock ownership')
            require(owner in (-1,0,1,2),'owner domain')
            for i,(p,c,g,committed,ran) in enumerate(callers):
                safe=not ran or (committed and g==1 and owner==i)
                if not safe:
                    if not legacy:raise ValueError('bot worker publication: unpublished worker execution')
                    if witness is None:witness=seen[state]
                require(g!=1 or committed or (c=='none' and p=='retained'),'release before commit')
                require(not committed or owner==i,'committed owner overwritten')
                executed+=ran;aborted+=g==0 and c=='done'
            targets=next_states(state,legacy)
            if not targets:
                terminals+=1
                require(lock==-1 and all(p in ('returned','retained','failed') and c in ('none','done','running') for p,c,_,_,_ in callers),'unexpected deadlock')
            for label,target in targets:
                require(rank(target)<rank(state),'nondecreasing protocol rank')
                require(owner==-1 or target[0]==owner,'existing registration overwritten')
                for old,new in zip(callers,target[2]):
                    require(old[2]==-1 or old[2]==new[2],'gate decision changed')
                    require(not old[3] or new[3],'commit revoked')
                edges+=1
                if target not in seen:seen[target]=seen[state]+[label];queue.append(target)
        require(terminals>0,'no terminal execution')
        if existing==-1:require(executed and aborted,'vacuous execution/abort coverage')
        else:require(executed==0,'preexisting owner did not reject')
        results.append({'initialOwner':existing,'states':len(seen),'transitions':edges,'maxShortestDepth':max(map(len,seen.values())),
                        'terminalStates':terminals,'executionStates':executed,'abortCompletionStates':aborted})
    require((witness is not None)==legacy,'counterexample expectation')
    return {'states':sum(r['states'] for r in results),'transitions':sum(r['transitions'] for r in results),
            'callers':2,'keys':1,'cases':results,'legacyCounterexample':witness,'initialRank':rank((-1,-1,(CALLER,CALLER))),
            'scope':'finite two-start publication protocol; running is terminal handoff, no action-termination or persistent ownership theorem'}


def conformance():
    with tempfile.TemporaryDirectory(prefix='trader-bot-publication-') as d:
        directory=Path(d);main=directory/'Main.hs';exe=directory/'check'
        main.write_text('module Main where\nimport Trader.Test.WorkerPublication\nmain :: IO ()\nmain = mapM_ (\\(name,action) -> action >> putStrLn name) workerPublicationSuite\n')
        subprocess.run(['ghc','-v0','-Wall','-Werror','-threaded','-i'+str(ROOT/'haskell/app'),'-i'+str(ROOT/'haskell/test'),
                        '-outputdir',d,'-o',str(exe),str(main)],check=True,capture_output=True,text=True,timeout=180)
        output=subprocess.run([str(exe),'+RTS','-N2'],check=True,capture_output=True,text=True,timeout=45).stdout.splitlines()
        require(len(output)==6 and len(set(output))==6,'compiled suite coverage')
    return {'compiledCases':6,'publicationFailureCuts':3,'concurrentCallers':2,'capabilities':['local MVar','local threads'],
            'legacyOrderingReproduced':True,'networkCalls':0,'liveOrders':0,'tests':output}


def check_publication():
    surface=extract();fixed=model();legacy=model(True)
    document=json.loads((ROOT/'formal/research/counterexamples.json').read_text())
    witness=next(e for e in document['entries'] if e['id']=='CE-BOT-UNPUBLISHED-WORKER')
    require(witness['trace']==legacy['legacyCounterexample'],'preserved counterexample drift')
    return {'surface':surface,'smt':gates(),'model':fixed,'legacyModel':legacy,'conformance':conformance()}


if __name__=='__main__':
    print(json.dumps(check_publication(),indent=2))
