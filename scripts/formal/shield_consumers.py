"""Source-derived replay consumer ownership and shield-origin composition.

Trusted Python/NumPy primitive contracts; not a compiler or numeric proof.
"""
import ast
from collections import deque
from dataclasses import dataclass, replace
from itertools import product
import json
from pathlib import Path
import sys
from unittest.mock import patch

import z3 as z
from data_composition import definition
from promotion_boundary import extract as promotion_extract, shape
from target_v2 import certify

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = 'formal/research/shield-consumer-source.json'
MODULES = {'sequential_env', 'sequential_evaluation', 'run_sequential_screen'}
FIELDS = {'actions','done','equity','failure','fills','horizon','modifications','observation',
          'pending','proposals','rejections','rows','start','step','stop','supported','t','units'}


def require(ok, message):
    if not ok: raise ValueError('shield consumers: ' + message)


def context(tree):
    parents = {c:n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    def owner(n):
        while n in parents:
            n = parents[n]
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)): return n.name
        return '<module>'
    return parents, owner


def roster(trees):
    uses = []; constructors = []; calls = []
    for module in sorted(MODULES):
        tree = trees[module]; parents, owner = context(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id in ('env','replay'):
                uses.append([module,owner(n),type(n.ctx).__name__,ast.unparse(parents[n])])
            if isinstance(n, ast.Call) and ast.unparse(n.func) == 'Replay':
                constructors.append([module,owner(n),ast.unparse(parents[n])])
            if isinstance(n, ast.Call) and ast.unparse(n.func) in ('env.step','replay.step','self._trade'):
                calls.append([module,owner(n),ast.unparse(n)])
    return {'receiverUses':sorted(uses), 'constructors':sorted(constructors), 'consumers':sorted(calls)}


def research_importers(sources=None):
    from promotion_boundary import MODULES as delivered
    sources = {str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'scripts/research').rglob('*.py'))} if sources is None else sources
    found=[]
    for path,source in sources.items():
        for n in ast.walk(ast.parse(source)):
            names = [a.name for a in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) else []
            if any(name in delivered for name in names):
                require(path in {'scripts/research/'+name+'.py' for name in delivered}, 'unregistered research consumer: '+path)
                found.append([path,ast.unparse(n)])
    return sorted(found)


def extract(trees=None, registry=None):
    real, surface = promotion_extract()
    trees = real if trees is None else trees
    registry = json.loads((ROOT/REGISTRY).read_text()) if registry is None else registry
    require(set(trees) == set(real), 'complete module coverage')
    require(set(registry) == {'schemaVersion','receiverUses','constructors','consumers'} and registry['schemaVersion'] == 1, 'registry schema')
    actual = roster(trees)
    require(actual == {k:registry[k] for k in actual}, 'receiver/consumer roster drift')
    require(all(shape(trees[k]) == shape(real[k]) for k in real), 'reviewed complete effects drift')
    require(len(actual['constructors']) == 4 and len(actual['consumers']) == 6, 'constructor/consumer count')
    # Every replay identity is an ordinary local binding, never passed to a
    # policy callback, thread, mutable global or unchecked helper. Metric helpers
    # are read-only and the sole returned instance is read only by runner rows.
    scopes = {'sequential_env':{'collect','_admit_training_transition'},
              'sequential_evaluation':{'_admit_economic_ledger','economic','replay_policy','short_ope'},
              'run_sequential_screen':{'run'}}
    for module in MODULES:
        parents, owner = context(trees[module])
        for n in ast.walk(trees[module]):
            if not isinstance(n, ast.Name) or n.id not in ('env','replay'): continue
            require(owner(n) in scopes[module], 'receiver escaped owner scope')
            p = parents[n]
            if isinstance(n.ctx, ast.Store):
                require(isinstance(p,(ast.Assign,ast.Tuple)), 'nonlocal receiver binding')
                continue
            if isinstance(p, ast.Attribute):
                require(isinstance(p.ctx,ast.Load) and p.attr in FIELDS, 'external state write/private call')
            elif isinstance(p, ast.Call):
                require(ast.unparse(p.func) in ('economic','_admit_economic_ledger','_admit_training_transition'), 'receiver callback escape')
            elif isinstance(p, ast.Compare):
                require(ast.unparse(p) == 'env is None' or ast.unparse(p) == 'env is not None', 'receiver comparison escape')
            elif isinstance(p, ast.Tuple):
                require(module == 'sequential_evaluation' and owner(n) == 'replay_policy' and
                        isinstance(parents[p],ast.Return) and ast.unparse(p) == '(env, result)', 'receiver return escape')
            else: raise ValueError('shield consumers: unreviewed receiver use')
    cls = next(n for n in trees['sequential_env'].body if isinstance(n,ast.ClassDef) and n.name == 'Replay')
    methods = {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    parents, owner = context(cls)
    writes = []
    for n in ast.walk(cls):
        if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store) and n.attr in ('pending','enabled'):
            require(ast.unparse(n.value)=='self', 'foreign instance mutation')
            writes.append([owner(n),ast.unparse(parents[n])])
    expected = [
        ['__init__','self.enabled = enabled'],
        ['__init__','self.pending: tuple[int, float] | None = None'],
        ['step','self.pending = (self.t + 1 + self.execution.extra_delay, proposal)'],
        ['step','self.pending = None'],['step','self.pending = None']]
    require(sorted(writes)==sorted(expected), 'pending origin/enabled write coverage')
    shield=definition(trees['sequential_env'],'shield')
    require(len(shield.body)==6 and isinstance(shield.body[0],ast.Expr), 'shield body shape')
    guards=[ast.unparse(n.test) for n in shield.body[1:5] if isinstance(n,ast.If)]
    require(guards==['enabled is not True','valid is not True or ownership is not True',
                    'not _finite_real(elapsed_ms) or not 0 <= elapsed_ms <= 20',
                    'not _finite_real(action) or action not in (-0.25, 0.0, 0.25)'], 'shield guard semantics')
    require(all(len(n.body)==1 and isinstance(n.body[0],ast.Return) and
                isinstance(n.body[0].value,ast.Tuple) and ast.literal_eval(n.body[0].value.elts[0]) is None
                for n in shield.body[1:5]), 'shield rejection semantics')
    require(ast.unparse(shield.body[-1])=="return (float(action), 'research_proposal_only')", 'shield accepted target identity')
    domain=list(ast.literal_eval(shield.body[4].test.values[1].comparators[0]))
    step=methods['step']
    prefix=ast.parse('''if self.done:
    raise ValueError('episode already terminated')
proposal, reason = shield(action, enabled=self.enabled, valid=valid, ownership=ownership, elapsed_ms=elapsed_ms)
if proposal is not None and self.observation() is None:
    proposal, reason = None, 'invalid_observation_or_position'
if proposal is None:
    self.rejections += 1
    self.failure, self.done = reason, True
    return None, 0.0, True
''').body
    require([shape(n) for n in step.body[:4]]==[shape(n) for n in prefix], 'call admission dominators')
    schedule=step.body[7]
    require(ast.unparse(schedule)=='''if self.pending is None:
    self.pending = (self.t + 1 + self.execution.extra_delay, proposal)
else:
    self.rejections += 1''', 'pending preservation transfer')
    require(isinstance(step.body[9],ast.While), 'consumer loop position')
    trades=[]
    for n in ast.walk(step):
        if isinstance(n,ast.Call) and ast.unparse(n.func)=='self._trade':
            ancestors=[];p=n
            while p in parents:
                p=parents[p]
                if isinstance(p,ast.If): ancestors.append(ast.unparse(p.test))
            trades.append({'call':ast.unparse(n),'guards':ancestors})
    expected_trades=[{'call':'self._trade(self.pending[1])','guards':['self.failure is None and self.pending is not None and (self.pending[0] <= self.t)']},
                     {'call':'self._trade(0.0, terminal=True)','guards':['isfinite(self.equity) and self.equity > 0','terminal']}]
    require(trades==expected_trades, 'trade dominators/targets')
    terminal=[n for n in ast.walk(step) if isinstance(n,ast.If) and ast.unparse(n.test)=='terminal']
    require(len(terminal)==1 and ast.unparse(terminal[0].body[0])=='self.pending = None', 'terminal cancellation precedence')
    # No catches resume a partially failed step or helper into another trade.
    require(not any(isinstance(n,(ast.AsyncFunctionDef,ast.Await,ast.Yield,ast.YieldFrom)) for n in ast.walk(cls)), 'asynchronous replay control')
    require(not any(isinstance(n,(ast.Try,ast.TryStar)) for method in ('step','_trade','_risk') for n in ast.walk(methods[method])), 'resuming replay control')
    return {'status':'exhaustively_checked', 'constructors':actual['constructors'], 'consumers':actual['consumers'],
            'receiverUses':len(actual['receiverUses']), 'pendingAndEnableWrites':writes, 'tradeGuards':trades,
            'modules':surface['moduleCount'],'callbacks':surface['callbacks'],'actionDomain':domain,'researchImporters':research_importers(),
            'scope':'all reviewed consumers, local receiver ownership, actual admission/target dominators; trusted primitive effects'}


def origins(surface):
    require(len(surface['constructors'])==4 and len(surface['tradeGuards'])==2, 'missing extracted transfers')
    has, tagged, accepted, observed, due, clear, terminal, solvent = z.Bools('has tagged accepted observed due clear terminal solvent')
    old, proposal = z.Reals('old proposal')
    admissible=lambda v:z.Or(*[v == z.RealVal(str(a)) for a in surface['actionDomain']])
    inv=z.Implies(has,z.And(tagged,admissible(old)))
    gate=z.And(accepted,observed)
    # Exact source transfer: only a missing pending slot is assigned proposal;
    # no later current proposal may replace an inherited pending value.
    inserted=z.And(gate,z.Not(has))
    next_has=z.Or(has,inserted)
    next_tag=z.If(inserted,accepted,tagged)
    next_value=z.If(inserted,proposal,old)
    premise=z.And(inv,z.Implies(accepted,admissible(proposal)))
    queries=[(premise,z.Implies(next_has,z.And(next_tag,admissible(next_value)))),
             (z.And(premise,has),next_value==old),
             (z.And(premise,z.Not(gate)),z.And(next_has==has,next_value==old)),
             (z.And(premise,gate,next_has,due,clear),z.And(accepted,observed,next_tag,admissible(next_value))),
             (z.And(premise,gate,terminal,solvent),z.And(accepted,observed,admissible(z.RealVal(0)))),
             (premise,z.Implies(z.Not(accepted),z.Not(gate)))]
    for i,(p,c) in enumerate(queries): certify('shield-origin-'+str(i),p,c,[])
    i,j=z.Ints('owner other'); states=z.Array('owners',z.IntSort(),z.IntSort()); value=z.Int('new_state')
    certify('shield-owner-frame',i!=j,z.Select(z.Store(states,i,value),j)==z.Select(states,j),[])
    return {'F-RL-SHIELD-ORIGIN':'unsat'}


@dataclass(frozen=True)
class State:
    phase: str = 'idle'
    pending: int = 2  # 2 absent; -1/0/1 denote exact quarter targets
    tagged: bool = False
    delay: int = 0
    proposal: int = 0
    admitted: bool = False
    observed: bool = False


def local_steps(s):
    go=lambda phase,**kw:replace(s,phase=phase,**kw)
    if s.phase=='idle':
        return [('reject',go('done',admitted=False))]+[('accept',go('observe',proposal=a,admitted=True)) for a in (-1,0,1)]
    if s.phase=='observe': return [('bad-observation',go('done')),('observe',go('schedule',observed=True))]
    if s.phase=='schedule':
        if s.pending==2: return [('schedule',go('bar',pending=s.proposal,tagged=True,delay=d)) for d in (1,2)]
        return [('retain',go('bar'))]
    if s.phase=='bar':
        return [('bad-market',go('done')),('mark-risk-clear',go('fill',delay=max(0,s.delay-1))),('risk-stop',go('cancel'))]
    if s.phase=='fill':
        if s.pending!=2 and s.delay==0: return [('target-fill',go('after',pending=2,tagged=False))]
        return [('not-due',go('after'))]
    if s.phase=='after':
        return [('next-bar',go('bar')),('return',go('idle',admitted=False,observed=False)),('endpoint-or-risk-stop',go('cancel'))]
    if s.phase=='cancel': return [('cancel',go('liquidate',pending=2,tagged=False))]
    if s.phase=='liquidate': return [('liquidate-zero',go('done')),('insolvent-no-fill',go('done'))]
    if s.phase=='done': return [('fresh-instance',State())]
    raise ValueError('shield consumers: unknown phase')


def edge(s,event,n):
    require(n.pending==2 or n.tagged, 'pending without shield origin')
    if event=='target-fill': require(s.admitted and s.observed and s.tagged and s.pending in (-1,0,1) and s.delay==0 and s.phase=='fill', 'fill bypass')
    if event=='liquidate-zero': require(s.admitted and s.observed and s.pending==2 and s.phase=='liquidate', 'terminal bypass')
    if event in ('schedule','retain','mark-risk-clear','risk-stop','target-fill','liquidate-zero'): require(s.admitted and s.observed, 'effect before shield/observation')
    if event=='retain': require(n.pending==s.pending and n.tagged==s.tagged, 'pending replaced')


def model(transition=local_steps):
    initial=State();queue=deque([initial]);depth={initial:0};edges=0;labels=set()
    while queue:
        s=queue.popleft();out=transition(s);require(out,'deadlock')
        for event,n in out:
            edge(s,event,n);labels.add(event);edges+=1
            if n not in depth:depth[n]=depth[s]+1;queue.append(n)
    require({'target-fill','retain','reject','liquidate-zero','fresh-instance'}<=labels,'vacuous model')
    # Exhaustively check component transitions framed by all reachable other
    # states. This is the complete Cartesian asynchronous product, factored to
    # avoid materializing identical independent interleavings. The SMT frame rule
    # discharges the arbitrary identity case; both receiver orders are checked.
    frames=0
    for s,other in product(depth,repeat=2):
        for event,n in transition(s):
            for owner in (0,1):
                before=(s,other) if owner==0 else (other,s)
                after=(n,other) if owner==0 else (other,n)
                require(after[1-owner]==before[1-owner], 'instance interference');frames+=1
    return {'states':len(depth), 'transitions':edges,'maxShortestDepth':max(depth.values()),
            'twoInstanceProductStates':len(depth)**2,'twoInstanceProductTransitions':frames,
            'actions':[-.25,0,.25],'pendingDelays':[0,1,2],'instances':2,
            'search':'reachable fixed point; factored Cartesian asynchronous product; no call-depth cutoff',
            'scope':'numeric-free precedence abstraction; no production races, asynchronous revocation or liveness theorem'}


def conformance():
    import numpy as np
    sys.path.insert(0,str(ROOT/'scripts/research'))
    import sequential_env as e
    import sequential_evaluation as ev
    prices=np.full(180,100.);funding=np.zeros(180);scale=e.Scale.fit([prices[:121]])
    actual_shield=e.shield;actual_trade=e.Replay._trade;actual_step=e.Replay.step
    accepted={};events=[];steps=0;active=[];frontiers={}
    visible={'accept','reject','target-fill','liquidate-zero'}
    def hidden(states):
        seen=set(states);queue=deque(states)
        while queue:
            for event,n in local_steps(queue.popleft()):
                if event not in visible and event!='fresh-instance' and n not in seen:
                    seen.add(n);queue.append(n)
        return seen
    def emit(env,event,target=None):
        candidates=set()
        for state in hidden(frontiers[env]):
            for label,n in local_steps(state):
                if label==event and (event!='accept' or n.proposal==round(target*4)) and (event!='target-fill' or state.pending==round(target*4)):
                    edge(state,label,n);candidates.add(n)
        require(candidates,'actual trace outside model: '+event)
        frontiers[env]=hidden(candidates)
    def screened(action,**kw):
        result=actual_shield(action,**kw);env=active[-1]
        emit(env,'reject' if result[0] is None else 'accept',result[0])
        if result[0] is not None:
            accepted['current']=result[0];accepted.setdefault(env,set()).add(result[0])
        return result
    def traded(env,target,terminal=False):
        require('current' in accepted,'concrete fill before shield')
        require(target==0 if terminal else target in accepted.get(env,set()), 'concrete target lacks origin')
        emit(env,'liquidate-zero' if terminal else 'target-fill',target)
        events.append((float(target),terminal));return actual_trade(env,target,terminal)
    def stepped(env,action,**kw):
        nonlocal steps
        accepted.pop('current',None);frontiers.setdefault(env,{State()});active.append(env);steps+=1
        try: result=actual_step(env,action,**kw)
        finally: active.pop()
        pending=2 if env.pending is None else round(env.pending[1]*4)
        due=0 if env.pending is None else max(0,env.pending[0]-env.t)
        expected='done' if env.done else 'idle'
        states={s for s in hidden(frontiers[env]) if s.phase==expected and s.pending==pending and (pending==2 or s.delay==due)}
        require(states,'actual successor outside model');frontiers[env]=states
        return result
    def fresh(h=1,d=0,remaining=10):
        return e.Replay(prices,funding,24,25+remaining,h,scale,e.Execution(extra_delay=d),enabled=True)
    cases=0
    with patch.object(e,'shield',screened),patch.object(e.Replay,'_trade',traded),patch.object(e.Replay,'step',stepped):
        for target,h,d in product((-.25,0.,.25),(1,3,6),(0,1)):
            env=fresh(h,d)
            while not env.done: env.step(target)
            cases+=1
        for invalid in ({'action':float('nan')},{'action':.25,'valid':False},{'action':.25,'ownership':False},{'action':.25,'elapsed_ms':21.}):
            env=fresh(d=1);env.step(.25);old=(env.units,len(env.rows),len(events));env.step(**invalid)
            require((env.units,len(env.rows),len(events))==old and env.done,'invalid call consumed pending')
            cases+=1
        env=fresh(d=1);env.step(.25);env.enabled=False;before=len(events);env.step(-.25)
        require(len(events)==before and env.done,'disabled call consumed pending');cases+=1
        for d in (0,1):
            left,right=fresh(d=d),fresh(d=d)
            while not (left.done and right.done):
                if not left.done:left.step(-.25)
                if not right.done:right.step(.25)
            cases+=1
        # Real registered consumer functions: no learning, market reads or IO.
        e.collect({'BTC':prices},{'BTC':funding},scale,1,11,12)
        ev.replay_policy(prices,funding,24,36,1,scale,lambda obs:(.25,0.))
        from sequential_learning import Network
        ev.short_ope({'BTC':prices},{'BTC':funding},scale,1,24,60,Network(11),11,episodes=2)
        cases+=3
    require(any(t for _,t in events) and any(not t for _,t in events),'missing ordinary/terminal coverage')
    return {'cases':cases,'stepCalls':steps,'tradeCalls':len(events),'ordinaryTrades':sum(not t for _,t in events),
            'terminalTrades':sum(t for _,t in events),'trainingRuns':0,'marketDataReads':0,
            'scope':'actual shield, fill helper, collector, replay and OPE on synthetic constant prices; conformance not numeric proof'}


def check_consumers(promotion=None):
    surface=extract()
    if promotion is None:
        from promotion_boundary import check_promotion
        promotion=check_promotion()
    require(promotion['surface']['moduleCount']==surface['modules']==13,'complete consumer composition')
    require(promotion['composition']['productionRoots']==['analyze-close-timing','lstm-bench','merge-top-combos','optimize-equity','outbox-publisher','trader-hs'],'production boundary composition')
    return json.loads(json.dumps({'surface':surface,'smt':origins(surface),'model':model(),'conformance':conformance()},allow_nan=False))


if __name__=='__main__': print(json.dumps(check_consumers(),indent=2))
