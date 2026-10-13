"""Scoped composition claims, finite lifecycle and actual Python/Haskell witnesses."""
import ast
from collections import deque
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import z3 as z

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT/'formal/research/rl-composition-source.json'


def require(ok, why):
    if not ok: raise ValueError('RL composition: '+why)


def extract():
    lock=json.loads(LOCK.read_text())
    for name,digest in lock['sha256'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'source drift: '+name)
    tree=ast.parse((ROOT/'scripts/research/rl_composition.py').read_text())
    names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    require(not names & {'open','eval','exec','socket','requests','subprocess','os','pickle'},'effect drift')
    require('reward = float(100 * delta)' in ast.unparse(tree),'economic reward drift')
    require('pairs = batch_v2(rows, 1.0, enabled=True)' in ast.unparse(tree),'GAE discount drift')
    require('self.session is self._owned' in ast.unparse(tree),'owned-state guard missing')
    return {'status':'exhaustively_checked','files':len(lock['sha256']),
        'scope':'reviewed complete source digests and direct effect names; trusted transitive NumPy/kernel/runtime, not OS capability security'}


def lemma(premise, conclusion):
    solver=z.Solver();solver.set(timeout=10000);solver.add(premise)
    require(solver.check()==z.sat,'vacuous or unknown premises')
    solver.add(z.Not(conclusion))
    require(solver.check()==z.unsat,'counterexample or unknown')


def prove():
    proposal,permitted,equity,gross,funding,cost,after=z.Reals('p a e g f c n')
    supported=z.Bool('support')
    domain=z.Or(proposal==-z.RealVal(1)/4,proposal==0,proposal==z.RealVal(1)/4)
    lemma(z.And(domain,permitted==z.If(supported,proposal,0)),z.And(permitted>=-z.RealVal(1)/4,permitted<=z.RealVal(1)/4))
    lemma(z.And(cost>=0,after==equity+gross+funding-cost),100*(after-equity)==100*(gross+funding-cost))
    e0,e1,e2=z.Reals('e0 e1 e2')
    lemma(z.BoolVal(True),100*(e1-e0)+100*(e2-e1)==100*(e2-e0))
    decision,close,available=z.Ints('decision close available')
    lemma(z.And(0<=decision,decision<close,close<=available),decision<available)
    start,stop,cutoff=z.Ints('start stop cutoff')
    lemma(z.And(start>=96,start<cutoff-32,stop==start+33),stop<=cutoff)
    due,pending,new=z.Bools('due pending new')
    admitted=z.And(new,z.Not(pending)); consumed=z.And(pending,due)
    count=z.If(pending,1,0)+z.If(admitted,1,0)-z.If(consumed,1,0)
    lemma(z.BoolVal(True),z.And(count>=0,count<=1))
    # A deterministic projection's action mass is the sum over its preimage.
    p0,p1,p2=z.Reals('p0 p1 p2')
    lemma(z.And(p0>=0,p1>=0,p2>=0,p0+p1+p2==1),z.If(supported,p1,p0+p1+p2)==z.If(supported,p1,1))
    return {'F-RL-COMPOSE-ARITH':'unsat'}


def model():
    # A single-process owner with two competing abstract proposals, one pending
    # slot, retry/consume/terminal; no live capability or promotion edge exists.
    initial=('ready',None,0,False)
    seen={initial};queue=deque([initial]);edges=0
    while queue:
        phase,owner,tick,authority=queue.popleft()
        require(authority is False and owner in (None,0,1),'authorization/ownership')
        if phase in ('disabled','terminal'): nxt=[]
        else:
            nxt=[('disabled',None,tick,False),('terminal',None,tick,False),(phase,owner,tick,False)]
            if tick<8:
                if owner is None: nxt += [('ready',p,tick,False) for p in (0,1)]
                else: nxt += [('ready',None,tick+1,False)]
        for state in nxt:
            edges+=1
            if state not in seen: seen.add(state);queue.append(state)
    require(any(t==8 and phase=='terminal' for phase,_,t,_ in seen),'unreachable terminal')
    return {'status':'model_checked','states':len(seen),'transitions':edges,'stepBound':8,
        'scope':'finite single-owner reservation/lifecycle, two proposal identities; no production concurrency or fairness proof'}


def conformance():
    sys.path.insert(0,str(ROOT/'scripts/research'))
    import rl_composition as r
    bars=tuple(r.causal.Bar('FIXTURE',i*1000,i*1000+1,F(10000+(i%7)*10,100)) for i in range(130))
    events=tuple(() for _ in bars)
    ep=r.Episode(bars,events,96,105,enabled=True)
    original=ep.session
    wire=[];expected=[]
    for j in range(97,105):
        before=ep.session
        action=2 if j%2 else 1
        pair=r.causal.step_v3(original,bars[j],(),r.ACTIONS[action],terminal=j==104,enabled=True)
        require(pair is not None,'reference rejection')
        ep.step(action)
        require(ep.session==pair[0],'cached construction differs from reference')
        receipt=pair[1]; c=receipt.costs
        values=(receipt.before.equity,receipt.before.units,receipt.before.price,receipt.after.price,
                F(0),c.fee,c.spread,c.slippage,c.impact)
        wire.append(str([(v.numerator,v.denominator) for v in values]))
        expected.append(str((receipt.after.equity.numerator,receipt.after.equity.denominator)).replace(' ',''))
        original=pair[0]
    out=subprocess.run(['runghc',str(ROOT/'formal/research/ReplayAccountingV2.hs')],
        input='\n'.join(wire)+'\n',capture_output=True,text=True,check=True,timeout=120)
    require(out.stdout.splitlines()==expected,'Haskell ledger mismatch')
    return {'status':'property_tested','referenceTransitions':8,'haskellLedgerRows':8,
            'historicalDataReads':0,'scope':'exact selected Python/kernel/Haskell differential cases, not compiler refinement'}


def check_composed():
    admission=check_admission()
    return {'source':extract(),'smt':{**prove(),'F-RL-ADMISSION-V2':'unsat'},
            'model':model(),'conformance':conformance(),'admissionV2':admission}


def check_admission():
    # Exact real/rational algebra; bounded Fraction implementation correspondence
    # is through conformance, not a proof about arbitrary Python execution.
    exposure,fill=z.Reals('admission_exposure admission_fill')
    magnitude=z.If(exposure>=0,exposure,-exposure)
    lemma(z.And(magnitude<=z.RealVal(7)/20,fill>0,fill<=1),
          fill*magnitude<=z.RealVal(1)/2)
    rate=z.Real('admission_cost_rate')
    denominator=1-rate*fill*magnitude
    lemma(z.And(magnitude<=z.RealVal(7)/20,fill>0,fill<=1,
                rate>=0,rate<=z.RealVal(1)/200),
          z.And(denominator>0,(1-fill)*magnitude<=z.RealVal(7)/20*denominator))
    sys.path.insert(0,str(ROOT/'scripts/research'))
    import rl_admission_v2 as v2
    r=v2.v1
    bars=tuple(r.causal.Bar('FIXTURE',i*1000,i*1000+1,F(10000+(i%7)*10,100)) for i in range(140))
    events=tuple(() for _ in bars)
    old=r.Episode(bars,events,96,129,enabled=True)
    new=v2.Episode(bars,events,96,129,enabled=True)
    rejected=0
    while not new.done:
        original=new.session
        action=0 if new.index%2 else 2
        if not old.done: old.step(action)
        new.step(action)
        row=new.rows[-1]
        if row['turnoverRejected']:
            expected=r.advance_owned(original,bars[new.index],(),F(0),terminal=False,
                                     multiplier=F(1),fill=F(1),enabled=True)
            require(expected is not None and new.session==expected[0], 'rejected preview committed')
            require(not new.done and row['applied']==0,'turnover reset')
            rejected+=1
    require(old.failure=='turnover_limit' and len(old.rows)==3,'counterexample drift')
    require(len(new.rows)==32 and new.failure is None and rejected==13,'continuation witness')
    return {'status':'smt_verified_and_conformance_tested','lemmas':2,
            'oldCounterexampleSteps':3,'continuedSteps':32,'rejectedPreviews':rejected,
            'scope':'zero-target turnover and post-cost exposure bounds under pre-fill exposure <=0.35, fill (0,1], linear cost <=50bp; exact selected immutable-preview witnesses. Capital/drawdown risk and broad Python/Haskell refinement remain outside proof.'}


if __name__=='__main__': print(json.dumps(check_composed(),indent=2))
