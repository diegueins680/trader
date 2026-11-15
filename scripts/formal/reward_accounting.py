"""Source-linked real accounting; bounded float conformance, not runtime proof."""
import ast
import hashlib
import json
from itertools import product
from math import isclose, prod
from pathlib import Path

import z3 as z
from target_v2 import certify

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = 'research-notes/registrations/reward-accounting-audit-engineering.json'
REGISTRATION_SHA256 = 'aa7677a5631e9080487b7ead87f16ca5aa45780abc35f243e874f75835a1e34b'
SOURCES = ('scripts/research/sequential_env.py','scripts/research/sequential_evaluation.py')
COSTS = ('fee','spread','slippage','impact')


def registration():
    raw = (ROOT/REGISTRATION).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTRATION_SHA256:
        raise ValueError('reward accounting: registration drift')
    return json.loads(raw)


def only(nodes):
    if len(nodes) != 1:
        raise ValueError('reward accounting: missing/repeated expression')
    return nodes[0]


def extract(sources, hashes):
    functions = {}
    for source in sources:
        for node in ast.parse(source).body:
            if isinstance(node,ast.ClassDef) and node.name == 'Replay':
                functions.update({'Replay.'+n.name:n for n in node.body if isinstance(n,ast.FunctionDef)})
            if isinstance(node,ast.FunctionDef): functions[node.name] = node
    for key,expected in hashes.items():
        if key not in functions or hashlib.sha256(ast.dump(functions[key],include_attributes=False).encode()).hexdigest() != expected:
            raise ValueError('reward accounting: source AST drift '+key)
    step,trade = functions['Replay.step'],functions['Replay._trade']
    augment = lambda fn,name,op: only([n.value for n in ast.walk(fn) if isinstance(n,ast.AugAssign) and
                                      ast.unparse(n.target) == name and isinstance(n.op,op)])
    row = only([n for n in ast.walk(step) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='rewardPenalty' for k in n.keys)])
    row_fields = {k.value:v for k,v in zip(row.keys,row.values) if isinstance(k,ast.Constant)}
    merge = only([n.value for n in ast.walk(step) if isinstance(n,ast.DictComp)])
    reward = only([n.value for n in ast.walk(step) if isinstance(n,ast.Assign) and
                   len(n.targets)==1 and ast.unparse(n.targets[0])=='reward'])
    initial = only([n.value for n in ast.walk(step) if isinstance(n,ast.Assign) and
                    len(n.targets)==1 and ast.unparse(n.targets[0])=='(before, penalty)'])
    if not isinstance(initial,ast.Tuple) or len(initial.elts)!=2:
        raise ValueError('reward accounting: penalty initializer shape')
    metric = only([v for n in ast.walk(functions['economic']) if isinstance(n,ast.Dict)
                   for k,v in zip(n.keys,n.values) if isinstance(k,ast.Constant) and k.value=='netReturn'])
    return {'mark':augment(step,'self.equity',ast.Add),'debit':augment(trade,'self.equity',ast.Sub),
            'merge':merge,'net':row_fields['net'],'rowPenalty':row_fields['rewardPenalty'],
            'penalty':augment(step,'penalty',ast.Add),'initialPenalty':initial.elts[1],
            'reward':reward,'metric':metric}


def real(node, atoms):
    key = ast.unparse(node)
    if key in atoms: return atoms[key]
    if isinstance(node,ast.Constant) and type(node.value) in (int,float):
        return z.RealVal(str(node.value))
    if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
        a,b = real(node.left,atoms),real(node.right,atoms)
        if isinstance(node.op,ast.Pow):
            if not isinstance(node.right,ast.Constant) or node.right.value != 2:
                raise ValueError('reward accounting: unsupported exponent')
            return a*a
        return {ast.Add:lambda:a+b,ast.Sub:lambda:a-b,ast.Mult:lambda:a*b,ast.Div:lambda:a/b}[type(node.op)]()
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='sum' and not node.keywords:
        gen = only(node.args)
        if not isinstance(gen,ast.GeneratorExp): raise ValueError('reward accounting: expected cost generator')
        comp = only(gen.generators)
        if (not isinstance(comp.iter,ast.Tuple) or comp.ifs or comp.is_async or
            ast.unparse(comp.target)!='k' or ast.unparse(gen.elt)!='terms[k]'):
            raise ValueError('reward accounting: unsupported cost generator')
        if not comp.iter.elts or any(not isinstance(k,ast.Constant) or k.value not in COSTS for k in comp.iter.elts):
            raise ValueError('reward accounting: unsupported cost key')
        return z.Sum(*[atoms['terms.'+k.value] for k in comp.iter.elts])
    raise ValueError('reward accounting: unsupported expression '+key)


def certificates(nodes):
    e,g,f = z.Reals('acct_e acct_g acct_f')
    c = dict(zip(COSTS,z.Reals('acct_c_fee acct_c_spread acct_c_slip acct_c_impact')))
    l = dict(zip(COSTS,z.Reals('acct_l_fee acct_l_spread acct_l_slip acct_l_impact')))
    debit = lambda terms: real(nodes['debit'],{'terms.'+k:v for k,v in terms.items()})
    ending = e+real(nodes['mark'],{'gross':g,'funding':f})-debit(c)-debit(l)
    merged = [real(nodes['merge'],{'costs[k]':c[k],'liquidation[k]':l[k]}) for k in COSTS]
    net = real(nodes['net'],{'self.equity':ending,'old_equity':e})
    certify('F-RL-ROW-RECONCILE',z.And(e>0,*[v>=0 for v in [*c.values(),*l.values()]]),
            z.And(ending==e+g+f-z.Sum(*merged),net==(g+f-z.Sum(*merged))/e),
            [e==1,g==0,f==0,*[v==0 for v in [*c.values(),*l.values()]]])
    initial,current,nxt,acc = z.Reals('fold_initial fold_current fold_next fold_acc')
    metric = real(nodes['metric'],{'equity[-1]':nxt})
    certify('F-RL-WEALTH-FOLD/base',initial>0,z.RealVal(1)==initial/initial,[initial==1])
    r = real(nodes['net'],{'self.equity':nxt,'old_equity':current})
    certify('F-RL-WEALTH-FOLD/step',z.And(initial>0,current>0,nxt>0,acc==current/initial),
            acc*(1+r)==nxt/initial,[initial==1,current==1,nxt==2,acc==1])
    certify('F-RL-WEALTH-FOLD/report',z.And(initial==1,nxt>0),metric==nxt/initial-1,[nxt==2])
    before,end,k,q,s,w,v = z.Reals('reward_before reward_end reward_k reward_q reward_s reward_w reward_v')
    term = real(nodes['penalty'],{'exposure':w,'x[5]':v})
    row_penalty = real(nodes['rowPenalty'],{'self.execution.risk_penalty':k,'exposure':w,'x[5]':v})
    initial_penalty = real(nodes['initialPenalty'],{})
    certify('F-RL-REWARD-RECONCILE/base',k>=0,
            z.And(initial_penalty>=0,z.RealVal(0)==100*k*initial_penalty),[k==0])
    certify('F-RL-REWARD-RECONCILE/step',z.And(k>=0,q>=0,s==100*k*q),
            z.And(q+term>=0,s+row_penalty==100*k*(q+term)),[k==1,q==0,s==0,w==1,v==1])
    reward = real(nodes['reward'],{'self.equity':end,'before':before,'self.execution.risk_penalty':k,'penalty':q})
    certify('F-RL-REWARD-RECONCILE/call',z.And(before>0,k>=0,q>=0,s==100*k*q),
            reward+s==100*(end/before-1),[before==1,end==1,k==0,q==0,s==0])
    return {'F-RL-ROW-RECONCILE':'unsat','F-RL-WEALTH-FOLD':'unsat','F-RL-REWARD-RECONCILE':'unsat'}


def nonadditive_certificate(nodes):
    r1,r2 = z.Reals('nonadditive_r1 nonadditive_r2')
    first = 1+r1; last = first*(1+r2)
    reward = lambda before,end: real(nodes['reward'],{'before':before,'self.equity':end,
                                                      'self.execution.risk_penalty':z.RealVal(0),'penalty':z.RealVal(0)})
    total = reward(z.RealVal(1),first)+reward(first,last)
    solver = z.Solver(); solver.set(timeout=10000,random_seed=0)
    solver.add(r1==z.RealVal('1/40'),r2==z.RealVal('-1/40'),first>0,last>0,
               total==0,last==z.RealVal('1599/1600'),total!=100*(last-1))
    if solver.check()!=z.sat:
        raise RuntimeError('reward accounting: prescribed nonadditivity witness failed/unknown')
    return {'requirement':'F-RL-REWARD-ADDITIVE','status':'refuted','solverResult':'sat',
            'summedReward':'0','finalEquity':'1599/1600','economicReturn':'-1/1600',
            'scope':'interpretation counterexample; not a defect in current economic reporting'}


def check_episode(env, target, tolerance, valid=True):
    from sequential_evaluation import economic
    rewards = []; positive = True; max_cash_error = 0.; max_reward_error = 0.
    while not env.done:
        before = env.equity; left = len(env.rows)
        _,reward,_ = env.step(target,valid=valid)
        rows = env.rows[left:]; penalties = sum(row['rewardPenalty'] for row in rows)
        expected = 100*(env.equity/before-1)-penalties
        if not isclose(reward,expected,rel_tol=tolerance,abs_tol=tolerance):
            raise RuntimeError('reward accounting: call reward mismatch')
        max_reward_error = max(max_reward_error,abs(reward-expected)); rewards.append(float(reward))
        previous = before
        for row in rows:
            cash_equity = previous+row['gross']+row['funding']-sum(row[k] for k in COSTS)
            if not isclose(row['equity'],cash_equity,rel_tol=tolerance,abs_tol=tolerance):
                raise RuntimeError('reward accounting: row cash mismatch')
            max_cash_error = max(max_cash_error,abs(row['equity']-cash_equity))
            positive = positive and row['equity']>0
            previous = row['equity']
    compounded = prod(1+row['net'] for row in env.rows)-1
    report = economic(env)
    if not isclose(compounded,env.equity-1,rel_tol=tolerance,abs_tol=tolerance):
        raise RuntimeError('reward accounting: compounded return mismatch')
    if env.rows and not isclose(report['netReturn'],env.equity-1,rel_tol=tolerance,abs_tol=tolerance):
        raise RuntimeError('reward accounting: economic report mismatch')
    return {'rewards':rewards,'equity':float(env.equity),'netReturn':report.get('netReturn'),
            'rows':len(env.rows),'failure':env.failure,'positiveWealth':bool(positive),
            'cashError':float(max_cash_error),'rewardError':float(max_reward_error)}


def replay_case(target,horizon,delay,remaining,cost,funding,pattern,penalty,scenario='ordinary'):
    import sys
    import numpy as np
    sys.path.insert(0,str(ROOT/'scripts/research'))
    from sequential_env import Replay,Scale,Execution
    prices = np.full(40,100.); settlements = np.zeros(40)
    for i in range(1,remaining+1):
        prices[24+i] = 100. if pattern=='flat' else 100.*(1+.01*(-1)**i)
        settlements[24+i] = funding
    env = Replay(prices,settlements,24,25+remaining,horizon,Scale.fit([prices[:25]]),
                 Execution(extra_delay=delay,cost_multiplier=cost,risk_penalty=penalty),enabled=True)
    if scenario=='invalid_market': prices[25]=np.nan
    if scenario in ('solvent_risk','insolvency'):
        env.units=.0025; settlements[25]=100. if scenario=='solvent_risk' else 800.
    return env


def check_conformance(reg):
    c = reg['conformance']; tol = c['absoluteTolerance']; count = 0; cash_error = reward_error = 0.
    for args in product(c['targets'],c['horizons'],c['extraDelays'],c['remainingBars'],c['costMultipliers'],
                        c['fundingCoefficients'],c['pricePatterns'],c['riskPenalties']):
        result = check_episode(replay_case(*args),args[0],tol)
        if result['failure'] is not None or not result['positiveWealth']:
            raise RuntimeError('reward accounting: unexpected synthetic grid failure')
        cash_error=max(cash_error,result['cashError']); reward_error=max(reward_error,result['rewardError']); count+=1
    failures = {name:check_episode(replay_case(.25,1,0,3,1.,0.,'flat',.01,name),.25,tol,name!='invalid_gate')
                for name in c['invalidScenarios']}
    env = replay_case(.25,1,0,3,0.,0.,'flat',0.)
    env.prices[25:28] = [100.,110.,99.]
    witness = check_episode(env,.25,tol)
    if not (isclose(sum(witness['rewards']),0.,abs_tol=tol) and
            isclose(witness['equity'],1599/1600,abs_tol=tol) and
            abs(sum(witness['rewards'])-100*witness['netReturn'])>.06):
        raise RuntimeError('reward accounting: actual nonadditivity witness mismatch')
    return {'gridEpisodes':count,'failureScenarios':failures,'nonadditiveReplay':witness,
            'absoluteTolerance':tol,'relativeTolerance':c['relativeTolerance'],
            'cashResidualWithinTolerance':cash_error<tol,'rewardResidualWithinTolerance':reward_error<tol,
            'scope':'deterministic synthetic tests; no universal binary64 error bound'}


def check_reward_accounting():
    reg=registration(); nodes=extract([(ROOT/p).read_text() for p in SOURCES],reg['sourceFunctionASTSha256'])
    return {'smt':certificates(nodes),'nonadditivity':nonadditive_certificate(nodes),
            'conformance':check_conformance(reg),'satUnsatPairs':7,
            'scope':'exact-real source arithmetic and induction obligations; no full runtime refinement'}
