"""Conditional value-objective algebra and prescribed binary64 witnesses."""
import ast
import copy
from fractions import Fraction
from pathlib import Path
import z3 as z
from causal_footprint import structure
from target_v2 import certify
from terminal_numerics import fixed_fp, finite

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/sequential_learning.py'
TEMPLATE = '''
def bellman_gradient(q: np.ndarray, actions: np.ndarray, targets: np.ndarray,
                     alpha: float) -> tuple[float, np.ndarray]:
    n, idx = len(actions), np.arange(len(actions))
    residual = RESIDUAL
    grad = np.zeros_like(q)
    grad[idx, actions] = SELECTED
    m = q.max(1)
    logsumexp = m + np.log(np.exp(q - m[:, None]).sum(1))
    penalty = softmax(q)
    penalty[idx, actions] -= 1
    grad += CONSERVATIVE
    loss = LOSS
    return float(loss), grad
'''
TARGET = '''
greedy = np.argmax(net.forward(nxt), axis=1)
targets = buffer["r"][idx] + 0.99**horizon * (~buffer["done"][idx]) * target.forward(nxt)[np.arange(64), greedy]
'''


def extract(source):
    tree = ast.parse(source)
    def unique(name):
        matches=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(matches)!=1:
            raise ValueError('value objective: missing/duplicate function')
        return matches[0]
    function=copy.deepcopy(unique('bellman_gradient'))
    expressions={}
    try:
        for index,name in ((1,'residual'),(3,'selected'),(8,'conservative'),(9,'loss')):
            expressions[name]=function.body[index].value
            function.body[index].value=ast.Name(id=name.upper(),ctx=ast.Load())
    except (AttributeError,IndexError):
        raise ValueError('value objective: loss skeleton drift')
    if structure(function)!=structure(ast.parse(TEMPLATE).body[0]):
        raise ValueError('value objective: loss skeleton drift')
    # Scope is this unique contiguous expression slice, not full train_q refinement.
    train=unique('train_q'); matches=[]
    for node in ast.walk(train):
        body=getattr(node,'body',None)
        if isinstance(body,list):
            for i,statement in enumerate(body[:-1]):
                if (isinstance(statement,ast.Assign) and len(statement.targets)==1 and
                    isinstance(statement.targets[0],ast.Name) and statement.targets[0].id=='greedy'):
                    matches.append(body[i:i+2])
    expected=ast.parse(TARGET).body
    if len(matches)!=1 or [structure(n) for n in matches[0]]!=[structure(n) for n in expected]:
        raise ValueError('value objective: target slice drift')
    expressions['target']=matches[0][1].value
    return expressions


def arithmetic(node,values,floating=False):
    key=ast.unparse(node)
    if key in values:
        return values[key]
    if isinstance(node,ast.Constant) and type(node.value) in (int,float):
        return fixed_fp(float(node.value).hex()) if floating else z.RealVal(str(Fraction(node.value)))
    if isinstance(node,ast.Call) and ast.unparse(node.func)=='np.mean' and len(node.args)==1 and not node.keywords:
        a=arithmetic(node.args[0],values,floating);b=values['n']
        return z.fpDiv(z.RNE(),a,b) if floating else a/b
    if isinstance(node,ast.BinOp):
        a=arithmetic(node.left,values,floating)
        if isinstance(node.op,ast.Pow) and isinstance(node.right,ast.Constant) and node.right.value==2:
            return z.fpMul(z.RNE(),a,a) if floating else a*a
        b=arithmetic(node.right,values,floating)
        if type(node.op) in (ast.Add,ast.Sub,ast.Mult,ast.Div):
            if floating:
                return {ast.Add:z.fpAdd,ast.Sub:z.fpSub,ast.Mult:z.fpMul,ast.Div:z.fpDiv}[type(node.op)](z.RNE(),a,b)
            return {ast.Add:lambda:a+b,ast.Sub:lambda:a-b,ast.Mult:lambda:a*b,ast.Div:lambda:a/b}[type(node.op)]()
    raise ValueError('value objective: unsupported arithmetic '+key)


def row(expressions,q,y,alpha,n,pminus,lse,floating=False):
    values={'q[idx, actions]':q,'targets':y,'alpha':alpha,'n':n,'penalty':pminus,'logsumexp':lse}
    for name in ('residual','selected','conservative','loss'):
        values[name]=arithmetic(expressions[name],values,floating)
    return values


def first_max(q):
    return z.If(z.And(q[0]>=q[1],q[0]>=q[2]),0,z.If(q[1]>=q[2],1,2))


def check_values(fixtures,source=None):
    expressions=extract((ROOT/SOURCE).read_text() if source is None else source)
    online=z.Reals('value_o0 value_o1 value_o2');target=z.Reals('value_t0 value_t1 value_t2')
    r,g=z.Reals('value_reward value_discount');done=z.Bool('value_done')
    chosen=first_max(online)
    selected=z.If(chosen==0,target[0],z.If(chosen==1,target[1],target[2]))
    values={'buffer[\'r\'][idx]':r,'0.99 ** horizon':g,'~buffer[\'done\'][idx]':z.If(done,0,1),
            'target.forward(nxt)[np.arange(64), greedy]':selected}
    actual=arithmetic(expressions['target'],values)
    discounts=[z.RealVal(str(Fraction(.99**h))) for h in (1,3,6)]
    premise=z.Or(*[g==v for v in discounts])
    first=z.And(*[z.Implies(chosen==i,z.And(*[online[i]>=v for v in online],
                          *[online[j]<online[i] for j in range(i)])) for i in range(3)])
    reference=z.If(done,r,r+g*selected)
    certify('F-RL-DOUBLE-TARGET',premise,z.And(first,actual==reference),[])
    q,y,alpha=z.Reals('value_q value_y value_alpha');n=z.Int('value_n');nr=z.ToReal(n)
    ps=z.Reals('value_p0 value_p1 value_p2');action=z.Int('value_action');lse=z.Real('value_lse')
    premise=z.And(n>=1,n<=256,alpha>=0,action>=0,action<=2,*[p>=0 for p in ps],sum(ps)==1)
    rows=[row(expressions,q,y,alpha,nr,p-z.If(action==j,1,0),lse) for j,p in enumerate(ps)]
    gradients=[z.If(action==j,v['selected'],0)+v['conservative'] for j,v in enumerate(rows)]
    formula=[z.If(action==j,(q-y)/nr,0)+alpha*(p-z.If(action==j,1,0))/nr for j,p in enumerate(ps)]
    conclusions=[actual==expected for actual,expected in zip(gradients,formula)]
    conclusions.extend([sum(gradients)==(q-y)/nr,sum(v['conservative'] for v in rows)==0])
    conclusions.extend(z.And(v['conservative']>=-alpha/nr,v['conservative']<=alpha/nr) for v in rows)
    # LSE itself and its analytic derivative are assumptions, not Z3 transcendental proofs.
    conclusions.append(rows[0]['loss']==(q-y)*(q-y)/(2*nr)+alpha*(lse-q)/nr)
    certify('F-RL-CQL-GRADIENT',premise,z.And(*conclusions),[])
    if fixtures.get('schemaVersion')!=1 or [e['id'] for e in fixtures['entries']]!=['CE-RL-014','CE-RL-015']:
        raise ValueError('value objective: counterexample roster drift')
    witnesses=[]
    for entry in fixtures['entries']:
        q,y,alpha,ell=[fixed_fp(entry[k]) for k in ('selectedQHex','targetHex','alphaHex','logIntermediateHex')]
        maximum=fixed_fp(entry['maximumHex']);one=fixed_fp(float(1).hex())
        lse=z.fpAdd(z.RNE(),maximum,ell)
        current=row(expressions,q,y,alpha,one,fixed_fp(float(0).hex()),lse,True)
        solver=z.Solver();solver.set(timeout=10000,random_seed=0)
        solver.add(*[finite(v) for v in (q,y,alpha,ell,maximum)],z.fpEQ(current['residual'],fixed_fp(float(0).hex())))
        if entry['id']=='CE-RL-014':
            solver.add(z.fpIsNaN(current['loss']))
        else:
            zero=fixed_fp(float(0).hex())
            original=row(expressions,zero,zero,alpha,one,zero,ell,True)
            solver.add(z.fpGT(ell,one),z.fpLT(ell,fixed_fp(float(2).hex())),
                       z.fpEQ(current['loss'],fixed_fp(entry['shiftedLossHex'])),
                       z.fpEQ(original['loss'],fixed_fp(entry['originalLossHex'])),
                       z.Not(z.fpEQ(current['loss'],original['loss'])))
        result=solver.check()
        if result!=z.sat:
            raise RuntimeError('value objective: prescribed witness not SAT '+entry['id']+' '+str(result))
        witnesses.append({'id':entry['id'],'result':'sat'})
    return {'smt':{'F-RL-DOUBLE-TARGET':'unsat','F-RL-CQL-GRADIENT':'unsat'},
            'queries':2,'premiseChecks':2,'actions':3,'horizons':[1,3,6],'batchBounds':[1,256],
            'counterexamples':witnesses,'transcendentalsVerified':False,'fullTrainingRefinement':False,
            'policyLowerBoundVerified':False,'binary64WitnessRounding':'separate RNE; prescribed log intermediate; one row'}
