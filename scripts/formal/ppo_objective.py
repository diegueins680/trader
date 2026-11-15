"""Source-linked PPO surrogate algebra and prescribed numeric counterexamples."""
import ast
import copy
from fractions import Fraction
from pathlib import Path
import z3 as z
from causal_footprint import structure
from target_v2 import certify
from terminal_numerics import finite, fixed_fp

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/sequential_learning.py'
TEMPLATE = '''
def ppo_gradient(logits: np.ndarray, actions: np.ndarray, old_prob: np.ndarray,
                 advantage: np.ndarray) -> tuple[float, np.ndarray]:
    probs = softmax(logits)
    idx = np.arange(len(actions))
    ratio = RATIO
    clipped = CLIPPED
    loss = LOSS
    active = ACTIVE
    gradient = probs.copy()
    gradient[idx, actions] -= 1
    gradient *= COEFFICIENT[:, None]
    return float(loss), gradient
'''


def extract(source):
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name=='ppo_gradient']
    if len(functions)!=1:
        raise ValueError('PPO objective: missing/duplicate function')
    function = copy.deepcopy(functions[0])
    expressions = {}
    try:
        for index,name in ((2,'ratio'),(3,'clipped'),(4,'loss'),(5,'active')):
            expressions[name] = function.body[index].value
            function.body[index].value = ast.Name(id=name.upper(),ctx=ast.Load())
        expressions['coefficient'] = function.body[8].value.value
        function.body[8].value.value = ast.Name(id='COEFFICIENT',ctx=ast.Load())
    except (AttributeError,IndexError):
        raise ValueError('PPO objective: control-flow drift')
    if structure(function)!=structure(ast.parse(TEMPLATE).body[0]):
        raise ValueError('PPO objective: source skeleton drift')
    return expressions


def scalar(node, values, floating=False):
    def number(value):
        if floating:
            return fixed_fp(float(value).hex())
        q = Fraction(value)
        return z.RealVal(q.numerator)/z.RealVal(q.denominator)
    def numeric(value):
        return z.If(value,number(1),number(0)) if z.is_bool(value) else value
    def binary(op,a,b):
        a,b = numeric(a),numeric(b)
        if floating:
            return {ast.Add:z.fpAdd,ast.Sub:z.fpSub,ast.Mult:z.fpMul,ast.Div:z.fpDiv}[op](z.RNE(),a,b)
        return {ast.Add:lambda:a+b,ast.Sub:lambda:a-b,ast.Mult:lambda:a*b,ast.Div:lambda:a/b}[op]()
    def compare(op,a,b):
        if floating:
            return {ast.Lt:z.fpLT,ast.LtE:z.fpLEQ,ast.Gt:z.fpGT,ast.GtE:z.fpGEQ}[op](a,b)
        return {ast.Lt:lambda:a<b,ast.LtE:lambda:a<=b,ast.Gt:lambda:a>b,ast.GtE:lambda:a>=b}[op]()
    key = ast.unparse(node)
    if key in values:
        return values[key]
    if isinstance(node,ast.Constant) and type(node.value) in (int,float):
        return number(node.value)
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
        value = scalar(node.operand,values,floating)
        return z.fpNeg(value) if floating else -value
    if isinstance(node,ast.BinOp):
        a,b = scalar(node.left,values,floating),scalar(node.right,values,floating)
        if isinstance(node.op,(ast.BitAnd,ast.BitOr)):
            return z.And(a,b) if isinstance(node.op,ast.BitAnd) else z.Or(a,b)
        if type(node.op) in (ast.Add,ast.Sub,ast.Mult,ast.Div):
            return binary(type(node.op),a,b)
    if isinstance(node,ast.Compare) and len(node.ops)==1 and type(node.ops[0]) in (ast.Lt,ast.LtE,ast.Gt,ast.GtE):
        return compare(type(node.ops[0]),scalar(node.left,values,floating),scalar(node.comparators[0],values,floating))
    if isinstance(node,ast.Call) and not node.keywords:
        name = ast.unparse(node.func)
        args = [scalar(arg,values,floating) for arg in node.args]
        if name=='np.clip' and len(args)==3:
            x,lo,hi = args
            return z.If(compare(ast.Lt,x,lo),lo,z.If(compare(ast.Gt,x,hi),hi,x))
        if name=='np.minimum' and len(args)==2:
            a,b = args
            out = z.If(compare(ast.Lt,a,b),a,b)
            return z.If(z.Or(z.fpIsNaN(a),z.fpIsNaN(b)),z.fpNaN(z.Float64()),out) if floating else out
        if isinstance(node.func,ast.Attribute) and node.func.attr=='mean' and not args:
            return binary(ast.Div,scalar(node.func.value,values,floating),values['len(actions)'])
    raise ValueError('PPO objective: unsupported expression '+key)


def evaluate(expressions,p,b,a,n,floating=False,ratio=None):
    values = {'probs[idx, actions]':p,'old_prob':b,'advantage':a,'len(actions)':n}
    for key in ('ratio','clipped','loss','active','coefficient'):
        values[key] = ratio if key=='ratio' and ratio is not None else scalar(expressions[key],values,floating)
    return values


def check_ppo(fixtures,source=None):
    expressions = extract((ROOT/SOURCE).read_text() if source is None else source)
    p,b,a,r = z.Reals('ppo_p ppo_b ppo_a ppo_r')
    n = z.Int('ppo_n'); nr=z.ToReal(n)
    lo,hi = [z.RealVal(str(Fraction(x))) for x in (.8,1.2)]
    actual = evaluate(expressions,p,b,a,nr)
    certify('F-RL-PPO-OBJECTIVE-ratio',z.And(p>=0,p<=1,b>0,b<=1),actual['ratio']==p/b,[])
    # Ratio lemma justifies scalar r abstraction. Mean is the row contribution.
    values = evaluate(expressions,p,b,a,nr,ratio=r)
    premise = z.And(r>=0,n>=1,n<=256)
    reference = -a*z.If(a>=0,z.If(r<hi,r,hi),z.If(r>lo,r,lo))/nr
    certify('F-RL-PPO-OBJECTIVE',premise,values['loss']==reference,[])
    active = z.Or(z.And(a>=0,r<=hi),z.And(a<0,r>=lo))
    coefficient = values['coefficient']
    certify('F-RL-PPO-COEFFICIENT',premise,z.And(
        coefficient==z.If(active,a*r/nr,0),
        z.Implies(a>=0,z.And(coefficient>=0,coefficient<=hi*a/nr)),
        z.Implies(a<0,coefficient<=0)),[])
    ps = z.Reals('ppo_p0 ppo_p1 ppo_p2'); c=z.Real('ppo_c'); action=z.Int('ppo_action')
    simplex=z.And(*[x>=0 for x in ps],sum(ps)==1,action>=0,action<=2)
    total=sum((x-z.If(action==i,1,0))*c for i,x in enumerate(ps))
    certify('F-RL-PPO-COEFFICIENT-simplex',simplex,total==0,[])
    if fixtures.get('schemaVersion')!=1 or [e['id'] for e in fixtures['entries']]!=['CE-RL-012','CE-RL-013']:
        raise ValueError('PPO objective: counterexample roster drift')
    results=[]
    for entry in fixtures['entries']:
        p,b,a=[fixed_fp(entry[k]) for k in ('probabilityHex','oldProbabilityHex','advantageHex')]
        values=evaluate(expressions,p,b,a,fixed_fp(float(1).hex()),True)
        solver=z.Solver();solver.set(timeout=10000,random_seed=0)
        solver.add(*[finite(x) for x in (p,b,a)],z.fpGT(b,z.FPVal(0,z.Float64())),
                   z.fpGEQ(p,z.FPVal(0,z.Float64())),z.fpLEQ(p,z.FPVal(1,z.Float64())),
                   z.fpLEQ(b,z.FPVal(1,z.Float64())),
                   z.fpEQ(values['loss'],fixed_fp(entry['lossHex'])))
        if entry['id']=='CE-RL-012':
            solver.add(z.fpIsInf(values['ratio']),z.fpIsNaN(values['coefficient']))
        else:
            expected=fixed_fp(entry['coefficientHex'])
            solver.add(z.fpEQ(values['coefficient'],expected),
                       z.fpGT(z.fpAbs(values['coefficient']),z.fpMul(z.RNE(),fixed_fp(float(1.2).hex()),z.fpAbs(a))))
            real_values=[z.RealVal(str(Fraction(float.fromhex(entry[k]))))
                         for k in ('probabilityHex','oldProbabilityHex','advantageHex')]
            real=evaluate(expressions,*real_values,z.RealVal(1))
            solver.add(z.Abs(real['coefficient'])>hi*z.Abs(real_values[2]))
        result=solver.check()
        if result!=z.sat:
            raise RuntimeError('PPO objective: prescribed witness not SAT: '+entry['id']+' '+str(result))
        results.append({'id':entry['id'],'result':'sat','lossHex':entry['lossHex']})
    return {'smt':{'F-RL-PPO-OBJECTIVE':'unsat','F-RL-PPO-COEFFICIENT':'unsat'},
            'queries':4,'premiseChecks':4,'counterexamples':results,
            'batchBounds':[1,256],'actions':3,'literalSemantics':'exact rationals of binary64 .8 and 1.2',
            'binary64WitnessRounding':'RNE separate operations; one-row mean',
            'softmaxImplementationVerified':False,'hardTrustRegionVerified':False,
            'universalRuntimeRefinement':False}
