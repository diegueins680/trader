"""Source-bound single-call staging model; explicitly non-atomic observer extension."""
import ast
import copy
from collections import deque
from fractions import Fraction
from pathlib import Path
import z3 as z
from causal_footprint import structure
from target_v2 import certify

ROOT=Path(__file__).resolve().parents[2]
SOURCE='scripts/research/sequential_learning.py'
TEMPLATE='''
class Network:
    def gradients(self, x: np.ndarray, dz: np.ndarray) -> dict:
        h = np.tanh(x @ self.p["w1"] + self.p["b1"])
        dh = (dz @ self.p["w2"].T) * (1 - h**2)
        return {"w1": x.T @ dh, "b1": dh.sum(0), "w2": h.T @ dz, "b2": dz.sum(0)}

    def update(self, x: np.ndarray, dz: np.ndarray, lr: float) -> None:
        if not _finite_real(lr) or lr <= 0 or not _integer(self.steps) or self.steps < 0:
            raise ValueError("invalid optimizer learning rate or step counter")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                grads = self.gradients(x, dz)
                if not all(np.isfinite(g).all() for g in grads.values()):
                    raise ValueError("non-finite gradient")
                norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
                if not np.isfinite(norm):
                    raise ValueError("non-finite gradient norm")
                steps = int(self.steps) + 1
                params, moments, variances = {}, {}, {}
                for k, grad in grads.items():
                    g = CLIP
                    moments[k] = 0.9 * self.m[k] + 0.1 * g
                    variances[k] = 0.999 * self.v[k] + 0.001 * g**2
                    m = moments[k] / (1 - 0.9**steps)
                    v = variances[k] / (1 - 0.999**steps)
                    params[k] = self.p[k] - lr * m / (np.sqrt(v) + 1e-8)
                if not all(np.isfinite(v).all() for state in (params, moments, variances)
                           for v in state.values()):
                    raise ValueError("non-finite optimizer state")
        except FloatingPointError as exc:
            raise ValueError("non-finite optimizer arithmetic") from exc
        self.p, self.m, self.v, self.steps = params, moments, variances, steps
'''


def extract(source):
    classes=[n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Network']
    if len(classes)!=1:
        raise ValueError('optimizer publication: missing/duplicate class')
    methods={}
    for name in ('gradients','update'):
        found=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(found)!=1:
            raise ValueError('optimizer publication: missing/duplicate method')
        methods[name]=copy.deepcopy(found[0])
    try:
        body=methods['update'].body[1].body[0].body
        clip=body[6].body[0].value
        body[6].body[0].value=ast.Name(id='CLIP',ctx=ast.Load())
    except (IndexError,AttributeError):
        raise ValueError('optimizer publication: staging drift')
    expected=ast.parse(TEMPLATE).body[0].body
    if any(structure(methods[n.name])!=structure(n) for n in expected):
        raise ValueError('optimizer publication: audited source drift')
    fields=[n.attr for n in methods['update'].body[-1].targets[0].elts]
    gates=['controls']+['gradient_computation','gradient_finiteness','norm_computation','norm_finiteness','step_increment','local_allocation']
    for key in ('w1','b1','w2','b2'):
        for statement in body[6].body:
            gates.append(key+':'+ast.unparse(statement.targets[0]))
    gates.append('candidate_finiteness')
    return clip,gates,fields


def real_expression(node,values):
    key=ast.unparse(node)
    if key in values:return values[key]
    if isinstance(node,ast.Constant) and type(node.value) in (int,float):
        return z.RealVal(str(Fraction(node.value)))
    if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Div,ast.Mult)):
        a,b=real_expression(node.left,values),real_expression(node.right,values)
        return a/b if isinstance(node.op,ast.Div) else a*b
    if isinstance(node,ast.Call) and ast.unparse(node.func)=='max' and len(node.args)==2 and not node.keywords:
        a,b=[real_expression(n,values) for n in node.args]
        return z.If(a>=b,a,b)
    raise ValueError('optimizer publication: unsupported clip '+key)


def publication_mask(pc,count):
    return (1<<max(0,pc-count))-1


def transitions(state,gates,fields,extended):
    pc,terminal,observed=state;count=len(gates);end=count+len(fields)
    if terminal or pc==end:
        yield 'terminal_stutter',state
        return
    if extended and observed<0:
        yield 'observe',(pc,False,publication_mask(pc,count))
    if pc<count:
        yield 'pass:'+gates[pc],(pc+1,False,observed)
        yield 'reject:'+gates[pc],(pc,True,observed)
    else:
        yield 'store:'+fields[pc-count],(pc+1,False,observed)
        if extended and count<pc<end:
            yield 'interrupt',(pc,True,observed)


def check_model(gates,fields,extended=False):
    count=len(gates);end=count+len(fields);initial=(0,False,-1)
    queue=deque([initial]);depth={initial:0};parent={};edges=0;witness=None;observations=set()
    def rank(s):return 0 if s[1] or s[0]==end else 2*(end-s[0])+int(extended and s[2]<0)
    while queue:
        state=queue.popleft();pc,terminal,seen=state
        mask=publication_mask(pc,count)
        if pc<count and mask!=0:raise RuntimeError('optimizer publication before validation')
        if pc==end and mask!=(1<<len(fields))-1:raise RuntimeError('optimizer incomplete normal publication')
        if terminal and not extended and mask!=0:raise RuntimeError('optimizer prepublication rejection mutated state')
        if seen>=0:observations.add(seen)
        if extended and terminal and mask==1 and seen==1 and witness is None:
            trace=[];cursor=state
            while cursor!=initial:
                previous,event=parent[cursor];trace.append(event);cursor=previous
            witness=list(reversed(trace))
        outgoing=list(transitions(state,gates,fields,extended))
        if not outgoing:raise RuntimeError('optimizer model deadlock')
        for event,nxt in outgoing:
            edges+=1
            if event!='terminal_stutter' and rank(nxt)>=rank(state):raise RuntimeError('optimizer progress rank failed')
            if event.startswith('reject:') and publication_mask(nxt[0],count)!=0:raise RuntimeError('optimizer rejection wrote state')
            if event.startswith('store:') and pc<count:raise RuntimeError('optimizer stores precede gates')
            if nxt not in depth:
                depth[nxt]=depth[state]+1;parent[nxt]=(state,event);queue.append(nxt)
    if extended and witness is None:raise RuntimeError('optimizer expected interruption witness absent')
    return {'states':len(depth),'transitions':edges,'maxShortestDepth':max(depth.values()),
            'progressRankBound':rank(initial),'gates':count,'publicationFields':fields,
            'parameterKeys':4,'calls':1,'observers':int(extended),'search':'reachable_fixed_point',
            'observedMasks':sorted(observations),'counterexample':witness}


def check_optimizer(fixtures,source=None):
    clip,gates,fields=extract((ROOT/SOURCE).read_text() if source is None else source)
    grad,norm=z.Reals('optimizer_grad optimizer_norm')
    value=real_expression(clip,{'grad':grad,'norm':norm})
    certify('F-RL-GRADIENT-CLIP',norm>=z.Abs(grad),z.And(
        z.Abs(value)<=1,z.Abs(value)<=z.Abs(grad),
        z.Implies(grad>0,value>0),z.Implies(grad<0,value<0),z.Implies(grad==0,value==0)),[])
    narrow=check_model(gates,fields);extended=check_model(gates,fields,True)
    if (fixtures.get('schemaVersion')!=1 or len(fixtures['entries'])!=1 or
        fixtures['entries'][0]['id']!='CE-RL-016' or fixtures['entries'][0]['trace']!=extended['counterexample'] or
        fixtures['entries'][0]['normalMasks']!=[0,1,3,7,15] or fixtures['entries'][0]['interruptedMask']!=1):
        raise ValueError('optimizer publication: witness fixture drift')
    return {'smt':{'F-RL-GRADIENT-CLIP':'unsat'},'queries':1,'premiseChecks':1,
            'singleWriter':narrow,'observerExtension':extended,
            'floatingNormPremiseVerified':False,'concurrentAtomicityVerified':False,'universalRuntimeRefinement':False}
