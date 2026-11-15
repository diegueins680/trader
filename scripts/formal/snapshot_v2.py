"""Source-bound finite publication model and SMT guards for immutable snapshots."""
import ast
from collections import deque
import dis
import hashlib
import importlib
import json
from pathlib import Path
import sys
import z3 as z

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/optimizer_snapshot_v2.py'
CONTRACT = 'formal/research/snapshot-v2-source.json'
CAP = 2**31 - 1


def require(ok, reason):
    if not ok:
        raise ValueError('snapshot v2: ' + reason)


def shape(node):
    return ast.dump(node, include_attributes=False)


def source_nodes(source):
    tree = ast.parse(source)
    nodes = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            require(n.name not in nodes, 'duplicate definition')
            nodes[n.name] = n
    return tree, nodes


def extract(source=None):
    tree, nodes = source_nodes((ROOT / SOURCE).read_text() if source is None else source)
    contract = json.loads((ROOT / CONTRACT).read_text())
    require(hashlib.sha256(shape(tree).encode()).hexdigest() == contract['astSha256'], 'audited source drift')
    methods = {n.name:n for n in nodes['_Optimizer'].body if isinstance(n, ast.FunctionDef)}
    require(shape(methods['snapshot'].body[0]) == shape(ast.parse('return self._state').body[0]), 'non-snapshot read')
    publish = methods['_publish']
    expected = ast.parse('''def _publish(self, expected, candidate):
    if GUARD:
        return None
    if not self._lock.acquire(blocking=False):
        return None
    try:
        if self._state is not expected:
            return None
        self._state = candidate
        return candidate
    finally:
        self._lock.release()
''').body[0]
    guard = publish.body[0].test
    # Check the whole control structure, not just presence of a lock/store token.
    expected.body[0].test = guard
    require(shape(publish) == shape(expected), 'publication control structure')
    stores = [(n.name, a.attr) for n in methods.values() for a in ast.walk(n)
              if isinstance(a, ast.Attribute) and isinstance(a.ctx, ast.Store)]
    require(stores == [('__init__','_state'),('__init__','_lock'),('_publish','_state')], 'extra mutable field')
    for name in ('update_v2','forward_v2'):
        reads = [n for n in ast.walk(nodes[name]) if isinstance(n,ast.Call) and ast.unparse(n.func)=='net.snapshot']
        require(len(reads)==1, 'consumer must capture one snapshot')
    require(shape(nodes['Snapshot'].decorator_list[0]) == shape(ast.parse('dataclass(frozen=True, slots=True)',mode='eval').body), 'mutable snapshot fields')
    slot_guard = next(n.test for n in ast.walk(nodes['_pack']) if isinstance(n, ast.If) and 'np.isfinite(value).all()' in ast.unparse(n.test))
    imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    calls = sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)})
    require(imports == contract['imports'] and calls == contract['calls'], 'unreviewed effect boundary')
    return guard, slot_guard


def integer_expression(node, values):
    label = ast.unparse(node)
    if label in values: return values[label]
    if isinstance(node,ast.Constant) and type(node.value) is int: return z.IntVal(node.value)
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
        return integer_expression(node.left,values)+integer_expression(node.right,values)
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not): return z.Not(integer_expression(node.operand,values))
    if isinstance(node,ast.BoolOp):
        return (z.Or if isinstance(node.op,ast.Or) else z.And)(*[integer_expression(v,values) for v in node.values])
    if isinstance(node,ast.Compare):
        ns=[node.left,*node.comparators];terms=[]
        for a,op,b in zip(ns,node.ops,ns[1:]):
            x,y=integer_expression(a,values),integer_expression(b,values)
            if isinstance(op,ast.Eq): q=x==y
            elif isinstance(op,ast.NotEq): q=x!=y
            elif isinstance(op,ast.Lt): q=x<y
            elif isinstance(op,ast.LtE): q=x<=y
            else: raise ValueError('unsupported comparison')
            terms.append(q)
        return z.And(*terms)
    raise ValueError('unsupported expression: '+label)


def certify(premise, conclusion):
    for expression, expected in ((premise,z.sat),(z.And(premise,z.Not(conclusion)),z.unsat)):
        solver=z.Solver();solver.set(timeout=10000,random_seed=0);solver.add(expression)
        require(solver.check()==expected,'unexpected SAT/UNKNOWN/vacuity')


def smt(guards):
    guard, slot_guard = guards
    old,new,current,width,newwidth=z.Ints('old new current width newwidth')
    same,held=z.Bools('expected_identity publication_lock')
    values={'candidate.step':new,'expected.step':old,'STEP_CAP':z.IntVal(CAP),
            'candidate.outputs':newwidth,'expected.outputs':width}
    accepted=z.And(z.Not(integer_expression(guard,values)),same,held)
    premise=z.And(accepted,z.Implies(same,current==old),z.Or(width==1,width==3))
    certify(premise,z.And(new==current+1,new>current,new<=CAP,new>=1,newwidth==width))
    certify(z.And(z.Not(same),old>=0),z.Not(accepted))
    value=z.FP('published_slot',z.Float64());is_variance=z.Bool('variance_slot')
    finite=z.And(z.Not(z.fpIsNaN(value)),z.Not(z.fpIsInf(value)))
    slot_values = {'type(value) is not np.ndarray': z.Bool('wrong_array'),
                   "value.dtype != np.dtype('float64')": z.Bool('wrong_dtype'),
                   'value.shape != shape': z.Bool('wrong_shape'),
                   'np.isfinite(value).all()': finite, 'group_index == 2': is_variance,
                   'np.any(value < 0)': z.fpLT(value,z.FPVal(0,z.Float64()))}
    gate=z.Not(integer_expression(slot_guard,slot_values))
    certify(gate,z.And(finite,z.Implies(is_variance,z.fpGEQ(value,z.FPVal(0,z.Float64())))))
    return {'F-RL-SNAPSHOT-CAS':'unsat','F-RL-SNAPSHOT-FINITE':'unsat'}


PHASES = {'capture':8,'captured':7,'ready':6,'locked':5,'checked':4,'stored':3,'release':2}


def rank(state):
    fields,owner,writers,read = state
    return int(read is None) + sum((2-call)*8+PHASES[phase] for call,phase,base,commits in writers if call<2)


def transitions(state, mutant=False):
    fields,owner,writers,read=state
    if read is None:
        yield 'read', (fields,owner,writers,fields)
    for index,(call,phase,base,commits) in enumerate(writers):
        if call>=2: continue
        def changed(newphase, newbase=base, newowner=owner, newfields=fields, newcommits=commits, finish=False):
            ws=list(writers)
            ws[index]=(call+int(finish),'capture' if finish else newphase,newbase,newcommits)
            return (newfields,newowner,tuple(ws),read)
        if phase=='capture':
            yield f'{index}:capture',changed('captured',newbase=fields[3])
        elif phase=='captured':
            yield f'{index}:stage',changed('ready')
            yield f'{index}:reject',changed(phase,finish=True)
        elif phase=='ready':
            if owner==-1: yield f'{index}:acquire',changed('locked',newowner=index)
            else: yield f'{index}:busy',changed(phase,finish=True)
            yield f'{index}:interrupt_before_lock',changed(phase,finish=True)
        elif phase=='locked':
            # Interrupted after successful acquire but before try/finally setup.
            yield f'{index}:orphan_lock',changed(phase,newowner=-2,finish=True)
            yield f'{index}:compare',changed('checked' if base==fields[3] else 'release')
        elif phase=='checked':
            newfields=(base+1,)*4 if not mutant else (base+1,*fields[1:])
            yield f'{index}:commit',changed('stored',newfields=newfields,newcommits=commits+1)
            yield f'{index}:interrupt_before_commit',changed('release')
        elif phase=='stored':
            yield f'{index}:ack_or_interrupt_after_commit',changed('release')
        elif phase=='release':
            yield f'{index}:release',changed(phase,newowner=-1,finish=True)


def check_model(mutant=False):
    initial=((0,0,0,0),-1,((0,'capture',-1,0),(0,'capture',-1,0)),None)
    depth={initial:0};q=deque([initial]);edges=0;orphan=0;stale=0;observed=set();max_generation=0
    while q:
        state=q.popleft();fields,owner,writers,read=state
        require(len(set(fields))==1,'mixed snapshot counterexample')
        require(fields[3]==sum(w[3] for w in writers),'lost/duplicated generation')
        require(0<=fields[3]<=4,'generation bound')
        max_generation=max(max_generation,fields[3])
        if read is not None:
            require(len(set(read))==1 and read[3]<=fields[3], 'incoherent retained read')
            observed.add(read[3])
        if owner==-2: orphan+=1
        for event,nxt in transitions(state,mutant):
            edges+=1
            require(rank(nxt)<rank(state),'progress rank')
            if event.endswith(':commit'):
                who=int(event[0]);require(owner==who and writers[who][2]==fields[3],'stale/unlocked publication')
            if event.endswith(':compare') and writers[int(event[0])][2]!=fields[3]: stale+=1
            if nxt not in depth:
                require(len(depth)<250000,'state-space bound exceeded')
                depth[nxt]=depth[state]+1;q.append(nxt)
    require(orphan>0 and stale>0 and max_generation==4,'missing adverse/success paths')
    return {'states':len(depth),'transitions':edges,'maxShortestDepth':max(depth.values()),
            'progressRankBound':rank(initial),'writers':2,'callsPerWriter':2,'readers':1,
            'orphanLockStates':orphan,'staleComparisons':stale,'generations':sorted(observed),
            'search':'reachable_fixed_point','scope':'atomic snapshots, not recovery or wall-clock termination'}


def check_snapshot():
    guard=extract()
    sys.path.insert(0,str(ROOT/'scripts/research'))
    module=importlib.import_module('optimizer_snapshot_v2')
    stores=[n.argval for n in dis.get_instructions(module._Optimizer._publish) if n.opname=='STORE_ATTR']
    require(stores==['_state'],'compiled publication not single store')
    require(sys.implementation.name=='cpython' and sys._is_gil_enabled(),'free-threaded runtime outside contract')
    return {'model':check_model(),'smt':smt(guard),'storeAttributes':stores,
            'isolation': {'requirement':'F-RL-SNAPSHOT-ISOLATION','status':'exhaustively_checked',
                          'calls':len(json.loads((ROOT/CONTRACT).read_text())['calls']),
                          'networkFileProcessOrderOperations':0},
            'stateRepresentation':'frozen slotted snapshot of native bytes tuples',
            'stepCap':CAP,'groups':3,'tensorsPerGroup':4,'outputWidths':[1,3],
            'floatSlotsPerSnapshot':[675,777],'sourceSha256':hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest(),
            'legacyCounterexampleResolved':False,'wallClockDeadlineVerified':False,
            'productionIntegration':False}
