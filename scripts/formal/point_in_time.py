"""Source-bound timestamp admission and finite publication protocol, not provider truth."""
import ast
from collections import deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import z3 as z
from ppo_successor import certify
from promotion_boundary import shape

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/point_in_time_v3.py'
REGISTRY = 'formal/research/point-in-time-source.json'
DEFINITIONS = {'Record', 'Witness', 'VintageBatch', 'TrainingV3', '_time', '_available',
               '_header', '_grid', 'admit_v3', 'train_point_in_time_v3'}


def require(ok, reason):
    if not ok: raise ValueError('point-in-time: ' + reason)


def extract(source=None, registry=None):
    tree = ast.parse((ROOT/SOURCE).read_text() if source is None else source)
    registry = json.loads((ROOT/REGISTRY).read_text()) if registry is None else registry
    nodes = {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    require(set(nodes) == DEFINITIONS and len(nodes) == sum(isinstance(n,(ast.FunctionDef,ast.ClassDef)) for n in tree.body), 'definition coverage')
    require(registry['schemaVersion'] == 1 and set(registry['definitions']) == DEFINITIONS, 'registry coverage')
    for name,node in nodes.items():
        require(shape(node) == registry['definitions'][name], 'reviewed body drift: '+name)
    require(hashlib.sha256(shape(tree).encode()).hexdigest() == registry['astSha256'], 'complete initialization/effect drift')
    for name in ('Record','Witness','VintageBatch','TrainingV3'):
        require([ast.unparse(n) for n in nodes[name].decorator_list] == ['dataclass(frozen=True, slots=True)'], 'immutable representation')
    guard=ast.parse('if enabled is not True or type(version) is not str or version != VERSION:\n return None').body[0]
    for name in ('admit_v3','train_point_in_time_v3'):
        require(shape(nodes[name].body[0]) == shape(guard), 'default/version gate')
        require([ast.unparse(n) for n in nodes[name].args.kw_defaults] == ['False','VERSION'], 'public defaults')
    actual_calls=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='train_ppo_v2']
    require(actual_calls == ['train_ppo_v2(prices, funding, horizon, seed, steps, enabled=True)'], 'training call coverage')
    train=nodes['train_point_in_time_v3']
    require(ast.unparse(train.body[1]) == 'admitted = admit_v3(records, symbols, closes, decisions, processing_us, enabled=True)' and
            ast.unparse(train.body[2]) == 'if admitted is None:\n    return None', 'admission dominates training')
    require(isinstance(train.body[3],ast.Try) and len(train.body)==4, 'publication/exception boundary')
    require(ast.unparse(train.body[3].body[-2]) == 'if trained is None:\n    return None' and
            ast.unparse(train.body[3].body[-1]) == 'return TrainingV3(VERSION, admitted, trained)', 'envelope-only publication')
    forbidden=(ast.AsyncFunctionDef,ast.Await,ast.Global,ast.Nonlocal,ast.Yield,ast.YieldFrom)
    require(not any(isinstance(n,forbidden) for n in ast.walk(tree)), 'shared/asynchronous effect')
    select=nodes['admit_v3'].body[3].body[3]
    require(isinstance(select,ast.For) and ast.unparse(select.target)=='record', 'record scan')
    require(ast.unparse(select.body[0])=='if not _header(record, symbols):\n    return None', 'header admission')
    require(ast.unparse(select.body[4])=='if available > decision:\n    continue', 'visibility predicate')
    require(ast.unparse(select.body[7])=='''if old is None or record.revision > old.revision:
    selected[key] = record
    ambiguous.discard(key)
elif record.revision == old.revision:
    ambiguous.add(key)''', 'unique maximum transfer')
    require(ast.unparse(nodes['_time'].body[0].value)=='type(value) is int and 0 <= value <= MAX_TIME', 'integer domain')
    constants={ast.unparse(n.targets[0]):ast.unparse(n.value) for n in tree.body if isinstance(n,ast.Assign)}
    require(constants == {'VERSION': "'point-in-time-training-v3'", 'MAX_TIME':'2 ** 63 - 1', 'MAX_RECORDS':'262144'}, 'version/numeric bounds')
    header=nodes['_header']
    require(ast.unparse(header.body[3]) == 'if not record.close <= record.first_seen <= record.collected:\n    return False', 'timestamp ordering')
    require(ast.unparse(header.body[5]) == 'if record.revision > 0 and record.revised is None:\n    return False', 'revision witness')
    require(ast.unparse(header.body[6]) == 'for witness in (record.released, record.revised):\n    if witness is not None and (not _time(witness) or not record.close <= witness <= record.first_seen):\n        return False', 'optional witness bounds')
    require(ast.unparse(nodes['_grid'].body[-1]) == 'return all((c <= d for c, d in zip(closes, decisions))) and all((d < c for d, c in zip(decisions[:-1], closes[1:])))', 'decision grid ordering')
    return nodes, {'status':'exhaustively_checked','definitions':len(nodes),'trainingCalls':len(actual_calls),
                   'defaultDisabledEntries':2,'immutableRepresentations':4,
                   'scope':'complete reviewed source, extracted admission/selection/publication transfers; trusted Python/NumPy primitives'}


def expression(node, atoms):
    label=ast.unparse(node)
    if label in atoms: return atoms[label]
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
        return expression(node.left,atoms)+expression(node.right,atoms)
    if isinstance(node,ast.IfExp):
        return z.If(expression(node.test,atoms),expression(node.body,atoms),expression(node.orelse,atoms))
    if isinstance(node,ast.Call) and ast.unparse(node.func) in ('max','min') and not node.keywords:
        values=[expression(n,atoms) for n in node.args]; result=values[0]
        for value in values[1:]:
            result=z.If(result>=value,result,value) if ast.unparse(node.func)=='max' else z.If(result<=value,result,value)
        return result
    raise ValueError('point-in-time: unsupported timestamp expression '+label)


def prove(nodes):
    close,seen,collected,released,revised,lag,decision=z.Ints('pit_close pit_seen pit_collected pit_released pit_revised pit_lag pit_decision')
    has_release,has_revision=z.Bools('pit_has_release pit_has_revision')
    atoms={'record.close':close,'record.first_seen':seen,'record.collected':collected,'record.released':released,
           'record.revised':revised,'processing_us':lag,'record.released is not None':has_release,'record.revised is not None':has_revision}
    available=expression(nodes['_available'].body[0].value,atoms)
    bound=2**63-1
    premise=z.And(close>=0,close<=seen,seen<=collected,collected<=bound,lag>=0,lag<=bound,
                  decision>=close,decision<=bound,z.Implies(has_release,z.And(released>=close,released<=seen)),
                  z.Implies(has_revision,z.And(revised>=close,revised<=seen)),available<=decision)
    certify(premise,z.And(close<=decision,seen<=decision,collected<=decision,
                         z.Implies(has_release,released<=decision),z.Implies(has_revision,revised<=decision)))
    certify(premise,z.And(available>=0,available<=bound,available>=collected+lag))
    certify(z.And(close>=0,seen>=close,collected>=seen,lag>=0,has_revision,revised>decision),available>decision)
    # Sequential maximum transfer corresponds to the extracted scan branch.
    old,new,older=z.Ints('pit_old pit_new pit_older'); tied,visible=z.Bools('pit_tied pit_visible')
    chosen=z.If(z.And(visible,new>old),new,old)
    duplicate=z.If(z.And(visible,new>old),False,z.If(z.And(visible,new==old),True,tied))
    certify(z.And(old>=-1,new>=0,older<=old),z.And(chosen>=old,chosen>=older,z.Implies(visible,chosen>=new)))
    certify(z.And(old>=0,new==old,visible),z.And(chosen==old,duplicate))
    certify(z.And(old>=-1,new>old,visible),z.And(chosen==new,z.Not(duplicate)))
    certify(z.And(old>=-1,new>=0,z.Not(visible)),z.And(chosen==old,duplicate==tied))
    return {'F-RL-PIT-TIME':'unsat'}


# State=(phase, best revisions for two keys, duplicate-max flags). Unavailable
# records stutter. Repeated scans cover arbitrary order/count; count cap is an
# implementation rejection not a claimed liveness premise.
def transitions(state, mutant=False):
    phase,best,ties=state
    if phase in ('absent','published'): return []
    if phase=='start': return [('disabled',('absent',best,ties)),('enabled',('scan',best,ties))]
    if phase=='scan':
        out=[('invalid-header',('absent',best,ties)),('unavailable',state)]
        for key in range(2):
            for revision in range(3):
                bs=list(best); ts=list(ties)
                if revision>best[key]: bs[key]=revision;ts[key]=False
                elif revision==best[key]: ts[key]=True
                out.append((f'record:{key}:{revision}',('scan',tuple(bs),tuple(ts))))
        admitted=all(b>=0 for b in best) and not any(ties)
        out.append(('finish',('ready' if admitted or mutant else 'absent',best,ties)))
        return out
    if phase=='ready': return [('training-start',('training',best,ties))]
    if phase=='training': return [('failure',('absent',best,ties)),('complete',('published',best,ties))]
    raise ValueError('unknown phase')


def check_model(mutant=False):
    initial=('start',(-1,-1),(False,False)); queue=deque([initial]); depth={initial:0}; edges=0
    while queue:
        state=queue.popleft();phase,best,ties=state
        if phase in ('ready','training','published'):
            require(all(b>=0 for b in best) and not any(ties), 'partial/ambiguous training publication')
        for _,nxt in transitions(state,mutant):
            edges+=1
            if nxt not in depth:depth[nxt]=depth[state]+1;queue.append(nxt)
    return {'status':'model_checked','states':len(depth),'transitions':edges,'maxShortestDepth':max(depth.values()),
            'slots':2,'revisionRanks':3,'search':'reachable_fixed_point','scope':'all record orders/counts in finite abstraction; no runtime/compiler/liveness theorem'}


def fixture():
    import point_in_time_v3 as p
    closes=tuple(1000*i for i in range(121));decisions=tuple(c+100 for c in closes)
    records=tuple(p.Record('FIXTURE',kind,c,0,None,c+10,c+20,None,100.0 if kind=='price' else 0.0)
                  for kind in ('price','funding') for c in closes)
    return records,('FIXTURE',),closes,decisions


def oracle(records,symbols,closes,decisions,lag):
    result=[]
    for symbol in symbols:
        for kind in ('price','funding'):
            for close,decision in zip(closes,decisions):
                eligible=[r for r in records if (r.symbol,r.kind,r.close)==(symbol,kind,close) and
                          max(t for t in (r.close,r.released,r.first_seen,r.collected,r.revised) if t is not None)+lag<=decision]
                if not eligible:return None
                best=max(r.revision for r in eligible); winners=[r for r in eligible if r.revision==best]
                if len(winners)!=1:return None
                result.append(winners[0])
    return result


def conformance():
    sys.path.insert(0,str(ROOT/'scripts/research'))
    import numpy as np
    import point_in_time_v3 as p
    args=fixture(); base=p.admit_v3(*args,5,enabled=True)
    require(base is not None and len(base.witnesses)==242,'baseline admission')
    rng=random.Random(20261006); cases=0
    for _ in range(64):
        records=list(args[0]);source=records[rng.randrange(len(records))]
        revision=replace(source,revision=1,released=source.close+30,revised=source.close+30,
                         first_seen=source.close+40,collected=source.close+50,value=101.0 if source.kind=='price' else .01)
        records.append(revision)
        if rng.randrange(3)==0:records.append(revision)
        if rng.randrange(2):records.append(replace(revision,revision=2,collected=source.close+101,value=float('nan')))
        rng.shuffle(records); expected=oracle(records,*args[1:],5)
        actual=p.admit_v3(tuple(records),*args[1:],5,enabled=True)
        require((actual is None)==(expected is None),'oracle refusal disagreement')
        if actual is not None:
            flattened=np.concatenate([np.frombuffer(b,dtype='<f8') for pair in zip(actual.prices,actual.funding) for b in pair])
            require(flattened.tolist()==[r.value for r in expected],'oracle values')
            require([w.revision for w in actual.witnesses]==[r.revision for r in expected],'oracle witnesses')
        cases+=1
    # Future payloads cannot affect earlier vintage selection; metadata must remain valid.
    for value in (float('nan'),float('inf'),-1.0,None,object()):
        later=replace(args[0][0],revision=1,revised=30,first_seen=40,collected=101,value=value)
        require(p.admit_v3(args[0]+(later,),*args[1:],5,enabled=True)==base,'future payload affected selection')
    with patch.object(p,'train_ppo_v2',side_effect=AssertionError('learner reached without admission')):
        require(p.train_point_in_time_v3(*args,1,31,1) is None,'default training')
        require(p.train_point_in_time_v3(args[0][1:],*args[1:],1,31,1,enabled=True) is None,'incomplete training')
    trained=p.train_point_in_time_v3(*args,1,31,1,5,enabled=True)
    require(trained is not None and trained.admitted==base and trained.training.steps==1,'actual synthetic training composition')
    require(all(w.available<=w.decision for w in trained.admitted.witnesses),'witness publication')
    return {'status':'property_tested','generatedOracleCases':cases,'futurePayloadCases':5,
            'syntheticTraining':{'seed':31,'steps':1,'bars':121,'symbols':1},
            'scope':'synthetic engineering conformance only, no financial data or efficacy result'}


def check_point_in_time():
    nodes,surface=extract()
    return {'surface':surface,'smt':prove(nodes),'model':check_model(),'conformance':conformance()}


if __name__=='__main__':
    print(json.dumps(check_point_in_time(),indent=2))
