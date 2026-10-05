"""Artifact identity composition; finite interleavings and source-bound predicates."""
import ast
import copy
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import z3 as z
from artifact_admission import extract as extract_v1, expression, prove, check_predicates

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = 'formal/research/artifact-composition-source.json'
LEGACY = 'formal/research/artifact-legacy-loader.json'
SOURCES = ('scripts/research/sequential_learning.py', 'scripts/research/ppo_artifact_v4.py')
V1 = ('_provenance_identity', 'validate_provenance', '_parameter_snapshots', 'load_policy')
V4 = ('_check', '_provenance', '_buffer', '_snapshot', '_restore', '_unique', '_json',
      'decode_artifact_v4', 'request_from_artifact_v4')


def require(ok, reason):
    if not ok:
        raise ValueError('artifact composition: ' + reason)


def definitions(source):
    return {n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}


def extract(sources=None, registry=None):
    sources = {p:(ROOT/p).read_text() for p in SOURCES} if sources is None else sources
    registry = json.loads((ROOT/REGISTRY).read_text()) if registry is None else registry
    require(set(sources)==set(SOURCES), 'reader coverage')
    trees = {p:definitions(s) for p,s in sources.items()}
    for p,names in zip(SOURCES,(V1,V4)):
        require(set(registry['definitions'][p])==set(names), 'helper coverage')
        for name in names:
            require(ast.dump(trees[p][name],include_attributes=False)==registry['definitions'][p][name], 'reviewed binding drift: '+name)
    predicates,_ = extract_v1(sources[SOURCES[0]])
    # These are executable data bindings, not a Boolean "safe loader" premise.
    first = ast.unparse(trees[SOURCES[0]]['load_policy'].body[0])
    require(first=='expected_identity = _provenance_identity(expected_provenance)', 'pre-read identity')
    helper=trees[SOURCES[0]]['_provenance_identity']
    require(ast.unparse(helper.body[1].body[0])=='identity = json.dumps(provenance, sort_keys=True, allow_nan=False)', 'immutable identity capture')
    require(ast.unparse(helper.body[2])=='validate_provenance(json.loads(identity))' and ast.unparse(helper.body[3])=='return identity', 'private snapshot validation')
    decoder=trees[SOURCES[1]]['decode_artifact_v4'].body[2].body
    require(ast.unparse(decoder[0])=='expected = _provenance(expected_provenance)', 'v4 capture ordering')
    require(ast.unparse(decoder[3])=='value = json.loads(raw, object_pairs_hook=_unique)', 'single-byte-snapshot decoding')
    require(ast.unparse(decoder[6])=="restored = _restore(value['result'])" and ast.unparse(decoder[8])=='return restored', 'validated object publication')
    require(ast.unparse(trees[SOURCES[1]]['_provenance'].body[-1])=='return value.copy()', 'v4 private identity')
    require(ast.unparse(trees[SOURCES[1]]['_restore'].body[8])=="actor, critic = (_snapshot(value['actor'], 3, step), _snapshot(value['critic'], 1, step))", 'nested snapshots')
    return trees,predicates


def identity(trees,predicates):
    check_predicates(predicates)
    nodes=trees[SOURCES[1]]; body=nodes['decode_artifact_v4'].body[2].body
    actual,expected,prov,wanted,schema,version,promotion=z.Strings('comp_digest comp_expected comp_prov comp_wanted comp_schema comp_version comp_promotion')
    contracts,disabled=z.Bools('comp_contracts comp_disabled')
    atoms={'hashlib.sha256(raw).hexdigest()':actual,'expected_sha256':expected,
           "value['schema']":schema,'VERSION':z.StringVal('ppo-artifact-v4'),
           "value['contracts'] != CONTRACTS":z.Not(contracts),"value['enabled'] is not False":z.Not(disabled),
           "value['promotion']":promotion,"_provenance(value['provenance'])":prov,'expected':wanted}
    reject=expression(body[5].test,atoms)
    prove(z.Not(expression(body[2].test,atoms)),actual==expected)
    prove(z.Not(reject),z.And(schema=='ppo-artifact-v4',contracts,disabled,promotion=='research-only',prov==wanted))
    # Decode-helper predicates are translated from actual checks, including nested versions.
    queries=4
    for name,literal in (('_restore','ppo-successor-v2'),('_snapshot','optimizer-snapshot-v2')):
        condition=nodes[name].body[1].value.args[0]
        other={ast.unparse(n):z.Bool('comp_'+name+'_'+str(i)) for i,n in enumerate(condition.values) if ast.unparse(n)!="value['version'] == '"+literal+"'"}
        other["value['version']"]=version
        prove(expression(condition,other),version==literal);queries+=1
    return {'F-RL-ARTIFACT-IDENTITY':'unsat'},queries


# File identity is (digest, provenance, supported version). Equality classes are
# relative to the expected digest/provenance; no cryptographic theorem is used.
FILES=tuple(itertools.product(range(2),range(2),(False,True)))


def model(legacy=False):
    # pc, caller, path, captured expected, captured bytes, checked digest, decoded
    initial=(0,0,0,-1,-1,-1,-1)
    queue=deque([initial]);seen={initial:[]};edges=0;accepted=0;counterexample=None
    while queue:
        state=queue.popleft();pc,caller,path,expected,raw,hashed,decoded=state
        if pc==6:
            accepted+=1
            if FILES[raw][0]!=0 or FILES[raw][1]!=expected or not FILES[raw][2]:
                if counterexample is None:counterexample=seen[state]
                if not legacy:raise ValueError('artifact composition: snapshot mismatch reached publication')
        targets=[]
        # Arbitrarily many concurrent replacements: reachable fixed point, no depth cutoff.
        for c in range(2):targets.append(('caller='+str(c),(pc,c,path,expected,raw,hashed,decoded)))
        for f in range(len(FILES)):targets.append(('path='+str(f),(pc,caller,f,expected,raw,hashed,decoded)))
        if pc==0:targets.append(('capture',(1,caller,path,caller,raw,hashed,decoded)))
        elif pc==1:targets.append(('read',(2,caller,path,expected,path,hashed,decoded)))
        elif pc==2:targets.append(('hash',(3 if FILES[raw][0]==0 else 7,caller,path,expected,raw,FILES[raw][0],decoded)))
        elif pc==3:targets.append(('decode',(4,caller,path,expected,raw,hashed,raw)))
        elif pc==4:
            ok=FILES[decoded][1]==(caller if legacy else expected) and FILES[decoded][2]
            targets.append(('metadata',(5 if ok else 7,caller,path,expected,raw,hashed,decoded)))
        elif pc==5:targets.append(('publish',(6,caller,path,expected,raw,hashed,decoded)))
        for event,target in targets:
            edges+=1
            if target not in seen:seen[target]=seen[state]+[event];queue.append(target)
    require(accepted>0,'vacuous publication')
    require((counterexample is not None)==legacy,'legacy witness missing or fixed model unsafe')
    return {'states':len(seen),'transitions':edges,'maxShortestDepth':max(map(len,seen.values())),
            'acceptedStates':accepted,'callerClasses':2,'fileClasses':8,'expectedDigestClass':0,
            'counterexample':counterexample,'fixedPoint':True,'unboundedRuntimeRefinement':False}


def fixtures():
    sys.path.insert(0,str(ROOT/'scripts/research'))
    import numpy as np
    import sequential_learning as learning
    import ppo_artifact_v4 as codec
    from optimizer_snapshot_v2 import Snapshot
    from ppo_successor_v2 import TrainingResult
    prov={'codeCommit':'a'*40,'registrationSha256':'b'*64,'dataSha256':'c'*64,'seed':11,'horizon':1,'algorithm':'ppo','fold':0}
    p4={k:('a'*40 if k=='codeCommit' else 'b'*64) for k in ('codeCommit','registrationSha256','dataSha256','fundingSha256','splitSha256','proofSha256')};p4['datasetRole']='development'
    def snapshot(outputs):
        buffers=tuple(np.zeros(n,dtype='<f8').tobytes() for n in (192,16,16*outputs,outputs))
        return Snapshot('optimizer-snapshot-v2',outputs,4,buffers,buffers,buffers)
    fixture=TrainingResult('ppo-successor-v2',11,1,1,('FIXTURE',),tuple(np.zeros(6,dtype='<f8').tobytes() for _ in range(4)),snapshot(3),snapshot(1),(0.0,)*4)
    return learning,codec,prov,p4,fixture


def conformance():
    import numpy as np
    learning,codec,prov,p4,fixture=fixtures()
    namespace=dict(vars(learning));legacy=json.loads((ROOT/LEGACY).read_text())
    exec(compile(legacy['function'],'<preserved-legacy-loader>','exec'),namespace)
    old=namespace['load_policy'];stable=0;rejected=0
    real_open=Path.open
    class ReadInterleaving:
        def __init__(self,stream,callback):self.stream=stream;self.callback=callback
        def __enter__(self):return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def read(self,n):
            value=self.stream.read(n);self.callback();return value
    def scheduled(loader,path,digest,expected,callback):
        def opened(p,*args,**kwargs):
            stream=real_open(p,*args,**kwargs)
            return ReadInterleaving(stream,callback) if args==('rb',) else stream
        with patch.object(Path,'open',opened):return loader(path,digest,expected)
    def rejects(call):
        try:call()
        except (ValueError,TypeError,OverflowError):return
        raise ValueError('artifact composition: invalid artifact accepted')
    with tempfile.TemporaryDirectory(prefix='trader-artifact-composition-') as d:
        path=Path(d)/'policy';net=learning.Network(11);digest=learning.save_policy(path,net,prov);raw=path.read_bytes()
        for extension in ({},{'extra':[1,'x',{'v':None}]},{'extra':(True,False,1.5)}):
            p=dict(prov,**extension);path.unlink();digest=learning.save_policy(path,net,p)
            a=old(path,digest,p);b=learning.load_policy(path,digest,p)
            require(all(np.array_equal(a.p[k],b.p[k]) for k in a.p),'stable-input compatibility');stable+=1
        path.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
        # Expected A differs from artifact B. I/O yields to an ordinary caller write.
        for loader,is_old in ((old,True),(learning.load_policy,False)):
            expected=dict(prov,fold=1)
            call=lambda:scheduled(loader,path,digest,expected,lambda:expected.update(fold=0))
            if is_old:require(call() is not None,'legacy race no longer reproduced')
            else:rejects(call)
        # Replacement of the path after read cannot change hashed or decoded bytes.
        expected=dict(prov)
        loaded=scheduled(learning.load_policy,path,digest,expected,lambda:path.write_bytes(b'corrupted replacement'))
        require(all(np.array_equal(net.p[k],loaded.p[k]) for k in net.p),'same-byte snapshot failure')
        doc=json.loads(raw)
        changes=[(['schema'],'future'),(['environment'],'future'),(['observation'],'future'),(['actions'],[-1,0,1]),
                 (['provenance','fold'],1),(['parameters','b2'],[0,0]),(['parameters','b2'],[0,0,float('nan')])]
        for field in prov:
            changed=copy.deepcopy(doc);del changed['provenance'][field];changes.append(([],changed))
        for keys,value in changes:
            changed=copy.deepcopy(doc)
            if not keys:changed=value
            else:
                target=changed
                for key in keys[:-1]:target=target[key]
                target[keys[-1]]=value
            bad=json.dumps(changed).encode();path.write_bytes(bad)
            rejects(lambda:learning.load_policy(path,hashlib.sha256(bad).hexdigest(),prov));rejected+=1
        path.write_bytes(raw);rejects(lambda:learning.load_policy(path,'0'*64,prov));rejected+=1
        with patch.object(Path,'open',side_effect=RuntimeError('I/O before expected validation')):
            for invalid in ({},dict(prov,horizon=2),dict(prov,extra=float('nan'))):
                rejects(lambda:learning.load_policy(path,digest,invalid));rejected+=1
    raw4=codec.encode_artifact_v4(fixture,p4,enabled=True);require(raw4 is not None,'v4 fixture encoding')
    digest4=hashlib.sha256(raw4).hexdigest();real_hash=hashlib.sha256
    expected=dict(p4,codeCommit='c'*40)
    def hash_interleave(value):expected.update(p4);return real_hash(value)
    with patch.object(codec.hashlib,'sha256',hash_interleave):
        require(codec.decode_artifact_v4(raw4,digest4,expected,enabled=True) is None,'v4 mutable reference accepted')
    changes=[(['schema'],'future'),(['result','version'],'future'),(['result','actor','version'],'future'),(['result','critic','version'],'future')]
    changes += [(['contracts',i],'future') for i in range(7)]
    for keys,value in changes:
        doc=json.loads(raw4);target=doc
        for key in keys[:-1]:target=target[key]
        target[keys[-1]]=value;bad=codec._json(doc)
        require(codec.decode_artifact_v4(bad,real_hash(bad).hexdigest(),p4,enabled=True) is None,'v4 incompatible version');rejected+=1
    return {'legacyRaceReproduced':True,'fixedRaceRejected':True,'v4SnapshotRejected':True,
            'postReadReplacementAcceptedOriginalBytes':True,'stableCompatibilityCases':stable,
            'rejections':rejected,'financialTrials':0,'holdoutAccess':False,
            'scheduling':'deterministic instrumentation at actual file-read/hash boundaries; no timing races'}


def check_composition():
    trees,predicates=extract();smt,queries=identity(trees,predicates)
    legacy=model(True)
    witnesses=json.loads((ROOT/'formal/research/counterexamples.json').read_text())['entries']
    witness=next(e for e in witnesses if e['id']=='CE-RL-ARTIFACT-PROVENANCE-RACE')
    require(legacy['counterexample']==witness['trace'],'preserved counterexample drift')
    return {'smt':smt,'queries':queries,'model':model(),'legacyModel':legacy,'conformance':conformance(),
            'surface':{'status':'exhaustively_checked','readerDefinitions':len(V1)+len(V4),
                       'maximumArtifactBytes':65536,'formats':['offline_policy_v1','ppo-artifact-v4'],
                       'sourceToPrimitiveCorrespondence':'reviewed assumption; not interpreter refinement'}}


if __name__=='__main__':
    print(json.dumps(check_composition(),indent=2))
