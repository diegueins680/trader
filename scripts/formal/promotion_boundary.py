"""Source-derived promotion composition, under reviewed primitive effect contracts.

This is not an OS sandbox or whole-language/production authorization proof.
"""
import ast
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile

import z3 as z
from artifact_admission import expression, extract as loader_extract, prove
from data_composition import definition, one_assignment
from default_paths import check_saved_default

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = 'formal/research/promotion-boundary-source.json'
ROOTS = {'run_sequential_screen', 'summarize_sequential_screen', 'ppo_artifact_v4', 'ess_rational_v2', 'point_in_time_v3', 'ope_rational_v2'}
MODULES = ROOTS | {'sequential_env', 'sequential_learning', 'sequential_evaluation', 'sequential_registry',
                   'optimizer_snapshot_v2', 'ppo_successor_v2', 'ppo_inference_v3', 'gae_targets_v2', 'report_bundle_v2'}
EXTERNAL = {'__future__', 'argparse', 'csv', 'dataclasses', 'datetime', 'hashlib', 'io', 'json', 'math',
            'pathlib', 'platform', 'resource', 'subprocess', 'time', 'numpy', 'pandas', 're', 'collections',
            'statistics', 'sys', 'threading', 'fractions', 'os', 'secrets', 'stat'}
FIELDS = {'promotion', 'enabled', 'promotionAllowed', 'liveAuthorization'}


def require(ok, reason):
    if not ok:
        raise ValueError('promotion boundary: ' + reason)


def shape(node):
    return ast.dump(node, include_attributes=False)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_path(name):
    return 'scripts/research/' + name + '.py'


def inventory(source):
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    def scope(node):
        names = []
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.insert(0, node.name)
        return '.'.join(names)
    calls = sorted((scope(n), ast.unparse(n.func), sha(shape(n))) for n in ast.walk(tree) if isinstance(n, ast.Call))
    imports = sorted(ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
    fields = sorted((scope(n), n.value, ast.unparse(parents[n])) for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and type(n.value) is str and n.value in FIELDS)
    return {'moduleASTHash': sha(shape(tree)), 'imports': imports, 'callSites': calls, 'controlFields': fields}


def extract(sources=None, registry=None):
    registry = json.loads((ROOT/REGISTRY).read_text()) if registry is None else registry
    sources = {name: (ROOT/source_path(name)).read_text() for name in MODULES} if sources is None else sources
    require(set(sources) == set(registry['modules']) == MODULES, 'mandatory module coverage')
    trees = {name: ast.parse(source) for name, source in sources.items()}
    graph = {}; call_count = field_count = 0
    for name, tree in trees.items():
        actual = inventory(sources[name])
        # Normalize tuples exactly as JSON does; inventories keep duplicate sites.
        require(json.loads(json.dumps(actual)) == registry['modules'][name], 'reviewed source/effect drift: ' + name)
        call_count += len(actual['callSites']); field_count += len(actual['controlFields'])
        imports = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports.extend(a.name for a in n.names)
            if isinstance(n, ast.ImportFrom):
                require(n.level == 0 and n.module is not None, 'relative/dynamic import')
                imports.append(n.module)
            require(not isinstance(n, (ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal)), 'unreviewed control construct')
            if isinstance(n, ast.Call):
                require(ast.unparse(n.func) not in {'eval','exec','__import__','compile','importlib.import_module'}, 'dynamic program input')
        require(set(imports) <= MODULES | EXTERNAL, 'unknown imported capability')
        graph[name] = sorted(set(imports) & MODULES)
    reached = set(ROOTS); queue = deque(sorted(ROOTS))
    while queue:
        for target in graph[queue.popleft()]:
            if target not in reached:
                reached.add(target); queue.append(target)
    require(reached == MODULES, 'unreached or omitted local source')
    # Every opened file has an extracted literal read/exclusive-create mode.
    opens = []; dirs = []; writes = []; processes = []; callbacks = []
    for name, tree in trees.items():
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            callee = ast.unparse(n.func)
            if callee.startswith('os.'):
                require(name == 'report_bundle_v2' and callee in {'os.open','os.write','os.read','os.fstat','os.fsync','os.link','os.unlink','os.close'}, 'unreviewed OS effect')
            if callee == 'os.open':
                require(name == 'report_bundle_v2', 'unreviewed descriptor open')
                # Full primitive flags, ordering and owned cleanup are checked below.
                continue
            if isinstance(n.func, ast.Attribute) and n.func.attr == 'open':
                require(len(n.args) == 1 and isinstance(n.args[0], ast.Constant) and n.args[0].value in ('x','xb','rb') and not n.keywords, 'nonexclusive/unreviewed file mode')
                opens.append((name, callee, n.args[0].value))
            if isinstance(n.func, ast.Attribute) and n.func.attr == 'mkdir':
                require(not n.args, 'directory argument drift')
                require({k.arg: ast.literal_eval(k.value) for k in n.keywords} in ({}, {'parents':True,'exist_ok':False}), 'nonexclusive directory creation')
                dirs.append((name, callee))
            if isinstance(n.func, ast.Attribute) and n.func.attr in ('write','write_bytes','write_text','flush','writerow','writerows','writeheader'):
                writes.append((name, callee))
            if callee == 'subprocess.check_output':
                require(isinstance(n.args[0],ast.List) and len(n.args[0].elts)==3, 'process argv drift')
                argv=n.args[0].elts
                require(isinstance(argv[0],ast.Constant) and argv[0].value=='git' and isinstance(argv[1],ast.Constant) and argv[1].value in ('rev-parse','show'), 'effectful process target')
                processes.append(ast.unparse(n))
            if callee == 'collect':
                policy=[k.value for k in n.keywords if k.arg=='policy']
                require(len(policy)<=1, 'ambiguous callback')
                callbacks.append((name, ast.unparse(policy[0]) if policy else 'None'))
            if callee == 'replay_policy':
                require(len(n.args)==8 and ast.unparse(n.args[6])=='choose', 'unreviewed replay callback')
                callbacks.append((name,'choose'))
    expected_callbacks = [('ppo_successor_v2','behavior'), ('run_sequential_screen','choose'),
                          ('sequential_evaluation','None'), ('sequential_learning','None'),
                          ('sequential_learning','explore'), ('sequential_learning','lambda s: softmax(net.forward(s))')]
    require(sorted(callbacks)==sorted(expected_callbacks), 'callback coverage/binding drift')
    require(sorted(opens)==[('run_sequential_screen',"(output / 'events.jsonl').open",'x'),
                            ('run_sequential_screen',"(output / 'returns.csv').open",'x'),
                            ('run_sequential_screen','path.open','x'),
                            ('sequential_learning','path.open','rb'),('sequential_learning','path.open','xb'),
                            ('summarize_sequential_screen',"(output / name).open",'xb'),
                            ('summarize_sequential_screen','path.open','rb')], 'file boundary coverage')
    require(sorted(dirs)==[('run_sequential_screen',"(output / 'policies').mkdir"),('run_sequential_screen','output.mkdir'),('summarize_sequential_screen','output.mkdir')], 'directory coverage')
    require(len(processes)==2 and all('cwd=ROOT' in p for p in processes), 'source-identity process coverage')
    require(sorted(writes)==[('report_bundle_v2','os.write'),('run_sequential_screen','ledger.flush'),('run_sequential_screen','ledger.write'),
                             ('run_sequential_screen','stream.write'),('run_sequential_screen','writer.writerow'),
                             ('run_sequential_screen','writer.writerow'),('sequential_learning','stream.write'),
                             ('summarize_sequential_screen','stream.write'),
                             ('summarize_sequential_screen','writer.writeheader'),('summarize_sequential_screen','writer.writerows')], 'write sink coverage')
    from report_bundle import extract as bundle_extract
    _, bundle_surface = bundle_extract(sources['report_bundle_v2'], exporter=sources['summarize_sequential_screen'])
    destinations = check_destinations(trees)
    check_saved_default(sources['sequential_learning'])
    # Complete reviewed callback bodies are pinned, not just their names.
    require(ast.unparse(definition(trees['ppo_successor_v2'],'train_ppo_v2').body[2].body[0].body[5]).startswith('def behavior('), 'successor callback skeleton')
    surface={'requirement':'F-RL-PROMOTION-SURFACE','status':'exhaustively_checked', 'moduleCount':len(reached),
             'localImportEdges':sum(map(len,graph.values())), 'callSites':call_count, 'controlFieldOccurrences':field_count,
             'callbacks':sorted(callbacks), 'fileOpens':sorted(opens), 'directoryCreates':sorted(dirs),
             'writeSites':sorted(writes), 'processes':processes, 'destinations':destinations,
             'bundlePublication':bundle_surface,
             'scope':'all finite reviewed source sites; primitive effect/type contracts, not whole-language or OS refinement'}
    return trees, surface


def check_destinations(trees):
    """Finite source correspondence for admitted path constructors and their callers.

    Path and exclusive-create primitives are assumptions; this is not a filesystem
    sandbox. Candidate numerical outputs cannot supply a destination in this code.
    """
    runner = definition(trees['run_sequential_screen'], 'run')
    calls = [n for n in ast.walk(runner) if isinstance(n, ast.Call)]
    destinations = sorted(ast.unparse(n.args[0]) for n in calls if ast.unparse(n.func) == 'write_json')
    names = ('manifest', 'planned-registry', 'training', 'evaluation', 'ope', 'summary', 'evidence-index')
    require(destinations == sorted("output / '" + name + ".json'" for name in names), 'runner destination drift')
    require(ast.unparse(one_assignment(runner, 'artifact')) == "output / 'policies' / (trial.replace('/', '_') + '.json')", 'policy destination drift')
    saves = [ast.unparse(n) for n in calls if ast.unparse(n.func) == 'save_policy']
    require(saves == ['save_policy(artifact, net, provenance)'], 'policy writer binding drift')
    # Both trial strings use fixed algorithms and registered scalar indices/seeds;
    # no network output is used in a filename. Full source roster checks bindings.
    trials = [ast.unparse(n.value) for n in ast.walk(runner) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'trial' for t in n.targets)]
    require(sorted(trials) == sorted(["f'{alg}/h{h}/f{fold}/s{seed}'", "f'{alg}/h{h}/f{fi}/s{seed}'"]), 'trial path input drift')
    render = definition(trees['summarize_sequential_screen'], 'render_reports')
    report_names = []
    for n in ast.walk(render):
        if isinstance(n, ast.Call) and ast.unparse(n.func) in ('js', 'csvfile'):
            require(n.args and isinstance(n.args[0], ast.Constant) and type(n.args[0].value) is str, 'policy-controlled report path')
            report_names.append(n.args[0].value)
    require(sorted(report_names) == sorted(['experiment-registry.csv','all-seed-results.csv','symbol-fold-base-results.csv',
        'multi-seed-training.json','ope-report.json','evaluation-summary.json','experiment-manifest.json']), 'report destination roster')
    exporter = definition(trees['summarize_sequential_screen'], 'export')
    require(ast.unparse(exporter.body[-2]) == 'output.mkdir(parents=True, exist_ok=False)', 'report directory must be fresh')
    require(ast.unparse(exporter.body[-1]) == "for name, content in reports.items():\n    with (output / name).open('xb') as stream:\n        stream.write(content)", 'report writer binding drift')
    # A fixed directory is created before any runner effect; the path cannot be
    # reassigned by results. External analyst path selection stays an assumption.
    for function in (runner, exporter):
        for n in ast.walk(function):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                bound = {v.id for t in targets for v in ast.walk(t) if isinstance(v, ast.Name)}
                if 'output' in bound:
                    require(function is exporter and ast.unparse(n) == 'source, output = (source.resolve(), output.resolve())', 'output path reassignment')
    # Metadata may be enriched by the exporter but never overridden after the
    # actual reconcile_disposition guard. Exact source inventory covers the guard.
    for n in ast.walk(render):
        if isinstance(n, ast.Call) and ast.unparse(n.func) in ('manifest.update','summary.update'):
            require(not n.args and all(k.arg is not None and k.arg not in FIELDS for k in n.keywords), 'exporter promotion override')
    return {'runnerJson':destinations, 'reportNames':sorted(report_names),
            'policyPath':ast.unparse(one_assignment(runner,'artifact')),
            'namespaceAssumption':'ordinary Paths and registered scalar trial identifiers; fresh stable directories; no hostile filesystem mutation'}


def record_fields(node):
    require(isinstance(node,ast.Dict) and all(isinstance(k,ast.Constant) and type(k.value) is str for k in node.keys), 'dynamic/unpacked control object')
    keys=[k.value for k in node.keys]; require(len(set(keys))==len(keys),'duplicate control field')
    return dict(zip(keys,node.values))


def metadata(trees):
    v1=record_fields(one_assignment(definition(trees['sequential_learning'],'save_policy'),'value'))
    v4=record_fields(one_assignment(definition(trees['ppo_artifact_v4'],'encode_artifact_v4'),'value'))
    records={'v1':v1,'v4':v4}; facts={}; queries=0
    for version,fields in records.items():
        promotion=ast.literal_eval(fields['promotion']); enabled=ast.literal_eval(fields['enabled'])
        expected='rejected_research_only' if version=='v1' else 'research-only'
        require(promotion==expected and enabled is False, 'writer promotes candidate')
        p=z.String('promotion_'+version); false=z.Bool('exact_false_'+version)
        premise=z.And(p==promotion,false==z.BoolVal(enabled is False))
        prove(premise,z.And(p==expected,false));queries+=1
        facts[version]={'promotion':promotion,'enabled':enabled,'keys':sorted(fields)}
    predicates,_=loader_extract(ast.unparse(trees['sequential_learning']))
    # All unrelated metadata atoms remain unconstrained, preserving arbitrary inputs.
    for version,node,container in [('v1',predicates[1],'a'),('v4',definition(trees['ppo_artifact_v4'],'decode_artifact_v4').body[2].body[5].test,'value')]:
        p=z.String('loaded_promotion_'+version); false=z.Bool('loaded_exact_false_'+version)
        atoms={f"{container}['promotion']":p,f"{container}['enabled'] is not False":z.Not(false)}
        comparisons=[n for n in ast.walk(node) if isinstance(n,ast.Compare)]
        for i,n in enumerate(comparisons):
            key=ast.unparse(n)
            if "['promotion']" not in key and "['enabled']" not in key:
                atoms[key]=z.Bool(version+'_other_rejection_'+str(i))
        reject=expression(node,atoms)
        prove(z.Not(reject),z.And(p==facts[version]['promotion'],false));queries+=1
    false_fields=[]
    runner=definition(trees['run_sequential_screen'],'run')
    for n in ast.walk(runner):
        if isinstance(n,ast.Call) and ast.unparse(n.func)=='write_json' and len(n.args)==2 and isinstance(n.args[1],ast.Dict):
            fields=record_fields(n.args[1])
            for key in ('liveAuthorization','promotionAllowed'):
                if key in fields:
                    require(isinstance(fields[key],ast.Constant) and fields[key].value is False,'runner promotion inference')
                    false_fields.append((ast.unparse(n.args[0]),key))
                    prove(z.BoolVal(True),z.Not(z.BoolVal(fields[key].value)));queries+=1
    require(sorted(false_fields)==[("output / 'manifest.json'",'liveAuthorization'),("output / 'manifest.json'",'promotionAllowed'),("output / 'summary.json'",'promotionAllowed')], 'manifest control coverage')
    return facts,{'queries':queries,'runnerFalseFields':sorted(false_fields),'smt':{'F-RL-PROMOTION-METADATA':'unsat'}}


def schema(facts,required):
    from capability_isolation import prove_disjoint
    require(set(required)=={'hiddenSize','params','trainBars','version'}, 'native schema domain drift')
    rows={key:prove_disjoint(value['keys'],required) for key,value in facts.items()}
    for row in rows.values():
        row['requirement']='F-RL-PROMOTION-SCHEMA'
    return {'formats':rows,'smt':{'F-RL-PROMOTION-SCHEMA':'unsat'}}


def lifecycle(facts, mutant=False):
    # Candidate format, current stage, persisted control metadata, live authority.
    initial=[(v,'fresh','none',False) for v in sorted(facts)]
    queue=deque((s,0) for s in initial);seen=set(initial);edges=depth=rejected=0
    while queue:
        state,d=queue.popleft();v,stage,p,live=state;depth=max(depth,d)
        require(not live and p in ('none',facts[v]['promotion']), 'promotion/authority reached')
        targets=[('failure',(v,'failed',p,False)),('disable',(v,'disabled',p,False))]
        if stage=='fresh':targets.append(('save',(v,'saved',facts[v]['promotion'],False)))
        if stage=='saved':
            for promotion,enabled in itertools.product(('research-only','rejected_research_only','shadow','paper','live','unknown'),(False,True,None)):
                good=promotion==facts[v]['promotion'] and enabled is False
                targets.append(('load' if good else 'reject-metadata',(v,'loaded' if good else 'failed',p,False)))
        if stage=='loaded':targets.append(('infer',(v,'proposed',p,False)))
        if stage=='proposed':
            targets.append(('successor',(v,'fresh',p,False)))
            if mutant:targets.append(('policy-promotes',(v,'proposed','live',True)))
        if stage in ('failed','disabled'):targets.append(('retry',(v,'fresh',p,False)))
        for label,target in targets:
            edges+=1;rejected+=label=='reject-metadata'
            if target not in seen:seen.add(target);queue.append((target,d+1))
    require(rejected and any(s[1]=='proposed' for s in seen),'vacuous progression')
    return {'states':len(seen),'transitions':edges,'maxShortestDepth':depth,'formats':2,
            'rejectedMetadataEdges':rejected,'promotionInputs':6,'enabledClasses':3,'terminalStuttering':False,
            'search':'reachable fixed point including retries and successors; no bounded retry cutoff',
            'scope':'source-derived metadata/capability abstraction; existing component/process certificates required'}


def conformance(facts):
    sys.path.insert(0,str(ROOT/'scripts/research'))
    import numpy as np
    import sequential_learning as learning
    import ppo_artifact_v4 as artifact
    # Compile the actual narrow writer declaration, with its two explicit globals.
    # Importing the campaign runner would add pandas to the proof-only toolchain.
    writer = definition(ast.parse((ROOT/source_path('run_sequential_screen')).read_text()), 'write_json')
    namespace = {'Path':Path, 'json':json}
    exec(compile(ast.Module(body=[writer],type_ignores=[]), '<actual-write_json>', 'exec'), namespace)
    from optimizer_snapshot_v2 import Snapshot
    from ppo_successor_v2 import TrainingResult
    prov={'codeCommit':'a'*40,'registrationSha256':'b'*64,'dataSha256':'c'*64,'seed':11,'horizon':1,'algorithm':'ppo','fold':0,
          'promotion':'live','enabled':True}  # Nested data must never override top-level controls.
    provenance={k:('a'*40 if k=='codeCommit' else 'b'*64) for k in ('codeCommit','registrationSha256','dataSha256','fundingSha256','splitSha256','proofSha256')};provenance['datasetRole']='development'
    def snapshot(outputs):
        buffers=tuple(np.zeros(n,dtype='<f8').tobytes() for n in (192,16,16*outputs,outputs))
        return Snapshot('optimizer-snapshot-v2',outputs,4,buffers,buffers,buffers)
    fixture=TrainingResult('ppo-successor-v2',11,1,1,('FIXTURE',),tuple(np.zeros(6,dtype='<f8').tobytes() for _ in range(4)),snapshot(3),snapshot(1),(0.0,)*4)
    raw4=artifact.encode_artifact_v4(fixture,provenance,enabled=True)
    require(raw4 is not None,'synthetic v4 fixture rejected')
    count=accepted=0
    with tempfile.TemporaryDirectory(prefix='trader-promotion-') as d:
        path=Path(d)/'policy.json';net=learning.Network(11);learning.save_policy(path,net,prov);raw1=path.read_bytes()
        for version,raw in (('v1',raw1),('v4',raw4)):
            original=json.loads(raw);require(original['promotion']==facts[version]['promotion'] and original['enabled'] is False,'actual writer mismatch')
            for promotion,enabled in itertools.product(('research-only','rejected_research_only','shadow','paper','live','unknown'),(False,True,0,None,'False',{})):
                value=dict(original,promotion=promotion,enabled=enabled)
                encoded=((json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode() if version=='v1' else artifact._json(value))
                digest=hashlib.sha256(encoded).hexdigest()
                if version=='v1':
                    candidate=Path(d)/('candidate'+str(count)+'.json');candidate.write_bytes(encoded)
                    try:learning.load_policy(candidate,digest,prov);ok=True
                    except ValueError:ok=False
                else:ok=artifact.decode_artifact_v4(encoded,digest,provenance,enabled=True) is not None
                expected=promotion==facts[version]['promotion'] and enabled is False
                require(ok==expected,'actual loader promotion mismatch');count+=1;accepted+=ok
        before=path.read_bytes()
        try:learning.save_policy(path,net,prov)
        except FileExistsError:pass
        else:raise ValueError('promotion boundary: overwrite admitted')
        require(path.read_bytes()==before,'existing artifact modified')
        try:namespace['write_json'](path,{'promotionAllowed':True})
        except FileExistsError:pass
        else:raise ValueError('promotion boundary: runner overwrite admitted')
        require(path.read_bytes()==before,'runner modified existing artifact')
    return {'metadataCases':count,'accepted':accepted,'rejected':count-accepted,'exclusiveCreateCases':2,
            'trainingRuns':0,'fixture':'hand-constructed structural codec fixture; not evidence of training or provenance truth'}


def check_promotion(isolation=None):
    trees,surface=extract();facts,meta=metadata(trees)
    if isolation is None:
        from capability_isolation import check_isolation
        isolation=check_isolation()
    required=isolation['schema']['requiredNativeKeys']
    schemas=schema(facts,required)
    result = {'surface':surface,'metadata':meta,'schema':schemas,'model':lifecycle(facts),
            'conformance':conformance(facts),'smt':{**meta['smt'],**schemas['smt']},
            'composition':{'productionRoots':sorted(isolation['graph']['roots']),
                           'productionAuthorizationVerified':False,'deployedImageVerified':False}}
    # Receipts are JSON values: Python tuple/list inequality must not make a
    # reproduced certificate disagree with its own serialized artifact.
    return json.loads(json.dumps(result, allow_nan=False))
