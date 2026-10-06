"""Source-bound exclusive bundle publication, crash model and filesystem conformance."""
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import z3 as z
from promotion_boundary import shape
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/report_bundle_v2.py'
REGISTRY = 'formal/research/report-bundle-source.json'
EXPORTER = 'scripts/research/summarize_sequential_screen.py'


def require(ok, reason):
    if not ok: raise ValueError('report bundle: '+reason)


def extract(source=None, registry=None, exporter=None):
    tree=ast.parse((ROOT/SOURCE).read_text() if source is None else source)
    reg=json.loads((ROOT/REGISTRY).read_text()) if registry is None else registry
    nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    require(set(nodes)=={'encode_bundle_v2','_matches','_write','publish_bundle_v2'},'definition coverage')
    require(reg['schemaVersion']==1 and set(reg['definitions'])==set(nodes),'registry coverage')
    for name,node in nodes.items():require(shape(node)==reg['definitions'][name],'reviewed body drift: '+name)
    require(hashlib.sha256(shape(tree).encode()).hexdigest()==reg['astSha256'],'module effects drift')
    constants={ast.unparse(n.targets[0]):ast.unparse(n.value) for n in tree.body if isinstance(n,ast.Assign)}
    require(constants==reg['constants'] and constants['VERSION']=="'report-bundle-v2'" and constants['TARGET']=="'report-bundle-v2.json'",'version/destination')
    require([constants[k] for k in ('MAX_REPORT','MAX_COMBINED','MAX_BUNDLE')]==['4194304','16777216','33554432'],'size bounds')
    public=nodes['publish_bundle_v2']
    require([ast.unparse(n) for n in public.args.kw_defaults]==['False','VERSION'],'public default gates')
    require(ast.unparse(public.body[0])=='if enabled is not True or type(version) is not str or version != VERSION:\n    return None','version/enable gate')
    require(ast.unparse(public.body[2])=='raw = encode_bundle_v2(reports)' and ast.unparse(public.body[3])=='if raw is None:\n    return None','encoding before effects')
    io=public.body[6];require(isinstance(io,ast.Try),'publication boundary')
    require(ast.unparse(io.body[-2])=='os.fsync(parent)' and ast.unparse(io.body[-1])=='return hashlib.sha256(raw).hexdigest()','durability before success')
    branch=io.body[1];require(ast.unparse(branch.test)=='not _matches(parent, raw)','retry comparison')
    require(ast.unparse(branch.body[1])=="fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384, dir_fd=parent)",'exclusive staging')
    require(ast.unparse(branch.body[2])=='owned = temporary','cleanup ownership acquired after open')
    require(ast.unparse(branch.body[3].body[0])=='_write(fd, raw)','complete staging before link')
    require(ast.unparse(branch.body[4].body[0])=='os.link(temporary, TARGET, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)','exclusive same-directory publication')
    require(ast.unparse(nodes['_write'].body[-1])=='os.fsync(fd)','file durable before linking')
    require(ast.unparse(nodes['_matches'].body[1].body[-2])=='os.fsync(fd)','reused bytes durable before success')
    require(ast.unparse(io.body[0])=='parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)', 'directory descriptor binding')
    require(ast.unparse(nodes['_matches'].body[0].body[0])=='fd = os.open(TARGET, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)', 'existing target read only')
    unlink=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='os.unlink']
    require(unlink==['os.unlink(owned, dir_fd=parent)'], 'only owned staging cleanup')
    effects=sorted(ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func).startswith('os.'))
    require(effects==reg['effects'],'complete filesystem effects')
    # Encoded metadata is fixed; arbitrary report text is nested as JSON strings.
    record=nodes['encode_bundle_v2'].body[3].body[0].value
    require(isinstance(record,ast.Dict),'bundle schema')
    fields={k.value:v for k,v in zip(record.keys,record.values)}
    require(set(fields)=={'schema','promotion','enabled','liveAuthorization','reports'},'closed bundle schema')
    require(ast.unparse(fields['promotion'])=="'research'" and ast.unparse(fields['enabled'])=='False' and ast.unparse(fields['liveAuthorization'])=='False','non-authorizing metadata')
    exp=ast.parse((ROOT/EXPORTER).read_text() if exporter is None else exporter)
    fn=next(n for n in exp.body if isinstance(n,ast.FunctionDef) and n.name=='export')
    require(shape(fn)==reg['exportFunction'],'complete exporter composition')
    require(ast.unparse(fn.args.kw_defaults[-1])=='False','export default')
    branch=fn.body[-3]
    require(ast.unparse(branch)=="if bundle_v2:\n    result = publish_bundle_v2(str(output), reports, enabled=True)\n    if result is None:\n        raise OSError('report bundle publication failed; retry only with identical verified reports')\n    return result",'admitted report publication binding')
    require(isinstance(fn.body[-4],ast.Try) and 'reconcile_disposition' in ast.unparse(fn.body[-4]),'reconciliation precedes publication')
    return nodes,{'status':'exhaustively_checked','definitions':4,'filesystemCalls':len(effects),
                  'exportCalls':1,'newProductionConsumers':0,'defaultDisabled':True,
                  'scope':'reviewed full source and primitive transfers; POSIX/fsync semantics assumed, not kernel refinement'}


def arithmetic(nodes):
    n,offset,count=z.Ints('bundle_n bundle_offset bundle_count')
    loop=nodes['_write'].body[1]
    require(ast.unparse(loop.test)=='offset < len(raw)' and ast.unparse(loop.body[1].test)=='not 0 < count <= min(65536, len(raw) - offset)','write admission guard')
    require(ast.unparse(loop.body[2])=='offset += count','cursor transfer')
    certify(z.And(n>=1,n<=33554432,offset>=0,offset<n,count>0,count<=65536,count<=n-offset),
            z.And(offset+count>offset,offset+count<=n,n-(offset+count)<n-offset))
    certify(z.And(n>=1,offset>=0,offset<=n,z.Not(offset<n)),offset==n)
    # Opaque immutable byte identity: no hash-equality assumption replaces equality.
    old,expected,staged=z.Ints('bundle_old bundle_expected bundle_staged')
    exists,complete=z.Bools('bundle_exists bundle_complete')
    published=z.If(exists,old,staged)
    success=z.And(z.If(exists,old==expected,z.And(complete,staged==expected)))
    certify(success,z.And(published==expected,z.Implies(exists,published==old)))
    certify(z.And(exists,old!=expected),z.Not(success))
    return {'F-RL-BUNDLE-ARITH':'unsat'}


# Global state=(visible target, durable directory target, writer0 phase, writer1
# phase, acknowledged mask). Content0=absent,1/2=different complete bundles.
# Private partial files are abstracted by phase; no target aliases a partial file.
# A crash may retain an unsynced link or roll it back to durable directory state.
def successors(s, contents, mutant=False):
    visible,durable,p0,p1,ack=s;phases=(p0,p1)
    for who,phase in enumerate(phases):
        wanted=contents[who]
        def go(label,next_phase,vis=visible,dur=durable,mask=ack):
            ps=list(phases);ps[who]=next_phase
            return label,(vis,dur,*ps,mask)
        if phase=='start':
            yield go('existing-match','matched') if visible==wanted else go('stage' if visible==0 else 'conflict','opened' if visible==0 else 'failed')
        elif phase=='opened':
            yield go('partial-write','partial')
            if mutant and visible==0:yield go('bad-partial-link','linked',vis=3)
        elif phase=='partial':yield go('write-complete','complete')
        elif phase=='complete':yield go('file-fsync','synced')
        elif phase=='synced':
            if visible==0:yield go('exclusive-link','linked',vis=wanted)
            else:yield go('race-match' if visible==wanted else 'race-conflict','matched' if visible==wanted else 'failed')
        elif phase in ('linked','matched'):yield go('directory-fsync','durable',dur=visible)
        elif phase=='durable':yield go('ack','done',mask=ack|(1<<who))
        elif phase in ('done','failed'):yield go('retry','start')
        # Failing a syscall never acknowledges; retries start with fresh staging.
        if phase not in ('done','failed'):yield go('io-failure','failed')
    for retained in {visible,durable}:
        yield 'crash',(retained,retained,'start','start',ack)


def check_model(mutant=False):
    total_states=total_edges=maximum=0;recovery=0;rank_edges=0;max_rank=0
    for contents in ((1,1),(1,2)):
        initial=(0,0,'start','start',0);queue=deque([initial]);depth={initial:0};edges=0
        while queue:
            s=queue.popleft();visible,durable,p0,p1,ack=s
            require(visible in (0,1,2) and durable in (0,1,2),'partial target publication')
            require(durable==0 or visible==durable,'durable target replaced')
            for who,value in enumerate(contents):
                require(not ack&(1<<who) or visible==durable==value,'acknowledged bytes lost or wrong')
            for label,nxt in successors(s,contents,mutant):
                edges+=1
                require(not visible or nxt[0]==visible or (label=='crash' and durable==0),'existing target overwritten')
                if nxt not in depth:depth[nxt]=depth[s]+1;queue.append(nxt)
        # Every same-intent reachable state has a finite crash-free/failure-free
        # path to both acknowledgements. Weak fairness and eventual healthy IO
        # are separate environmental assumptions, not proved scheduler behavior.
        if contents==(1,1):
            ranks={'start':8,'opened':7,'partial':6,'complete':5,'synced':4,'linked':3,'matched':3,'durable':2,'done':0,'failed':9}
            def rank(state):return sum(0 if state[-1]&(1<<i) else ranks[state[2+i]] for i in range(2))
            max_rank=max(rank(state) for state in depth)
            for state in depth:
                for label,nxt in successors(state,contents):
                    if label in ('crash','io-failure'):continue
                    rank_edges+=1
                    pending_actor=any(not state[-1]&(1<<i) and state[2+i]!=nxt[2+i] for i in range(2))
                    require(rank(nxt)<rank(state) if pending_actor else rank(nxt)==rank(state),'healthy recovery rank does not decrease')
            reverse={s:[] for s in depth};winning={s for s in depth if s[-1]==3}
            for s in depth:
                for label,nxt in successors(s,contents):
                    if label not in ('crash','io-failure'):reverse[nxt].append(s)
            queue=deque(winning)
            while queue:
                for previous in reverse[queue.popleft()]:
                    if previous not in winning:winning.add(previous);queue.append(previous)
            require(winning==set(depth),'recoverable state permanently locked out')
            recovery=len(winning)
        total_states+=len(depth);total_edges+=edges;maximum=max(maximum,max(depth.values()))
    return {'status':'model_checked','states':total_states,'transitions':total_edges,'maxShortestDepth':maximum,
            'writers':2,'contentCases':2,'recoverableSameIntentStates':recovery,
            'healthyRankEdges':rank_edges,'maximumRecoveryRank':max_rank,
            'search':'reachable fixed point including arbitrary crashes/retries',
            'scope':'durable file before exclusive link; modeled fsync/crash semantics assumed; eventual healthy IO and fairness for recovery'}


def conformance():
    sys.path.insert(0,str(ROOT/'scripts/research'))
    import report_bundle_v2 as b
    import summarize_sequential_screen as exporter
    from champion_archive import prepare_fixture
    with tempfile.TemporaryDirectory(prefix='trader-bundle-') as tmp:
        root=Path(tmp);source,kwargs,reports=prepare_fixture(root)
        output=root/'output';output.mkdir()
        expected=b.encode_bundle_v2(reports);require(expected is not None,'fixture encoding')
        digest=exporter.export(source,output,bundle_v2=True,**kwargs)
        path=output/b.TARGET;inode=path.stat().st_ino
        require(path.read_bytes()==expected and digest==hashlib.sha256(expected).hexdigest(),'exporter bundle bytes')
        require(exporter.export(source,output,bundle_v2=True,**kwargs)==digest and path.stat().st_ino==inode,'retry identity')
        decoded=json.loads(expected)
        require({k:v.encode() for k,v in decoded['reports'].items()}==reports,'legacy report parity')
        with patch.object(exporter,'publish_bundle_v2',side_effect=AssertionError('unverified publication')):
            try:exporter.export(source,output,bundle_v2=True,**dict(kwargs,expected_index_sha256='wrong-hash'))
            except ValueError:pass
            else:raise ValueError('report bundle: invalid source accepted')
            # Re-hash a synthetic contradictory summary so hash admission passes
            # and the actual disposition/reconciliation guard must reject it.
            summary_path=source/'summary.json';summary=json.loads(summary_path.read_bytes())
            summary['promotionAllowed']=True;summary_path.write_text(json.dumps(summary))
            index_path=source/'evidence-index.json';index=json.loads(index_path.read_bytes())
            index['summary.json']=hashlib.sha256(summary_path.read_bytes()).hexdigest()
            index_path.write_text(json.dumps(index));bad_hash=hashlib.sha256(index_path.read_bytes()).hexdigest()
            try:exporter.export(source,output,bundle_v2=True,**dict(kwargs,expected_index_sha256=bad_hash))
            except ValueError:pass
            else:raise ValueError('report bundle: unreconciled evidence published')
    return {'status':'property_tested','verifiedExporterCases':4,'reportByteParity':7,
            'marketDataReads':0,'powerFailureHardwareTested':False}


def check_bundle():
    reg=json.loads((ROOT/'research-notes/registrations/report-bundle-engineering.json').read_text())
    require([reg[k] for k in ('maxReportBytes','maxCombinedBytes','maxBundleBytes')]==[4194304,16777216,33554432],'registered sizes')
    require(reg['financialTrials']==reg['historicalDataReads']==0 and reg['holdoutOpened'] is False,'research boundary')
    nodes,surface=extract()
    return {'surface':surface,'smt':arithmetic(nodes),'queries':4,'premiseChecks':4,'model':check_model(),'conformance':conformance()}
