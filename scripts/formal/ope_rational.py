"""Exact OPE recurrences and finite publication model with source conformance."""
import ast
from collections import deque
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys

import z3 as z
from promotion_boundary import shape
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/ope_rational_v2.py'
REGISTRY = 'formal/research/ope-rational-source.json'
NAMES = {'Episode', 'EpisodeEstimate', 'Estimate', '_bounded', '_add', '_mul', '_div',
         '_episode_valid', '_episode', 'estimate_v2'}


def require(ok, reason):
    if not ok: raise ValueError('exact OPE: ' + reason)


def extract(source=None, registry=None):
    tree = ast.parse((ROOT/SOURCE).read_text() if source is None else source)
    registry = json.loads((ROOT/REGISTRY).read_text()) if registry is None else registry
    nodes = {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    require(set(nodes)==NAMES and len(nodes)==sum(isinstance(n,(ast.FunctionDef,ast.ClassDef)) for n in tree.body), 'definition coverage')
    require(set(registry['definitions'])==NAMES and registry['schemaVersion']==1, 'registry coverage')
    for name,node in nodes.items(): require(shape(node)==registry['definitions'][name], 'reviewed body drift: '+name)
    require(hashlib.sha256(shape(tree).encode()).hexdigest()==registry['astSha256'], 'module effects drift')
    constants={ast.unparse(n.targets[0]):ast.unparse(n.value) for n in tree.body if isinstance(n,ast.Assign)}
    require(constants=={'VERSION':"'ope-rational-v2'",'MAX_EPISODES':'256','MAX_DECISIONS':'32','MAX_BITS':'8192'}, 'bounds/version')
    for name in ('Episode','EpisodeEstimate','Estimate'):
        require([ast.unparse(n) for n in nodes[name].decorator_list]==['dataclass(frozen=True, slots=True)'], 'immutable envelope')
    public=nodes['estimate_v2']
    require([ast.unparse(n) for n in public.args.kw_defaults]==['False','VERSION'], 'disabled defaults')
    require(ast.unparse(public.body[0])=='if enabled is not True or type(version) is not str or version != VERSION:\n    return None', 'explicit activation')
    require(ast.unparse(public.body[6])=='if not all((_episode_valid(episode, horizon) for episode in episodes)):\n    return None', 'whole batch admission')
    require(isinstance(public.body[7],ast.Try) and len(public.body)==8, 'private publication boundary')
    require(ast.unparse(nodes['_bounded'].body[0].test)=='value.numerator.bit_length() > MAX_BITS or value.denominator.bit_length() > MAX_BITS', 'numeric rejection predicate')
    for name,op in (('_add','a + b'),('_mul','a * b'),('_div','a / b')):
        require(ast.unparse(nodes[name].body[0])=='return _bounded('+op+')', 'checked primitive '+name)
    loop=nodes['_episode'].body[5]
    require(isinstance(loop,ast.For) and ast.unparse(loop.iter)=='range(len(episode.rewards))', 'decision loop')
    assignments={ast.unparse(n.targets[0]):n.value for n in loop.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    require(set(assignments)=={'weight','total','pdis','residual','dr','discount'}, 'recurrence coverage')
    batch=public.body[7].body
    require(isinstance(batch[3],ast.For) and ast.unparse(batch[3].iter)=='episodes', 'batch loop')
    aggregates={ast.unparse(n.targets[0]):n.value for n in batch[3].body if isinstance(n,ast.Assign)}
    result=nodes['_episode'].body[-1].value
    require(isinstance(result,ast.Call) and ast.unparse(result.func)=='EpisodeEstimate', 'episode result')
    return (assignments,aggregates,result,batch), {'status':'exhaustively_checked','definitions':len(nodes),
            'checkedPrimitives':3,'defaultDisabledEntries':1,'immutableRepresentations':3,
            'scope':'reviewed complete module plus extracted recurrences; primitive runtime semantics assumed'}


def expression(node, atoms):
    label=ast.unparse(node)
    if label in atoms:return atoms[label]
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -expression(node.operand,atoms)
    if isinstance(node,ast.Call) and ast.unparse(node.func) in ('_add','_mul','_div') and len(node.args)==2 and not node.keywords:
        a,b=(expression(n,atoms) for n in node.args)
        return {'_add':lambda:a+b,'_mul':lambda:a*b,'_div':lambda:a/b}[ast.unparse(node.func)]()
    raise ValueError('exact OPE: unsupported expression '+label)


def prove(extracted):
    assignments,aggregates,result,batch=extracted
    w,d,g,r,b,p,q,v,total,pdis,dr=z.Reals('ov_w ov_d ov_g ov_r ov_b ov_p ov_q ov_v ov_total ov_pdis ov_dr')
    atoms=dict(zip(('weight','discount','gamma','reward','behavior','target','q','next_v','total','pdis','dr'),
                   (w,d,g,r,b,p,q,v,total,pdis,dr)))
    premise=z.And(w>=0,d>=0,d<=1,g>=0,g<=1,b>0,b<=1,p>=0,p<=1)
    for name,node in assignments.items():atoms[name]=expression(node,atoms)
    certify(premise,z.And(atoms['weight']==w*p/b,atoms['weight']>=0,
                         atoms['discount']==d*g,atoms['discount']>=0,atoms['discount']<=1))
    certify(premise,z.And(atoms['total']==total+d*r,atoms['pdis']==pdis+w*p/b*d*r,
                         atoms['dr']==dr+w*p/b*d*(r+g*v-q)))
    certify(z.And(premise,w>0,p>0),atoms['weight']>0)
    certify(z.And(premise,p==0),atoms['weight']==0)
    certify(premise,expression(result.args[2],{'weight':w,'total':total})==w*total)
    s,ss,i,pp,dd,wi,ii,pi,di=z.Reals('ov_s ov_ss ov_i ov_pp ov_dd ov_wi ov_ii ov_pi ov_di')
    accumulator={'weights':s,'squares':ss,'ordinary':i,'per_decision':pp,'robust':dd,
                 'value.weight':wi,'value.ordinary_is':ii,'value.per_decision_is':pi,'value.doubly_robust':di}
    for name in ('weights','squares','ordinary','per_decision','robust'):
        accumulator[name]=expression(aggregates[name],accumulator)
    certify(wi>=0,z.And(accumulator['weights']==s+wi,accumulator['squares']==ss+wi*wi,
                         accumulator['ordinary']==i+ii,accumulator['per_decision']==pp+pi,accumulator['robust']==dd+di))
    # Integer bit-length bound: abs numerator and denominator <2^8192.
    # Fraction binary operations need products below B^2 and signed sum below2B^2.
    bound=z.IntVal(2**8192);a1,a2,c1,c2=z.Ints('ov_a1 ov_a2 ov_c1 ov_c2')
    sized=z.And(a1>=0,a1<bound,a2>=0,a2<bound,c1>=1,c1<bound,c2>=1,c2<bound)
    certify(sized,z.And(a1*a2<bound*bound,c1*c2<bound*bound,a1*c2+a2*c1<2*bound*bound))
    from ess_v2 import invariant
    k=z.Real('ov_prefix_count')
    certify(z.BoolVal(True), invariant(0,z.RealVal(0),z.RealVal(0)))
    certify(z.And(invariant(k,s,ss),wi>=0),
            invariant(k+1,accumulator['weights'],accumulator['squares']))
    publication=batch[-1].value
    require(isinstance(publication,ast.Call) and ast.unparse(publication.func)=='Estimate', 'batch result')
    count=z.Real('ov_count')
    final_atoms={'ordinary':i,'per_decision':pp,'robust':dd,'count':count}
    certify(z.And(count>=1,count<=256),z.And(
        expression(publication.args[1],final_atoms)==i/count,
        expression(publication.args[2],final_atoms)==pp/count,
        expression(publication.args[4],final_atoms)==dd/count))
    # Strict positivity guards are extracted from actual publication branches.
    wis=next(n.value for n in batch if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='wis')
    ess=next(n.value for n in batch if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='ess')
    require(ast.unparse(wis.test)=='weights > 0' and ast.unparse(wis.orelse)=='None', 'WIS no-support branch')
    require(ast.unparse(ess.test)=='squares > 0' and ast.unparse(ess.orelse)=='Fraction(0)', 'ESS no-support branch')
    k=z.Real('ov_k');domain=z.And(k>=1,k<=256,s>=0,ss>0,ss<=s*s,s*s<=k*ss)
    value=expression(ess.body,{'weights':s,'squares':ss})
    certify(domain,z.And(value>=1,value<=k))
    certify(s>0,expression(wis.body,{'ordinary':i,'weights':s})==i/s)
    return {'F-RL-OPE-V2-ARITH':'unsat'}


# Model state=(phase, validated episodes, completed episodes, current decisions).
# Batch size2, horizon3. Invalid data rejects before arithmetic; arithmetic/resource
# failure rejects after any prefix. No partial envelope or external effects.
def transitions(state, mutant=False):
    phase,valid,done,step=state
    if phase in ('absent','published'):return []
    out=[('reject',('absent',valid,done,step))]
    if phase=='start':out.append(('enable',('validate',0,0,0)))
    elif phase=='validate':
        out.append(('validate',('validate',valid+1,0,0)) if valid<2 else ('begin',('compute',valid,0,0)))
    elif phase=='compute':
        if step<3:out.append(('step',('compute',valid,done,step+1)))
        elif done<1:out.append(('episode',('compute',valid,done+1,0)))
        else:out.append(('publish',('published',valid,2,0)))
        if mutant:out.append(('partial-publish',('published',valid,done,step)))
    else:raise ValueError('unknown OPE model state')
    return out


def check_model(mutant=False):
    initial=('start',0,0,0);queue=deque([initial]);depth={initial:0};edges=0
    while queue:
        state=queue.popleft();phase,valid,done,step=state
        require(phase!='compute' or valid==2, 'arithmetic before whole admission')
        require(phase!='published' or (valid,done,step)==(2,2,0), 'partial publication')
        successors=transitions(state,mutant)
        require(bool(successors) or phase in ('absent','published'), 'nonterminal deadlock')
        for _,nxt in successors:
            edges+=1
            if nxt not in depth:depth[nxt]=depth[state]+1;queue.append(nxt)
    require(any(s[0]=='published' for s in depth),'vacuous publication')
    return {'status':'model_checked','states':len(depth),'transitions':edges,'maxShortestDepth':max(depth.values()),
            'episodes':2,'decisions':3,'search':'reachable_fixed_point','orderTransitions':0,
            'scope':'bounded publication protocol; no whole-language refinement or wall-clock liveness claim'}


def oracle(episodes,gamma):
    g=Fraction.from_float(gamma);rows=[]
    for e in episodes:
        r,b,p,q,v=[tuple(Fraction.from_float(x) for x in xs) for xs in (e.rewards,e.behavior,e.target,e.q,e.v)]
        def product(xs):
            result=Fraction(1)
            for x in xs:result*=x
            return result
        ws=[product(p[j]/b[j] for j in range(t+1)) for t in range(len(r))]
        total=sum((g**t*r[t] for t in range(len(r))),Fraction(0))
        rows.append((total,ws[-1],ws[-1]*total,
                     sum((ws[t]*g**t*r[t] for t in range(len(r))),Fraction(0)),
                     v[0]+sum((ws[t]*g**t*(r[t]+g*v[t+1]-q[t]) for t in range(len(r))),Fraction(0))))
    s=sum(row[1] for row in rows);ss=sum(row[1]**2 for row in rows);n=len(rows)
    return (sum(row[2] for row in rows)/n,sum(row[3] for row in rows)/n,
            sum(row[2] for row in rows)/s if s else None,sum(row[4] for row in rows)/n,
            s*s/ss if ss else Fraction(0),tuple(rows))


def conformance():
    sys.path.insert(0,str(ROOT/'scripts/research'))
    import ope_rational_v2 as o
    rng=random.Random(20261006)
    for _ in range(96):
        n=rng.randrange(1,9);h=rng.randrange(1,9);episodes=[]
        for _ in range(n):
            signed=lambda:tuple(rng.choice((-2.0,-0.125,0.0,0.25,1.0)) for _ in range(h))
            episodes.append(o.Episode(signed(),tuple(rng.randrange(3) for _ in range(h)),
                                     tuple(rng.choice((0.125,0.25,0.75,1.0)) for _ in range(h)),
                                     tuple(rng.choice((0.0,0.125,0.75,1.0)) for _ in range(h)),signed(),signed()+(0.0,)))
        batch=tuple(episodes);gamma=rng.choice((0.0,0.25,0.99,1.0));got=o.estimate_v2(batch,gamma,enabled=True)
        require(got is not None,'oracle publication')
        actual=(got.ordinary_is,got.per_decision_is,got.weighted_is,got.doubly_robust,got.effective_sample_size,
                tuple((e.discounted_return,e.weight,e.ordinary_is,e.per_decision_is,e.doubly_robust) for e in got.episodes))
        require(actual==oracle(batch,gamma),'exact independent oracle')
        require(got==o.estimate_v2(batch,gamma,enabled=True) and got.reliable is False,'deterministic descriptive envelope')
    return {'status':'property_tested','oracleCases':96,'seed':20261006,'deterministicRepeats':96,
            'marketDataReads':0,'statisticalReliabilityVerified':False}


def check_ope():
    registration=json.loads((ROOT/'research-notes/registrations/ope-rational-engineering.json').read_text())
    require((registration['maximumEpisodes'],registration['maximumDecisions'],registration['maximumRationalBits'])==(256,32,8192), 'registered bounds')
    require(registration['financialTrials']==0 and registration['historicalDataReads']==0 and registration['holdoutOpened'] is False,'research isolation')
    extracted,surface=extract()
    return {'surface':surface,'smt':prove(extracted),'queries':12,'premiseChecks':12,
            'model':check_model(),'conformance':conformance()}
