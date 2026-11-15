"""Source-bound exact replay arithmetic and bounded terminal-publication model."""
import ast
from collections import deque
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import z3 as z
from promotion_boundary import shape
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/replay_accounting_v2.py'
REGISTRY = 'formal/research/replay-accounting-v2-source.json'


def require(ok, reason):
    if not ok:
        raise ValueError('exact replay: ' + reason)


def extract(source=None, registry=None):
    tree = ast.parse((ROOT / SOURCE).read_text() if source is None else source)
    lock = json.loads((ROOT / REGISTRY).read_text()) if registry is None else registry
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    require(len(nodes) == sum(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body), 'duplicate definition')
    require({k: shape(v) for k, v in nodes.items()} == lock['definitions'], 'reviewed definition drift')
    require(hashlib.sha256(shape(tree).encode()).hexdigest() == lock['astSha256'], 'module effect drift')
    for path, value in lock['supportHashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value, 'support drift: ' + path)
    constants = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    require(constants == {'VERSION': "'replay-accounting-v2'", 'MAX_BITS': '8192', 'MAX_TICKS': '4096',
                         'MAX_EVENTS': '128', 'SQRT_GRID': '2 ** 32',
                         '__all__': "['VERSION', 'State', 'Costs', 'Receipt', 'initial_v2', 'advance_v2']"}, 'version/bounds/exports')
    require([ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))] ==
            ['from __future__ import annotations', 'from dataclasses import dataclass',
             'from fractions import Fraction as F', 'from math import isqrt'], 'effects import boundary')
    for name in ('State', 'Costs', 'Receipt'):
        require([ast.unparse(n) for n in nodes[name].decorator_list] == ['dataclass(frozen=True, slots=True)'], 'immutable output')
    for name in ('initial_v2', 'advance_v2'):
        node = nodes[name]
        defaults = dict(zip((x.arg for x in node.args.kwonlyargs), map(ast.unparse, node.args.kw_defaults)))
        require(defaults['enabled'] == 'False' and defaults['version'] == 'VERSION', 'disabled default')
        require(ast.unparse(node.body[0]) == "if enabled is not True or type(version) is not str or version != VERSION:\n    return None", 'activation first')
    for name, op in (('_add', 'a + b'), ('_mul', 'a * b'), ('_div', 'a / b')):
        require(ast.unparse(nodes[name].body[0]) == 'return _bounded(' + op + ')', 'checked primitive')
    require(ast.unparse(nodes['_bounded'].body[0].test) == 'not _valid(x)', 'numeric guard')
    require(ast.unparse(nodes['_valid'].body[0].value) ==
            'type(x) is F and x.numerator.bit_length() <= MAX_BITS and (x.denominator.bit_length() <= MAX_BITS)', 'exact admission bound')
    return nodes, {'status': 'exhaustively_checked', 'definitions': len(nodes), 'checkedPrimitives': 3,
                   'defaultDisabledEntries': 2, 'scope': 'complete reviewed AST and extracted arithmetic; runtime semantics assumed'}


def expression(node, atoms):
    label = ast.unparse(node)
    if label in atoms:
        return atoms[label]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -expression(node.operand, atoms)
    if isinstance(node, ast.Call) and ast.unparse(node.func) in ('_add', '_mul', '_div'):
        require(len(node.args) == 2 and not node.keywords, 'primitive shape')
        a, b = (expression(v, atoms) for v in node.args)
        return {'_add': lambda: a+b, '_mul': lambda: a*b, '_div': lambda: a/b}[ast.unparse(node.func)]()
    raise ValueError('exact replay: unsupported expression ' + label)


def prove(nodes):
    assignments = {ast.unparse(n.targets[0]): n.value for n in nodes['_advance'].body if isinstance(n, ast.Assign)}
    equity, units, p0, p1, fund, fee, spread, slip, impact = z.Reals('ac_e ac_u ac_p0 ac_p1 ac_f ac_fee ac_spread ac_slip ac_impact')
    atoms = {'state.equity': equity, 'state.units': units, 'price': p1, 'state.price': p0, 'funding_per_unit': fund}
    for name in ('gross', 'funding', 'equity'):
        atoms[name] = expression(assignments[name], atoms)
    charges = {'costs.fee': fee, 'costs.spread': spread, 'costs.slippage': slip, 'costs.impact': impact}
    debit = expression(nodes['_debit'].body[0].value, charges)
    certify(z.And(p0 > 0, p1 > 0), atoms['equity'] - debit == equity + units*(p1-p0) - units*fund - fee-spread-slip-impact)
    certify(z.And(fee >= 0, spread >= 0, slip >= 0, impact >= 0), z.And(debit >= 0, atoms['equity']-debit <= atoms['equity']))
    # Inductive composition: a rebalance followed by liquidation debits both once.
    f2, s2, l2, i2 = z.Reals('ac_f2 ac_s2 ac_l2 ac_i2')
    debit2 = f2+s2+l2+i2
    certify(z.BoolVal(True), (atoms['equity']-debit)-debit2 == atoms['equity']-((fee+f2)+(spread+s2)+(slip+l2)+(impact+i2)))
    target, fill, old, e, p = z.Reals('ac_target ac_fill ac_old ac_eq ac_price')
    branch = next(n for n in nodes['_advance'].body if isinstance(n, ast.If) and ast.unparse(n.test) == 'not end')
    terms = {ast.unparse(n.targets[0]): n.value for n in branch.body if isinstance(n, ast.Assign)}
    a = {'target': target, 'equity': e, 'price': p, 'units': old, 'fill': fill}
    a['desired'] = expression(terms['desired'], a)
    a['new'] = expression(terms['new'], a)
    certify(z.And(e>0, p>0, fill==1), a['new']*p == target*e)
    certify(z.And(e>0,p>0,fill>0,fill<=1,old==a['desired']), a['new']==old)
    # Fraction primitives: signed products and sums before GCD reduction.
    b=z.IntVal(2**8192);n,m,d,q=z.Ints('ac_n ac_m ac_d ac_q')
    size=z.And(n>=0,n<b,m>=0,m<b,d>=1,d<b,q>=1,q<b)
    certify(size,z.And(n*m < b*b,d*q < b*b,n*q+m*d < 2*b*b))
    # isqrt primitive contract floor(sqrt(a/b)): k^2*b <=a<(k+1)^2*b.
    num, den, k = z.Ints('ac_num ac_den ac_k')
    floor=z.And(num>=0,den>0,k>=0,k*k*den<=num,num<(k+1)*(k+1)*den)
    upper=z.If(k*k*den==num,k,k+1)
    certify(floor,z.And(upper*upper*den>=num,upper>=0))
    certify(floor,z.Or(upper==0,(upper-1)*(upper-1)*den<num))
    # Exact post-cost risk test is detection, not an impossible gap-prevention claim.
    peak=z.Real('ac_peak')
    certify(z.And(e>0,peak>=e,peak>0,e>=z.RealVal('4/5'),1-e/peak<=z.RealVal('3/20'),
                  old*p/e<=z.RealVal('7/20'),old*p/e>=-z.RealVal('7/20')),
            z.And(e>=z.RealVal('4/5'),e>=z.RealVal('17/20')*peak,old*p<=z.RealVal('7/20')*e,old*p>=-z.RealVal('7/20')*e))
    return {'F-RL-ACCOUNT-V2-ARITH': 'unsat'}


# Model=(phase, terminal, solvent, flat, liquidation, receipt).
# Arithmetic failure at every private phase discards all candidate state.
def successors(state, mutant=False):
    phase,end,solvent,flat,liquidation,published=state
    if phase in ('absent','published'):
        return []
    out=[('reject',('absent',end,solvent,flat,liquidation,False))]
    if phase=='start':
        out += [('mark',('marked',e,s,f,'not_required',False)) for e in (False,True) for s in (False,True) for f in (False,True) if s or e]
    elif phase=='marked':
        if end:
            out.append(('close',('closing',True,solvent,flat,'not_required',False)))
        else:
            out += [('rebalance',('rebalanced',e,s,f,'not_required',False)) for e in (False,True) for s in (False,True) for f in (False,True) if s or e]
    elif phase=='rebalanced':
        out.append(('close',('closing',True,solvent,flat,'not_required',False)) if end else
                   ('publish',('published',False,True,flat,'not_required',True)))
    elif phase=='closing':
        if flat or solvent:
            out += [('liquidate',('published',True,s,True,'flat',True)) for s in (False,True)]
        else:
            out.append(('failed-liquidation',('published',True,False,False,'failed',True)))
    else:
        raise ValueError('exact replay: unknown model phase')
    if mutant and phase=='marked':
        out.append(('publish-incomplete',('published',True,False,False,'not_required',True)))
    return out


def model(mutant=False):
    initial=('start',False,True,True,'not_required',False)
    depth={initial:0};queue=deque([initial]);edges=0
    while queue:
        state=queue.popleft();phase,end,solvent,flat,liquidation,published=state
        require((phase=='published') == published,'partial publication')
        if published:
            require((not end and solvent and liquidation=='not_required') or
                    (end and ((flat and liquidation=='flat') or (not flat and not solvent and liquidation=='failed'))), 'terminal accounting status')
        following=successors(state,mutant)
        require(bool(following) or phase in ('absent','published'),'nonterminal deadlock')
        for _,nxt in following:
            edges+=1
            if nxt not in depth:
                depth[nxt]=depth[state]+1;queue.append(nxt)
    require(any(s[0]=='published' and s[4]=='failed' for s in depth),'vacuous failed-liquidation coverage')
    return {'status':'model_checked','states':len(depth),'transitions':edges,'maxShortestDepth':max(depth.values()),
            'fundingEventsAbstracted':True,'orderTransitions':0,'scope':'finite terminal/publication abstraction; no whole-language refinement or physical liveness'}


def conformance():
    sys.path.insert(0,str(ROOT/'scripts/research'))
    import replay_accounting_v2 as a
    rng=random.Random(20261006);wire=[];expected=[];count=0
    for case in range(128):
        state=a.initial_v2(F(100),enabled=True)
        for tick in range(12):
            price=F(rng.randrange(70,131))
            events=((price,F(rng.randrange(-3,4),10000)),)
            target=rng.choice((F(-1,4),F(0),F(1,4)))
            fill=rng.choice((F(1),F(1,2)))
            result=a.advance_v2(state,price,events,target,
                                terminal=tick==11,fill=fill,impact=F(1,10000),enabled=True)
            require(result is not None,'ordinary path absent')
            require(result==a.advance_v2(state,price,events,target,terminal=tick==11,
                    fill=fill,impact=F(1,10000),enabled=True),'deterministic transition')
            c=result.costs
            values=(state.equity,state.units,state.price,price,sum((m*r for m,r in events),F(0)),c.fee,c.spread,c.slippage,c.impact)
            wire.append(str([(v.numerator,v.denominator) for v in values]))
            expected.append(str((result.after.equity.numerator,result.after.equity.denominator)).replace(' ',''))
            require(result.after.equity==state.equity+result.gross+result.funding-c.fee-c.spread-c.slippage-c.impact,'wealth identity')
            require(not result.after.terminal or result.liquidation in ('flat','failed'),'terminal receipt')
            count+=1;state=result.after
            if state.terminal:
                require(a.advance_v2(state,price,(),F(0),enabled=True) is None,'terminal continuation')
                break
    cmd=['runghc','-i'+str(ROOT/'haskell/app'),str(ROOT/'formal/research/ReplayAccountingV2.hs')]
    out=subprocess.run(cmd,input='\n'.join(wire)+'\n',text=True,capture_output=True,timeout=60,check=True)
    require(out.stdout.splitlines()==expected,'Haskell exact ledger conformance')
    return {'status':'property_tested','generatedEpisodes':128,'transitions':count,'seed':20261006,
            'haskellExactRows':count,'marketDataReads':0,'economicEvidence':False}


def check_accounting():
    registered=json.loads((ROOT/'research-notes/registrations/replay-accounting-v2-engineering.json').read_text())
    require((registered['maximumTicks'],registered['maximumFundingEvents'],registered['maximumRationalBits'],registered['sqrtFractionalBits'])==(4096,128,8192,32),'registration')
    nodes,source=extract()
    return {'source':source,'smt':prove(nodes),'model':model(),'conformance':conformance()}
