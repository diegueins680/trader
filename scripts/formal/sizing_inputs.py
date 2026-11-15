"""Source-bound sizing admission, retry control and compiled pure Main fragments."""
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import struct
import subprocess
import tempfile
import textwrap
import z3 as z
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
CORE = 'haskell/app/Trader/QuantityRounding.hs'
MAIN = 'haskell/app/Main.hs'
REGISTRY = 'formal/research/sizing-input-source.json'
FIXTURE = 'formal/research/sizing-input-counterexamples.json'
ERRORS = ['Invalid minimum notional.', 'Invalid quantity filter.', 'Invalid quantity bounds.',
          'Invalid sizing price.', 'Missing sizing price for minimum notional.']
FUNCTIONS = ['validateProbeQuote','normalizeProbeQty','effectiveStep','effectiveMinQty','effectiveMaxQty',
             'stepValue','quantizeUp','minTradeQty','minTradeQtyMaybe','isLongSpot','isTooSmallQtyError',
             'normalizeQty','normalizeEntryQty']


def require(ok, message):
    if not ok:
        raise ValueError('sizing inputs: '+message)


def fragments(main):
    a=main.index('    validateProbeQuote mSf qq ='); b=main.index('\ncomputeCoinbaseKeysStatusFromArgs',a)
    c=main.index('    effectiveStep sf =',main.index('    effectiveStep sf =')+1)
    d=main.index('    sendMarketOrderWithMaker ::',c)
    return {'probe':main[a:b], 'sizing':main[c:d]}


def extract(core=None,main=None,registry=None):
    core=(ROOT/CORE).read_text() if core is None else core
    main=(ROOT/MAIN).read_text() if main is None else main
    registry=json.loads((ROOT/REGISTRY).read_text()) if registry is None else registry
    require(hashlib.sha256(core.encode()).hexdigest()==registry['coreSha256'],'core drift')
    require(fragments(main)==registry['fragments'],'caller drift')
    require(registry['functions']==FUNCTIONS,'caller roster drift')
    for line in [
        'validateSizingInputs (effectiveMinQty sf) (effectiveMaxQty sf) (sfMinNotional sf) (Just price)',
        'validateSizingInputs (mSf >>= effectiveMinQty) (mSf >>= effectiveMaxQty) (mSf >>= sfMinNotional) mPrice',
        'validateMinimumNotional (mSf >>= sfMinNotional)',
        'validateQuantityInput Nothing qty2']:
        require(main.count(line)==1,'missing/duplicate sizing boundary '+line)
    require('| isNaN x || isInfinite x || x < 0 = Left "Invalid quantity input."' in core,'nonnegative raw input guard')
    require(core.count('finiteNonnegative x = not (isNaN x || isInfinite x) && x >= 0')==1,'finite metadata guard')
    require('Just p\n                    | isNaN p || isInfinite p || p <= 0 -> Left "Invalid sizing price."' in core,'price guard')
    require('(Just lo, Just hi) | lo > hi -> Left "Invalid quantity bounds."' in core,'bounds ordering')
    require('maybe False (> 0) minimumNotional -> Left "Missing sizing price for minimum notional."' in core,'required availability')
    require(re.findall(r'Left "([^"]+)"',core)==['Invalid quantity input.','Invalid quantity step.',*ERRORS],'error roster')
    return registry['fragments']


def prove():
    fp=z.Float64(); lo,hi,n,p,q=[z.FP('size_'+s,fp) for s in ['lo','hi','n','p','q']]; zero=z.FPVal(0,fp)
    hl,hh,hn,hp=z.Bools('size_has_lo size_has_hi size_has_n size_has_p')
    finite=lambda x:z.And(z.Not(z.fpIsNaN(x)),z.Not(z.fpIsInf(x)))
    nonnegative=lambda x:z.And(finite(x),z.fpGEQ(x,zero))
    valid=z.And(z.Implies(hl,nonnegative(lo)),z.Implies(hh,nonnegative(hi)),z.Implies(hn,nonnegative(n)),
                z.Implies(z.And(hl,hh),z.Not(z.fpGT(lo,hi))),
                z.If(hp,z.And(finite(p),z.fpGT(p,zero)),z.Not(z.And(hn,z.fpGT(n,zero)))))
    certify(valid,z.And(z.Implies(hl,nonnegative(lo)),z.Implies(hh,nonnegative(hi)),z.Implies(hn,nonnegative(n))))
    certify(z.And(valid,hl,hh),z.fpLEQ(lo,hi))
    certify(z.And(hn,z.fpGT(n,zero),z.Not(hp)),z.Not(valid))
    certify(z.And(hp,z.Or(z.Not(finite(p)),z.fpLEQ(p,zero))),z.Not(valid))
    certify(z.And(z.Not(hl),z.Not(hh),z.Not(hn),z.Not(hp)),valid)
    published=z.And(valid,finite(q),z.Not(z.fpLEQ(q,zero)),z.Implies(hl,z.Not(z.fpLT(q,lo))),z.Implies(hh,z.Not(z.fpGT(q,hi))))
    certify(published,z.And(finite(q),z.fpGT(q,zero),z.Implies(hl,z.fpGEQ(q,lo)),z.Implies(hh,z.fpLEQ(q,hi))))
    quote_ok=z.And(finite(q),z.fpGT(q,zero),z.Implies(hn,nonnegative(n)))
    quote=z.If(z.And(hn,z.fpLT(q,n)),n,q)
    certify(quote_ok,z.And(finite(quote),z.fpGT(quote,zero),z.Implies(hn,z.fpGEQ(quote,n))))
    for error in ERRORS+['Invalid quantity input.','Invalid quantity step.']:
        msg=z.StringVal(error)
        retry=z.Or(msg==z.StringVal('Quantity rounds to 0.'),z.PrefixOf(z.StringVal('Quantity below minQty'),msg),z.PrefixOf(z.StringVal('Notional below minNotional'),msg))
        certify(z.BoolVal(True),z.Not(retry))
    return {'F-SIZING-INPUT':'unsat','F-SIZING-PUBLISH':'unsat','F-SIZING-RETRY':'unsat'}


def model():
    initial={(quantity,metadata,result,retry,'start') for quantity,metadata,retry in itertools.product((False,True),repeat=3)
             for result in ('accepted','small','other')}
    seen=set(initial); todo=list(initial); edges=0; depth={s:0 for s in initial}
    while todo:
        state=todo.pop();quantity,metadata,result,retry,phase=state
        require(phase not in ('normal','retry','published') or quantity and metadata,'invalid input reached sizing/retry')
        require(phase!='retry' or result=='small','non-retryable error reached minimum expansion')
        if phase=='start': dest='normal' if quantity and metadata else 'rejected'
        elif phase=='normal':dest={'accepted':'published','small':'retry','other':'rejected'}[result]
        elif phase=='retry':dest='published' if retry else 'rejected'
        else:continue
        nxt=(*state[:-1],dest);edges+=1
        if nxt not in seen:seen.add(nxt);todo.append(nxt);depth[nxt]=depth[state]+1
    require(len(seen)==56 and edges==32 and max(depth.values())==3,'model bounds drift')
    return {'status':'model_checked','states':56,'initialStates':24,'terminalStates':24,'transitions':32,'maximumDepth':3,
            'bounds':'one pure normalizeEntryQty call; quantity/metadata admission Booleans, three normalization outcomes and retry-success Boolean; no order effects'}


def word(x):return struct.unpack('<Q',struct.pack('<d',x))[0]
def value(w):return struct.unpack('<d',struct.pack('<Q',w))[0]

def samples():
    boundary=[0,1,2**63,0x7fefffffffffffff,0x0010000000000000,0x7ff0000000000000,0xfff0000000000000,
              0x7ff8000000000000,word(.3),word(.1),word(1.),word(-1.),word(100.)]
    rows=[]
    for w in boundary:
        for slot in range(5):
            values=list(map(word,[1.,100.,.01,10.,50.]));values[slot]=w
            for hp,hl,hh,hn,hf,hg in itertools.product((False,True),repeat=6):
                rows.append((*values,hp,hl,hh,hn,hf,hg))
    rng=random.Random(20261005)
    rows += [tuple([rng.getrandbits(64) for _ in range(5)]+[bool(rng.getrandbits(1)) for _ in range(6)]) for _ in range(2048)]
    require(len(rows)==6208,'row budget')
    return rows


def admission(lo,hi,n,p):
    good=lambda x:x is None or math.isfinite(x) and x>=0
    return (all(good(x) for x in (lo,hi,n)) and (lo is None or hi is None or lo<=hi)
            and (math.isfinite(p) and p>0 if p is not None else n is None or n<=0))


def fields(row):
    qw,pw,lw,hw,nw,hp,hl,hh,hn,hf,hg=row
    q,p=value(qw),value(pw)
    lo,hi,n=[value(w) if present and hf else None for w,present in [(lw,hl),(hw,hh),(nw,hn)]]
    return q,p,lo,hi,n,p if hp else None


def fixture_rows(fixture):
    def num(x):return {'NaN':math.nan,'Infinity':math.inf}[x] if isinstance(x,str) else x
    rows=[]
    for e in fixture['entries']:
        vals=[e.get(k) for k in ['quantity','price','minimum','maximum','notional']]
        rows.append(tuple([word(float(num(x))) if x is not None else word(1.) for x in vals]+
                          [vals[1] is not None,vals[2] is not None,vals[3] is not None,vals[4] is not None,True,e['grid']]))
    return rows


def conformance(current):
    fixture=json.loads((ROOT/FIXTURE).read_text())
    require([e['id'] for e in fixture['entries']]==[f'CE-SIZING-00{i}' for i in range(1,7)],'witness roster')
    old='\n'.join(textwrap.dedent(fixture[k]) for k in ['probe','sizing','quantityValidator'])
    pattern=r'\b('+'|'.join(FUNCTIONS+['validateQuantityInput'])+r')\b'
    old=re.sub(pattern,lambda m:'old_'+m.group(),old)
    new='\n'.join(textwrap.dedent(current[k]) for k in ['probe','sizing'])
    entry=new[new.index('normalizeEntryQty ::'):]
    entry=entry.replace('normalizeEntryQty','noRetryEntry').replace('minTradeQty sf price','forbiddenMinimum sf price')
    adapter=(ROOT/'haskell/app/Trader/Binance.hs').read_text()
    delegate=adapter[adapter.index('quantizeDown ::'):adapter.index('\ndata SymbolFilters')]
    code='''module Main (main) where
import Control.Applicative ((<|>))
import Data.Maybe (fromMaybe)
import Data.List (isPrefixOf)
import Data.Word (Word64)
import GHC.Float (castWord64ToDouble,castDoubleToWord64)
import Trader.QuantityRounding

data Step = Step {stepScale :: Integer,stepInt :: Integer}
data SymbolFilters = SymbolFilters
 {sfLotMinQty :: Maybe Double,sfLotMaxQty :: Maybe Double,sfLotStepSize :: Maybe Step,
  sfMarketMinQty :: Maybe Double,sfMarketMaxQty :: Maybe Double,sfMarketStepSize :: Maybe Step,
  sfMinNotional :: Maybe Double}
type Row = (Word64,Word64,Word64,Word64,Word64,Bool,Bool,Bool,Bool,Bool,Bool)
type Result = (Bool,Word64,Bool,String)
encode :: Either String Double -> Result
encode = encodeEntry . fmap (\\q -> (q,False))
encodeEntry :: Either String (Double,Bool) -> Result
encodeEntry r = case r of {Left e -> (False,0,False,e);Right (q,b) -> (True,castDoubleToWord64 q,b,"")}
forbiddenMinimum :: SymbolFilters -> Double -> Maybe Double
forbiddenMinimum _ _ = error "Unexpected minimum-size retry after invalid admission"
main :: IO ()
main = getContents >>= mapM_ (run . read) . lines
run :: Row -> IO ()
run (qw,pw,lw,hw,nw,hp,hl,hh,hn,hf,hg) = do
 let q=castWord64ToDouble qw;p=castWord64ToDouble pw
     opt flag w = if flag && hf then Just (castWord64ToDouble w) else Nothing
     grid=if hg && hf then Just (Step 10 1) else Nothing
     sf=SymbolFilters (opt hl lw) (opt hh hw) grid Nothing Nothing Nothing (opt hn nw)
     mSf=if hf then Just sf else Nothing
     mp=if hp then Just p else Nothing
     ok=either (const False) (const True)
     normalGate=validateSizingInputs (effectiveMinQty sf) (effectiveMaxQty sf) (sfMinNotional sf) (Just p)
     probeGate=validateSizingInputs (effectiveMinQty sf) (effectiveMaxQty sf) (sfMinNotional sf) mp
     quoteGate=validateQuantityInput Nothing q >> validateMinimumNotional (sfMinNotional sf)
     noRetry=if not (ok normalGate) || isNaN q || isInfinite q || q < 0
             then either (not . isTooSmallQtyError) (const False) (noRetryEntry sf p q) else True
 print ([encode (normalizeQty sf p q),encodeEntry (normalizeEntryQty sf p q),encode (normalizeProbeQty mSf mp q),encode (validateProbeQuote mSf q),
         encode (old_normalizeQty sf p q),encodeEntry (old_normalizeEntryQty sf p q),encode (old_normalizeProbeQty mSf mp q),encode (old_validateProbeQuote mSf q)],
        [ok normalGate,ok probeGate,ok quoteGate],noRetry)
'''+delegate+'\n'+new+'\n'+old+'\n'+entry
    rows=samples()+fixture_rows(fixture)
    with tempfile.TemporaryDirectory(prefix='trader-sizing-') as tmp:
        p=Path(tmp);(p/'Main.hs').write_text(code);exe=p/'check'
        built=subprocess.run(['ghc','-v0','-O2','-ihaskell/app','-outputdir',tmp,str(p/'Main.hs'),'-o',str(exe)],cwd=ROOT,capture_output=True,text=True,timeout=120)
        require(built.returncode==0,'driver compilation: '+built.stderr)
        text=''.join(str(r).replace(' ','')+'\n' for r in rows)
        output=subprocess.check_output([str(exe)],input=text,text=True,timeout=60)
    actual=[ast.literal_eval(line) for line in output.splitlines()]
    require(len(actual)==len(rows),'missing result rows')
    successes=[0]*4;compatibility=[0]*4
    for row,(results,gates,no_retry) in zip(rows,actual):
        q,p,lo,hi,n,mp=fields(row)
        normal=admission(lo,hi,n,p);probe=admission(lo,hi,n,mp)
        quote=math.isfinite(q) and q>=0 and (n is None or math.isfinite(n) and n>=0)
        require(gates==[normal,probe,quote],'independent admission mismatch')
        require(no_retry,'compiled invalid-input retry marker')
        permits=[math.isfinite(q) and q>=0 and normal,math.isfinite(q) and q>=0 and normal,math.isfinite(q) and q>=0 and probe,quote]
        for i,permit in enumerate(permits):
            new,old=results[i],results[i+4]
            if not permit:require(not new[0],'invalid input published')
            if new[0]:
                out=value(new[1]);successes[i]+=1
                require(math.isfinite(out) and out>0,'non-finite/nonpositive publication')
                if i<3:require((lo is None or out>=lo) and (hi is None or out<=hi),'quantity bound publication')
                else:require(n is None or out>=n,'quote notional publication')
                require(new==old,'changed admitted output bits/entry flag')
            if permit and old[0] and math.isfinite(value(old[1])):
                require(new==old,'valid legacy result lost');compatibility[i]+=1
    require(all(n>0 for n in successes),'empty success coverage')
    witnesses=actual[-6:]
    for index,paths in enumerate([[0,1,2],[0,1,2],[2],[2],[3],[1]]):
        results=witnesses[index][0]
        for path in paths:require(not results[path][0] and results[path+4][0],'counterexample not reproduced')
    return {'status':'property_tested','rows':len(rows),'boundaryRows':4160,'generatedRows':2048,'witnessRows':6,'seed':20261005,
            'currentFunctionCases':len(rows)*4,'legacyFunctionCases':len(rows)*4,'admittedResultsByFunction':successes,
            'compatibleResultsByFunction':compatibility,'functions':['normalizeQty','normalizeEntryQty','normalizeProbeQty','validateProbeQuote'],
            'counterexamples':[e['id'] for e in fixture['entries']],
            'network':'Only pure Main fragments and production numeric kernels linked; no exchange module, credentials or HTTP.'}


def check_sizing():
    current=extract()
    return {'smt':prove(),'model':model(),'conformance':conformance(current),
            'scope':'effective metadata/price admission, finite bounded publication and local retry exclusion; no parser/freshness, exact notional, complete caller or wire-cap proof'}
