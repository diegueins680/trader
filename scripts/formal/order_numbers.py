"""Finite Binance numeric preflight: conditional local model and compiled prefixes."""
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
import z3 as z
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
CORE = 'haskell/app/Trader/OrderNumeric.hs'
ADAPTER = 'haskell/app/Trader/Binance.hs'
REGISTRY = 'formal/research/order-number-source.json'
ROSTER = ['placeFuturesPostOnlyLimitOrder','placeMarketOrder','placeFuturesMarketOrderWithPositionSide',
          'placeFuturesTriggerMarketOrder','placeFuturesAlgoTriggerMarketOrder']
CREDENTIAL = '    apiKey <- maybe (throwIO (userError "Missing BINANCE_API_KEY")) pure (beApiKey env)'


def require(ok, message):
    if not ok:
        raise ValueError('order number: '+message)


def prefixes(source):
    result={}
    for name in ROSTER:
        match=re.search(r'^'+name+r' [^\n]+ = do\n',source,re.M)
        require(match is not None, 'missing constructor '+name)
        end=source.index(CREDENTIAL,match.end())
        result[name]=source[match.end():end]
    return result


def extract(core=None, adapter=None, registry=None):
    core=(ROOT/CORE).read_text() if core is None else core
    adapter=(ROOT/ADAPTER).read_text() if adapter is None else adapter
    registry=json.loads((ROOT/REGISTRY).read_text()) if registry is None else registry
    require(hashlib.sha256(core.encode()).hexdigest()==registry['coreSha256'],'core drift')
    require(hashlib.sha256(adapter.encode()).hexdigest()==registry['adapterSha256'],'adapter drift')
    require(list(registry['prefixes'])==ROSTER,'constructor coverage')
    require(prefixes(adapter)==registry['prefixes'],'pre-credential control-flow drift')
    require('import Trader.OrderNumeric (validateMarketNumbers, validateOrderNumber)' in adapter,'validator import')
    require('isNaN value || isInfinite value || value <= 0 = Left message' in core,'numeric guard')
    require('Just q -> validateOrderNumber "MARKET quantity must be finite and > 0" q' in core,'base priority')
    require('case quantity of\n        Just q' in core and '| futures -> Left ' in core,'selection structure')
    require(adapter.count('either (throwIO . userError) pure (validateOrderNumber ')==5,'numeric call coverage')
    require(adapter.count('either (throwIO . userError) pure (validateMarketNumbers ')==1,'market call coverage')
    # Exact full adapter binding also covers unchanged selection and request tails.
    return registry['prefixes']


def prove():
    fp=z.Float64(); b=z.FP('order_base',fp); q=z.FP('order_quote',fp); zero=z.FPVal(0,fp)
    finite=lambda x:z.And(z.Not(z.fpIsNaN(x)),z.Not(z.fpIsInf(x)))
    valid=lambda x:z.Not(z.Or(z.fpIsNaN(x),z.fpIsInf(x),z.fpLEQ(x,zero)))
    certify(z.BoolVal(True),z.Implies(valid(b),z.And(finite(b),z.fpGT(b,zero))))
    hb,hq,f=z.Bools('order_has_base order_has_quote order_futures')
    accepted=z.If(hb,valid(b),z.And(z.Not(f),hq,valid(q)))
    certify(z.And(hb,z.Not(valid(b))),z.Not(accepted))
    certify(z.And(z.Not(hb),f),z.Not(accepted))
    certify(z.And(accepted,z.Not(hb)),z.And(z.Not(f),hq,finite(q),z.fpGT(q,zero)))
    certify(z.And(hb,valid(b)),accepted)
    return {'F-ORDER-NUMBER-FINITE':'unsat','F-ORDER-NUMBER-SELECTION':'unsat'}


def model():
    initial={(name,numeric,context,'start') for name in ROSTER for numeric in (False,True) for context in (False,True)}
    seen=set(initial); work=list(initial); edges=0
    while work:
        name,numeric,context,phase=work.pop()
        successors=[(name,numeric,context,'credentials' if numeric and context else 'rejected')] if phase=='start' else []
        for nxt in successors:
            edges+=1
            require(nxt[3]!='credentials' or (numeric and context),'invalid preflight reached credentials')
            if nxt not in seen:
                seen.add(nxt);work.append(nxt)
    require(len(seen)==40 and edges==20,'finite coverage')
    return {'status':'model_checked','states':40,'initialStates':20,'terminalStates':20,'transitions':20,
            'maximumDepth':1,'bounds':'five constructors; numeric/context prefix-validity Booleans; credential boundary is terminal, not an order capability'}


def number(w):
    return struct.unpack('<d',struct.pack('<Q',w))[0]


def samples():
    boundary=[0,1,2**63,0x7fefffffffffffff,0x0010000000000000,0x7ff0000000000000,
              0xfff0000000000000,0x7ff8000000000000,0x3fd3333333333333,0x3fb999999999999a,
              0x3ff0000000000000,0x3fefffffffffffff,0x3ff0000000000001]
    one=0x3ff0000000000000
    rows=[(b,q,hb,hq,m,test,typ) for x in boundary for b,q in [(x,one),(one,x)]
          for hb,hq,test,typ in itertools.product((False,True),repeat=4) for m in range(3)]
    rng=random.Random(20261005)
    rows += [(rng.getrandbits(64),rng.getrandbits(64),bool(rng.getrandbits(1)),bool(rng.getrandbits(1)),
              rng.randrange(3),bool(rng.getrandbits(1)),bool(rng.getrandbits(1))) for _ in range(4096)]
    return rows


def oracle(row):
    wb,wq,hb,hq,m,test,typ=row; b,q=number(wb),number(wq)
    valid=lambda x:math.isfinite(x) and x>0
    vb,vq=valid(b),valid(q); f=m==2
    selected=vb if hb else not f and hq and vq
    current=[f and vb and vq,selected,f and vb,f and vb and typ,f and not test and vb and typ]
    # Historical guards: not (x<=0) passes NaN; market prefix had no numeric guard.
    oldb,oldq=not (b<=0),not (q<=0)
    legacy=[f and oldb and oldq,True,f and oldb,f and oldb and typ,f and not test and oldb and typ]
    return (vb,selected,current,legacy)


def conformance(current):
    fixture=json.loads((ROOT/'formal/research/order-number-counterexamples.json').read_text())
    require([e['id'] for e in fixture['entries']]==['CE-ORDER-NUM-001','CE-ORDER-NUM-002'],'regression roster')
    legacy=fixture['legacyPrefixes'];require(list(legacy)==ROSTER,'legacy coverage')
    signatures=[('env quantity price','Env -> Double -> Double'),('env quantity quoteOrderQty','Env -> Maybe Double -> Maybe Double'),
                ('env quantity','Env -> Double'),('env stopPrice orderType','Env -> Double -> String'),
                ('env mode triggerPrice orderType','Env -> Mode -> Double -> String')]
    defs=[]
    for tag,ps in [('cur',current),('old',legacy)]:
        for i,name in enumerate(ROSTER):
            args,typ=signatures[i]
            defs.append(f'{tag}{i} :: {typ} -> IO ()\n{tag}{i} {args} = do\n'+ps[name]+'    pure ()\n')
    code='''module Main (main) where
import Control.Exception (SomeException,try,throwIO)
import qualified Control.Monad
import Data.Word (Word64)
import GHC.Float (castWord64ToDouble)
import Trader.OrderNumeric
import Data.Char (isSpace)
data Market = MarketSpot | MarketMargin | MarketFutures deriving (Eq)
data Mode = OrderTest | OrderLive deriving (Eq)
newtype Env = Env {beMarket :: Market}
allowed :: IO () -> IO Bool
allowed action = either (const False) (const True) <$> (try action :: IO (Either SomeException ()))
main :: IO ()
main = getContents >>= mapM_ (run . read) . lines
run :: (Word64,Word64,Bool,Bool,Int,Bool,Bool) -> IO ()
run (wb,wq,hb,hq,m,test,typ) = do
 let b=castWord64ToDouble wb; q=castWord64ToDouble wq
     market=case m of {0 -> MarketSpot; 1 -> MarketMargin; _ -> MarketFutures}
     env=Env market; mode=if test then OrderTest else OrderLive
     qty=if hb then Just b else Nothing; quote=if hq then Just q else Nothing
     orderType=if typ then "STOP_MARKET" else "   "
     ok=either (const False) (const True)
 current <- mapM allowed [cur0 env b q,cur1 env qty quote,cur2 env b,cur3 env b orderType,cur4 env mode b orderType]
 legacy <- mapM allowed [old0 env b q,old1 env qty quote,old2 env b,old3 env b orderType,old4 env mode b orderType]
 print (ok (validateOrderNumber "invalid" b),ok (validateMarketNumbers (market==MarketFutures) qty quote),current,legacy)
'''+ '\n'.join(defs)+'trim :: String -> String\n'+(ROOT/ADAPTER).read_text().split('trim :: String -> String\n',1)[1]
    rows=samples()
    with tempfile.TemporaryDirectory(prefix='trader-order-number-') as tmp:
        p=Path(tmp);(p/'Main.hs').write_text(code);exe=p/'check'
        subprocess.run(['ghc','-v0','-O2','-ihaskell/app','-outputdir',tmp,str(p/'Main.hs'),'-o',str(exe)],
                       cwd=ROOT,check=True,timeout=120,capture_output=True)
        data=''.join(str(r).replace(' ','')+'\n' for r in rows)
        out=subprocess.check_output([str(exe)],input=data,text=True,timeout=60)
    actual=[ast.literal_eval(line) for line in out.splitlines()]
    expected=[oracle(row) for row in rows]
    require(actual==expected,'compiled preflight/oracle mismatch')
    for item in fixture['entries']:
        witness=(item['inputWord64'],0x3ff0000000000000,True,True,2,False,True)
        require(oracle(witness)[2]==[False]*5 and oracle(witness)[3]==[True]*5,'legacy non-finite witness')
    return {'status':'property_tested','rows':len(rows),'boundaryRows':len(rows)-4096,'generatedRows':4096,
            'boundaryWords':13,'seed':20261005,'constructors':5,'currentPrefixCases':len(rows)*5,
            'legacyPrefixCases':len(rows)*5,'compilerOptimization':'-O2','counterexamples':['CE-ORDER-NUM-001','CE-ORDER-NUM-002'],
            'network':'No Binance module or credential/request/HTTP code linked into the prefix driver.'}


def check_order_numbers():
    current=extract()
    return {'smt':prove(),'model':model(),'conformance':conformance(current),
            'scope':'selected numeric validity before credentials; no wire, cap, caller-retry or complete order-authorization proof'}
