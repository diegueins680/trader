"""Source-bound closed-trade numeric admission and exact index reconstruction."""
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import textwrap
import z3 as z
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
CORE = 'haskell/app/Trader/BotSnapshotRecovery.hs'
REGISTRY = 'formal/research/closed-trade-recovery-source.json'
FIXTURE = 'formal/research/closed-trade-recovery-counterexamples.json'


def require(ok, message):
    if not ok:
        raise ValueError('closed-trade recovery: '+message)


def extract(source=None):
    source=(ROOT/CORE).read_text() if source is None else source
    registry=json.loads((ROOT/REGISTRY).read_text())
    require(hashlib.sha256(source.encode()).hexdigest()==registry['sha256'],'source drift')
    for line in ['unless (validTradeMetadata holdingPeriods entryHighVolProb)',
                 'pure (checkedTradeReturn entryEquity exitEquity mReturn)',
                 'takeLast (tmscTradeLimit context) (mapMaybe tradeFromSnapshotValue (V.toList tradesV))',
                 'reindexRestoredTrades = fromMaybe [] . go 0',
                 'if idx < 0 || exitIdx > toInteger (maxBound :: Int)',
                 'go (exitIdx + 1) rest']:
        require(source.count(line)==1,'missing/duplicate boundary '+line)
    require(source[source.index('snapshotMatchesTradeMemoryContext ::'):source.index('parseTradeEntrySourceCode ::')]==registry['identityAndSelection'],'identity/selection drift')
    return source[source.index('checkedTradeReturn ::'):]


def prove():
    fp=z.Float64();zero=z.FPVal(0,fp);one=z.FPVal(1,fp)
    entry,exit,raw,p=[z.FP('recovery_'+n,fp) for n in ('entry','exit','raw','p')]
    present,hp=z.Bools('recovery_present recovery_hp');hold=z.Int('recovery_hold')
    finite=lambda x:z.And(z.Not(z.fpIsNaN(x)),z.Not(z.fpIsInf(x)))
    # Overapproximate division/subtraction by any binary64 result; the publication guard must reject every non-finite result.
    derived=z.FP('recovery_derived',fp)
    value=z.If(present,raw,derived)
    accepted=z.And(finite(entry),z.fpGT(entry,zero),finite(exit),finite(value))
    certify(accepted,z.And(finite(value),finite(entry),finite(exit),z.fpGT(entry,zero)))
    certify(z.And(present,z.Not(finite(raw))),z.Not(accepted))
    metadata=z.And(hold>=0,z.Implies(hp,z.And(finite(p),z.fpGEQ(p,zero),z.fpLEQ(p,one))))
    certify(metadata,z.And(hold>=0,z.Implies(hp,z.And(finite(p),z.fpGEQ(p,zero),z.fpLEQ(p,one)))))
    idx,duration,bound=z.Ints('recovery_idx recovery_duration recovery_bound')
    end=idx+z.If(duration>1,duration,1)
    admitted=z.And(idx>=0,end<=bound)
    certify(admitted,z.And(idx>=0,end>idx,end<=bound,idx<=bound))
    certify(z.And(admitted,bound==2**63-1),z.And(idx<2**63,end<2**63))
    certify(z.And(admitted,bound==2**31-1),z.And(idx<2**31,end<2**31))
    certify(admitted,end+1>end)
    return {'F-RECOVERY-NUMERIC':'unsat','F-RECOVERY-INDEX':'unsat'}


def model():
    # One record's eligibility and the selected-history index gate; pure publication.
    initial={(identity,numeric,index,'start') for identity,numeric,index in itertools.product((False,True),repeat=3)}
    seen=set(initial);todo=list(initial);edges=0;depth={s:0 for s in initial}
    while todo:
        s=todo.pop();identity,numeric,index,phase=s
        require(phase!='published' or identity and numeric and index,'invalid publication')
        if phase=='start':nxt='decoded' if identity else 'rejected'
        elif phase=='decoded':nxt='selected' if numeric else 'rejected'
        elif phase=='selected':nxt='published' if index else 'rejected'
        else:nxt=phase  # Retried pure publication/rejection is stable.
        t=(*s[:-1],nxt);edges+=1
        if t not in seen:seen.add(t);todo.append(t);depth[t]=depth[s]+1
    histories=0;accepted=0
    for n in range(5):
        for holds in itertools.product(range(4),repeat=n):
            result=index_oracle(holds,7);histories+=1
            if result:
                accepted+=1
                require(all(0<=a<b<=7 for a,b,_ in result),'bounded recurrence')
                require(all(result[i][1]<result[i+1][0] for i in range(len(result)-1)),'overlap')
    return {'status':'model_checked','states':len(seen),'initialStates':len(initial),'transitions':edges,'maximumDepth':max(depth.values()),
            'finiteHistories':histories,'nonemptyAdmittedHistories':accepted,'indexBound':7,'maximumHistoryLength':4,'durations':[0,1,2,3],
            'scope':'one-record eligibility and selected-history index gate; no IO, crashes, shared ownership or scheduler'}


def word(x):return struct.unpack('<Q',struct.pack('<d',x))[0]
def value(w):return struct.unpack('<d',struct.pack('<Q',w))[0]

def numeric_rows():
    boundary=[0,1,2**63,0x7fefffffffffffff,0x0010000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000000,word(-1.),word(1.),word(.1),word(1e-300),word(1e300)]
    rows=[]
    for w,slot,present,hp,hold in itertools.product(boundary,range(4),(False,True),(False,True),(-1,0,3)):
        nums=list(map(word,[1.,1.1,.1,.5]));nums[slot]=w
        rows.append((nums[0],nums[1],present,nums[2],hold,hp,nums[3]))
    rng=random.Random(20261005)
    for _ in range(2048):
        e,x,r,p=[rng.getrandbits(64) for _ in range(4)]
        rows.append((e,x,bool(rng.getrandbits(1)),r,rng.choice([-1,0,3,2**63-1]),bool(rng.getrandbits(1)),p))
    rows.extend([(word(1e-300),word(1e300),False,word(0.),1,False,word(0.)),
                 (word(1.),word(1.1),True,word(math.inf),1,False,word(0.)),
                 (word(math.inf),word(1.1),True,word(.1),1,False,word(0.))])
    return rows


def numeric_oracle(row):
    ew,xw,present,rw,hold,hp,pw=row;e,x,r,p=map(value,[ew,xw,rw,pw])
    result=None
    if math.isfinite(e) and e>0 and math.isfinite(x):
        q=r if present else x/e-1
        if math.isfinite(q):result=word(q)
    metadata=hold>=0 and (not hp or math.isfinite(p) and 0<=p<=1)
    return (result is not None,0 if result is None else result),metadata


def index_oracle(holds,bound=2**63-1):
    out=[];idx=0
    for h in holds:
        end=idx+max(1,h)
        if idx<0 or end>bound:return []
        out.append((idx,end,h));idx=end+1
    return out


def conformance(source):
    fixture=json.loads((ROOT/FIXTURE).read_text());old=fixture['oldSource']
    old_index=old[old.index('reindexRestoredTrades ::'):old.index('takeLast ::')].replace('reindexRestoredTrades','oldReindex')
    block=old[old.index('                tradeReturn ='):old.index('                exitReason =')]
    old_return='oldReturn entryEquity exitEquity mReturn =\n'+textwrap.indent(textwrap.dedent('\n'.join(block.splitlines()[1:])),'    ')+'\n'
    code='''module Main (main) where
import Data.List (mapAccumL)
import Data.Maybe (fromMaybe)
import Data.Word (Word64)
import GHC.Float (castWord64ToDouble,castDoubleToWord64)
import System.Environment (getArgs)
data Trade = Trade {trEntryIndex :: Int,trExitIndex :: Int,trHoldingPeriods :: Int} deriving (Eq,Show)
type Row = (Word64,Word64,Bool,Word64,Int,Bool,Word64)
number :: Row -> ((Bool,Word64),Bool,Word64)
number (ew,xw,hr,rw,h,hp,pw) =
 let e=castWord64ToDouble ew;x=castWord64ToDouble xw
     r=if hr then Just (castWord64ToDouble rw) else Nothing
     p=if hp then Just (castWord64ToDouble pw) else Nothing
     encode Nothing=(False,0)
     encode (Just v)=(True,castDoubleToWord64 v)
 in (encode (checkedTradeReturn e x r),validTradeMetadata h p,castDoubleToWord64 (oldReturn e x r))
indices :: [Int] -> ([(Int,Int,Int)],[(Int,Int,Int)])
indices hs =
 let rows=map (Trade 0 0) hs
     encode=map (\\t -> (trEntryIndex t,trExitIndex t,trHoldingPeriods t))
 in (encode (reindexRestoredTrades rows),encode (oldReindex rows))
main :: IO ()
main = do
 args <- getArgs
 case args of
  ["numeric"] -> interact (unlines . map (show . number . read) . lines)
  ["indices"] -> interact (unlines . map (show . indices . read) . lines)
  _ -> error "invalid driver mode"
'''+source+'\n'+old_index+'\n'+old_return
    rows=numeric_rows();rng=random.Random(20261005)
    histories=[[],[0],[1],[2**63-1],[2**63-1,1],[2**63-2,0],[0,2**63-1]]
    histories += [[rng.choice([0,1,3,2**63-1,2**62]) for _ in range(rng.randrange(9))] for _ in range(1024)]
    with tempfile.TemporaryDirectory(prefix='trader-recovery-') as tmp:
        p=Path(tmp);(p/'Main.hs').write_text(code);exe=p/'check'
        built=subprocess.run(['ghc','-v0','-O2','-outputdir',tmp,str(p/'Main.hs'),'-o',str(exe)],cwd=ROOT,capture_output=True,text=True,timeout=120)
        require(built.returncode==0,'compilation: '+built.stderr)
        def run(mode,values):
            text=''.join(str(v).replace(' ','')+'\n' for v in values)
            out=subprocess.check_output([str(exe),mode],input=text,text=True,timeout=60)
            return [ast.literal_eval(line) for line in out.splitlines()]
        actual=run('numeric',rows);indexes=run('indices',histories)
        require(actual==run('numeric',rows) and indexes==run('indices',histories),'replay mismatch')
    require(len(actual)==len(rows) and len(indexes)==len(histories),'missing driver results')
    compatible=0
    for row,(got,meta,old_word) in zip(rows,actual):
        expected,m=numeric_oracle(row)
        require(got==expected and meta==m,'numeric oracle disagreement')
        if got[0]:require(got[1]==old_word,'valid return compatibility');compatible+=1
    for hs,(got,previous) in zip(histories,indexes):
        require(got==index_oracle(hs),'index oracle disagreement')
        if got:require(got==previous,'valid index compatibility')
    require(not actual[-3][0][0] and math.isinf(value(actual[-3][2])),'derived overflow witness')
    require(not actual[-2][0][0] and math.isfinite(value(actual[-2][2])),'invalid explicit return witness')
    require(not actual[-1][0][0] and math.isfinite(value(actual[-1][2])),'invalid equity witness')
    require(not indexes[4][0] and any(a<0 or b<0 for a,b,_ in indexes[4][1]),'index overflow witness')
    return {'status':'property_tested','numericRows':len(rows),'boundaryRows':624,'generatedRows':2048,'numericWitnessRows':3,
            'compatibleFiniteReturns':compatible,'indexHistories':len(histories),'generatedHistories':1024,'seed':20261005,
            'replayRunsPerMode':2,'intBits':64,'counterexamples':[e['id'] for e in fixture['entries']],
            'scope':'actual pure helpers with projected Trade fields; real Aeson integration is tested in Haskell suite, not this base-only driver'}


def check_recovery():
    return {'smt':prove(),'model':model(),'conformance':conformance(extract())}
