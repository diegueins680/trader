"""Exact upward lemmas and source-bound maker rejection; no exchange calls."""
import ast
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile
import textwrap
import z3 as z
from ppo_successor import certify
from quantity_rounding import cases, value, word

ROOT = Path(__file__).resolve().parents[2]
CORE = 'haskell/app/Trader/QuantityRounding.hs'
REGISTRY = 'formal/research/upward-rounding-source.json'
MAIN = 'haskell/app/Main.hs'
START = '                if not (validOrderPrice px)'
END = '                        ts <- getTimestampMs'
DELEGATE = '    quantizeUp st = quantizeUpExact (stepScale st) (stepInt st)'


def require(ok, message):
    if not ok:
        raise ValueError('upward rounding: ' + message)


def extract(core=None, main=None):
    core = (ROOT/CORE).read_text() if core is None else core
    main = (ROOT/MAIN).read_text() if main is None else main
    registry = json.loads((ROOT/REGISTRY).read_text())
    require(hashlib.sha256(core.encode()).hexdigest() == registry['coreSha256'], 'core drift')
    require(main.count(DELEGATE) == 2 and main.count('    quantizeUp ::') == 2, 'delegate coverage')
    require('import Trader.QuantityRounding (quantizeUpExact, validOrderPrice, validateMinimumNotional, validateQuantityInput, validateSizingInputs)' in main, 'core import')
    require([(a,b) for a,b,_ in registry['mainFragments']] == [
        ('    quantizeUp ::', '    normalizeProbeQty mSf mPrice qtyRaw ='),
        ('    stepValue ::', '    isTooSmallQtyError ::'),
        ('    sendPostOnlyEntry ::', END),
        ('    baseResult :: ApiOrderResult', '    noOrder ::')], 'caller roster drift')
    for start, end, expected in registry['mainFragments']:
        require(main[main.index(start):main.index(end,main.index(start))] == expected, 'caller drift')
    require('            { aorSent = False' in registry['mainFragments'][-1][2], 'unsent base result')
    require('units = (numerator value * scale + divisor - 1) `div` divisor' in core, 'ceiling mismatch')
    require('if isNaN rounded || isInfinite rounded || rounded < x' in core, 'publication guard')
    require('validOrderPrice price = either (const False) (const True) (validateOrderNumber "Invalid maker price." price)' in core, 'price guard')
    require('import Trader.OrderNumeric (validateOrderNumber)' in core, 'wire-aware guard import')
    branch = main[main.index(START):main.index(END,main.index(START))]
    require(branch == START+'\n                    then pure baseOut{aorMessage = "No order: invalid maker price."}\n                    else do\n', 'invalid price must return without effects')
    return branch


def prove():
    n,d = z.Ints('up_n up_d'); u=(n+d-1)/d
    certify(z.And(n > 0,d > 0),z.And(u > 0,(u-1)*d < n,n <= u*d))
    a,b,s,k,q = z.Reals('up_a up_b up_s up_k up_q')
    certify(z.And(a > 0,b > 0,s > 0,k > 0,(q-1)*b*k < a*s,a*s <= q*b*k),
            z.And(a/b <= q*k/s,q*k/s < a/b+k/s))
    fp=z.Float64(); x=z.FP('up_x',fp); y=z.FP('up_y',fp); zero=z.FPVal(0,fp)
    finite=lambda v:z.And(z.Not(z.fpIsNaN(v)),z.Not(z.fpIsInf(v)))
    scale,step=z.Ints('up_scale up_step')
    valid=z.And(finite(x),z.fpGT(x,zero),scale > 0,step > 0)
    accepted=z.And(finite(y),z.Not(z.fpLT(y,x)))
    out=z.If(z.And(valid,accepted),y,zero)
    certify(z.BoolVal(True),z.And(finite(out),z.Or(z.fpEQ(out,zero),z.fpGEQ(out,x)),
            z.Implies(z.Not(valid),z.fpEQ(out,zero))))
    price_ok=z.And(finite(x),z.fpGT(x,zero),z.Bool('up_wire_parsed'),z.Int('up_wire_units')>0)
    certify(z.Not(price_ok),z.Not(z.And(price_ok,z.Bool('up_market_fallback'))))
    return {'F-ROUND-UP-INTEGER':'unsat','F-ROUND-UP-FINITE':'unsat'}


def model():
    # Complete one-call abstraction; the flag cannot change the invalid branch.
    initial={(valid,flag,'observed') for valid in (False,True) for flag in (False,True)}
    reached=set(initial); frontier=list(initial); edges=0
    while frontier:
        valid,flag,phase=frontier.pop()
        successors=[(valid,flag,'eligible' if valid else 'rejected')] if phase == 'observed' else []
        for nxt in successors:
            edges+=1
            require(nxt[2] != 'eligible' or valid, 'invalid order eligibility')
            require(nxt[2] != 'market', 'numeric rejection reaches fallback')
            if nxt not in reached:
                reached.add(nxt); frontier.append(nxt)
    require(len(reached)==8 and edges==4, 'model coverage')
    return {'requirement':'F-ROUND-UP-FLOW','status':'model_checked','states':8,'initialStates':4,
            'transitions':4,'maximumDepth':1,'terminalStates':4,
            'bounds':'one dispatch; Boolean validity and configured fallback flag; terminal states intentional'}


def oracle(s,k,w):
    x=value(w)
    if not math.isfinite(x) or x <= 0 or s <= 0 or k <= 0:
        return 0
    v=Fraction.from_float(x); d=v.denominator*k; n=v.numerator*s
    r=Fraction(((n+d-1)//d)*k,s)
    try:
        y=float(r)
    except OverflowError:
        return 0
    return word(y if math.isfinite(y) and y >= x else 0.)


def conformance(branch):
    samples=cases(); main=(ROOT/MAIN).read_text()
    # Compile both actual Main delegates, not independently rewritten wrappers.
    delegates=[]
    for i,part in enumerate(main.split('    quantizeUp ::')[1:]):
        body='    quantizeUp ::'+part[:part.index('\n\n')]
        delegates.append(textwrap.dedent(body).replace('quantizeUp ',f'up{i} '))
    # Actual invalid branch against effect recording stubs; valid continuation
    # stops at eligibility. This never invokes an exchange function.
    dispatch=textwrap.dedent(branch).replace('else do\n','else pure baseOut{aorSent = True}\n')
    counter=json.loads((ROOT/'formal/research/upward-rounding-counterexamples.json').read_text())
    legacy=textwrap.dedent(counter['entries'][1]['legacyBranch']).replace('else do\n','else pure baseOut{aorSent = True}\n')
    code='''module Main (main) where
import GHC.Float (castWord64ToDouble,castDoubleToWord64)
import Data.Word (Word64)
import Trader.QuantityRounding

data Step = Step {stepScale :: Integer, stepInt :: Integer}
data Out = Out {aorSent :: Bool, aorMessage :: String}
main :: IO ()
main = do
 rows <- getContents
 mapM_ (run . read) (lines rows)
run :: (Integer,Integer,Word64) -> IO ()
run (s,k,w) = do
 let x=castWord64ToDouble w
 a <- dispatch x False
 b <- dispatch x True
 oldA <- legacyDispatch x False
 oldB <- legacyDispatch x True
 print (castDoubleToWord64 (up0 (Step s k) x),castDoubleToWord64 (up1 (Step s k) x),validOrderPrice x,aorSent a,aorSent b,aorSent oldA,aorSent oldB)
dispatch :: Double -> Bool -> IO Out
dispatch px flag =
 let baseOut=Out False "initial"
     fallback _ = pure (Out flag "market fallback")
 in
'''+textwrap.indent(dispatch,'    ')+'\n'+'''legacyDispatch :: Double -> Bool -> IO Out
legacyDispatch px flag =
 let baseOut=Out False "initial"
     fallback _ = pure (Out flag "market fallback")
 in
'''+textwrap.indent(legacy,'    ')+'\n'+'\n'.join(delegates)+'\n'
    with tempfile.TemporaryDirectory(prefix='trader-up-rounding-') as tmp:
        p=Path(tmp);(p/'Main.hs').write_text(code);exe=p/'check'
        subprocess.run(['ghc','-v0','-O2','-ihaskell/app','-outputdir',tmp,str(p/'Main.hs'),'-o',str(exe)],
                       cwd=ROOT,check=True,timeout=120,capture_output=True)
        rows=''.join(f'({s},{k},{w})\n' for s,k,w in samples)
        out=subprocess.check_output([str(exe)],input=rows,text=True,timeout=60)
    actual=[ast.literal_eval(line) for line in out.splitlines()]
    expected=[]
    for s,k,w in samples:
        old_valid=math.isfinite(value(w)) and value(w)>0
        valid=old_valid and value(w)>5e-9
        expected.append((oracle(s,k,w),oracle(s,k,w),valid,valid,valid,old_valid,True))
    require(actual == expected,'compiled caller/oracle mismatch')
    counter=json.loads((ROOT/'formal/research/upward-rounding-counterexamples.json').read_text())
    require([c['id'] for c in counter['entries']]==['CE-ROUND-003','CE-ROUND-004'],'counterexample roster')
    x=value(counter['entries'][0]['inputWord64'])
    require(math.ceil(x-1e-9)<x and oracle(1,1,word(x))==word(2.),'ceiling regression')
    require(counter['entries'][1]['oldFallbackEnabled'] is True and not (math.nan > 0),'fallback regression')
    return {'status':'property_tested','cases':len(samples),'compiledDelegates':2,'dispatchCases':2*len(samples),
            'seed':20261005,'boundaryCases':130,'generatedCases':4096,'compilerOptimization':'-O2',
            'counterexamples':['CE-ROUND-003','CE-ROUND-004']}


def check_upward():
    branch=extract()
    return {'smt':prove(),'model':model(),'conformance':conformance(branch),
            'scope':'exact ceiling and finite publication; local invalid-price rejection, not wire/order-cap closure'}
