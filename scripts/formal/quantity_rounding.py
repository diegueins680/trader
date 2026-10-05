"""Source-bound downward rounding lemmas and compiled binary64 conformance."""
import ast
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import z3 as z
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'haskell/app/Trader/QuantityRounding.hs'
REGISTRY = 'formal/research/quantity-rounding-source.json'
ADAPTER = 'haskell/app/Trader/Binance.hs'


def require(ok, message):
    if not ok:
        raise ValueError('quantity rounding: ' + message)


def extract(source=None, adapter=None, main=None):
    source = (ROOT/SOURCE).read_text() if source is None else source
    adapter = (ROOT/ADAPTER).read_text() if adapter is None else adapter
    main = (ROOT/'haskell/app/Main.hs').read_text() if main is None else main
    registry = json.loads((ROOT/REGISTRY).read_text())
    require(hashlib.sha256(source.encode()).hexdigest() == registry['coreSha256'], 'core drift')
    declaration = adapter[adapter.index('quantizeDown ::'):adapter.index('\ndata SymbolFilters')]
    require(declaration == registry['adapterDeclaration'], 'adapter drift')
    require('import Trader.QuantityRounding (quantizeDownExact)' in adapter, 'missing core import')
    require('units = (numerator value * scale) `div` (denominator value * increment)' in source, 'quotient mismatch')
    require('| isNaN x || isInfinite x || x <= 0 || scale <= 0 || increment <= 0 = 0' in source, 'input guard')
    require('if isNaN rounded || isInfinite rounded || rounded < 0 || rounded > x' in source, 'output guard')
    require([(a,b) for a,b,_ in registry['mainFragments']] == [('    normalizeProbeQty mSf mPrice qtyRaw =','\ncomputeCoinbaseKeysStatusFromArgs'),('    isTooSmallQtyError ::','    sendMarketOrderWithMaker ::')], 'caller coverage omitted')
    require('import Trader.QuantityRounding (validateQuantityInput)' in main, 'missing preflight import')
    for start, end, expected in registry['mainFragments']:
        require(main[main.index(start):main.index(end,main.index(start))] == expected, 'caller control-flow drift')
    for name, grid in [('normalizeQty sf price qtyRaw', '(effectiveStep sf)'),
                       ('normalizeProbeQty mSf mPrice qtyRaw', '(mSf >>= effectiveStep)')]:
        prefix = '    '+name+' = do\n        validateQuantityInput (fmap (\\st -> (stepScale st, stepInt st)) '+grid+') qtyRaw\n'
        require(prefix in main, 'preflight not first')
    return registry


def prove():
    n, d = z.Ints('round_n round_d')
    # Euclidean division on arbitrary nonnegative numerator / positive divisor.
    certify(z.And(n >= 0, d > 0), z.And(n/d >= 0, (n/d)*d <= n, n < (n/d+1)*d))
    # Reconstruction over exact rationals, conditional on quotient bounds.
    a, b, s, k, u = z.Reals('round_a round_b round_s round_k round_u')
    certify(z.And(a >= 0, b > 0, s > 0, k > 0, u >= 0,
                  u*b*k <= a*s, a*s < (u+1)*b*k),
            z.And(u*k/s >= 0, u*k/s <= a/b, a/b < (u+1)*k/s))
    fp = z.Float64(); x = z.FP('round_x',fp); y = z.FP('round_y',fp)
    zero = z.FPVal(0,fp)
    finite = lambda v: z.And(z.Not(z.fpIsNaN(v)),z.Not(z.fpIsInf(v)))
    valid = z.And(finite(x), z.fpGT(x,zero))
    accepted = z.And(finite(y), z.Not(z.fpLT(y,zero)), z.Not(z.fpGT(y,x)))
    out = z.If(z.And(valid,accepted),y,zero)
    certify(z.BoolVal(True), z.And(finite(out),z.fpGEQ(out,zero),
            z.Implies(valid,z.fpLEQ(out,x)), z.Implies(z.Not(valid),z.fpEQ(out,zero))))
    # A nonpositive scale or increment selects the zero branch before division.
    scale, step = z.Ints('round_scale round_step')
    invalid = z.Or(scale <= 0, step <= 0)
    result = z.If(invalid,zero,out)
    certify(invalid,z.fpEQ(result,zero))
    has_grid = z.Bool('round_has_grid')
    error = z.Or(z.Not(finite(x)),z.And(has_grid,z.Or(scale <= 0,step <= 0)))
    certify(z.Not(error),z.And(finite(x),z.Implies(has_grid,z.And(scale > 0,step > 0))))
    return {'F-ROUND-DOWN-INTEGER':'unsat' ,'F-ROUND-DOWN-FINITE':'unsat'}


def word(x):
    return struct.unpack('<Q',struct.pack('<d',x))[0]


def value(w):
    return struct.unpack('<d',struct.pack('<Q',w))[0]


def oracle(scale,step,w):
    x = value(w)
    if not math.isfinite(x) or x <= 0 or scale <= 0 or step <= 0:
        return 0
    exact = Fraction.from_float(x)
    units = (exact.numerator*scale)//(exact.denominator*step)
    y = float(Fraction(units*step,scale))
    return word(y if math.isfinite(y) and 0 <= y <= x else 0.)


def cases():
    rng = random.Random(20261005)
    grid = [(1,1),(10,1),(100,3),(100000000,1),(10**400,1),(1,10**400),
            (0,1),(1,0),(-1,1),(1,-1)]
    boundaries = [0,1,2**63,0x7fefffffffffffff,0x0010000000000000,
                  0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000000,
                  word(.3),word(.1),word(1.),word(math.nextafter(1.,0.)),word(math.nextafter(1.,2.))]
    result = [(s,k,w) for s,k in grid for w in boundaries]
    result += [(*rng.choice(grid),rng.getrandbits(64)) for _ in range(4096)]
    return result


def conformance():
    samples = cases()
    code = '''module Main (main) where
import GHC.Float (castWord64ToDouble, castDoubleToWord64)
import Data.Word (Word64)
import Trader.QuantityRounding (quantizeDownExact, validateQuantityInput)
main :: IO ()
main = interact (unlines . map (show . run . read) . lines)
run :: (Integer,Integer,Word64) -> (Word64,Bool,Bool)
run (s,k,w) =
    let x = castWord64ToDouble w
        ok = either (const False) (const True)
    in (castDoubleToWord64 (quantizeDownExact s k x),
        ok (validateQuantityInput (Just (s,k)) x),
        ok (validateQuantityInput Nothing x))
'''
    with tempfile.TemporaryDirectory(prefix='trader-rounding-') as tmp:
        p = Path(tmp); (p/'Main.hs').write_text(code); exe=p/'rounding'
        subprocess.run(['ghc','-v0','-O2','-ihaskell/app','-outputdir',tmp,str(p/'Main.hs'),'-o',str(exe)],
                       cwd=ROOT,check=True,timeout=120,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        rows = ''.join(f'({s},{k},{w})\n' for s,k,w in samples)
        output = subprocess.check_output([str(exe)],input=rows,text=True,timeout=60)
    actual = [ast.literal_eval(w) for w in output.splitlines()]
    expected = [(oracle(s,k,w),math.isfinite(value(w)) and s > 0 and k > 0,math.isfinite(value(w))) for s,k,w in samples]
    require(actual == expected,'compiled core differs from Fraction oracle')
    counter = json.loads((ROOT/'formal/research/quantity-rounding-counterexamples.json').read_text())
    fixture = counter['entries'][0]
    x = value(fixture['inputWord64'])
    require(math.floor(x+1e-9) == fixture['oldOutput'] > x,'old counterexample lost')
    require(oracle(1,1,fixture['inputWord64']) == fixture['newOutputWord64'],'new regression')
    require(max(1.,value(oracle(1,1,counter['entries'][1]['inputWord64']))) == 1., 'intermediate fallback counterexample lost')
    return {'cases':len(samples),'generatedCases':4096,'boundaryCases':130,
            'seed':20261005,'counterexamples':['CE-ROUND-001','CE-ROUND-002'],
            'preflightCases':2*len(samples),
            'compilerOptimization':'-O2','status':'property_tested'}


def check_rounding():
    extract()
    return {'smt':prove(),'conformance':conformance(),
            'scope':'exact quotient/reconstruction lemmas and guarded binary64 helper; no wire/order-cap theorem'}
