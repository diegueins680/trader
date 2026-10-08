"""Obligation 9: minimum-size increases respect the order cap, and grid values reach the wire exactly.

CE-ROUND-003: a fraction-sized futures entry capped by maxOrderQuote could be raised to the exchange minimum above
that cap. Source certificate: the only quantity increase is normalizeEntryQty's minimum bump; its spot caller cannot
carry maxOrderQuote (which requires fraction sizing, while the spot bump needs an explicit quantity) and its futures
caller refuses a bump above the cap. SMT: over the reals, the published entry notional never exceeds the cap.
Wire theorem: every 8-decimal grid value below 2^24 is recovered exactly by the 8-decimal renderer from its binary64
value, and binary64 preserves the order of distinct grid values, so cap comparisons on doubles are comparisons on the
wire decimals. Conformance: the compiled actual renderOrderNumber round-trips seeded grid values.
"""
from fractions import Fraction as F
from pathlib import Path
import random
import struct
import re
import subprocess
import tempfile
import z3 as z

ROOT = Path(__file__).resolve().parents[2]
MAIN = 'haskell/app/Main.hs'
ARGS = 'haskell/app/Trader/App/Args.hs'
NUMERIC = 'haskell/app/Trader/OrderNumeric.hs'
GRID_DIGITS = 8
GRID_LIMIT_EXP = 24  # values below 2^24 (about 1.68e7 base or quote units)


def require(ok, reason):
    if not ok:
        raise ValueError('min-size cap: ' + reason)


def definitions(text):
    lines = text.split('\n')
    heads = [(i, m.group(1)) for i, line in enumerate(lines) for m in [re.match(r'^ *([a-z]\w*) ::', line)] if m]
    bodies = {}
    for k, (i, name) in enumerate(heads):
        j = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        bodies.setdefault(name, '\n'.join(line.split('--')[0] for line in lines[i:j]))
    return bodies


def bind(sources=None):
    sources = sources or {}
    read = lambda p: sources.get(p, (ROOT / p).read_text())
    main, args, numeric = read(MAIN), read(ARGS), read(NUMERIC)
    bodies = definitions(main)
    entry = bodies['normalizeEntryQty']
    require('Right qMin -> Right (qMin, qMin > qtyRaw + 1e-9)' in entry and 'Right q -> Right (q, False)' in entry,
            'minimum bump site changed')
    require(len(re.findall(r'\bminTradeQty sf price\b', entry)) == 1, 'bump does not come from the venue minimum')
    callers = [m.start() for m in re.finditer(r'normalizeEntryQty sf currentPrice', main)]
    require(len(callers) == 2, f'unreviewed minimum-bump caller ({len(callers)})')
    # Spot: the bump applies only to an explicit orderQuantity; maxOrderQuote requires fraction sizing.
    spot = main[callers[0] - 1500:callers[0]]
    require('(Just qRaw, _) ->' in spot and 'qtyArgBuy = fmap (* entryScale) qtyArg' in main, 'spot bump path changed')
    require('Just _ -> ensure "--max-order-quote requires --order-quote-fraction" fracOn' in args,
            'maxOrderQuote no longer requires fraction sizing')
    # Futures: a bump above the cap is refused.
    plan = bodies['buildFuturesEntryPlan']
    for fragment in ('case normalizeFuturesEntryQty quantity of',
                     '| Just cap <- positiveMaxOrderQuote\n                                            , fractionSized && toRational q * toRational currentPrice > toRational cap ->',
                     'quote = maybe quote0 (`min` quote0) positiveMaxOrderQuote',
                     'not (maybe False (> 0) (argOrderQuantity args))\n                                && not (maybe False (> 0) (argOrderQuote args))'):
        require(fragment in plan, 'futures cap guard changed: ' + fragment.split('\n')[0].strip())
    require('Just sf -> normalizeEntryQty sf currentPrice qRaw' in main, 'futures normalizer changed')
    for fragment in ('| otherwise = trimTrailingZeros (renderWireUnits (orderWireUnits x))',
                     'orderWireUnits = gridUnits 8',
                     'if fractionDigits <= scaleExp\n                    then mantissa * 10 ^ (scaleExp - fractionDigits)\n                    else truncate (toRational (abs x) * 10 ^ scaleExp)',
                     '(digits, exponent) = floatToDigits 10 (abs x)'):
        require(fragment in numeric, 'wire renderer changed')
    coinbase = read('haskell/app/Trader/Coinbase.hs')
    require('renderDoubleText = renderOrderNumber' in coinbase and 'showFFloat' not in coinbase,
            'Coinbase sizes bypass the shared wire renderer')
    order = coinbase[coinbase.index('placeCoinbaseMarketOrder env product'):]
    require(0 <= order.find('validateOrderNumber "Coinbase order size rounds to 0 at 8 decimals." q') < order.find('let body =') and
            'validateOrderNumber "Coinbase order funds round to 0 at 8 decimals." q' in order, 'Coinbase sends amounts that truncate to zero')
    dex = read('haskell/app/Trader/Dex.hs')
    require('let scaled = gridUnits decimals amt' in dex and 'floor (scaledRaw' not in dex, 'DEX token amounts bypass exact grid truncation')
    binance = read('haskell/app/Trader/Binance.hs')
    require('renderDouble = BS.pack . renderOrderNumber' in binance, 'Binance wire renderer changed')
    return {'status': 'exhaustively_checked', 'bumpSites': 1, 'bumpCallers': 2}


def prove_cap():
    """Over the reals: a fraction-sized entry is at most the cap whether or not the minimum bump applies."""
    balance, fraction, cap, price, step, quote, desired, q, minimum = z.Reals('balance fraction cap price step quote desired q minimum')
    k = z.Int('k')
    bumped = z.Bool('bumped')
    premise = z.And(balance >= 0, fraction > 0, cap > 0, price > 0, step > 0,
                    quote == z.If(balance * fraction <= cap, balance * fraction, cap), desired == quote / price,
                    # quantizeDown: the largest grid multiple not above the desired quantity
                    k >= 0, z.ToReal(k) * step <= desired, desired < (z.ToReal(k) + 1) * step,
                    z.If(bumped, q == minimum, q == z.ToReal(k) * step), minimum > z.ToReal(k) * step,
                    # The guard refuses any fraction-sized order above the cap (exact comparison), bumped or not.
                    q * price <= cap)
    s = z.Solver()
    s.set(timeout=20000, random_seed=0)
    s.add(premise)
    require(s.check() == z.sat, 'cap premise inconsistent')
    s.add(z.Not(q * price <= cap))
    result = s.check()
    require(result == z.unsat, f'cap violation or unknown ({result})')
    # Without the guard the bump does exceed the cap (CE-ROUND-003 is reachable in the model).
    m = z.Solver()
    m.add(z.And(balance >= 0, fraction > 0, cap > 0, price > 0, step > 0, quote == cap, desired == quote / price,
                k >= 0, z.ToReal(k) * step <= desired, desired < (z.ToReal(k) + 1) * step,
                bumped, q == minimum, minimum > z.ToReal(k) * step, q * price > cap))
    require(m.check() == z.sat, 'unguarded bump unexpectedly safe')
    return {'F-ROUND-MIN-CAP': 'unsat', 'unguardedWitness': 'sat'}


def prove_wire_grid():
    """Per binade [2^e, 2^(e+1)) with e < 24: half an ulp is below half a grid step, and an ulp is below a step."""
    grid = F(1, 10 ** GRID_DIGITS)
    for e in range(-GRID_DIGITS * 4, GRID_LIMIT_EXP):
        ulp = F(2) ** (e - 52)
        require(ulp / 2 < grid / 2, f'binade 2^{e}: nearest rendering may not recover the grid value')
        require(ulp < grid, f'binade 2^{e}: distinct grid values may share a double')
    return {'status': 'proved', 'binades': GRID_LIMIT_EXP + GRID_DIGITS * 4, 'limit': f'2^{GRID_LIMIT_EXP}'}


def acceptable_units(units, x, decimals=GRID_DIGITS):
    """gridUnits contract: the exact truncation, or a shortest round-trip decimal of x that fits the grid
    (shortest digits may tie between two decimals; either reads back as x). Its binary64 never exceeds x."""
    from decimal import Decimal
    value = F(units, 10 ** decimals)
    if float(value) > x if x >= 0 else float(value) < x:
        return False
    if units == grid_units(x, decimals):
        return True
    exact = F(abs(x)) * 10 ** decimals
    truncated = (exact.numerator // exact.denominator) * (-1 if x < 0 else 1)
    if units == truncated:
        return True
    shortest_len = len(Decimal(repr(abs(x))).as_tuple().digits)
    candidate = Decimal(abs(units)).scaleb(-decimals).normalize()
    return float(value) == x and len(candidate.as_tuple().digits) == shortest_len


def grid_units(x, decimals=GRID_DIGITS):
    """Oracle for gridUnits: the shortest round-trip decimal if it fits the grid exactly, else exact truncation."""
    from decimal import Decimal
    if x == 0:
        return 0
    shortest = Decimal(repr(abs(x)))
    sign = -1 if x < 0 else 1
    if -shortest.as_tuple().exponent <= decimals:
        return sign * int(shortest.scaleb(decimals))
    exact = F(abs(x)) * 10 ** decimals
    return sign * (exact.numerator // exact.denominator)


PROGRAM = r'''
import Trader.OrderNumeric (gridUnits, renderOrderNumber)
main :: IO ()
main = getContents >>= mapM_ (putStrLn . render . words) . lines
  where
    render ["wire", x] = renderOrderNumber (read x)
    render [d, x] = show (gridUnits (read d) (read x))
    render _ = error "bad line"
'''


def cases(seed=20261008, count=4000):
    rng = random.Random(seed)
    values = []
    for _ in range(count):
        digits = rng.randint(0, GRID_DIGITS)
        whole = rng.choice([0, 1, 7, 99, 12345, 2 ** 20, rng.randint(0, 2 ** GRID_LIMIT_EXP - 1)])
        frac = rng.randint(0, 10 ** digits - 1) if digits else 0
        value = F(whole) + F(frac, 10 ** digits)
        if 0 < value < 2 ** GRID_LIMIT_EXP:
            values.append(value)
    values += [F(1, 10 ** GRID_DIGITS), F(29, 100), F(1, 10), F(2 ** GRID_LIMIT_EXP - 1) + F(99999999, 10 ** 8)]
    return values


def raw_cases(seed=20261009, count=3000):
    """Arbitrary finite positive doubles below 2^53 from random bit patterns, plus non-grid fractions."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        x = struct.unpack('<d', struct.pack('<Q', rng.getrandbits(64)))[0]
        if x == x and 0 < x < 2.0 ** 53 and x != float('inf'):
            out.append(x)
    return out + [1 / 3, 2 / 3, 99.999999999, 4.0e-9, 0.123456789, 123456789.5]


def conformance():
    values = cases()
    raw = raw_cases()
    doubles = [float(v) for v in values] + raw
    with tempfile.TemporaryDirectory(prefix='trader-wire-grid-') as directory:
        path = Path(directory) / 'Wire.hs'
        path.write_text(PROGRAM)
        dex = [(d, x) for d in (6, 18) for x in raw[:1500] + [float(v) for v in values[:500]]]
        lines = ['wire ' + repr(d) for d in doubles] + [f'{d} {x!r}' for d, x in dex]
        out = subprocess.run(['runghc', '-i' + str(ROOT / 'haskell/app'), str(path)], input='\n'.join(lines) + '\n',
                             capture_output=True, text=True, timeout=600, check=True).stdout.split()
    require(len(out) == len(doubles) + len(dex), 'renderer output count')
    for (d, x), text in zip(dex, out[len(doubles):]):
        require(acceptable_units(int(text), x, d), f'gridUnits {d} {x!r} = {text}, oracle {grid_units(x, d)}')
        require(float(F(int(text), 10 ** d)) <= x, f'token amount above the checked value for {x!r} at {d} decimals')
    for value, text in zip(values, out):
        require(F(text) == value, f'grid value {value} rendered as {text}')
    for x, text in zip(raw, out[len(values):]):
        wire = F(text)
        require(wire.denominator <= 10 ** GRID_DIGITS and acceptable_units(int(wire * 10 ** GRID_DIGITS), x),
                f'renderer differs from the wire contract for {x!r}: {text}')
        require(float(wire) <= x, f'wire value above the checked value for {x!r}: {text}')
    ordered = sorted(set(values))
    require(all(float(a) < float(b) for a, b in zip(ordered, ordered[1:])), 'binary64 does not preserve grid order')
    return {'status': 'property_tested', 'cases': len(values), 'rawCases': len(raw), 'tokenCases': len(dex)}


def check_min_size_cap():
    return {'source': bind(), 'smt': prove_cap(), 'wireGrid': prove_wire_grid(), 'conformance': conformance()}
