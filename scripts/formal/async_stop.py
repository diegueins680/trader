"""CE-LIVE-001: a killed bot worker must not keep acting. Whole-tree source binding: no module under haskell/app
can capture an asynchronous exception except through reviewed forwarding sites, plus a compiled regression
against the actual Trader.App.AsyncSafe module."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MAIN = 'haskell/app/Main.hs'
BINANCE = 'haskell/app/Trader/Binance.hs'
ASYNC = 'haskell/app/Trader/App/AsyncSafe.hs'
COUNTEREXAMPLE = 'formal/research/async-stop-counterexample.json'
# Modules whose import lists could bring an exception-capturing combinator into scope.
EXCEPTION_MODULES = ('Control.Exception', 'Control.Exception.Base', 'Control.Exception.Safe', 'Control.Monad.Catch',
                     'GHC.IO', 'UnliftIO', 'UnliftIO.Exception')
CAPTURES = {'try', 'tryJust', 'catch', 'catchJust', 'handle', 'handleJust', 'catches', 'Handler', 'catchException',
            'catchAny', 'tryAny', 'handleAny', 'catchAll', 'handleAll'}
# The only modules allowed to import a capture, with what they may import and the fragment that makes it async-safe.
REVIEWED_IMPORTS = {
    ASYNC: ({'try', 'catch'}, 'Left ex | isAsyncException (toException ex) -> throwIO ex'),
    'haskell/app/Trader/App/BacktestGate.hs': ({'tryJust'}, 'case fromException ex :: Maybe SomeAsyncException of\n        Just _ -> Nothing'),
}
# Threads whose last act is handing their outcome to a waiter; a rethrown kill would leave the waiter blocked forever.
FORWARDING = {
    MAIN: ['catchForwardingAll\n                            (action item >>= putMVar resultVar . Right)'],
    'haskell/app/Trader/App/GracefulShutdown.hs': ['result <- tryForwardingAll action\n            putMVar done result'],
    BINANCE: ['result <- tryForwardingAll action\n        putMVar resultVar result'],
    'haskell/app/Trader/MarketContext.hs': ['res <- tryForwardingAll (action x) `finally` signalQSem sem\n            case res of\n                Left _ -> putMVar mv Nothing'],
    'haskell/app/Trader/Optimizer/Optimize.hs': ['_ <- tryForwardingAll loop\n            _ <- tryForwardingAll (hClose h)\n            putMVar done ()'],
}
IMPORT_HEAD = re.compile(r'^import\s+(qualified\s+)?([A-Z][\w.]*)(\s+qualified)?(\s+as\s+[A-Z][\w.]*)?(\s+hiding)?', re.M)


def imports(text):
    """Yield (module, qualified_or_aliased_or_hiding, explicit list or None) with a balanced-parenthesis list."""
    for m in IMPORT_HEAD.finditer(text):
        i = m.end()
        while i < len(text) and text[i] in ' \t\n' and not text.startswith('\nimport', i) and not (text[i] == '\n' and i + 1 < len(text) and text[i + 1] not in ' \t'):
            i += 1
        listed = None
        if i < len(text) and text[i] == '(':
            depth, j = 0, i
            while True:
                depth += {'(': 1, ')': -1}.get(text[j], 0)
                j += 1
                if depth == 0:
                    break
            listed = text[i:j]
        yield m.group(2), bool(m.group(1) or m.group(3) or m.group(4) or m.group(5)), listed


def require(ok, reason):
    if not ok:
        raise ValueError('async stop: ' + reason)


def bind(sources=None):
    sources = sources or {}
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'haskell/app').rglob('*.hs'))
    for name in sources:
        if name not in files:
            files.append(name)
    counts = {'trySync': 0, 'catchSync': 0}
    for name in files:
        text = sources.get(name) if name in sources else (ROOT / name).read_text()
        for module, restricted, listed in imports(text):
            if module not in EXCEPTION_MODULES and not module.startswith('UnliftIO'):
                continue
            require(not restricted and listed, f'{name}: {module} must be imported with an explicit unqualified list')
            captured = set(re.findall(r"[A-Za-z_][\w']*", listed)) & CAPTURES
            allowed, fragment = REVIEWED_IMPORTS.get(name, (set(), None))
            require(captured <= allowed, f'{name}: imports capture {sorted(captured - allowed)} from {module}')
            if captured:
                require(fragment in text, f'{name}: reviewed async-safe fragment changed')
        body = re.sub(r'^import Trader\.App\.AsyncSafe \([^)]*\)\n', '', text, flags=re.M)
        forwarding = len(re.findall(r'\b(?:try|catch)ForwardingAll\b', body))
        if name == ASYNC:
            require('trySync :: (Exception e) => IO a -> IO (Either e a)' in text and
                    'if isAsyncException (toException ex) then throwIO ex else handler ex' in text and
                    'isAsyncException ex = isJust (fromException ex :: Maybe SomeAsyncException)' in text,
                    'trySync/catchSync rethrow every async exception')
            continue
        reviewed = FORWARDING.get(name, [])
        require(all(text.count(f) == 1 for f in reviewed) and
                forwarding == sum(len(re.findall(r'\b(?:try|catch)ForwardingAll\b', f)) for f in reviewed),
                f'{name}: unreviewed capture of asynchronous exceptions')
        for key in counts:
            counts[key] += len(re.findall(r'\b' + key + r'\b', body))
    require(counts['trySync'] >= 150 and counts['catchSync'] >= 3, f'implausibly few converted sites: {counts}')
    return {'status': 'exhaustively_checked', 'modules': len(files), 'sites': counts,
            'forwardingSites': sum(len(v) for v in FORWARDING.values()),
            'scope': 'every Haskell module under haskell/app: no exception-capturing combinator is imported outside '
                     'Trader.App.AsyncSafe and the async-filtering BacktestGate; the compiler rejects any other capture'}


PROGRAM = r'''
import Control.Concurrent
import Control.Exception
import Data.IORef
import Trader.App.AsyncSafe (trySync)

routine :: (IO String -> IO (Either SomeException String)) -> IO [String]
routine capture = do
  steps <- newIORef []
  done <- newEmptyMVar
  ready <- newEmptyMVar
  tid <- forkIO $ (do
      r <- capture (putMVar ready () >> threadDelay 5000000 >> pure "entry")
      modifyIORef' steps (either (const "entry failed") id r :)
      modifyIORef' steps ("follow-up order" :)) `finally` putMVar done ()
  takeMVar ready
  killThread tid
  takeMVar done
  reverse <$> readIORef steps

main :: IO ()
main = do
  legacy <- routine try
  fixed <- routine trySync
  print (legacy, fixed)
'''


def conformance():
    with tempfile.TemporaryDirectory(prefix='trader-async-stop-') as directory:
        path = Path(directory) / 'AsyncStop.hs'
        path.write_text(PROGRAM)
        out = subprocess.run(['runghc', '-i' + str(ROOT / 'haskell/app'), str(path)],
                             capture_output=True, text=True, timeout=120, check=True).stdout.strip()
    recorded = json.loads((ROOT / COUNTEREXAMPLE).read_text())
    require(out == recorded['observed'], 'compiled regression differs from the preserved counterexample: ' + out)
    return {'status': 'property_tested', 'observed': out}


def check_async_stop():
    return {'source': bind(), 'conformance': conformance(),
            'helperSha256': hashlib.sha256((ROOT / ASYNC).read_bytes()).hexdigest()}
