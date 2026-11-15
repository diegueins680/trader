-- Research-only executable. Not a Cabal production component or artifact loader.
module Main (main) where

import Control.Concurrent (threadDelay)
import Control.Exception (IOException, catch, mask, onException)
import Control.Monad (join, void)
import qualified Data.ByteString.Char8 as BS
import Data.List (transpose)
import GHC.Clock (getMonotonicTimeNSec)
import SnapshotRequestV3 (decodeSnapshot, snapshotBits)
import System.Environment (getArgs, getExecutablePath)
import System.IO (BufferMode (NoBuffering), Handle, hClose, hFlush, hGetChar, hSetBuffering, stdin, stdout)
import System.Posix.Signals (sigKILL, signalProcess)
import System.Process (CreateProcess (..), ProcessHandle, StdStream (CreatePipe, NoStream), createProcess, getPid, getProcessExitCode, proc, terminateProcess)
import System.Timeout (timeout)
import Text.Read (readMaybe)

data Decision = Absent | QuarterTarget Int deriving (Eq, Show)

type Request = ([Double], [Double])

budgetNS :: Integer
budgetNS = 20000000

clockNS :: IO Integer
clockNS = toInteger <$> getMonotonicTimeNSec

-- The whole acceptance guard is independently translated to SMT.
admission :: Integer -> Bool -> String -> Decision
admission elapsed clean frame
    | not clean || elapsed < 0 || elapsed >= budgetNS = Absent
    | frame == "IPV2 -1" = QuarterTarget (-1)
    | frame == "IPV2 0" = QuarterTarget 0
    | frame == "IPV2 1" = QuarterTarget 1
    | otherwise = Absent

safeIO :: a -> IO a -> IO a
safeIO fallback action = catch action (handler fallback)
  where
    handler :: a -> IOException -> IO a
    handler value _ = pure value

-- No lazy hGetContents or unbounded line read crosses the pipe boundary.
frameRead :: Int -> Handle -> IO (Maybe String)
frameRead limit handle = safeIO Nothing (go limit [])
  where
    go remaining chars
        | remaining <= 0 = pure Nothing
        | otherwise = do
            char <- hGetChar handle
            if char == '\n'
                then pure (Just (reverse chars))
                else go (remaining - 1) (char : chars)

validRequest :: Request -> Bool
validRequest (observation, parameters) =
    length observation == 12
        && length parameters == 259
        && all finiteBounded (observation ++ parameters)
  where
    finiteBounded x = not (isNaN x || isInfinite x) && abs x <= 1000

chunks :: Int -> [a] -> [[a]]
chunks size values
    | size <= 0 || null values = []
    | otherwise = let (prefix, suffix) = splitAt size values in prefix : chunks size suffix

multiply :: [Double] -> [[Double]] -> [Double]
multiply values rows = map (sum . zipWith (*) values) (transpose rows)

workerCompute :: Request -> Maybe Int
workerCompute request@(observation, parameters)
    | not (validRequest request) = Nothing
    | otherwise =
        let (w1, rest1) = splitAt 192 parameters
            (b1, rest2) = splitAt 16 rest1
            (w2, b2) = splitAt 48 rest2
            hidden = map tanh (zipWith (+) (multiply observation (chunks 16 w1)) b1)
            scores = zipWith (+) (multiply hidden (chunks 3 w2)) b2
         in case scores of
                [a, b, c]
                    | any (\x -> isNaN x || isInfinite x) scores -> Nothing
                    | a > b && a > c -> Just (-1)
                    | b > a && b > c -> Just 0
                    | c > a && c > b -> Just 1
                _ -> Nothing

worker :: IO ()
worker = do
    hSetBuffering stdout NoBuffering
    putStrLn "IPV2 ready"
    raw <- frameRead 32768 stdin
    let answer = raw >>= readMaybe >>= workerCompute
    putStrLn (maybe "IPV2 absent" (("IPV2 " ++) . show) answer)

waitExit :: Int -> ProcessHandle -> IO Bool
waitExit micros child = do
    started <- clockNS
    let loop = do
            exited <- getProcessExitCode child
            case exited of
                Just _ -> pure True
                Nothing -> do
                    now <- clockNS
                    if now - started >= toInteger micros * 1000 || now < started
                        then pure False
                        else threadDelay 1000 >> loop
    safeIO False loop

-- Only this parent reaps its child. Never signal a PID after a successful reap.
stopChild :: ProcessHandle -> IO Bool
stopChild child = safeIO False $ do
    exited <- getProcessExitCode child
    case exited of
        Just _ -> pure True
        Nothing -> do
            pid <- getPid child
            safeIO () (terminateProcess child)
            stopped <- waitExit 50000 child
            if stopped
                then pure True
                else case pid of
                    Nothing -> pure False
                    Just identity -> do
                        safeIO () (signalProcess sigKILL identity)
                        waitExit 50000 child

closePipe :: Maybe Handle -> IO Bool
closePipe Nothing = pure True
closePipe (Just handle) = safeIO False (hClose handle >> pure True)

cleanup :: (Maybe Handle, Maybe Handle, Maybe Handle, ProcessHandle) -> IO Bool
cleanup (input, output, errors, child) = do
    stopped <- stopChild child
    -- NoBuffering means there is no deferred request to flush on close.
    a <- closePipe input
    b <- closePipe output
    c <- closePipe errors
    pure (stopped && a && b && c)

exchange :: (String -> Maybe Request) -> Integer -> Handle -> Handle -> IO (Maybe String)
exchange decode started input output = do
    now <- clockNS
    let remaining = budgetNS - (now - started)
    if now < started || remaining <= 0
        then pure Nothing
        else do
            reply <- timeout (fromInteger ((remaining + 999) `div` 1000)) $ do
                raw <- frameRead 32768 stdin
                case raw >>= decode of
                    Just request | validRequest request -> do
                        let payload = BS.pack (show (request :: Request))
                        if BS.length payload >= 32768
                            then pure Nothing
                            else do
                                BS.hPutStrLn input payload
                                hFlush input
                                frameRead 16 output
                    _ -> pure Nothing
            pure (join reply)

supervise :: (String -> Maybe Request) -> IO (Decision, Bool)
supervise decode = mask $ \restore -> do
    executable <- getExecutablePath
    handles@(input, output, _, _) <- createProcess (proc executable ["--offline-inference-v2", "--worker"]){std_in = CreatePipe, std_out = CreatePipe, std_err = NoStream, env = Just [], close_fds = True, create_group = True}
    let session = case (input, output) of
            (Just requestPipe, Just replyPipe) -> do
                hSetBuffering requestPipe NoBuffering
                ready <- safeIO Nothing (timeout 1000000 (frameRead 16 replyPipe))
                started <- clockNS
                reply <- if ready == Just (Just "IPV2 ready") then safeIO Nothing (exchange decode started requestPipe replyPipe) else pure Nothing
                pure (started, reply)
            _ -> do
                now <- clockNS
                pure (now, Nothing)
    (started, reply) <- restore session `onException` void (cleanup handles)
    clean <- cleanup handles
    ended <- clockNS
    pure (maybe Absent (admission (ended - started) clean) reply, clean)

offline :: (String -> Maybe Request) -> IO ()
offline decode = do
    result <- safeIO (Absent, False) (supervise decode)
    print result

-- The contract probe runs the same pure guard; it cannot start a child.
contractProbe :: IO ()
contractProbe = do
    raw <- frameRead 32768 stdin
    case raw >>= readMaybe of
        Just (elapsed, clean, frame) -> print (admission elapsed clean frame)
        Nothing -> print Absent

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["--offline-inference-v2"] -> offline readMaybe
        ["--offline-snapshot-v3"] -> offline decodeSnapshot
        ["--snapshot-contract-v3"] -> frameRead 32768 stdin >>= print . (>>= snapshotBits)
        ["--offline-inference-v2", "--worker"] -> worker
        ["--offline-inference-v2", "--contract"] -> contractProbe
        _ -> putStrLn "(Absent,True)"
