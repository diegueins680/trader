{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.GraduationTrace (graduationTraceSuite) where

import Control.Concurrent (threadDelay)
import Control.Exception (IOException, MaskingState (..), bracket, getMaskingState, mask_, try)
import Control.Monad (forM_, unless)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString as BS
import Data.IORef (modifyIORef', newIORef, readIORef, writeIORef)
import Data.List (sort)
import Data.Maybe (isNothing)
import System.Directory (getTemporaryDirectory, removeFile)
import System.IO (SeekMode (AbsoluteSeek), hClose, hFileSize, hFlush, hSeek, openBinaryTempFile)
import System.IO.Error (ioeGetErrorString)
import System.Timeout (timeout)
import Trader.Ops.GraduationTrace

graduationTraceSuite :: [(String, IO ())]
graduationTraceSuite =
    [ ("preserves success, laziness and masking", testSuccess)
    , ("records failure and cancellation lifetimes", testUnwinding)
    , ("observer failure preserves action behavior and cancellation", testObserverFailure)
    , ("connection interval contains all five ordered query phases", testNestedQueries)
    , ("emits only the fixed numeric trace schema", testWireSchema)
    ]

check :: String -> Bool -> IO ()
check message ok = unless ok (ioError (userError message))

testSuccess :: IO ()
testSuccess = do
    events <- newIORef []
    let collect event = modifyIORef' events (++ [event])
    value <- withGraduationPhaseObserved GraduationBaseline collect (pure (7 :: Int))
    check "action result changed" (value == 7)
    rows <- readIORef events
    checkPair GraduationBaseline rows
    _ <- withGraduationPhaseObserved GraduationDaily collect (pure (error "result must remain lazy" :: Int))
    unmasked <- withGraduationPhaseObserved GraduationOrders collect getMaskingState
    inherited <- mask_ (withGraduationPhaseObserved GraduationStatus collect getMaskingState)
    check "action masking changed" (unmasked == Unmasked && inherited == MaskedInterruptible)

checkPair :: GraduationPhase -> [GraduationPhaseEvent] -> IO ()
checkPair phase rows = check "missing or malformed lifetime pair" $ case rows of
    [GraduationPhaseStarted started, GraduationPhaseFinished ended elapsed] ->
        started == phase && ended == phase && not (isNaN elapsed || isInfinite elapsed) && elapsed >= 0
    _ -> False

testUnwinding :: IO ()
testUnwinding = do
    events <- newIORef []
    let collect event = modifyIORef' events (++ [event])
    failed <- try (withGraduationPhaseObserved GraduationBaseline collect (ioError (userError "private failure"))) :: IO (Either IOException ())
    check "action exception swallowed" (either (const True) (const False) failed)
    readIORef events >>= checkPair GraduationBaseline
    writeIORef events []
    cancelled <- timeout 20000 (withGraduationPhaseObserved GraduationDaily collect (threadDelay 1000000))
    check "action cancellation swallowed" (isNothing cancelled)
    readIORef events >>= checkPair GraduationDaily

testObserverFailure :: IO ()
testObserverFailure = do
    let unavailable _ = ioError (userError "observer unavailable")
    value <- withGraduationPhaseObserved GraduationLatest unavailable (pure (11 :: Int))
    check "observer changed success" (value == 11)
    failed <- try (withGraduationPhaseObserved GraduationLatest unavailable (ioError (userError "original action"))) :: IO (Either IOException ())
    check "observer replaced original error" (either ((== "original action") . ioeGetErrorString) (const False) failed)
    ran <- newIORef False
    cancelled <- timeout 20000 (withGraduationPhaseObserved GraduationLatest (const (threadDelay 1000000)) (writeIORef ran True))
    acted <- readIORef ran
    check "observer cancellation swallowed" (isNothing cancelled && not acted)

testNestedQueries :: IO ()
testNestedQueries = do
    events <- newIORef []
    let collect event = modifyIORef' events (++ [event])
        phases = [GraduationBaseline, GraduationDaily, GraduationOrders, GraduationStatus, GraduationLatest]
    withGraduationPhaseObserved GraduationConnection collect $
        forM_ phases $
            \phase -> withGraduationPhaseObserved phase collect (pure ())
    rows <- readIORef events
    check "parent connection did not enclose ordered queries" $
        case rows of
            GraduationPhaseStarted GraduationConnection : rest ->
                case reverse rest of
                    GraduationPhaseFinished GraduationConnection _ : queries ->
                        map startedPhase (reverse queries) == concatMap (\p -> [Just p, Nothing]) phases
                    _ -> False
            _ -> False
  where
    startedPhase event = case event of
        GraduationPhaseStarted phase -> Just phase
        GraduationPhaseFinished _ _ -> Nothing

testWireSchema :: IO ()
testWireSchema = do
    directory <- getTemporaryDirectory
    bracket (openBinaryTempFile directory "graduation-trace.jsonl") (\(path, handle) -> hClose handle >> removeFile path) $ \(_path, handle) -> do
        withLoggedGraduationConnection handle $ \attemptId ->
            forM_ [GraduationBaseline, GraduationDaily, GraduationOrders, GraduationStatus, GraduationLatest] $ \phase ->
                withLoggedGraduationPhase handle attemptId phase (pure ())
        hFlush handle
        hSeek handle AbsoluteSeek 0
        size <- hFileSize handle
        bytes <- BS.hGet handle (fromIntegral size)
        let linesOfJson = filter (not . BS.null) (BS.split 10 bytes)
        check "unexpected number of trace records" (length linesOfJson == 12)
        objects <- traverse decode linesOfJson
        let ids = map (KM.lookup "attemptId") objects
        check "nested trace attempt IDs differ" $ case ids of
            first : rest -> all (== first) rest
            [] -> False
        forM_ objects $ \obj -> do
            check "unexpected fields in trace" (sort (KM.keys obj) == sort ["event", "attemptId", "atMs", "phase", "state", "elapsedMs"])
            check "wrong event name" (KM.lookup "event" obj == Just (Aeson.String "portfolio_graduation_query"))
            check "missing numeric timestamp" $ case KM.lookup "atMs" obj of Just (Aeson.Number n) -> n > 0; _ -> False
            check "unexpected phase" (KM.lookup "phase" obj `elem` map (Just . Aeson.String) ["connection", "baseline", "daily", "orders", "status", "latest"])
            check "invalid state or duration" $ case (KM.lookup "state" obj, KM.lookup "elapsedMs" obj) of
                (Just (Aeson.String "started"), Just Aeson.Null) -> True
                (Just (Aeson.String "finished"), Just (Aeson.Number n)) -> n >= 0
                _ -> False
  where
    decode bytes = case Aeson.eitherDecodeStrict' bytes of
        Right (Aeson.Object obj) -> pure obj
        _ -> ioError (userError "malformed trace JSON")
