{-# LANGUAGE OverloadedStrings #-}

module Trader.Ops.GraduationTrace (
    GraduationPhase (..),
    GraduationPhaseEvent (..),
    withGraduationPhaseObserved,
    writeGraduationPhaseEvent,
    withLoggedGraduationConnection,
    withLoggedGraduationPhase,
) where

import Control.Concurrent (myThreadId)
import Control.Exception (SomeException, finally, mask)
import Data.Aeson (object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int64)
import Data.Time.Clock.POSIX (getPOSIXTime)
import GHC.Clock (getMonotonicTimeNSec)
import System.IO (Handle)
import Trader.App.AsyncSafe (trySync)

data GraduationPhase
    = GraduationConnection
    | GraduationBaseline
    | GraduationDaily
    | GraduationOrders
    | GraduationStatus
    | GraduationLatest
    deriving (Eq, Show)

data GraduationPhaseEvent
    = GraduationPhaseStarted !GraduationPhase
    | GraduationPhaseFinished !GraduationPhase !Double
    deriving (Eq, Show)

{- | Observe lifetime, including failure or cancellation, without forcing the
result. Synchronous observer failures leave the action's behavior intact.
-}
withGraduationPhaseObserved :: GraduationPhase -> (GraduationPhaseEvent -> IO ()) -> IO a -> IO a
withGraduationPhaseObserved phase observe action = mask $ \restore -> do
    startedAt <- getMonotonicTimeNSec
    let emit event = do
            _ <- trySync (observe event) :: IO (Either SomeException ())
            pure ()
        finish = do
            endedAt <- getMonotonicTimeNSec
            let elapsedMs = fromIntegral (toInteger endedAt - toInteger startedAt) / 1000000
            emit (GraduationPhaseFinished phase elapsedMs)
    restore (emit (GraduationPhaseStarted phase) >> action) `finally` finish

-- | Only internal attempt IDs, fixed enum labels and numeric timing fields.
writeGraduationPhaseEvent :: Handle -> String -> Int64 -> GraduationPhaseEvent -> IO ()
writeGraduationPhaseEvent handle attemptId atMs event = do
    let (phase, state, elapsedMs) = case event of
            GraduationPhaseStarted value -> (value, "started", Nothing)
            GraduationPhaseFinished value elapsed -> (value, "finished", Just elapsed)
        label = case phase of
            GraduationConnection -> "connection"
            GraduationBaseline -> "baseline"
            GraduationDaily -> "daily"
            GraduationOrders -> "orders"
            GraduationStatus -> "status"
            GraduationLatest -> "latest"
        record =
            object
                [ "event" .= ("portfolio_graduation_query" :: String)
                , "attemptId" .= attemptId
                , "atMs" .= atMs
                , "phase" .= (label :: String)
                , "state" .= (state :: String)
                , "elapsedMs" .= elapsedMs
                ]
    BS.hPut handle (BL.toStrict (Aeson.encode record) <> "\n")

{- | Call only inside the shared provider callback. A connection event then
identifies a holder interval, rather than time waiting to obtain the lock.
-}
withLoggedGraduationConnection :: Handle -> (String -> IO a) -> IO a
withLoggedGraduationConnection handle action = do
    startedAt <- getMonotonicTimeNSec
    thread <- myThreadId
    let attemptId = show thread ++ ":" ++ show startedAt
    withLoggedGraduationPhase handle attemptId GraduationConnection (action attemptId)

withLoggedGraduationPhase :: Handle -> String -> GraduationPhase -> IO a -> IO a
withLoggedGraduationPhase handle attemptId phase =
    withGraduationPhaseObserved phase $ \event -> do
        atMs <- floor . (* 1000) <$> getPOSIXTime
        writeGraduationPhaseEvent handle attemptId atMs event
