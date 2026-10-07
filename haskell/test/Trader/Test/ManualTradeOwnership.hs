{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.ManualTradeOwnership (manualTradeOwnershipSuite) where

import Control.Concurrent (forkIO, threadDelay)
import Control.Concurrent.MVar (MVar, modifyMVar, newEmptyMVar, newMVar, putMVar, readMVar, takeMVar)
import Control.Exception (ErrorCall (..), SomeException, throwIO, try)
import Control.Monad (forM_, replicateM_)
import Data.IORef (atomicModifyIORef', newIORef, readIORef)
import qualified Data.Map.Strict as Map
import System.Timeout (timeout)
import Trader.App.ManualTradeOwnership

-- Tenant -> owner identities of its live bots, standing in for the bot runtime map.
type Runtime = Map.Map String [OwnershipKey]

manualTradeOwnershipSuite :: [(String, IO ())]
manualTradeOwnershipSuite =
    [ ("manual trade admitted when no bot owns the identity", admittedWhenFree)
    , ("manual trade admitted for another account on the same symbol", admittedForOtherAccount)
    , ("manual trade refused when any tenant's bot owns the identity", refusedWhenOwned)
    , ("bot start sees the claim only while the trade runs", claimVisibleDuringTrade)
    , ("second bot on the same account is refused across tenants", secondBotRefused)
    , ("claim is released after exceptions and timeouts", claimReleasedOnFailure)
    , ("bot start and manual trade never both own an identity", raceNeverDoubleOwns)
    ]

expect :: (Eq a, Show a) => String -> a -> a -> IO ()
expect label wanted actual =
    if wanted == actual then pure () else ioError (userError (label ++ ": expected " ++ show wanted ++ ", got " ++ show actual))

btcA, btcB, ethA :: OwnershipKey
btcA = ("binance:account-a", "BTCUSDT")
btcB = ("binance:account-b", "BTCUSDT")
ethA = ("binance:account-a", "ETHUSDT")

owned :: Runtime -> [OwnershipKey]
owned = concat . Map.elems

-- The bot-start publication step: under the lock, refuse a claimed or owned identity, else publish the bot.
startBot :: MVar Runtime -> ManualTradeClaims -> String -> OwnershipKey -> IO Bool
startBot lock claims tenant key =
    modifyMVar lock $ \rt -> do
        claimed <- manualTradeClaimed claims key
        if claimed || key `elem` owned rt
            then pure (rt, False)
            else pure (Map.insertWith (++) tenant [key] rt, True)

admittedWhenFree :: IO ()
admittedWhenFree = do
    lock <- newMVar (Map.singleton "t1" [ethA])
    claims <- newManualTradeClaims
    r <- withManualTradeClaim lock owned claims btcA (pure ("sent" :: String))
    expect "free identity" "sent" r

admittedForOtherAccount :: IO ()
admittedForOtherAccount = do
    lock <- newMVar (Map.singleton "t1" [btcA])
    claims <- newManualTradeClaims
    r <- withManualTradeClaim lock owned claims btcB (pure ("sent" :: String))
    expect "other account" "sent" r

refusedWhenOwned :: IO ()
refusedWhenOwned = do
    lock <- newMVar (Map.singleton "header-tenant" [btcA])
    claims <- newManualTradeClaims
    ran <- newIORef False
    r <- try (withManualTradeClaim lock owned claims btcA (atomicModifyIORef' ran (const (True, ()))))
    case r of
        Left (ManualTradeOwnershipConflict _) -> pure ()
        Right () -> ioError (userError "owned identity was not refused")
    readIORef ran >>= expect "refused trade never ran" False
    manualTradeClaimed claims btcA >>= expect "refusal leaves no claim" False

claimVisibleDuringTrade :: IO ()
claimVisibleDuringTrade = do
    lock <- newMVar Map.empty
    claims <- newManualTradeClaims
    during <- withManualTradeClaim lock owned claims btcA (startBot lock claims "t1" btcA)
    expect "bot start refused during manual trade" False during
    after <- startBot lock claims "t1" btcA
    expect "bot start allowed after manual trade" True after

secondBotRefused :: IO ()
secondBotRefused = do
    lock <- newMVar Map.empty
    claims <- newManualTradeClaims
    first <- startBot lock claims "server-tenant" btcA
    second <- startBot lock claims "header-tenant" btcA
    other <- startBot lock claims "header-tenant" btcB
    expect "first, same-account second, other-account" (True, False, True) (first, second, other)

claimReleasedOnFailure :: IO ()
claimReleasedOnFailure = do
    lock <- newMVar Map.empty
    claims <- newManualTradeClaims
    _ <- try (withManualTradeClaim lock owned claims btcA (throwIO (ErrorCall "venue rejected"))) :: IO (Either SomeException ())
    manualTradeClaimed claims btcA >>= expect "released after exception" False
    r <- timeout 20000 (withManualTradeClaim lock owned claims btcA (threadDelay 2000000))
    expect "timed out" True (null r)
    manualTradeClaimed claims btcA >>= expect "released after timeout" False

-- Many rounds of a bot start racing a manual trade on one identity: at most one may own it at any instant.
raceNeverDoubleOwns :: IO ()
raceNeverDoubleOwns =
    replicateM_ 200 $ do
        lock <- newMVar Map.empty
        claims <- newManualTradeClaims
        overlap <- newIORef False
        botDone <- newEmptyMVar
        tradeDone <- newEmptyMVar
        _ <- forkIO $ do
            _ <- startBot lock claims "t1" btcA
            putMVar botDone ()
        _ <- forkIO $ do
            _ <-
                try
                    ( withManualTradeClaim lock owned claims btcA $ do
                        threadDelay 50
                        rt <- readMVar lock
                        atomicModifyIORef' overlap (\o -> (o || btcA `elem` owned rt, ()))
                    ) ::
                    IO (Either ManualTradeOwnershipConflict ())
            putMVar tradeDone ()
        forM_ [botDone, tradeDone] takeMVar
        readIORef overlap >>= expect "bot and manual trade overlapped" False
