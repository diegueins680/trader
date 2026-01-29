{-# LANGUAGE ScopedTypeVariables #-}

module Trader.Cache (
    TtlCache,
    newTtlCache,
    fetchWithCache,
    insertCache,
) where

import Control.Concurrent.MVar (MVar, modifyMVar, newMVar, readMVar)
import Control.Exception (SomeException, throwIO, try)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Time.Clock (NominalDiffTime, UTCTime, diffUTCTime, getCurrentTime)

newtype TtlCache k v = TtlCache (MVar (Map k (UTCTime, v)))

newTtlCache :: IO (TtlCache k v)
newTtlCache = TtlCache <$> newMVar Map.empty

fetchWithCache :: (Ord k) => TtlCache k v -> NominalDiffTime -> NominalDiffTime -> k -> IO v -> IO v
fetchWithCache cache freshTtl staleTtl key action = do
    now <- getCurrentTime
    mEntry <- readEntry cache key
    case mEntry of
        Just (ts, val) | diffUTCTime now ts <= freshTtl -> pure val
        _ -> do
            res <- try action
            case res of
                Right val -> do
                    insertEntry cache key val
                    pure val
                Left (err :: SomeException) ->
                    case mEntry of
                        Just (ts, val) | diffUTCTime now ts <= staleTtl -> pure val
                        _ -> throwIO err

insertCache :: (Ord k) => TtlCache k v -> k -> v -> IO ()
insertCache = insertEntry

readEntry :: (Ord k) => TtlCache k v -> k -> IO (Maybe (UTCTime, v))
readEntry (TtlCache ref) key = Map.lookup key <$> readMVar ref

insertEntry :: (Ord k) => TtlCache k v -> k -> v -> IO ()
insertEntry (TtlCache ref) key val = do
    now <- getCurrentTime
    _ <- modifyMVar ref $ \m -> pure (Map.insert key (now, val) m, ())
    pure ()
