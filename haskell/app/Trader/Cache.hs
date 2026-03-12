{-# LANGUAGE ScopedTypeVariables #-}

module Trader.Cache (
    TtlCache,
    newTtlCache,
    fetchWithCache,
    insertCache,
    cacheSize,
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
    mEntry <- readFreshEntry cache staleTtl now key
    case mEntry of
        Just (ts, val) | diffUTCTime now ts <= freshTtl -> pure val
        _ -> do
            res <- try action
            case res of
                Right val -> do
                    insertEntry cache key val
                    pure val
                Left (err :: SomeException) -> do
                    nowAfter <- getCurrentTime
                    mFallback <- readFreshEntry cache staleTtl nowAfter key
                    case mFallback of
                        Just (_, val) -> pure val
                        _ -> throwIO err

insertCache :: (Ord k) => TtlCache k v -> k -> v -> IO ()
insertCache = insertEntry

cacheSize :: TtlCache k v -> IO Int
cacheSize (TtlCache ref) = Map.size <$> readMVar ref

readEntry :: (Ord k) => TtlCache k v -> k -> IO (Maybe (UTCTime, v))
readEntry (TtlCache ref) key = Map.lookup key <$> readMVar ref

readFreshEntry :: (Ord k) => TtlCache k v -> NominalDiffTime -> UTCTime -> k -> IO (Maybe (UTCTime, v))
readFreshEntry (TtlCache ref) staleTtl now key =
    modifyMVar ref $ \m ->
        let pruned = pruneExpiredEntries staleTtl now m
         in pure (pruned, Map.lookup key pruned)

pruneExpiredEntries :: NominalDiffTime -> UTCTime -> Map k (UTCTime, v) -> Map k (UTCTime, v)
pruneExpiredEntries staleTtl now = Map.filter (\(ts, _) -> diffUTCTime now ts <= staleTtl)

insertEntry :: (Ord k) => TtlCache k v -> k -> v -> IO ()
insertEntry (TtlCache ref) key val = do
    now <- getCurrentTime
    _ <- modifyMVar ref $ \m -> pure (Map.insert key (now, val) m, ())
    pure ()
