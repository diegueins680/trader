{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Trader.Cache (
    TtlCache,
    TtlCacheStats (..),
    newTtlCache,
    newTtlCacheWithMaxEntries,
    fetchWithCache,
    insertCache,
    cacheSize,
    cacheStats,
) where

import Control.Concurrent.MVar (MVar, modifyMVar, newMVar, readMVar)
import Control.Exception (SomeException, throwIO, try)
import Data.Aeson (ToJSON (..), object, (.=))
import Data.List (sortOn)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Time.Clock (NominalDiffTime, UTCTime, diffUTCTime, getCurrentTime)

data TtlCache k v = TtlCache
    { tcMaxEntries :: !(Maybe Int)
    , tcEntriesRef :: !(MVar (Map k (UTCTime, v)))
    }

data TtlCacheStats = TtlCacheStats
    { tcsEntries :: !Int
    , tcsMaxEntries :: !(Maybe Int)
    }
    deriving (Eq, Show)

instance ToJSON TtlCacheStats where
    toJSON stats =
        object
            [ "entries" .= tcsEntries stats
            , "maxEntries" .= tcsMaxEntries stats
            ]

newTtlCache :: IO (TtlCache k v)
newTtlCache = newTtlCacheWithMaxEntries (-1)

newTtlCacheWithMaxEntries :: Int -> IO (TtlCache k v)
newTtlCacheWithMaxEntries maxEntries =
    TtlCache (normalizeMaxEntries maxEntries) <$> newMVar Map.empty

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
                    insertEntry cache staleTtl key val
                    pure val
                Left (err :: SomeException) -> do
                    nowAfter <- getCurrentTime
                    mFallback <- readFreshEntry cache staleTtl nowAfter key
                    case mFallback of
                        Just (_, val) -> pure val
                        _ -> throwIO err

insertCache :: (Ord k) => TtlCache k v -> NominalDiffTime -> k -> v -> IO ()
insertCache = insertEntry

cacheSize :: TtlCache k v -> IO Int
cacheSize cache = Map.size <$> readMVar (tcEntriesRef cache)

cacheStats :: TtlCache k v -> NominalDiffTime -> IO TtlCacheStats
cacheStats cache staleTtl = do
    now <- getCurrentTime
    let ref = tcEntriesRef cache
    modifyMVar ref $ \m ->
        let pruned = pruneExpiredEntries staleTtl now m
         in pure (pruned, TtlCacheStats (Map.size pruned) (tcMaxEntries cache))

readEntry :: (Ord k) => TtlCache k v -> k -> IO (Maybe (UTCTime, v))
readEntry cache key = Map.lookup key <$> readMVar (tcEntriesRef cache)

readFreshEntry :: (Ord k) => TtlCache k v -> NominalDiffTime -> UTCTime -> k -> IO (Maybe (UTCTime, v))
readFreshEntry cache staleTtl now key =
    modifyMVar ref $ \m ->
        let pruned = pruneExpiredEntries staleTtl now m
         in pure (pruned, Map.lookup key pruned)
  where
    ref = tcEntriesRef cache

pruneExpiredEntries :: NominalDiffTime -> UTCTime -> Map k (UTCTime, v) -> Map k (UTCTime, v)
pruneExpiredEntries staleTtl now = Map.filter (\(ts, _) -> diffUTCTime now ts <= staleTtl)

insertEntry :: (Ord k) => TtlCache k v -> NominalDiffTime -> k -> v -> IO ()
insertEntry cache staleTtl key val = do
    now <- getCurrentTime
    _ <-
        modifyMVar ref $ \m ->
            let pruned = pruneExpiredEntries staleTtl now m
                inserted = Map.insert key (now, val) pruned
                bounded = enforceMaxEntries (tcMaxEntries cache) inserted
             in pure (bounded, ())
    pure ()
  where
    ref = tcEntriesRef cache

normalizeMaxEntries :: Int -> Maybe Int
normalizeMaxEntries maxEntries =
    if maxEntries < 0
        then Nothing
        else Just maxEntries

enforceMaxEntries :: (Ord k) => Maybe Int -> Map k (UTCTime, v) -> Map k (UTCTime, v)
enforceMaxEntries mMaxEntries entries =
    case mMaxEntries of
        Nothing -> entries
        Just maxEntries
            | maxEntries <= 0 -> Map.empty
            | Map.size entries <= maxEntries -> entries
            | otherwise ->
                let keepCount = max 0 maxEntries
                    survivors =
                        drop
                            (Map.size entries - keepCount)
                            (sortOn (\(k, (ts, _)) -> (ts, k)) (Map.toList entries))
                 in Map.fromList survivors
