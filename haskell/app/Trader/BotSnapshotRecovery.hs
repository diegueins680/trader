{-# LANGUAGE OverloadedStrings #-}

module Trader.BotSnapshotRecovery (
    TradeMemorySnapshotContext (..),
    restoreTradeMemoryFromStatus,
    snapshotMatchesTradeMemoryContext,
) where

import Control.Monad (unless)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import Data.Char (toLower)
import Data.Maybe (fromMaybe, mapMaybe)
import qualified Data.Text as T
import qualified Data.Vector as V
import Trader.Text (trim)
import Trader.Trading (Trade (..), TradeEntrySource (..), exitReasonFromCode)

{- | Identity and bounded-history contract for closed-trade memory recovery.
Position and open-trade state are deliberately absent: startup exposure is
established from the venue, never from a persisted status snapshot.
-}
data TradeMemorySnapshotContext = TradeMemorySnapshotContext
    { tmscSymbol :: !String
    , tmscInterval :: !String
    , tmscMarket :: !String
    , tmscMethod :: !String
    , tmscTradeLimit :: !Int
    }
    deriving (Eq, Show)

snapshotMatchesTradeMemoryContext :: TradeMemorySnapshotContext -> Aeson.Value -> Bool
snapshotMatchesTradeMemoryContext context statusValue =
    case statusValue of
        Aeson.Object o ->
            let getText key = KM.lookup key o >>= AT.parseMaybe Aeson.parseJSON
             in getText "symbol" == Just (tmscSymbol context)
                    && getText "interval" == Just (tmscInterval context)
                    && getText "market" == Just (tmscMarket context)
                    && getText "method" == Just (tmscMethod context)
        _ -> False

restoreTradeMemoryFromStatus :: TradeMemorySnapshotContext -> Aeson.Value -> [Trade]
restoreTradeMemoryFromStatus context statusValue
    | not (snapshotMatchesTradeMemoryContext context statusValue) = []
    | otherwise =
        case statusValue of
            Aeson.Object o ->
                case KM.lookup "trades" o of
                    Just (Aeson.Array tradesV) ->
                        reindexRestoredTrades
                            (takeLast (tmscTradeLimit context) (mapMaybe tradeFromSnapshotValue (V.toList tradesV)))
                    _ -> []
            _ -> []

parseTradeEntrySourceCode :: String -> Maybe TradeEntrySource
parseTradeEntrySourceCode raw =
    case map toLower (trim raw) of
        "signal" -> Just TradeEntrySignal
        "adopted" -> Just TradeEntryAdopted
        "post_direction_gates" -> Just TradeEntryPostDirectionGates
        "post-direction-gates" -> Just TradeEntryPostDirectionGates
        "postdirectiongates" -> Just TradeEntryPostDirectionGates
        _ -> Nothing

tradeFromSnapshotValue :: Aeson.Value -> Maybe Trade
tradeFromSnapshotValue =
    AT.parseMaybe $
        Aeson.withObject "Trade" $ \o -> do
            entryEquity <- o Aeson..: "entryEquity"
            exitEquity <- o Aeson..: "exitEquity"
            mReturn <- o Aeson..:? "return"
            holdingPeriods <- fromMaybe 0 <$> (o Aeson..:? "holdingPeriods")
            entryHighVolProb <- o Aeson..:? "entryHighVolProb"
            entrySourceRaw <- o Aeson..:? "entrySource"
            exitReasonRaw <- o Aeson..:? "exitReason"
            entryIp <- o Aeson..:? "entryIp"
            exitIp <- o Aeson..:? "exitIp"
            let entrySource = fromMaybe TradeEntrySignal (entrySourceRaw >>= parseTradeEntrySourceCode)
                exitReason = exitReasonRaw >>= exitReasonFromCode
            unless (validTradeMetadata holdingPeriods entryHighVolProb) (fail "Invalid closed-trade metadata.")
            tradeReturn <- maybe (fail "Invalid closed-trade equity or return.") pure (checkedTradeReturn entryEquity exitEquity mReturn)
            pure
                Trade
                    { trEntryIndex = 0
                    , trExitIndex = max 1 holdingPeriods
                    , trEntryEquity = entryEquity
                    , trExitEquity = exitEquity
                    , trReturn = tradeReturn
                    , trHoldingPeriods = holdingPeriods
                    , trEntryHighVolProb = entryHighVolProb
                    , trEntrySource = entrySource
                    , trExitReason = exitReason
                    , trEntryIp = entryIp
                    , trExitIp = exitIp
                    , trFeeCost = 0
                    }

-- | Missing return may be derived; explicitly invalid evidence never becomes zero.
checkedTradeReturn :: Double -> Double -> Maybe Double -> Maybe Double
checkedTradeReturn entryEquity exitEquity supplied
    | not (finite entryEquity && entryEquity > 0 && finite exitEquity) = Nothing
    | otherwise =
        let value = fromMaybe (exitEquity / entryEquity - 1) supplied
         in if finite value then Just value else Nothing

validTradeMetadata :: Int -> Maybe Double -> Bool
validTradeMetadata holdingPeriods probability =
    holdingPeriods >= 0 && maybe True (\p -> finite p && p >= 0 && p <= 1) probability

finite :: Double -> Bool
finite x = not (isNaN x || isInfinite x)

reindexRestoredTrades :: [Trade] -> [Trade]
reindexRestoredTrades = fromMaybe [] . go 0
  where
    go :: Integer -> [Trade] -> Maybe [Trade]
    go _ [] = Just []
    go idx (tr : rest) =
        let exitIdx = idx + max 1 (toInteger (trHoldingPeriods tr))
         in if idx < 0 || exitIdx > toInteger (maxBound :: Int)
                then Nothing
                else
                    (tr{trEntryIndex = fromInteger idx, trExitIndex = fromInteger exitIdx} :)
                        <$> go (exitIdx + 1) rest

takeLast :: Int -> [a] -> [a]
takeLast n xs
    | n <= 0 = []
    | otherwise = drop (max 0 (length xs - n)) xs
