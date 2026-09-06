{-# LANGUAGE OverloadedStrings #-}

module Trader.BotSnapshotRecovery (
    TradeMemorySnapshotContext (..),
    restoreTradeMemoryFromStatus,
    snapshotMatchesTradeMemoryContext,
) where

import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import Data.Char (toLower)
import Data.List (mapAccumL)
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
                tradeReturn =
                    case mReturn of
                        Just r | finite r -> r
                        _ ->
                            if finite entryEquity && finite exitEquity && entryEquity > 0
                                then exitEquity / entryEquity - 1
                                else 0
                exitReason = exitReasonRaw >>= exitReasonFromCode
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
  where
    finite x = not (isNaN x || isInfinite x)

reindexRestoredTrades :: [Trade] -> [Trade]
reindexRestoredTrades trades =
    snd (mapAccumL reindex 0 trades)
  where
    reindex idx tr =
        let hold = max 1 (trHoldingPeriods tr)
            entryIdx = idx
            exitIdx = idx + hold
         in (exitIdx + 1, tr{trEntryIndex = entryIdx, trExitIndex = exitIdx})

takeLast :: Int -> [a] -> [a]
takeLast n xs
    | n <= 0 = []
    | otherwise = drop (max 0 (length xs - n)) xs
