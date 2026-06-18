module Trader.BinanceTradeAnalysis (
    BinanceTradeMaxPnl (..),
    attachBinanceTradeMaxPnl,
    binanceTradeMaxPnlKlineRanges,
) where

import Data.Char (toUpper)
import Data.Int (Int64)
import Data.List (foldl', sortOn)
import qualified Data.Map.Strict as M
import Data.Maybe (mapMaybe)
import Trader.Binance (BinanceTrade (..), Kline (..))

data BinanceTradeMaxPnl = BinanceTradeMaxPnl
    { btmpPnl :: !Double
    , btmpCloseTime :: !Int64
    }
    deriving (Eq, Show)

data TradeSide = SideBuy | SideSell
    deriving (Eq, Show)

data PositionSideMode = PositionLong | PositionShort | PositionBoth
    deriving (Eq, Show)

data TradeDirection = DirectionLong | DirectionShort
    deriving (Eq, Show)

type TradeKey = (String, Int64)

data OpenLot = OpenLot
    { olTradeKey :: !TradeKey
    , olSymbol :: !String
    , olDirection :: !TradeDirection
    , olEntryTime :: !Int64
    , olEntryPrice :: !Double
    , olOriginalQty :: !Double
    , olRemainingQty :: !Double
    }
    deriving (Eq, Show)

data TargetLot = TargetLot
    { tlDirection :: !TradeDirection
    , tlEntryTime :: !Int64
    , tlEntryPrice :: !Double
    , tlQty :: !Double
    }
    deriving (Eq, Show)

data PnlTarget = PnlTarget
    { ptTradeKey :: !TradeKey
    , ptSymbol :: !String
    , ptDirection :: !TradeDirection
    , ptLots :: ![TargetLot]
    , ptLowerTime :: !Int64
    , ptUpperTime :: !Int64
    , ptActualExit :: !(Maybe (Int64, Double))
    }
    deriving (Eq, Show)

data AnalysisState = AnalysisState
    { asLongLots :: !(M.Map String [OpenLot])
    , asShortLots :: !(M.Map String [OpenLot])
    , asNetPositions :: !(M.Map String Double)
    , asTargets :: ![PnlTarget]
    }
    deriving (Eq, Show)

emptyAnalysisState :: AnalysisState
emptyAnalysisState =
    AnalysisState
        { asLongLots = M.empty
        , asShortLots = M.empty
        , asNetPositions = M.empty
        , asTargets = []
        }

qtyEps :: Double
qtyEps = 1e-12

attachBinanceTradeMaxPnl :: M.Map String [Kline] -> [BinanceTrade] -> [BinanceTrade]
attachBinanceTradeMaxPnl klineMap trades =
    let resultMap =
            foldl'
                insertTargetResult
                M.empty
                (collectPnlTargets trades)
     in map (attachResult resultMap) trades
  where
    insertTargetResult acc target =
        case scoreTarget klineMap target of
            Nothing -> acc
            Just result -> M.insertWith chooseBetterResult (ptTradeKey target) result acc
    attachResult resultMap trade =
        case M.lookup (tradeKey trade) resultMap of
            Nothing -> trade
            Just result ->
                trade
                    { btMaxPnl = Just (btmpPnl result)
                    , btMaxPnlCloseTime = Just (btmpCloseTime result)
                    }

binanceTradeMaxPnlKlineRanges :: [BinanceTrade] -> M.Map String (Int64, Int64)
binanceTradeMaxPnlKlineRanges =
    foldl' addTargetRange M.empty . collectPnlTargets
  where
    addTargetRange acc target =
        M.insertWith
            mergeRange
            (ptSymbol target)
            (ptLowerTime target, ptUpperTime target)
            acc
    mergeRange (aStart, aEnd) (bStart, bEnd) = (min aStart bStart, max aEnd bEnd)

collectPnlTargets :: [BinanceTrade] -> [PnlTarget]
collectPnlTargets trades =
    reverse . asTargets $
        foldl' stepTrade emptyAnalysisState (sortTrades trades)

sortTrades :: [BinanceTrade] -> [BinanceTrade]
sortTrades =
    sortOn (\t -> (finiteTimeOrMax (btTime t), btTradeId t))

finiteTimeOrMax :: Int64 -> Int64
finiteTimeOrMax t
    | t >= 0 = t
    | otherwise = maxBound

stepTrade :: AnalysisState -> BinanceTrade -> AnalysisState
stepTrade st trade =
    case normalizedTradeInputs trade of
        Nothing -> st
        Just (sym, key, side, posSide, qty, price, tradeTime) ->
            case posSide of
                PositionLong ->
                    case side of
                        SideBuy -> pushLongLot (sym ++ "::LONG") (mkLot key sym DirectionLong tradeTime price qty) st
                        SideSell -> closeLongLots (sym ++ "::LONG") key sym qty price tradeTime st
                PositionShort ->
                    case side of
                        SideSell -> pushShortLot (sym ++ "::SHORT") (mkLot key sym DirectionShort tradeTime price qty) st
                        SideBuy -> closeShortLots (sym ++ "::SHORT") key sym qty price tradeTime st
                PositionBoth -> stepBothPosition sym key side qty price tradeTime st

normalizedTradeInputs :: BinanceTrade -> Maybe (String, TradeKey, TradeSide, PositionSideMode, Double, Double, Int64)
normalizedTradeInputs trade = do
    side <- tradeSide trade
    let sym = normalizeSymbolKey (btSymbol trade)
        key = (sym, btTradeId trade)
        posSide = positionSideMode (btPositionSide trade)
        qty = btQty trade
        price = btPrice trade
        tradeTime = btTime trade
    if null sym || not (positiveFinite qty) || not (positiveFinite price) || tradeTime < 0
        then Nothing
        else Just (sym, key, side, posSide, qty, price, tradeTime)

stepBothPosition :: String -> TradeKey -> TradeSide -> Double -> Double -> Int64 -> AnalysisState -> AnalysisState
stepBothPosition sym key side qty price tradeTime st =
    let netKey = sym ++ "::BOTH"
        longKey = sym ++ "::BOTH:LONG"
        shortKey = sym ++ "::BOTH:SHORT"
        net = M.findWithDefault 0 netKey (asNetPositions st)
     in case side of
            SideBuy ->
                let (stAfterClose, openQty) =
                        if net < -qtyEps
                            then
                                let closeQty = min qty (abs net)
                                 in (closeShortLots shortKey key sym closeQty price tradeTime st, qty - closeQty)
                            else (st, qty)
                    stAfterOpen =
                        if openQty > qtyEps
                            then pushLongLot longKey (mkLot key sym DirectionLong tradeTime price openQty) stAfterClose
                            else stAfterClose
                 in stAfterOpen{asNetPositions = M.insert netKey (net + qty) (asNetPositions stAfterOpen)}
            SideSell ->
                let (stAfterClose, openQty) =
                        if net > qtyEps
                            then
                                let closeQty = min qty net
                                 in (closeLongLots longKey key sym closeQty price tradeTime st, qty - closeQty)
                            else (st, qty)
                    stAfterOpen =
                        if openQty > qtyEps
                            then pushShortLot shortKey (mkLot key sym DirectionShort tradeTime price openQty) stAfterClose
                            else stAfterClose
                 in stAfterOpen{asNetPositions = M.insert netKey (net - qty) (asNetPositions stAfterOpen)}

mkLot :: TradeKey -> String -> TradeDirection -> Int64 -> Double -> Double -> OpenLot
mkLot key sym dir entryTime entryPrice qty =
    OpenLot
        { olTradeKey = key
        , olSymbol = sym
        , olDirection = dir
        , olEntryTime = entryTime
        , olEntryPrice = entryPrice
        , olOriginalQty = qty
        , olRemainingQty = qty
        }

pushLongLot :: String -> OpenLot -> AnalysisState -> AnalysisState
pushLongLot key lot st =
    st{asLongLots = pushLot key lot (asLongLots st)}

pushShortLot :: String -> OpenLot -> AnalysisState -> AnalysisState
pushShortLot key lot st =
    st{asShortLots = pushLot key lot (asShortLots st)}

pushLot :: String -> OpenLot -> M.Map String [OpenLot] -> M.Map String [OpenLot]
pushLot key lot =
    M.alter (Just . maybe [lot] (++ [lot])) key

closeLongLots :: String -> TradeKey -> String -> Double -> Double -> Int64 -> AnalysisState -> AnalysisState
closeLongLots lotKey closeKey sym qty exitPrice exitTime st =
    let (nextLots, targets) = closeLots closeKey sym qty exitPrice exitTime (M.findWithDefault [] lotKey (asLongLots st))
     in st
            { asLongLots = setLots lotKey nextLots (asLongLots st)
            , asTargets = targets ++ asTargets st
            }

closeShortLots :: String -> TradeKey -> String -> Double -> Double -> Int64 -> AnalysisState -> AnalysisState
closeShortLots lotKey closeKey sym qty exitPrice exitTime st =
    let (nextLots, targets) = closeLots closeKey sym qty exitPrice exitTime (M.findWithDefault [] lotKey (asShortLots st))
     in st
            { asShortLots = setLots lotKey nextLots (asShortLots st)
            , asTargets = targets ++ asTargets st
            }

setLots :: String -> [OpenLot] -> M.Map String [OpenLot] -> M.Map String [OpenLot]
setLots key lots store
    | null lots = M.delete key store
    | otherwise = M.insert key lots store

closeLots :: TradeKey -> String -> Double -> Double -> Int64 -> [OpenLot] -> ([OpenLot], [PnlTarget])
closeLots closeKey sym qty exitPrice exitTime lots =
    let (_remainingQty, nextLots, consumedLots, openTargets) =
            foldl'
                consumeOne
                (qty, [], [], [])
                lots
        closeTargets =
            case closeTarget closeKey sym exitPrice exitTime (reverse consumedLots) of
                Nothing -> []
                Just target -> [target]
     in (reverse nextLots, closeTargets ++ openTargets)
  where
    consumeOne (remaining, kept, consumed, targets) lot
        | remaining <= qtyEps = (remaining, lot : kept, consumed, targets)
        | otherwise =
            let takeQty = min remaining (olRemainingQty lot)
                remaining' = remaining - takeQty
                leftover = olRemainingQty lot - takeQty
                targetLot = TargetLot (olDirection lot) (olEntryTime lot) (olEntryPrice lot) takeQty
             in if takeQty <= qtyEps
                    then (remaining', lot : kept, consumed, targets)
                    else
                        if leftover > qtyEps
                            then
                                let lot' = lot{olRemainingQty = leftover}
                                 in (remaining', lot' : kept, targetLot : consumed, targets)
                            else
                                let openTarget = completedOpenTarget lot exitPrice exitTime
                                 in (remaining', kept, targetLot : consumed, maybe targets (: targets) openTarget)

completedOpenTarget :: OpenLot -> Double -> Int64 -> Maybe PnlTarget
completedOpenTarget lot exitPrice exitTime =
    let lower = olEntryTime lot
        upper = closeWindowUpper lower exitTime
     in if upper < lower || not (positiveFinite (olOriginalQty lot))
            then Nothing
            else
                Just
                    PnlTarget
                        { ptTradeKey = olTradeKey lot
                        , ptSymbol = olSymbol lot
                        , ptDirection = olDirection lot
                        , ptLots = [TargetLot (olDirection lot) (olEntryTime lot) (olEntryPrice lot) (olOriginalQty lot)]
                        , ptLowerTime = lower
                        , ptUpperTime = upper
                        , ptActualExit = Just (exitTime, exitPrice)
                        }

closeTarget :: TradeKey -> String -> Double -> Int64 -> [TargetLot] -> Maybe PnlTarget
closeTarget closeKey sym exitPrice exitTime lots =
    case lots of
        [] -> Nothing
        firstLot : _ ->
            let lower = maximum (map tlEntryTime lots)
                upper = minimum (map (\lot -> closeWindowUpper (tlEntryTime lot) exitTime) lots)
                dir = tlDirection firstLot
             in if upper < lower
                    then Nothing
                    else
                        Just
                            PnlTarget
                                { ptTradeKey = closeKey
                                , ptSymbol = sym
                                , ptDirection = dir
                                , ptLots = lots
                                , ptLowerTime = lower
                                , ptUpperTime = upper
                                , ptActualExit = Just (exitTime, exitPrice)
                                }

scoreTarget :: M.Map String [Kline] -> PnlTarget -> Maybe BinanceTradeMaxPnl
scoreTarget klineMap target =
    let klines = M.findWithDefault [] (ptSymbol target) klineMap
        candidates =
            entryCandidates target
                ++ actualExitCandidates target
                ++ mapMaybe (klineCandidate target) klines
        scored = mapMaybe (scoreCandidate target) candidates
     in case scored of
            [] -> Nothing
            best : rest ->
                let (bestTime, bestPnl) = foldl' chooseBetterScore best rest
                 in Just (BinanceTradeMaxPnl bestPnl bestTime)

entryCandidates :: PnlTarget -> [(Int64, Double)]
entryCandidates target =
    [ (tlEntryTime lot, tlEntryPrice lot)
    | lot <- ptLots target
    , tlEntryTime lot >= ptLowerTime target
    , tlEntryTime lot <= ptUpperTime target
    , positiveFinite (tlEntryPrice lot)
    ]

actualExitCandidates :: PnlTarget -> [(Int64, Double)]
actualExitCandidates target =
    case ptActualExit target of
        Just (t, price)
            | t >= ptLowerTime target
                && t <= ptUpperTime target
                && positiveFinite price ->
                [(t, price)]
        _ -> []

klineCandidate :: PnlTarget -> Kline -> Maybe (Int64, Double)
klineCandidate target k =
    let t = kOpenTime k
        price =
            case ptDirection target of
                DirectionLong -> kHigh k
                DirectionShort -> kLow k
     in if t < ptLowerTime target || t > ptUpperTime target || not (positiveFinite price)
            then Nothing
            else Just (t, price)

scoreCandidate :: PnlTarget -> (Int64, Double) -> Maybe (Int64, Double)
scoreCandidate target (t, price) =
    let pnl = sum (map (lotPnl (ptDirection target) price) (ptLots target))
     in if isFinite pnl then Just (t, pnl) else Nothing

lotPnl :: TradeDirection -> Double -> TargetLot -> Double
lotPnl dir closePrice lot =
    case dir of
        DirectionLong -> (closePrice - tlEntryPrice lot) * tlQty lot
        DirectionShort -> (tlEntryPrice lot - closePrice) * tlQty lot

chooseBetterScore :: (Int64, Double) -> (Int64, Double) -> (Int64, Double)
chooseBetterScore best cur
    | snd cur > snd best = cur
    | snd cur < snd best = best
    | fst cur < fst best = cur
    | otherwise = best

chooseBetterResult :: BinanceTradeMaxPnl -> BinanceTradeMaxPnl -> BinanceTradeMaxPnl
chooseBetterResult cur prev =
    let (bestTime, bestPnl) = chooseBetterScore (btmpCloseTime prev, btmpPnl prev) (btmpCloseTime cur, btmpPnl cur)
     in BinanceTradeMaxPnl bestPnl bestTime

closeWindowUpper :: Int64 -> Int64 -> Int64
closeWindowUpper entryTime exitTime =
    let entry = toInteger entryTime
        exit = max entry (toInteger exitTime)
        upper = entry + 2 * (exit - entry)
     in if upper > toInteger (maxBound :: Int64)
            then maxBound
            else fromInteger upper

tradeSide :: BinanceTrade -> Maybe TradeSide
tradeSide trade =
    case fmap normalizeText (btSide trade) of
        Just "BUY" -> Just SideBuy
        Just "SELL" -> Just SideSell
        _ ->
            case btIsBuyer trade of
                Just True -> Just SideBuy
                Just False -> Just SideSell
                Nothing -> Nothing

positionSideMode :: Maybe String -> PositionSideMode
positionSideMode raw =
    case fmap normalizeText raw of
        Just "LONG" -> PositionLong
        Just "SHORT" -> PositionShort
        _ -> PositionBoth

tradeKey :: BinanceTrade -> TradeKey
tradeKey trade = (normalizeSymbolKey (btSymbol trade), btTradeId trade)

normalizeSymbolKey :: String -> String
normalizeSymbolKey = map toUpper . trim

normalizeText :: String -> String
normalizeText = map toUpper . trim

trim :: String -> String
trim = dropWhileEndAscii . dropWhile isAsciiSpace

dropWhileEndAscii :: String -> String
dropWhileEndAscii = reverse . dropWhile isAsciiSpace . reverse

isAsciiSpace :: Char -> Bool
isAsciiSpace c = c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'

positiveFinite :: Double -> Bool
positiveFinite x = x > 0 && isFinite x

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)
