module Trader.Predictors.CrossExchangeFeaturesV2 (
    CrossExchangeCloseV2 (..),
    CrossExchangeInputsV2,
    crossExchangeModelFeatureSchemaIdV2,
    crossExchangeModelFeatureSchemaVersionV2,
    crossExchangeModelFeatureNamesV2,
    crossExchangeModelFeatureSignatureV2,
    crossExchangeInputsV2,
    crossExchangeFeatureRowsV2,
) where

import Control.Monad (guard, join)
import Data.Char (isAlphaNum, isAscii, toUpper)
import Data.Int (Int64)
import Data.List (intercalate)
import Data.Maybe (isJust)
import qualified Data.Vector as V

import Trader.Coinbase (coinbaseProductFromBinance)
import Trader.Predictors.FeatureSchema (
    FeatureField (..),
    FeatureRequirement (OptionalFeature),
    FeatureRowV2,
    TimedFeatureValue (..),
    featureAvailabilitySchemaIdV2,
    mkFeatureRowV2,
 )

data CrossExchangeCloseV2 = CrossExchangeCloseV2
    { cec2BarOpenTimeMs :: !Int64
    , cec2EventTimeMs :: !Int64
    , cec2AvailabilityTimeMs :: !Int64
    , cec2Close :: !Double
    }
    deriving (Eq, Show)

data CrossExchangeInputsV2 = CrossExchangeInputsV2
    { cei2OpenTimesMs :: !(V.Vector Int64)
    , cei2DecisionTimesMs :: !(V.Vector Int64)
    , cei2IntervalMs :: !Int64
    , cei2BinanceCloses :: !(V.Vector CrossExchangeCloseV2)
    , cei2CoinbaseCloses :: !(V.Vector (Maybe CrossExchangeCloseV2))
    }
    deriving (Eq, Show)

crossExchangeModelFeatureSchemaIdV2 :: String
crossExchangeModelFeatureSchemaIdV2 = "coinbase_cross_exchange_model_features_v2"

crossExchangeModelFeatureSchemaVersionV2 :: Int
crossExchangeModelFeatureSchemaVersionV2 = 2

{- | Fixed value order. It deliberately matches the legacy Coinbase feature
block when its inputs are complete. Availability masks are appended only by
'Trader.Predictors.FeatureSchema.featureRowModelInputs'.
-}
crossExchangeModelFeatureNamesV2 :: [String]
crossExchangeModelFeatureNamesV2 =
    [ "coinbase_binance_basis.level"
    , "coinbase_binance_basis.delta"
    , "coinbase_binance_basis.zscore"
    , "coinbase.return_1"
    , "coinbase_binance.return_spread_1"
    ]

crossExchangeModelFeatureSignatureV2 :: String
crossExchangeModelFeatureSignatureV2 =
    crossExchangeModelFeatureSchemaIdV2
        ++ "|"
        ++ featureAvailabilitySchemaIdV2
        ++ "|"
        ++ intercalate "," (map (++ ":optional") crossExchangeModelFeatureNamesV2)

{- | Construct an exact-grid, timestamp-preserving same-asset close bundle.

The primary Binance series is required and must be causally usable at every bar
decision. Coinbase cells are optional: invalid values or incoherent timestamps
remain unavailable, but their vector position must still name the exact bar.
That structural rule forbids implicit forward-fill across a missing Coinbase
bucket. Source adapters must supply real event and availability witnesses; this
constructor does not derive them from exchange candle open times. Decisions are
explicit and must occur from the bucket end through the end of the following
bucket, so a source's real processing delay is preserved without admitting a
later completed bar.
-}
crossExchangeInputsV2 ::
    String ->
    String ->
    V.Vector Int64 ->
    V.Vector Int64 ->
    Int64 ->
    V.Vector CrossExchangeCloseV2 ->
    V.Vector (Maybe CrossExchangeCloseV2) ->
    Maybe CrossExchangeInputsV2
crossExchangeInputsV2 binanceSymbol coinbaseProduct openTimes decisionTimes intervalMs binanceCloses coinbaseCloses = do
    guard (validBinanceSymbol binanceSymbol)
    guard (coinbaseProductFromBinance binanceSymbol == Just coinbaseProduct)
    guard (contiguousGrid intervalMs (V.toList openTimes))
    let rowCount = V.length openTimes
    guard
        ( V.length decisionTimes == rowCount
            && V.length binanceCloses == rowCount
            && V.length coinbaseCloses == rowCount
        )
    guard (validDecisionSchedule intervalMs (V.toList openTimes) (V.toList decisionTimes))
    guard (and (V.toList (V.zipWith exactBar openTimes binanceCloses)))
    guard (and (V.toList (V.zipWith optionalExactBar openTimes coinbaseCloses)))
    guard
        ( and
            [ isJust (usableClose intervalMs openTime decision close)
            | (openTime, decision, close) <- zip3 (V.toList openTimes) (V.toList decisionTimes) (V.toList binanceCloses)
            ]
        )
    pure
        CrossExchangeInputsV2
            { cei2OpenTimesMs = openTimes
            , cei2DecisionTimesMs = decisionTimes
            , cei2IntervalMs = intervalMs
            , cei2BinanceCloses = binanceCloses
            , cei2CoinbaseCloses = coinbaseCloses
            }
  where
    exactBar openTime close = cec2BarOpenTimeMs close == openTime
    optionalExactBar openTime = maybe True ((== openTime) . cec2BarOpenTimeMs)

{- | Build the availability-aware form of the five legacy same-asset
cross-exchange features. All formulas match the legacy block on complete
inputs. The z-score is deliberately unavailable until its entire configured
window is present; incomplete windows and missing Coinbase bars are not
silently treated as evidence.
-}
crossExchangeFeatureRowsV2 :: Int -> CrossExchangeInputsV2 -> Maybe [FeatureRowV2]
crossExchangeFeatureRowsV2 shortBars inputs = do
    guard (shortBars > 0)
    traverse buildRow (zip [0 ..] (V.toList (cei2DecisionTimesMs inputs)))
  where
    buildRow (index, decision) =
        mkFeatureRowV2
            decision
            ( zipWith
                (`FeatureField` OptionalFeature)
                crossExchangeModelFeatureNamesV2
                [ basisAt inputs index decision
                , basisDeltaAt inputs index decision
                , basisZScoreAt shortBars inputs index decision
                , returnAt coinbaseAt inputs index decision
                , returnSpreadAt inputs index decision
                ]
            )

basisAt :: CrossExchangeInputsV2 -> Int -> Int64 -> Maybe TimedFeatureValue
basisAt inputs index decision = do
    binance <- binanceAt inputs index decision
    coinbase <- coinbaseAt inputs index decision
    let denominator = tfvValue binance
        value = (tfvValue coinbase - denominator) / denominator
    combineTimed value [binance, coinbase]

basisDeltaAt :: CrossExchangeInputsV2 -> Int -> Int64 -> Maybe TimedFeatureValue
basisDeltaAt inputs index decision = do
    guard (index > 0)
    previousDecision <- decisionAt inputs (index - 1)
    previous <- basisAt inputs (index - 1) previousDecision
    current <- basisAt inputs index decision
    combineTimed (tfvValue current - tfvValue previous) [previous, current]

basisZScoreAt :: Int -> CrossExchangeInputsV2 -> Int -> Int64 -> Maybe TimedFeatureValue
basisZScoreAt shortBars inputs index decision = do
    let start = index - shortBars + 1
    guard (start >= 0)
    window <-
        traverse
            ( \offset -> do
                historicalDecision <- decisionAt inputs offset
                basisAt inputs offset historicalDecision
            )
            [start .. index]
    current <- lastMaybe window
    let values = map tfvValue window
        (average, deviation) = meanStd values
        value = if deviation <= 1.0e-12 then 0 else (tfvValue current - average) / deviation
    combineTimed value window

returnAt :: (CrossExchangeInputsV2 -> Int -> Int64 -> Maybe TimedFeatureValue) -> CrossExchangeInputsV2 -> Int -> Int64 -> Maybe TimedFeatureValue
returnAt observationAt inputs index decision = do
    guard (index > 0)
    previousDecision <- decisionAt inputs (index - 1)
    previous <- observationAt inputs (index - 1) previousDecision
    current <- observationAt inputs index decision
    let value = tfvValue current / tfvValue previous - 1
    combineTimed value [previous, current]

returnSpreadAt :: CrossExchangeInputsV2 -> Int -> Int64 -> Maybe TimedFeatureValue
returnSpreadAt inputs index decision = do
    coinbaseReturn <- returnAt coinbaseAt inputs index decision
    binanceReturn <- returnAt binanceAt inputs index decision
    combineTimed (tfvValue coinbaseReturn - tfvValue binanceReturn) [coinbaseReturn, binanceReturn]

binanceAt :: CrossExchangeInputsV2 -> Int -> Int64 -> Maybe TimedFeatureValue
binanceAt inputs index decision = do
    openTime <- cei2OpenTimesMs inputs V.!? index
    close <- cei2BinanceCloses inputs V.!? index
    usableClose (cei2IntervalMs inputs) openTime decision close

coinbaseAt :: CrossExchangeInputsV2 -> Int -> Int64 -> Maybe TimedFeatureValue
coinbaseAt inputs index decision = do
    openTime <- cei2OpenTimesMs inputs V.!? index
    close <- join (cei2CoinbaseCloses inputs V.!? index)
    usableClose (cei2IntervalMs inputs) openTime decision close

decisionAt :: CrossExchangeInputsV2 -> Int -> Maybe Int64
decisionAt inputs index = cei2DecisionTimesMs inputs V.!? index

usableClose :: Int64 -> Int64 -> Int64 -> CrossExchangeCloseV2 -> Maybe TimedFeatureValue
usableClose intervalMs openTime decision close = do
    eventBoundary <- barEndTime intervalMs openTime
    let eventTime = cec2EventTimeMs close
        availabilityTime = cec2AvailabilityTimeMs close
        value = cec2Close close
    guard
        ( cec2BarOpenTimeMs close == openTime
            && eventTime == eventBoundary
            && eventTime <= availabilityTime
            && availabilityTime <= decision
            && value > 0
            && finite value
        )
    pure (timedClose close)

timedClose :: CrossExchangeCloseV2 -> TimedFeatureValue
timedClose close =
    TimedFeatureValue
        { tfvEventTimeMs = cec2EventTimeMs close
        , tfvAvailabilityTimeMs = cec2AvailabilityTimeMs close
        , tfvValue = cec2Close close
        }

combineTimed :: Double -> [TimedFeatureValue] -> Maybe TimedFeatureValue
combineTimed value observations = do
    guard (finite value)
    eventTime <- maximumMaybe (map tfvEventTimeMs observations)
    availabilityTime <- maximumMaybe (map tfvAvailabilityTimeMs observations)
    pure
        TimedFeatureValue
            { tfvEventTimeMs = eventTime
            , tfvAvailabilityTimeMs = availabilityTime
            , tfvValue = value
            }

meanStd :: [Double] -> (Double, Double)
meanStd values =
    case values of
        [] -> (0, 0)
        _ ->
            let count = length values
                average = sum values / fromIntegral count
                variance =
                    if count < 2
                        then 0
                        else
                            sum (map (\value -> (value - average) * (value - average)) values)
                                / fromIntegral (count - 1)
             in (average, sqrt (variance + 1.0e-12))

lastMaybe :: [a] -> Maybe a
lastMaybe [] = Nothing
lastMaybe values = Just (last values)

maximumMaybe :: (Ord a) => [a] -> Maybe a
maximumMaybe [] = Nothing
maximumMaybe values = Just (maximum values)

contiguousGrid :: Int64 -> [Int64] -> Bool
contiguousGrid intervalMs openTimes =
    intervalMs > 0
        && not (null openTimes)
        && all (>= 0) openTimes
        && and
            [ toInteger current == toInteger previous + toInteger intervalMs
            | (previous, current) <- zip openTimes (drop 1 openTimes)
            ]

barEndTime :: Int64 -> Int64 -> Maybe Int64
barEndTime intervalMs openTime = do
    guard (intervalMs > 0 && openTime >= 0)
    let candidate = toInteger openTime + toInteger intervalMs
    guard (candidate <= toInteger (maxBound :: Int64))
    pure (fromInteger candidate)

validDecisionSchedule :: Int64 -> [Int64] -> [Int64] -> Bool
validDecisionSchedule intervalMs openTimes decisionTimes =
    length openTimes == length decisionTimes
        && strictlyAscending decisionTimes
        && and
            [ validDecision openTime decision
            | (openTime, decision) <- zip openTimes decisionTimes
            ]
  where
    validDecision openTime decision =
        case barEndTime intervalMs openTime of
            Nothing -> False
            Just barEnd ->
                let latestDecision = toInteger barEnd + toInteger intervalMs - 1
                 in decision >= barEnd && toInteger decision <= latestDecision

strictlyAscending :: (Ord a) => [a] -> Bool
strictlyAscending values = and (zipWith (<) values (drop 1 values))

validBinanceSymbol :: String -> Bool
validBinanceSymbol symbol =
    not (null symbol)
        && symbol == map toUpper symbol
        && all (\character -> isAscii character && isAlphaNum character) symbol

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)
