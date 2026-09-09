module Trader.Predictors.MissingnessAwareFeaturesV1 (
    MissingnessAwareFeaturePanelV1 (..),
    missingnessAwareFeaturePanelSchemaIdV1,
    missingnessAwareFeaturePanelSchemaVersionV1,
    missingnessAwareFeatureNamesV1,
    missingnessAwareFeatureSignatureV1,
    missingnessAwarePriceFeatureNamesV1,
    missingnessAwareLookbackBarsV1,
    missingnessAwareShortWindowBarsV1,
    missingnessAwareRegisteredSymbolsV1,
    missingnessAwareFeaturePanelV1,
) where

import Control.Monad (guard, join)
import Data.Int (Int64)
import Data.List (elemIndex, intercalate)
import Data.Maybe (isNothing)
import qualified Data.Vector as V

import Trader.Predictors.CrossExchangeFeaturesV2 (
    CrossExchangeInputsV2,
    crossExchangeFeatureRowsV2,
    crossExchangeGridV2,
    crossExchangeModelFeatureNamesV2,
    crossExchangeScopeV2,
 )
import Trader.Predictors.DerivativesFeaturesV2 (
    derivativesFeatureRowsV2,
    derivativesModelFeatureNamesV2,
 )
import Trader.Predictors.DerivativesPanelSchema (DerivativesPanelRowV2 (..))
import Trader.Predictors.FeatureSchema (
    FeatureField (..),
    FeatureRequirement (..),
    FeatureRowV2,
    TimedFeatureValue (..),
    featureAvailabilitySchemaIdV2,
    frv2AvailabilityTimesMs,
    frv2Available,
    frv2DecisionTimeMs,
    frv2EventTimesMs,
    frv2Names,
    frv2Required,
    frv2SchemaId,
    frv2Values,
    mkFeatureRowV2,
 )
import Trader.Predictors.OhlcvInputsV2 (
    CompleteOhlcvInputsV2,
    completeOhlcvGridV2,
    completeOhlcvRowsV2,
    completeOhlcvScopeV2,
 )

{- | A development-only, scope-bound feature panel. It is not a fitted model,
predictor, trading signal, or production artifact.
-}
data MissingnessAwareFeaturePanelV1 = MissingnessAwareFeaturePanelV1
    { mafp1Scope :: !String
    , mafp1IntervalMs :: !Int64
    , mafp1OpenTimesMs :: !(V.Vector Int64)
    , mafp1Rows :: ![FeatureRowV2]
    }
    deriving (Eq, Show)

missingnessAwareFeaturePanelSchemaIdV1 :: String
missingnessAwareFeaturePanelSchemaIdV1 = "missingness_aware_calibrated_shallow_feature_panel_v1"

missingnessAwareFeaturePanelSchemaVersionV1 :: Int
missingnessAwareFeaturePanelSchemaVersionV1 = 1

missingnessAwareLookbackBarsV1 :: Int
missingnessAwareLookbackBarsV1 = 24

missingnessAwareShortWindowBarsV1 :: Int
missingnessAwareShortWindowBarsV1 = 6

missingnessAwareRegisteredSymbolsV1 :: [String]
missingnessAwareRegisteredSymbolsV1 =
    [ "BTCUSDT"
    , "ETHUSDT"
    , "SOLUSDT"
    , "BNBUSDT"
    , "XRPUSDT"
    , "AVAXUSDT"
    , "UNIUSDT"
    , "SUIUSDT"
    , "ETCUSDT"
    , "ADAUSDT"
    ]

-- | Exact required price-feature order frozen before prospective collection.
missingnessAwarePriceFeatureNamesV1 :: [String]
missingnessAwarePriceFeatureNamesV1 =
    [ "price.return_1"
    , "price.return_3"
    , "price.return_6"
    , "price.return_24"
    , "price.mean_return_6"
    , "price.return_volatility_6"
    , "price.mean_return_24"
    , "price.return_volatility_24"
    , "price.return_spread_6_24"
    , "price.mean_reversion_1_6"
    , "price.volatility_ratio_6_24"
    , "price.trend_slope_6_24"
    ]

missingnessAwareFeatureNamesV1 :: [String]
missingnessAwareFeatureNamesV1 =
    missingnessAwarePriceFeatureNamesV1
        ++ derivativesModelFeatureNamesV2
        ++ crossExchangeModelFeatureNamesV2

missingnessAwareFeatureSignatureV1 :: String
missingnessAwareFeatureSignatureV1 =
    missingnessAwareFeaturePanelSchemaIdV1
        ++ "|"
        ++ featureAvailabilitySchemaIdV2
        ++ "|"
        ++ intercalate
            ","
            ( map (++ ":required") missingnessAwarePriceFeatureNamesV1
                ++ map (++ ":optional") (derivativesModelFeatureNamesV2 ++ crossExchangeModelFeatureNamesV2)
            )

registeredDatasetStartMsV1 :: Int64
registeredDatasetStartMsV1 = 1800489600000

registeredDevelopmentEndMsV1 :: Int64
registeredDevelopmentEndMsV1 = 1821481200000

registeredIntervalsMsV1 :: [Int64]
registeredIntervalsMsV1 = [3600000, 14400000, 28800000]

{- | Compose the exact price, derivatives, and same-symbol spot-context feature
order for the preregistered shallow challenger. Complete OHLCV is required.
Derivatives and cross-exchange panels are optional as whole inputs and retain
one observed mask per field. A present panel must carry the exact target scope,
bar grid, and a source decision no later than the primary decision.

Only genuinely future, pre-holdout development rows are accepted. The first 24
bars are retained solely as causal lookback and do not produce model rows.
Missing optional values remain absent here; fold-local training code must learn
imputation values and may not call 'featureRowModelInputs' as an imputation
shortcut.
-}
missingnessAwareFeaturePanelV1 ::
    CompleteOhlcvInputsV2 ->
    Maybe [DerivativesPanelRowV2] ->
    Maybe CrossExchangeInputsV2 ->
    Maybe MissingnessAwareFeaturePanelV1
missingnessAwareFeaturePanelV1 ohlcvInputs maybeDerivatives maybeCrossExchange = do
    let scope = completeOhlcvScopeV2 ohlcvInputs
        (openTimes, decisionTimes, _, intervalMs) = completeOhlcvGridV2 ohlcvInputs
        ohlcvRows = completeOhlcvRowsV2 ohlcvInputs
        rowCount = V.length openTimes
    guard (scope `elem` missingnessAwareRegisteredSymbolsV1)
    guard (intervalMs `elem` registeredIntervalsMsV1)
    guard (rowCount == V.length decisionTimes && rowCount == length ohlcvRows)
    guard (rowCount > missingnessAwareLookbackBarsV1)
    eventTimes <- traverse (fmap tfvEventTimeMs . marketCloseWitness) ohlcvRows
    guard (validRegisteredEventTimes intervalMs eventTimes)
    priceRows <-
        traverse
            (priceFeatureRowV1 ohlcvRows)
            [missingnessAwareLookbackBarsV1 .. rowCount - 1]
    derivativeRows <- sourceDerivativeRows scope intervalMs openTimes decisionTimes maybeDerivatives
    crossExchangeRows <- sourceCrossExchangeRows scope intervalMs openTimes decisionTimes maybeCrossExchange
    combinedRows <-
        sequence
            [ combineRows primaryDecision priceRow derivativeRow crossExchangeRow
            | (primaryDecision, priceRow, derivativeRow, crossExchangeRow) <-
                zip4
                    (drop missingnessAwareLookbackBarsV1 (V.toList decisionTimes))
                    priceRows
                    (drop missingnessAwareLookbackBarsV1 derivativeRows)
                    (drop missingnessAwareLookbackBarsV1 crossExchangeRows)
            ]
    guard (length combinedRows == rowCount - missingnessAwareLookbackBarsV1)
    pure
        MissingnessAwareFeaturePanelV1
            { mafp1Scope = scope
            , mafp1IntervalMs = intervalMs
            , mafp1OpenTimesMs = V.drop missingnessAwareLookbackBarsV1 openTimes
            , mafp1Rows = combinedRows
            }

priceFeatureRowV1 :: [FeatureRowV2] -> Int -> Maybe FeatureRowV2
priceFeatureRowV1 rows index = do
    guard (index >= missingnessAwareLookbackBarsV1)
    window <- traverse (atMay rows) [index - missingnessAwareLookbackBarsV1 .. index]
    witnesses <- traverse marketCloseWitness window
    let closes = map tfvValue witnesses
        oneBarReturns = zipWith simpleReturn closes (drop 1 closes)
    returns <- sequence oneBarReturns
    return1 <- totalReturn closes 1
    return3 <- totalReturn closes 3
    return6 <- totalReturn closes missingnessAwareShortWindowBarsV1
    return24 <- totalReturn closes missingnessAwareLookbackBarsV1
    let shortReturns = drop (length returns - missingnessAwareShortWindowBarsV1) returns
        (mean6, volatility6) = meanVolatility shortReturns
        (mean24, volatility24) = meanVolatility returns
        returnSpread = return6 - return24
        meanReversion = return1 - mean6
        volatilityRatio =
            if volatility24 <= 1.0e-12
                then 0
                else volatility6 / volatility24
        trendSlope = mean6 - mean24
        values =
            [ return1
            , return3
            , return6
            , return24
            , mean6
            , volatility6
            , mean24
            , volatility24
            , returnSpread
            , meanReversion
            , volatilityRatio
            , trendSlope
            ]
        eventTime = maximum (map tfvEventTimeMs witnesses)
        availabilityTime = maximum (map tfvAvailabilityTimeMs witnesses)
    currentRow <- atMay rows index
    let decisionTime = frv2DecisionTimeMs currentRow
    guard (all finite values && availabilityTime <= decisionTime)
    mkFeatureRowV2
        decisionTime
        [ FeatureField
            name
            RequiredFeature
            (Just (TimedFeatureValue eventTime availabilityTime value))
        | (name, value) <- zip missingnessAwarePriceFeatureNamesV1 values
        ]

sourceDerivativeRows ::
    String ->
    Int64 ->
    V.Vector Int64 ->
    V.Vector Int64 ->
    Maybe [DerivativesPanelRowV2] ->
    Maybe [Maybe FeatureRowV2]
sourceDerivativeRows _ _ openTimes _ Nothing =
    pure (replicate (V.length openTimes) Nothing)
sourceDerivativeRows scope intervalMs openTimes decisionTimes (Just sourceRows) = do
    guard (length sourceRows == V.length openTimes)
    guard (map dpr2OpenTimeMs sourceRows == V.toList openTimes)
    guard (all ((== scope) . dpr2Symbol) sourceRows)
    guard (and (zipWith noLaterDecision (map dpr2DecisionTimeMs sourceRows) (V.toList decisionTimes)))
    rows <- derivativesFeatureRowsV2 intervalMs sourceRows
    guard (length rows == V.length openTimes)
    pure (map Just rows)

sourceCrossExchangeRows ::
    String ->
    Int64 ->
    V.Vector Int64 ->
    V.Vector Int64 ->
    Maybe CrossExchangeInputsV2 ->
    Maybe [Maybe FeatureRowV2]
sourceCrossExchangeRows _ _ openTimes _ Nothing =
    pure (replicate (V.length openTimes) Nothing)
sourceCrossExchangeRows scope intervalMs openTimes decisionTimes (Just source) = do
    let (binanceSymbol, _) = crossExchangeScopeV2 source
        (sourceOpenTimes, sourceDecisionTimes, sourceIntervalMs) = crossExchangeGridV2 source
    guard (binanceSymbol == scope)
    guard (sourceIntervalMs == intervalMs)
    guard (sourceOpenTimes == openTimes)
    guard (V.length sourceDecisionTimes == V.length decisionTimes)
    guard (and (V.toList (V.zipWith noLaterDecision sourceDecisionTimes decisionTimes)))
    rows <- crossExchangeFeatureRowsV2 missingnessAwareShortWindowBarsV1 source
    guard (length rows == V.length openTimes)
    pure (map Just rows)

combineRows ::
    Int64 ->
    FeatureRowV2 ->
    Maybe FeatureRowV2 ->
    Maybe FeatureRowV2 ->
    Maybe FeatureRowV2
combineRows decisionTime priceRow derivativeRow crossExchangeRow = do
    priceFields <- fieldsFromRow decisionTime missingnessAwarePriceFeatureNamesV1 RequiredFeature priceRow
    derivativeFields <- optionalFields decisionTime derivativesModelFeatureNamesV2 derivativeRow
    crossExchangeFields <- optionalFields decisionTime crossExchangeModelFeatureNamesV2 crossExchangeRow
    mkFeatureRowV2 decisionTime (priceFields ++ derivativeFields ++ crossExchangeFields)

optionalFields :: Int64 -> [String] -> Maybe FeatureRowV2 -> Maybe [FeatureField]
optionalFields _ names Nothing =
    pure [FeatureField name OptionalFeature Nothing | name <- names]
optionalFields decisionTime names (Just row) =
    fieldsFromRow decisionTime names OptionalFeature row

fieldsFromRow :: Int64 -> [String] -> FeatureRequirement -> FeatureRowV2 -> Maybe [FeatureField]
fieldsFromRow decisionTime expectedNames requirement row = do
    let names = frv2Names row
        values = frv2Values row
        available = frv2Available row
        required = frv2Required row
        eventTimes = frv2EventTimesMs row
        availabilityTimes = frv2AvailabilityTimesMs row
        width = length expectedNames
    guard (frv2SchemaId row == featureAvailabilitySchemaIdV2)
    guard (frv2DecisionTimeMs row <= decisionTime)
    guard (names == expectedNames)
    guard (required == replicate width (requirement == RequiredFeature))
    guard (length values == width)
    guard (length available == width)
    guard (length eventTimes == width)
    guard (length availabilityTimes == width)
    traverse
        (fieldFromCell decisionTime requirement)
        (zip5 names values available eventTimes availabilityTimes)

fieldFromCell ::
    Int64 ->
    FeatureRequirement ->
    (String, Double, Bool, Maybe Int64, Maybe Int64) ->
    Maybe FeatureField
fieldFromCell decisionTime requirement (name, value, available, eventTime, availabilityTime)
    | available = do
        event <- eventTime
        observedAt <- availabilityTime
        guard (finite value && event >= 0 && event <= observedAt && observedAt <= decisionTime)
        pure (FeatureField name requirement (Just (TimedFeatureValue event observedAt value)))
    | otherwise = do
        guard (requirement == OptionalFeature)
        guard (value == 0 && isNothing eventTime && isNothing availabilityTime)
        pure (FeatureField name requirement Nothing)

marketCloseWitness :: FeatureRowV2 -> Maybe TimedFeatureValue
marketCloseWitness row = do
    guard (frv2SchemaId row == featureAvailabilitySchemaIdV2)
    index <- elemIndex "market.close" (frv2Names row)
    value <- atMay (frv2Values row) index
    available <- atMay (frv2Available row) index
    required <- atMay (frv2Required row) index
    eventTime <- join (atMay (frv2EventTimesMs row) index)
    availabilityTime <- join (atMay (frv2AvailabilityTimesMs row) index)
    guard (available && required)
    guard
        ( value > 0
            && finite value
            && eventTime >= 0
            && eventTime <= availabilityTime
            && availabilityTime <= frv2DecisionTimeMs row
        )
    pure (TimedFeatureValue eventTime availabilityTime value)

validRegisteredEventTimes :: Int64 -> [Int64] -> Bool
validRegisteredEventTimes intervalMs eventTimes =
    not (null eventTimes)
        && all
            ( \eventTime ->
                eventTime >= registeredDatasetStartMsV1
                    && eventTime <= registeredDevelopmentEndMsV1
                    && gridAligned registeredDatasetStartMsV1 intervalMs eventTime
            )
            eventTimes

gridAligned :: Int64 -> Int64 -> Int64 -> Bool
gridAligned anchor intervalMs value =
    intervalMs > 0
        && value >= anchor
        && (toInteger value - toInteger anchor) `mod` toInteger intervalMs == 0

noLaterDecision :: Int64 -> Int64 -> Bool
noLaterDecision sourceDecision primaryDecision =
    sourceDecision >= 0 && sourceDecision <= primaryDecision

simpleReturn :: Double -> Double -> Maybe Double
simpleReturn old new = do
    guard (old > 0 && new > 0 && finite old && finite new)
    let value = new / old - 1
    guard (value > -1 && finite value)
    pure value

totalReturn :: [Double] -> Int -> Maybe Double
totalReturn closes lag = do
    guard (lag > 0 && length closes > lag)
    old <- atMay closes (length closes - lag - 1)
    new <- atMay closes (length closes - 1)
    simpleReturn old new

meanVolatility :: [Double] -> (Double, Double)
meanVolatility values =
    let count = fromIntegral (length values)
        average = sum values / count
        variance = sum [(value - average) * (value - average) | value <- values] / count
     in (average, sqrt (max 0 variance))

atMay :: [a] -> Int -> Maybe a
atMay values index
    | index < 0 = Nothing
    | otherwise =
        case drop index values of
            value : _ -> Just value
            [] -> Nothing

zip4 :: [a] -> [b] -> [c] -> [d] -> [(a, b, c, d)]
zip4 (a : as) (b : bs) (c : cs) (d : ds) = (a, b, c, d) : zip4 as bs cs ds
zip4 _ _ _ _ = []

zip5 :: [a] -> [b] -> [c] -> [d] -> [e] -> [(a, b, c, d, e)]
zip5 (a : as) (b : bs) (c : cs) (d : ds) (e : es) = (a, b, c, d, e) : zip5 as bs cs ds es
zip5 _ _ _ _ _ = []

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)
