module Trader.Predictors.MarketContextFeaturesV2 (
    MarketContextPeerReturnV2 (..),
    marketContextFactorSchemaIdV2,
    marketContextFactorSchemaVersionV2,
    marketContextFactorFeatureNamesV2,
    marketContextFactorSchemaSignatureV2,
    marketContextFactorRowV2,
    marketContextFactorRowsV2,
) where

import Control.Monad (guard)
import Data.Char (isAlphaNum, isAscii, toUpper)
import Data.Int (Int64)
import Data.List (foldl', intercalate)
import qualified Data.Vector as V

import Trader.PointInTimeUniverseV2 (
    PointInTimeUniverseSelectionV2,
    UniverseMemberV2 (..),
    pitSelectionAvailabilityTimeMsV2,
    pitSelectionDecisionTimeMsV2,
    pitSelectionEventTimeMsV2,
    pitSelectionMembersV2,
    pitSelectionQuoteV2,
 )
import Trader.Predictors.FeatureSchema (
    FeatureField (..),
    FeatureRequirement (OptionalFeature),
    FeatureRowV2,
    TimedFeatureValue (..),
    featureAvailabilitySchemaIdV2,
    mkFeatureRowV2,
 )

data MarketContextPeerReturnV2 = MarketContextPeerReturnV2
    { mcpr2Symbol :: !String
    , mcpr2BarOpenTimeMs :: !Int64
    , mcpr2EventTimeMs :: !Int64
    , mcpr2AvailabilityTimeMs :: !Int64
    , mcpr2SimpleReturn :: !Double
    }
    deriving (Eq, Show)

marketContextFactorSchemaIdV2 :: String
marketContextFactorSchemaIdV2 = "point_in_time_market_context_factor_v2"

marketContextFactorSchemaVersionV2 :: Int
marketContextFactorSchemaVersionV2 = 2

marketContextFactorFeatureNamesV2 :: [String]
marketContextFactorFeatureNamesV2 = ["market.return_1"]

marketContextFactorSchemaSignatureV2 :: String
marketContextFactorSchemaSignatureV2 =
    marketContextFactorSchemaIdV2
        ++ "|"
        ++ featureAvailabilitySchemaIdV2
        ++ "|"
        ++ intercalate "," (map (++ ":optional") marketContextFactorFeatureNamesV2)

{- | Build one point-in-time, quote-volume-weighted peer return. The selection
and the peer vector are positional: every present peer must identify the same
symbol as the corresponding ranked universe member and the exact requested
bar. The target is removed before the first @requiredPeerCount@ peers are
chosen, and their weights are renormalized after that removal.

Malformed scope, shape, alignment, or decision timing rejects the row. Missing,
late, non-finite, or otherwise unusable evidence for a chosen peer instead
creates an unavailable optional feature, so absence is never interpreted as a
directional zero. A caller that may select the target must request at least one
additional universe member upstream.
-}
marketContextFactorRowV2 ::
    String ->
    Int ->
    Int64 ->
    Int64 ->
    PointInTimeUniverseSelectionV2 ->
    V.Vector (Maybe MarketContextPeerReturnV2) ->
    Maybe FeatureRowV2
marketContextFactorRowV2 targetSymbol requiredPeerCount barOpenTime intervalMs selection peers = do
    guard (validSymbolForQuote (pitSelectionQuoteV2 selection) targetSymbol)
    guard (requiredPeerCount > 0)
    eventBoundary <- barEndTime intervalMs barOpenTime
    let decisionTime = pitSelectionDecisionTimeMsV2 selection
        members = pitSelectionMembersV2 selection
    guard (validDecision intervalMs eventBoundary decisionTime)
    guard (V.length peers == length members)
    guard (and (zipWith (structurallyAligned barOpenTime) members (V.toList peers)))
    let chosen = take requiredPeerCount (filter ((/= targetSymbol) . um2Symbol . fst) (zip members (V.toList peers)))
    guard (length chosen == requiredPeerCount)
    let observation = factorObservation eventBoundary decisionTime selection chosen
    mkFeatureRowV2
        decisionTime
        [FeatureField "market.return_1" OptionalFeature observation]

{- | Construct an exact contiguous series. Each row has its own universe
decision and peer witnesses; no terminal membership or weight vector is
back-applied to earlier bars.
-}
marketContextFactorRowsV2 ::
    String ->
    Int ->
    Int64 ->
    V.Vector Int64 ->
    V.Vector PointInTimeUniverseSelectionV2 ->
    V.Vector (V.Vector (Maybe MarketContextPeerReturnV2)) ->
    Maybe [FeatureRowV2]
marketContextFactorRowsV2 targetSymbol requiredPeerCount intervalMs openTimes selections peerRows = do
    guard (contiguousGrid intervalMs (V.toList openTimes))
    let rowCount = V.length openTimes
    guard (V.length selections == rowCount && V.length peerRows == rowCount)
    guard (strictlyAscending (map pitSelectionDecisionTimeMsV2 (V.toList selections)))
    traverse buildRow [0 .. rowCount - 1]
  where
    buildRow index = do
        openTime <- openTimes V.!? index
        selection <- selections V.!? index
        peers <- peerRows V.!? index
        marketContextFactorRowV2 targetSymbol requiredPeerCount openTime intervalMs selection peers

factorObservation ::
    Int64 ->
    Int64 ->
    PointInTimeUniverseSelectionV2 ->
    [(UniverseMemberV2, Maybe MarketContextPeerReturnV2)] ->
    Maybe TimedFeatureValue
factorObservation eventBoundary decisionTime selection chosen = do
    observations <- traverse (usablePeer eventBoundary decisionTime) chosen
    value <- stableWeightedMean [(um2QuoteVolume member, mcpr2SimpleReturn peer) | (member, peer) <- observations]
    guard (finite value)
    pure
        TimedFeatureValue
            { tfvEventTimeMs = maximum (pitSelectionEventTimeMsV2 selection : map (mcpr2EventTimeMs . snd) observations)
            , tfvAvailabilityTimeMs = maximum (pitSelectionAvailabilityTimeMsV2 selection : map (mcpr2AvailabilityTimeMs . snd) observations)
            , tfvValue = value
            }

usablePeer ::
    Int64 ->
    Int64 ->
    (UniverseMemberV2, Maybe MarketContextPeerReturnV2) ->
    Maybe (UniverseMemberV2, MarketContextPeerReturnV2)
usablePeer eventBoundary decisionTime (member, maybePeer) = do
    peer <- maybePeer
    let eventTime = mcpr2EventTimeMs peer
        availabilityTime = mcpr2AvailabilityTimeMs peer
        value = mcpr2SimpleReturn peer
    guard
        ( eventTime == eventBoundary
            && eventTime <= availabilityTime
            && availabilityTime <= decisionTime
            && value > -1
            && finite value
        )
    pure (member, peer)

structurallyAligned :: Int64 -> UniverseMemberV2 -> Maybe MarketContextPeerReturnV2 -> Bool
structurallyAligned _ _ Nothing = True
structurallyAligned barOpenTime member (Just peer) =
    mcpr2Symbol peer == um2Symbol member
        && mcpr2BarOpenTimeMs peer == barOpenTime

-- A positive-weight online mean avoids overflow from summing raw volumes or
-- same-sign high-magnitude weighted returns. Returns are bounded below by -1.
stableWeightedMean :: [(Double, Double)] -> Maybe Double
stableWeightedMean [] = Nothing
stableWeightedMean ((firstWeight, firstValue) : rest) = do
    guard (firstWeight > 0 && finite firstWeight && finite firstValue)
    (_, totalScaledWeight, average) <- foldl' step (Just (firstWeight, 1, firstValue)) rest
    guard (totalScaledWeight > 0 && finite totalScaledWeight && finite average)
    pure average
  where
    step Nothing _ = Nothing
    step (Just (weightScale, totalScaledWeight, average)) (weight, value) = do
        guard (weight > 0 && finite weight && finite value)
        let nextScale = max weightScale weight
            rescaledTotal = totalScaledWeight * (weightScale / nextScale)
            scaledWeight = weight / nextScale
            combined = rescaledTotal + scaledWeight
            nextAverage = average + (scaledWeight / combined) * (value - average)
        guard (combined > 0 && finite nextAverage)
        pure (nextScale, combined, nextAverage)

validDecision :: Int64 -> Int64 -> Int64 -> Bool
validDecision intervalMs eventBoundary decisionTime =
    let latestDecision = toInteger eventBoundary + toInteger intervalMs - 1
     in decisionTime >= eventBoundary && toInteger decisionTime <= latestDecision

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

strictlyAscending :: (Ord a) => [a] -> Bool
strictlyAscending values = and (zipWith (<) values (drop 1 values))

validSymbolForQuote :: String -> String -> Bool
validSymbolForQuote quote symbol =
    validIdentifier quote
        && validIdentifier symbol
        && length symbol > length quote
        && drop (length symbol - length quote) symbol == quote

validIdentifier :: String -> Bool
validIdentifier value =
    not (null value)
        && value == map toUpper value
        && all (\character -> isAscii character && isAlphaNum character) value

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)
