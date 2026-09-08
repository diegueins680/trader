module Trader.PointInTimeUniverseV2 (
    UniverseMemberV2 (..),
    PointInTimeUniverseSnapshotV2,
    PointInTimeUniverseSelectionV2,
    pointInTimeUniverseSchemaIdV2,
    pointInTimeUniverseSchemaVersionV2,
    pointInTimeUniverseSchemaSignatureV2,
    pointInTimeUniverseSnapshotV2,
    pointInTimeUniverseSelectionV2,
    pitSnapshotQuoteV2,
    pitSnapshotEventTimeMsV2,
    pitSnapshotAvailabilityTimeMsV2,
    pitSnapshotMembersV2,
    pitSelectionQuoteV2,
    pitSelectionEventTimeMsV2,
    pitSelectionAvailabilityTimeMsV2,
    pitSelectionDecisionTimeMsV2,
    pitSelectionMembersV2,
    pitSelectionWeightsV2,
) where

import Control.Monad (guard)
import Data.Char (isAlphaNum, isAscii, isAsciiLower)
import Data.Int (Int64)
import Data.List (isSuffixOf, nub, sortOn)
import Data.Ord (Down (..), comparing)

data UniverseMemberV2 = UniverseMemberV2
    { um2Symbol :: !String
    , um2QuoteVolume :: !Double
    , um2Eligible :: !Bool
    }
    deriving (Eq, Show)

data PointInTimeUniverseSnapshotV2 = PointInTimeUniverseSnapshotV2
    { pus2Quote :: !String
    , pus2EventTimeMs :: !Int64
    , pus2AvailabilityTimeMs :: !Int64
    , pus2Members :: ![UniverseMemberV2]
    }
    deriving (Eq, Show)

data PointInTimeUniverseSelectionV2 = PointInTimeUniverseSelectionV2
    { pui2Quote :: !String
    , pui2EventTimeMs :: !Int64
    , pui2AvailabilityTimeMs :: !Int64
    , pui2DecisionTimeMs :: !Int64
    , pui2Members :: ![UniverseMemberV2]
    }
    deriving (Eq, Show)

pointInTimeUniverseSchemaIdV2 :: String
pointInTimeUniverseSchemaIdV2 = "point_in_time_liquidity_universe_v2"

pointInTimeUniverseSchemaVersionV2 :: Int
pointInTimeUniverseSchemaVersionV2 = 2

pointInTimeUniverseSchemaSignatureV2 :: String
pointInTimeUniverseSchemaSignatureV2 =
    pointInTimeUniverseSchemaIdV2
        ++ "|quote,event_time_ms,availability_time_ms,decision_time_ms,symbol,quote_volume,eligible"

{- | Construct one coherent cross-sectional liquidity snapshot. Event time is
the end of the source measurement window; availability is the real first-seen
time of the complete snapshot. Every member must belong to the exact quote
scope. Eligibility is explicit source evidence rather than an inference from a
current symbol list or token-name heuristic.

This constructor validates structure, but it cannot prove that a caller
supplied the complete venue population. A source artifact must bind that
completeness before the snapshot is admissible for research.
-}
pointInTimeUniverseSnapshotV2 ::
    String ->
    Int64 ->
    Int64 ->
    [UniverseMemberV2] ->
    Maybe PointInTimeUniverseSnapshotV2
pointInTimeUniverseSnapshotV2 quote eventTime availabilityTime members = do
    guard (validIdentifier quote)
    guard (eventTime >= 0 && eventTime <= availabilityTime)
    guard (not (null members))
    guard (all (validMember quote) members)
    guard (allUnique (map um2Symbol members))
    pure
        PointInTimeUniverseSnapshotV2
            { pus2Quote = quote
            , pus2EventTimeMs = eventTime
            , pus2AvailabilityTimeMs = availabilityTime
            , pus2Members = sortOn um2Symbol members
            }

{- | Select the freshest economic snapshot that was actually available at the
decision. Freshness is measured from event time, not cache or download time.
For a revision of the same event, the latest revision available by the decision
wins. Appending a later snapshot or revision therefore cannot change an
earlier selection.

The selected population contains exactly @topN@ positive-volume eligible
members. Ties are deterministic by canonical symbol. Missing, stale,
incomplete, mixed-scope, or ambiguous snapshot evidence returns 'Nothing'.
-}
pointInTimeUniverseSelectionV2 ::
    String ->
    Int ->
    Int64 ->
    Int64 ->
    [PointInTimeUniverseSnapshotV2] ->
    Maybe PointInTimeUniverseSelectionV2
pointInTimeUniverseSelectionV2 quote topN maxEventAgeMs decisionTime snapshots = do
    guard (validIdentifier quote)
    guard (topN > 0 && maxEventAgeMs >= 0 && decisionTime >= 0)
    guard (not (null snapshots))
    guard (all ((== quote) . pus2Quote) snapshots)
    let usableSnapshots = filter usableSnapshot snapshots
    guard (allUnique (map snapshotIdentity usableSnapshots))
    snapshot <- maximumMaybeBy (comparing snapshotIdentity) usableSnapshots
    let ranked =
            take topN
                . sortOn (\member -> (Down (um2QuoteVolume member), um2Symbol member))
                . filter (\member -> um2Eligible member && um2QuoteVolume member > 0)
                $ pus2Members snapshot
    guard (length ranked == topN)
    pure
        PointInTimeUniverseSelectionV2
            { pui2Quote = quote
            , pui2EventTimeMs = pus2EventTimeMs snapshot
            , pui2AvailabilityTimeMs = pus2AvailabilityTimeMs snapshot
            , pui2DecisionTimeMs = decisionTime
            , pui2Members = ranked
            }
  where
    usableSnapshot snapshot =
        pus2AvailabilityTimeMs snapshot <= decisionTime
            && toInteger decisionTime - toInteger (pus2EventTimeMs snapshot)
                <= toInteger maxEventAgeMs

pitSnapshotQuoteV2 :: PointInTimeUniverseSnapshotV2 -> String
pitSnapshotQuoteV2 = pus2Quote

pitSnapshotEventTimeMsV2 :: PointInTimeUniverseSnapshotV2 -> Int64
pitSnapshotEventTimeMsV2 = pus2EventTimeMs

pitSnapshotAvailabilityTimeMsV2 :: PointInTimeUniverseSnapshotV2 -> Int64
pitSnapshotAvailabilityTimeMsV2 = pus2AvailabilityTimeMs

pitSnapshotMembersV2 :: PointInTimeUniverseSnapshotV2 -> [UniverseMemberV2]
pitSnapshotMembersV2 = pus2Members

pitSelectionQuoteV2 :: PointInTimeUniverseSelectionV2 -> String
pitSelectionQuoteV2 = pui2Quote

pitSelectionEventTimeMsV2 :: PointInTimeUniverseSelectionV2 -> Int64
pitSelectionEventTimeMsV2 = pui2EventTimeMs

pitSelectionAvailabilityTimeMsV2 :: PointInTimeUniverseSelectionV2 -> Int64
pitSelectionAvailabilityTimeMsV2 = pui2AvailabilityTimeMs

pitSelectionDecisionTimeMsV2 :: PointInTimeUniverseSelectionV2 -> Int64
pitSelectionDecisionTimeMsV2 = pui2DecisionTimeMs

pitSelectionMembersV2 :: PointInTimeUniverseSelectionV2 -> [UniverseMemberV2]
pitSelectionMembersV2 = pui2Members

-- | Normalize selected quote-volume weights without overflowing their raw sum.
pitSelectionWeightsV2 :: PointInTimeUniverseSelectionV2 -> [(String, Double)]
pitSelectionWeightsV2 selection =
    let members = pui2Members selection
        scale = maximum (map um2QuoteVolume members)
        scaled = [(um2Symbol member, um2QuoteVolume member / scale) | member <- members]
        denominator = sum (map snd scaled)
     in [(symbol, value / denominator) | (symbol, value) <- scaled]

snapshotIdentity :: PointInTimeUniverseSnapshotV2 -> (Int64, Int64)
snapshotIdentity snapshot = (pus2EventTimeMs snapshot, pus2AvailabilityTimeMs snapshot)

validMember :: String -> UniverseMemberV2 -> Bool
validMember quote member =
    validIdentifier (um2Symbol member)
        && quote `isSuffixOf` um2Symbol member
        && length (um2Symbol member) > length quote
        && finite (um2QuoteVolume member)
        && um2QuoteVolume member >= 0

validIdentifier :: String -> Bool
validIdentifier value =
    not (null value)
        && value == map toUpperAscii value
        && all (\character -> isAscii character && isAlphaNum character) value

toUpperAscii :: Char -> Char
toUpperAscii character =
    if isAsciiLower character
        then toEnum (fromEnum character - 32)
        else character

allUnique :: (Eq a) => [a] -> Bool
allUnique values = length values == length (nub values)

maximumMaybeBy :: (a -> a -> Ordering) -> [a] -> Maybe a
maximumMaybeBy _ [] = Nothing
maximumMaybeBy compareValues (first : rest) = Just (foldl choose first rest)
  where
    choose best candidate =
        case compareValues candidate best of
            GT -> candidate
            _ -> best

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)
