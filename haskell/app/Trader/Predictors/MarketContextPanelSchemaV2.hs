{-# LANGUAGE OverloadedStrings #-}

module Trader.Predictors.MarketContextPanelSchemaV2 (
    MarketContextPanelMemberV2,
    MarketContextPanelSnapshotV2,
    mcpm2Symbol,
    mcpm2QuoteVolume,
    mcpm2Eligible,
    mcpm2TickerOpenTimeMs,
    mcpm2TickerCloseTimeMs,
    mcpm2PeerReturn,
    mcps2SourceManifestSha256,
    mcps2Quote,
    mcps2IntervalMs,
    mcps2BarOpenTimeMs,
    mcps2DecisionTimeMs,
    mcps2UniverseEventTimeMs,
    mcps2UniverseAvailabilityTimeMs,
    mcps2Members,
    marketContextPanelSchemaIdV2,
    marketContextPanelSchemaVersionV2,
    marketContextPanelSourceIdV2,
    marketContextPanelColumnsV2,
    decodeMarketContextPanelV2,
    marketContextPanelUniverseSnapshotV2,
    marketContextPanelSelectionInputsV2,
) where

import Control.Monad (guard, unless, when)
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isAlphaNum, isAscii, isAsciiLower, isDigit, isHexDigit)
import qualified Data.Csv as Csv
import qualified Data.HashMap.Strict as HM
import Data.Int (Int64)
import Data.List (groupBy, nub)
import qualified Data.Map.Strict as Map
import qualified Data.Vector as V
import Text.Read (readMaybe)

import Trader.PointInTimeUniverseV2 (
    PointInTimeUniverseSelectionV2,
    PointInTimeUniverseSnapshotV2,
    UniverseMemberV2 (..),
    pitSelectionMembersV2,
    pointInTimeUniverseSelectionV2,
    pointInTimeUniverseSnapshotV2,
 )
import Trader.Predictors.MarketContextFeaturesV2 (MarketContextPeerReturnV2 (..))
import Trader.Text (trim)

data MarketContextPanelMemberV2 = MarketContextPanelMemberV2
    { mcpm2Symbol :: !String
    , mcpm2QuoteVolume :: !Double
    , mcpm2Eligible :: !Bool
    , mcpm2TickerOpenTimeMs :: !Int64
    , mcpm2TickerCloseTimeMs :: !Int64
    , mcpm2PeerReturn :: !(Maybe MarketContextPeerReturnV2)
    }
    deriving (Eq, Show)

data MarketContextPanelSnapshotV2 = MarketContextPanelSnapshotV2
    { mcps2SourceManifestSha256 :: !String
    , mcps2Quote :: !String
    , mcps2IntervalMs :: !Int64
    , mcps2BarOpenTimeMs :: !Int64
    , mcps2DecisionTimeMs :: !Int64
    , mcps2UniverseEventTimeMs :: !Int64
    , mcps2UniverseAvailabilityTimeMs :: !Int64
    , mcps2Members :: ![MarketContextPanelMemberV2]
    }
    deriving (Eq, Show)

marketContextPanelSchemaIdV2 :: String
marketContextPanelSchemaIdV2 = "binance_usdm_market_context_panel_v2"

marketContextPanelSchemaVersionV2 :: Int
marketContextPanelSchemaVersionV2 = 2

marketContextPanelSourceIdV2 :: String
marketContextPanelSourceIdV2 = "binance-usdm-public-market-data"

marketContextPanelColumnsV2 :: [String]
marketContextPanelColumnsV2 =
    [ "schemaId"
    , "sourceId"
    , "sourceManifestSha256"
    , "quote"
    , "intervalMs"
    , "barOpenTime"
    , "decisionTime"
    , "universeEventTime"
    , "universeAvailabilityTime"
    , "populationCount"
    , "memberOrdinal"
    , "symbol"
    , "quoteVolume"
    , "eligible"
    , "tickerOpenTime"
    , "tickerCloseTime"
    , "peerObserved"
    , "peerEventTime"
    , "peerAvailabilityTime"
    , "peerSimpleReturn"
    ]

{- | Decode a dedicated prospective market-context panel. Each snapshot binds
the exact derived population to a frozen source-manifest digest, preserves the
rolling-volume window and first-seen clocks, and carries an optional exact-bar
peer return for every population member.

This decoder checks the declared digest syntax and all derived row semantics;
it cannot prove that the referenced manifest exists, matches its digest, or
contains complete raw exchange-info, ticker, and kline responses. Research
admission must verify those raw bytes independently before calling this
boundary. Missing peer evidence stays 'Nothing' and is never made into a zero.
-}
decodeMarketContextPanelV2 :: BL.ByteString -> Either String [MarketContextPanelSnapshotV2]
decodeMarketContextPanelV2 bytes = do
    (header, records) <-
        Csv.decodeByName bytes :: Either String (Csv.Header, V.Vector Csv.NamedRecord)
    unless (V.toList header == map BS.pack marketContextPanelColumnsV2) $
        Left (marketContextPanelSchemaIdV2 ++ " header is incompatible")
    rows <- traverse decodeRow (V.toList records)
    when (null rows) $
        Left (marketContextPanelSchemaIdV2 ++ " has no rows")
    snapshots <- traverse decodeSnapshot (groupBy sameSnapshot rows)
    validatePanel snapshots
    pure snapshots

marketContextPanelUniverseSnapshotV2 :: MarketContextPanelSnapshotV2 -> Maybe PointInTimeUniverseSnapshotV2
marketContextPanelUniverseSnapshotV2 snapshot =
    pointInTimeUniverseSnapshotV2
        (mcps2Quote snapshot)
        (mcps2UniverseEventTimeMs snapshot)
        (mcps2UniverseAvailabilityTimeMs snapshot)
        [ UniverseMemberV2
            { um2Symbol = mcpm2Symbol member
            , um2QuoteVolume = mcpm2QuoteVolume member
            , um2Eligible = mcpm2Eligible member
            }
        | member <- mcps2Members snapshot
        ]

{- | Apply the canonical v2 liquidity ranker, then reorder optional peer cells
to exactly match its selected members. This function does not define a second
ranking rule or verify the referenced source manifest; research callers must
complete that external verification first.
-}
marketContextPanelSelectionInputsV2 ::
    Int ->
    Int64 ->
    MarketContextPanelSnapshotV2 ->
    Maybe (PointInTimeUniverseSelectionV2, V.Vector (Maybe MarketContextPeerReturnV2))
marketContextPanelSelectionInputsV2 topN maxEventAgeMs panel = do
    snapshot <- marketContextPanelUniverseSnapshotV2 panel
    selection <-
        pointInTimeUniverseSelectionV2
            (mcps2Quote panel)
            topN
            maxEventAgeMs
            (mcps2DecisionTimeMs panel)
            [snapshot]
    let peerBySymbol =
            Map.fromList
                [(mcpm2Symbol member, mcpm2PeerReturn member) | member <- mcps2Members panel]
    peers <- traverse (flip Map.lookup peerBySymbol . um2Symbol) (pitSelectionMembersV2 selection)
    pure (selection, V.fromList peers)

data DecodedRow = DecodedRow
    { drSourceManifestSha256 :: !String
    , drQuote :: !String
    , drIntervalMs :: !Int64
    , drBarOpenTimeMs :: !Int64
    , drDecisionTimeMs :: !Int64
    , drUniverseEventTimeMs :: !Int64
    , drUniverseAvailabilityTimeMs :: !Int64
    , drPopulationCount :: !Int
    , drMemberOrdinal :: !Int
    , drMember :: !MarketContextPanelMemberV2
    }
    deriving (Eq, Show)

decodeRow :: Csv.NamedRecord -> Either String DecodedRow
decodeRow record = do
    schemaId <- textCell "schemaId" record
    unless (schemaId == marketContextPanelSchemaIdV2) $
        Left (marketContextPanelSchemaIdV2 ++ " row has an incompatible schema id")
    sourceId <- textCell "sourceId" record
    unless (sourceId == marketContextPanelSourceIdV2) $
        Left (marketContextPanelSchemaIdV2 ++ " row has an incompatible source id")
    sourceManifestSha256 <- textCell "sourceManifestSha256" record
    unless (validSha256 sourceManifestSha256) $
        Left (marketContextPanelSchemaIdV2 ++ " source manifest digest is invalid")
    quote <- textCell "quote" record
    unless (validIdentifier quote) $
        Left (marketContextPanelSchemaIdV2 ++ " quote is invalid")
    intervalMs <- timestampCell "intervalMs" record
    unless (intervalMs > 0) $
        Left (marketContextPanelSchemaIdV2 ++ " interval is invalid")
    barOpenTime <- timestampCell "barOpenTime" record
    decisionTime <- timestampCell "decisionTime" record
    universeEventTime <- timestampCell "universeEventTime" record
    universeAvailabilityTime <- timestampCell "universeAvailabilityTime" record
    populationCount <- positiveIntCell "populationCount" record
    memberOrdinal <- nonNegativeIntCell "memberOrdinal" record
    symbol <- textCell "symbol" record
    unless (validSymbolForQuote quote symbol) $
        Left (marketContextPanelSchemaIdV2 ++ " symbol is outside the quote scope")
    quoteVolume <- finiteCell "quoteVolume" record
    unless (quoteVolume >= 0) $
        Left (marketContextPanelSchemaIdV2 ++ " quote volume is negative")
    eligible <- maskCell "eligible" record
    tickerOpenTime <- timestampCell "tickerOpenTime" record
    tickerCloseTime <- timestampCell "tickerCloseTime" record
    peer <- decodePeer record symbol barOpenTime
    pure
        DecodedRow
            { drSourceManifestSha256 = sourceManifestSha256
            , drQuote = quote
            , drIntervalMs = intervalMs
            , drBarOpenTimeMs = barOpenTime
            , drDecisionTimeMs = decisionTime
            , drUniverseEventTimeMs = universeEventTime
            , drUniverseAvailabilityTimeMs = universeAvailabilityTime
            , drPopulationCount = populationCount
            , drMemberOrdinal = memberOrdinal
            , drMember =
                MarketContextPanelMemberV2
                    { mcpm2Symbol = symbol
                    , mcpm2QuoteVolume = quoteVolume
                    , mcpm2Eligible = eligible
                    , mcpm2TickerOpenTimeMs = tickerOpenTime
                    , mcpm2TickerCloseTimeMs = tickerCloseTime
                    , mcpm2PeerReturn = peer
                    }
            }

decodePeer :: Csv.NamedRecord -> String -> Int64 -> Either String (Maybe MarketContextPeerReturnV2)
decodePeer record symbol barOpenTime = do
    observed <- maskCell "peerObserved" record
    rawEvent <- requiredCell "peerEventTime" record
    rawAvailability <- requiredCell "peerAvailabilityTime" record
    rawReturn <- requiredCell "peerSimpleReturn" record
    if observed
        then do
            eventTime <- parseTimestamp "peerEventTime" rawEvent
            availabilityTime <- parseTimestamp "peerAvailabilityTime" rawAvailability
            simpleReturn <- parseFinite "peerSimpleReturn" rawReturn
            pure
                ( Just
                    MarketContextPeerReturnV2
                        { mcpr2Symbol = symbol
                        , mcpr2BarOpenTimeMs = barOpenTime
                        , mcpr2EventTimeMs = eventTime
                        , mcpr2AvailabilityTimeMs = availabilityTime
                        , mcpr2SimpleReturn = simpleReturn
                        }
                )
        else do
            unless (all blank [rawEvent, rawAvailability, rawReturn]) $
                Left (marketContextPanelSchemaIdV2 ++ " unavailable peer cell is not blank")
            pure Nothing

decodeSnapshot :: [DecodedRow] -> Either String MarketContextPanelSnapshotV2
decodeSnapshot [] = Left (marketContextPanelSchemaIdV2 ++ " internal empty snapshot")
decodeSnapshot rows@(first : _) = do
    let members = map drMember rows
        populationCount = drPopulationCount first
        eventTime = drUniverseEventTimeMs first
        availabilityTime = drUniverseAvailabilityTimeMs first
        decisionTime = drDecisionTimeMs first
        intervalMs = drIntervalMs first
        barOpenTime = drBarOpenTimeMs first
    unless (length rows == populationCount) $
        Left (marketContextPanelSchemaIdV2 ++ " population count is incomplete")
    unless (map drMemberOrdinal rows == [0 .. populationCount - 1]) $
        Left (marketContextPanelSchemaIdV2 ++ " member ordinals are incomplete or unordered")
    unless (allUnique (map mcpm2Symbol members)) $
        Left (marketContextPanelSchemaIdV2 ++ " snapshot contains duplicate symbols")
    barEnd <- checkedAdd barOpenTime intervalMs
    nextBarEnd <- checkedAdd barEnd intervalMs
    unless (decisionTime >= barEnd && decisionTime < nextBarEnd) $
        Left (marketContextPanelSchemaIdV2 ++ " decision is outside the bar window")
    unless (eventTime == maximum (map mcpm2TickerCloseTimeMs members)) $
        Left (marketContextPanelSchemaIdV2 ++ " universe event does not match the latest ticker close")
    unless (eventTime <= availabilityTime && availabilityTime <= decisionTime) $
        Left (marketContextPanelSchemaIdV2 ++ " universe timestamps are not causal")
    unless (all (validMemberTimes availabilityTime) members) $
        Left (marketContextPanelSchemaIdV2 ++ " ticker timestamps are not causal")
    unless (all (validPeer barEnd decisionTime) members) $
        Left (marketContextPanelSchemaIdV2 ++ " peer evidence is not causal or finite")
    pure
        MarketContextPanelSnapshotV2
            { mcps2SourceManifestSha256 = drSourceManifestSha256 first
            , mcps2Quote = drQuote first
            , mcps2IntervalMs = intervalMs
            , mcps2BarOpenTimeMs = barOpenTime
            , mcps2DecisionTimeMs = decisionTime
            , mcps2UniverseEventTimeMs = eventTime
            , mcps2UniverseAvailabilityTimeMs = availabilityTime
            , mcps2Members = members
            }

validatePanel :: [MarketContextPanelSnapshotV2] -> Either String ()
validatePanel [] = Left (marketContextPanelSchemaIdV2 ++ " has no snapshots")
validatePanel snapshots@(first : _) = do
    unless (all ((== mcps2Quote first) . mcps2Quote) snapshots) $
        Left (marketContextPanelSchemaIdV2 ++ " mixes quote scopes")
    unless (all ((== mcps2IntervalMs first) . mcps2IntervalMs) snapshots) $
        Left (marketContextPanelSchemaIdV2 ++ " mixes intervals")
    unless (strictlyIncreasing (map mcps2BarOpenTimeMs snapshots)) $
        Left (marketContextPanelSchemaIdV2 ++ " bar opens are not strictly increasing")
    unless (strictlyIncreasing (map mcps2DecisionTimeMs snapshots)) $
        Left (marketContextPanelSchemaIdV2 ++ " decisions are not strictly increasing")
    unless (allUnique (map mcps2SourceManifestSha256 snapshots)) $
        Left (marketContextPanelSchemaIdV2 ++ " reuses a source manifest across decisions")

sameSnapshot :: DecodedRow -> DecodedRow -> Bool
sameSnapshot left right = snapshotIdentity left == snapshotIdentity right

snapshotIdentity :: DecodedRow -> (String, String, Int64, Int64, Int64, Int64, Int64, Int)
snapshotIdentity row =
    ( drSourceManifestSha256 row
    , drQuote row
    , drIntervalMs row
    , drBarOpenTimeMs row
    , drDecisionTimeMs row
    , drUniverseEventTimeMs row
    , drUniverseAvailabilityTimeMs row
    , drPopulationCount row
    )

validMemberTimes :: Int64 -> MarketContextPanelMemberV2 -> Bool
validMemberTimes availabilityTime member =
    mcpm2TickerOpenTimeMs member <= mcpm2TickerCloseTimeMs member
        && mcpm2TickerCloseTimeMs member <= availabilityTime

validPeer :: Int64 -> Int64 -> MarketContextPanelMemberV2 -> Bool
validPeer barEnd decisionTime member =
    case mcpm2PeerReturn member of
        Nothing -> True
        Just peer ->
            mcpr2EventTimeMs peer == barEnd
                && mcpr2EventTimeMs peer <= mcpr2AvailabilityTimeMs peer
                && mcpr2AvailabilityTimeMs peer <= decisionTime
                && mcpr2SimpleReturn peer > -1
                && finite (mcpr2SimpleReturn peer)

requiredCell :: String -> Csv.NamedRecord -> Either String BS.ByteString
requiredCell name record =
    maybe
        (Left (marketContextPanelSchemaIdV2 ++ " is missing " ++ name))
        Right
        (HM.lookup (BS.pack name) record)

textCell :: String -> Csv.NamedRecord -> Either String String
textCell name record = do
    value <- trim . BS.unpack <$> requiredCell name record
    when (null value) $
        Left (marketContextPanelSchemaIdV2 ++ " " ++ name ++ " is blank")
    pure value

timestampCell :: String -> Csv.NamedRecord -> Either String Int64
timestampCell name record = requiredCell name record >>= parseTimestamp name

positiveIntCell :: String -> Csv.NamedRecord -> Either String Int
positiveIntCell name record = do
    value <- intCell name record
    unless (value > 0) $
        Left (marketContextPanelSchemaIdV2 ++ " " ++ name ++ " is not positive")
    pure value

nonNegativeIntCell :: String -> Csv.NamedRecord -> Either String Int
nonNegativeIntCell name record = do
    value <- intCell name record
    unless (value >= 0) $
        Left (marketContextPanelSchemaIdV2 ++ " " ++ name ++ " is negative")
    pure value

intCell :: String -> Csv.NamedRecord -> Either String Int
intCell name record = do
    raw <- trim . BS.unpack <$> requiredCell name record
    case readMaybe raw of
        Just value | canonicalUnsignedInteger raw -> Right value
        _ -> Left (marketContextPanelSchemaIdV2 ++ " " ++ name ++ " is invalid")

finiteCell :: String -> Csv.NamedRecord -> Either String Double
finiteCell name record = requiredCell name record >>= parseFinite name

maskCell :: String -> Csv.NamedRecord -> Either String Bool
maskCell name record = do
    raw <- trim . BS.unpack <$> requiredCell name record
    case raw of
        "0" -> Right False
        "1" -> Right True
        _ -> Left (marketContextPanelSchemaIdV2 ++ " " ++ name ++ " is not a binary mask")

parseTimestamp :: String -> BS.ByteString -> Either String Int64
parseTimestamp name raw =
    let value = trim (BS.unpack raw)
     in case readMaybe value of
            Just timestamp | canonicalUnsignedInteger value -> Right timestamp
            _ -> Left (marketContextPanelSchemaIdV2 ++ " " ++ name ++ " is not a canonical timestamp")

parseFinite :: String -> BS.ByteString -> Either String Double
parseFinite name raw =
    case readMaybe (trim (BS.unpack raw)) of
        Just value | finite value -> Right value
        _ -> Left (marketContextPanelSchemaIdV2 ++ " " ++ name ++ " is not finite")

canonicalUnsignedInteger :: String -> Bool
canonicalUnsignedInteger raw =
    not (null raw)
        && all isDigit raw
        && (raw == "0" || head raw /= '0')

validSha256 :: String -> Bool
validSha256 digest =
    length digest == 64
        && all (\character -> isHexDigit character && not (character >= 'A' && character <= 'F')) digest

validSymbolForQuote :: String -> String -> Bool
validSymbolForQuote quote symbol =
    validIdentifier symbol
        && length symbol > length quote
        && drop (length symbol - length quote) symbol == quote

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

strictlyIncreasing :: (Ord a) => [a] -> Bool
strictlyIncreasing values = and (zipWith (<) values (drop 1 values))

checkedAdd :: Int64 -> Int64 -> Either String Int64
checkedAdd left right =
    let result = toInteger left + toInteger right
     in if result > toInteger (maxBound :: Int64)
            then Left (marketContextPanelSchemaIdV2 ++ " timestamp arithmetic overflow")
            else Right (fromInteger result)

blank :: BS.ByteString -> Bool
blank = null . trim . BS.unpack

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)
