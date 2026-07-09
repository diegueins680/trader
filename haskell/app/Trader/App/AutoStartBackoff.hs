{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE OverloadedStrings #-}

{- |
Module: Trader.App.AutoStartBackoff

Engineering rationale
=====================

The live-bot auto-start loop polls Binance every @TRADER_BOT_COMBOS_POLL_SEC@
seconds (default 30 s) and, for each missing symbol, calls
@futures/positionRisk@ (via 'resolveAdoptionRequirement') to decide whether to
start a runtime. When the API key, IP allow-list, or DNS is broken, the loop
hammers Binance forever at the same cadence. This module provides the pure
control-plane logic that turns that retry storm into:

  * exponentially backed-off per-symbol retries for transient errors
    (DNS failures, 5xx, 429, request timeouts), and
  * a per-symbol circuit breaker that holds open for a long, deterministic
    window when Binance reports a permanent auth/IP failure
    (HTTP 401, codes @-2015@, @-2014@, @-1022@, @-1021@), and
  * a global auth circuit breaker that opens once enough distinct symbols
    have entered the auth state — protecting against full-key invalidation
    that would otherwise burn rate limit / risk an IP ban.

The module exposes pure functions only so they can be exercised from unit
tests without touching @IO@.

Invariants tested in 'Trader.Test.AutoStartBackoff':

  * 'classifyAutoStartError' is total and stable: identical inputs produce
    identical outputs.
  * 'nextBackoff' is monotone in the consecutive-failure count for transient
    errors, up to 'bpMaxDelaySec'.
  * For auth errors the next allowed timestamp is at least the auth-circuit
    open duration away from "now".
  * For successful recovery ('clearBackoff') the symbol is permitted to
    attempt immediately.
  * 'shouldAttempt' is true iff @now >= sbNextAllowedAtMs@.
  * 'shouldOpenCircuit' depends only on the symbols currently classified as
    'ErrAuth' (not on transient flap).
-}
module Trader.App.AutoStartBackoff (
    BackoffPolicy (..),
    CircuitPolicy (..),
    CircuitState (..),
    ErrorClass (..),
    SymbolBackoff (..),
    classifyAutoStartError,
    clearBackoff,
    defaultBackoffPolicy,
    defaultCircuitPolicy,
    errorClassDescription,
    initialBackoff,
    nextBackoff,
    normalizeAutoStartErrorMessage,
    shouldAttempt,
    shouldOpenCircuit,
    summarizeAuthSymbols,
) where

import Data.Char (isAsciiLower, isAsciiUpper, isDigit, isHexDigit, toLower)
import qualified Data.HashMap.Strict as HM
import Data.Int (Int64)
import Data.List (isInfixOf, sort, stripPrefix)
import Data.Maybe (fromMaybe)

import Trader.App.BinanceProbe (
    BinanceErrorInfo (..),
    binanceAuthFailureFromMessage,
    parseBinanceError,
 )

-- | Coarse classification of a per-symbol failure observed during auto-start.
data ErrorClass
    = -- | Network/DNS, 5xx, 429, timeouts — retry with exponential backoff.
      ErrTransient
    | {- | Binance rejected with @-2015@/@-2014@/@-1022@/@-1021@ or HTTP 401/403.
      Permanent until operator rotates keys or fixes IP allow-list.
      -}
      ErrAuth
    | {- | Binance order/account rejection that will not heal without a config
      change (e.g. @-1013@ LOT_SIZE, @-1100@ illegal characters). Per-symbol
      permanent until operator intervention.
      -}
      ErrPermanent
    | -- | Unrecognized error. Treated like a long-but-finite transient.
      ErrUnknown
    deriving (Eq, Show)

errorClassDescription :: ErrorClass -> String
errorClassDescription ErrTransient = "transient (will retry)"
errorClassDescription ErrAuth = "auth (Binance key/IP/permissions; circuit open)"
errorClassDescription ErrPermanent = "permanent (Binance rejected configuration)"
errorClassDescription ErrUnknown = "unknown (treated as transient)"

-- | Per-symbol backoff control block. All timestamps are epoch milliseconds.
data SymbolBackoff = SymbolBackoff
    { sbLastErrorAtMs :: !Int64
    , sbLastErrorClass :: !ErrorClass
    , sbConsecutiveFails :: !Int
    , sbNextAllowedAtMs :: !Int64
    -- ^ The earliest epoch-ms at which the loop may retry the symbol.
    , sbLastErrorMessage :: !String
    -- ^ Stable message fingerprint to drive log-dedup decisions upstream.
    }
    deriving (Eq, Show)

data BackoffPolicy = BackoffPolicy
    { bpInitialDelaySec :: !Int
    -- ^ First retry delay after a transient error, in seconds.
    , bpMaxDelaySec :: !Int
    -- ^ Cap for the exponential schedule, in seconds.
    , bpAuthCircuitOpenSec :: !Int
    -- ^ How long to hold per-symbol auth circuit open (seconds).
    , bpPermanentOpenSec :: !Int
    -- ^ How long to hold per-symbol permanent circuit open (seconds).
    , bpMultiplier :: !Double
    -- ^ Exponential multiplier (e.g. 2.0).
    }
    deriving (Eq, Show)

defaultBackoffPolicy :: BackoffPolicy
defaultBackoffPolicy =
    BackoffPolicy
        { bpInitialDelaySec = 60
        , bpMaxDelaySec = 1_800
        , bpAuthCircuitOpenSec = 3_600
        , bpPermanentOpenSec = 21_600 -- 6h: needs an operator change anyway
        , bpMultiplier = 2.0
        }

data CircuitPolicy = CircuitPolicy
    { cpAuthThreshold :: !Int
    {- ^ Open the global circuit once at least this many symbols are
    classified as auth-failed.
    -}
    , cpOpenDurationSec :: !Int
    -- ^ How long to keep the global circuit open before re-probing.
    }
    deriving (Eq, Show)

defaultCircuitPolicy :: CircuitPolicy
defaultCircuitPolicy =
    CircuitPolicy
        { cpAuthThreshold = 3
        , cpOpenDurationSec = 3_600
        }

data CircuitState
    = CircuitClosed
    | CircuitOpen
        { coOpenedAtMs :: !Int64
        , coReopenAtMs :: !Int64
        , coAuthSymbols :: ![String]
        }
    deriving (Eq, Show)

-- | Classify a free-form error message from 'recordError' into 'ErrorClass'.
classifyAutoStartError :: String -> ErrorClass
classifyAutoStartError raw
    -- Auth/IP/permissions failures from Binance — permanent until operator
    -- rotates keys, fixes IP allow-list, or repairs the clock.
    | isJustVal (binanceAuthFailureFromMessage raw) = ErrAuth
    -- Network-level failures (DNS, TCP reset, TLS handshake, IP unreachable).
    | any (`isInfixOf` lowerRaw) transientNetworkPhrases = ErrTransient
    -- HTTP 5xx / 408 / 429 from Binance — back off, do not circuit-open.
    | isTransientByCodeOrSummary code summary = ErrTransient
    -- Order/account validation rejections (LOT_SIZE, NOTIONAL, illegal chars).
    | isPermanentByCode code || isPermanentBySummary summary = ErrPermanent
    -- Everything else is unknown — give it a moderate transient policy.
    | otherwise = ErrUnknown
  where
    parsed = parseBinanceError raw
    code = beiCode parsed
    summary = beiSummary parsed
    lowerRaw = map toLower raw
    isJustVal :: Maybe a -> Bool
    isJustVal Nothing = False
    isJustVal (Just _) = True

transientNetworkPhrases :: [String]
transientNetworkPhrases =
    [ "getaddrinfo"
    , "connectionfailure"
    , "connection refused"
    , "connection reset"
    , "network is unreachable"
    , "no route to host"
    , "tls handshake"
    , "responsetimeout"
    , "tlsexception"
    , "does not exist (nodename"
    ]

isTransientByCodeOrSummary :: Maybe Int -> String -> Bool
isTransientByCodeOrSummary mCode summary =
    maybe False (`elem` transientCodes) mCode
        || any (`isInfixOf` lowerSummary) transientPhrases
  where
    lowerSummary = map toLower summary
    transientCodes :: [Int]
    transientCodes = [408, 429, 500, 502, 503, 504]
    transientPhrases =
        [ "timed out"
        , "timeout"
        , "too many requests"
        , "temporarily unavailable"
        , "service unavailable"
        , "bad gateway"
        , "internal error"
        ]

isPermanentByCode :: Maybe Int -> Bool
isPermanentByCode = maybe False go
  where
    go c =
        c == (-1_013)
            || c == (-2_010)
            || c == (-2_011)
            || c == (-2_022)
            || c == (-4_164)
            || (c <= (-1_100) && c >= (-1_199))
            || (c <= (-4_000) && c >= (-4_999))

isPermanentBySummary :: String -> Bool
isPermanentBySummary summary = any (`isInfixOf` lowerSummary) phrases
  where
    lowerSummary = map toLower summary
    phrases =
        [ "filter failure"
        , "invalid symbol"
        , "lot_size"
        , "lot size"
        , "min_qty"
        , "min qty"
        , "notional"
        , "mandatory parameter"
        , "illegal characters found"
        , "precision is over the maximum"
        ]

-- | Initial backoff record for a never-failed symbol.
initialBackoff :: Int64 -> ErrorClass -> String -> BackoffPolicy -> SymbolBackoff
initialBackoff nowMs cls msg policy =
    let delaySec = case cls of
            ErrAuth -> bpAuthCircuitOpenSec policy
            ErrPermanent -> bpPermanentOpenSec policy
            ErrTransient -> bpInitialDelaySec policy
            ErrUnknown -> bpInitialDelaySec policy
     in SymbolBackoff
            { sbLastErrorAtMs = nowMs
            , sbLastErrorClass = cls
            , sbConsecutiveFails = 1
            , sbNextAllowedAtMs = nowMs + fromIntegral delaySec * 1_000
            , sbLastErrorMessage = normalizeAutoStartErrorMessage msg
            }

{- | Advance the backoff record after a fresh failure.
For auth/permanent errors the schedule jumps to its full circuit-open delay
regardless of prior count; for transient errors the delay grows
geometrically up to 'bpMaxDelaySec'.
-}
nextBackoff ::
    BackoffPolicy ->
    Int64 ->
    Maybe SymbolBackoff ->
    ErrorClass ->
    String ->
    SymbolBackoff
nextBackoff policy nowMs mPrev cls msg =
    case mPrev of
        Nothing -> initialBackoff nowMs cls msg policy
        Just prev ->
            let prevCount = sbConsecutiveFails prev
                count' = prevCount + 1
                base =
                    case cls of
                        ErrAuth -> bpAuthCircuitOpenSec policy
                        ErrPermanent -> bpPermanentOpenSec policy
                        ErrTransient -> transientDelaySec policy count'
                        ErrUnknown -> transientDelaySec policy count'
             in SymbolBackoff
                    { sbLastErrorAtMs = nowMs
                    , sbLastErrorClass = cls
                    , sbConsecutiveFails = count'
                    , sbNextAllowedAtMs = nowMs + fromIntegral base * 1_000
                    , sbLastErrorMessage = normalizeAutoStartErrorMessage msg
                    }

-- | Geometric schedule capped by 'bpMaxDelaySec'.
transientDelaySec :: BackoffPolicy -> Int -> Int
transientDelaySec policy consecutiveCount =
    let !c = max 1 consecutiveCount
        !mult = bpMultiplier policy
        !initial = max 1 (bpInitialDelaySec policy)
        !cap = max initial (bpMaxDelaySec policy)
        raw :: Double
        raw = fromIntegral initial * (mult ** fromIntegral (c - 1))
        clipped = min (fromIntegral cap) (max (fromIntegral initial) raw)
     in min cap (max initial (truncate clipped))

{- | Clear backoff after a confirmed success. Returns 'Nothing' so callers can
drop the symbol from the in-memory map.
-}
clearBackoff :: Maybe SymbolBackoff
clearBackoff = Nothing

-- | Decide whether the loop may attempt a symbol now.
shouldAttempt :: Int64 -> Maybe SymbolBackoff -> Bool
shouldAttempt _ Nothing = True
shouldAttempt nowMs (Just sb) = nowMs >= sbNextAllowedAtMs sb

-- | Decide whether the global auth circuit should be open.
shouldOpenCircuit :: CircuitPolicy -> HM.HashMap String SymbolBackoff -> Bool
shouldOpenCircuit policy m =
    HM.size authMap >= cpAuthThreshold policy
  where
    authMap = HM.filter ((== ErrAuth) . sbLastErrorClass) m

-- | List symbols currently classified as auth-failed (sorted for log stability).
summarizeAuthSymbols :: HM.HashMap String SymbolBackoff -> [String]
summarizeAuthSymbols =
    sort . map fst . HM.toList . HM.filter ((== ErrAuth) . sbLastErrorClass)

{- | Normalize a raw error message into a stable fingerprint suitable for
log-deduplication and 'sbLastErrorMessage'.

Real Binance auth replies embed the client's egress IP
@request ip: 148.227.107.253@; on a host whose IP rotates (VPN flap,
captive portal, DHCP renewal) the otherwise-identical @-2015@ failure
produces a fresh raw message every rotation. The 2026-06-10 launchd log
shows the same auth failure logged five times per symbol because the
host's egress IP cycled through five values in one day.

This function strips known volatile noise (request IPs, timestamps,
signatures, request-ids, port numbers, listenKeys, integer 'request ip:'
tails, and trailing whitespace) so the fingerprint depends only on the
stable parts of the failure (HTTP code, Binance error code, message
text). It is __idempotent__ and __total__ over arbitrary strings.

Invariants exercised in 'Trader.Test.AutoStartBackoff':

  * The fingerprint of two raw messages that differ only in 'request ip'
    is identical.
  * The fingerprint of two raw messages that differ only in a
    signature/timestamp/listenKey/request-id token is identical.
  * Normalization is idempotent: @normalize . normalize = normalize@.
  * Normalization preserves the classified 'ErrorClass'.
-}
normalizeAutoStartErrorMessage :: String -> String
normalizeAutoStartErrorMessage =
    stripTrailing
        . collapseSpaces
        . scrubAll
  where
    stripTrailing = reverse . dropWhile isWsOrPunct . reverse
    isWsOrPunct c = c == ' ' || c == '\t' || c == '.' || c == ','

scrubAll :: String -> String
scrubAll = scrubLoop
  where
    scrubLoop [] = []
    scrubLoop s@(c : rest) =
        case tryScrub s of
            Just (replacement, leftover) -> replacement ++ scrubLoop leftover
            Nothing -> c : scrubLoop rest

    -- \| Try to scrub one volatile token at the head of the string.
    --   Each pattern is keyed off a stable English/Binance phrase so we
    --   only redact what we know is noise. The 'consumeRedacted' suffix on
    --   every branch keeps normalization idempotent: re-running the function
    --   on an already-redacted string still produces the same output.
    tryScrub :: String -> Maybe (String, String)
    tryScrub s
        | Just rest <- stripPrefixCI "request ip: " s =
            Just ("request ip: <REDACTED>", consumeRedacted (dropIpish rest))
        | Just rest <- stripPrefixCI "request ip:" s =
            Just ("request ip: <REDACTED>", consumeRedacted (dropIpish (dropWhile (== ' ') rest)))
        | Just rest <- stripPrefixCI "client ip: " s =
            Just ("client ip: <REDACTED>", consumeRedacted (dropIpish rest))
        | Just rest <- stripPrefixCI "signature=" s =
            Just ("signature=<REDACTED>", consumeRedacted (dropWhile isHexDigit rest))
        | Just rest <- stripPrefixCI "signature: " s =
            Just ("signature: <REDACTED>", consumeRedacted (dropWhile isHexDigit rest))
        | Just rest <- stripPrefixCI "listenkey=" s =
            Just ("listenKey=<REDACTED>", consumeRedacted (dropWhile isListenKeyChar rest))
        | Just rest <- stripPrefixCI "timestamp=" s =
            Just ("timestamp=<REDACTED>", consumeRedacted (dropWhile isDigit rest))
        | Just rest <- stripPrefixCI "recvwindow=" s =
            Just ("recvWindow=<REDACTED>", consumeRedacted (dropWhile isDigit rest))
        | Just rest <- stripPrefixCI "servertime=" s =
            Just ("serverTime=<REDACTED>", consumeRedacted (dropWhile isDigit rest))
        | Just rest <- stripPrefixCI "x-mbx-uuid=" s =
            Just ("x-mbx-uuid=<REDACTED>", consumeRedacted (dropWhile isUuidChar rest))
        | Just rest <- stripPrefixCI "requestid=" s =
            Just ("requestId=<REDACTED>", consumeRedacted (dropWhile isUuidChar rest))
        | Just rest <- stripPrefixCI "request-id: " s =
            Just ("request-id: <REDACTED>", consumeRedacted (dropWhile isUuidChar rest))
        | otherwise = Nothing

    -- \| Consume a leading literal '<REDACTED>' marker so normalization is
    --   idempotent under re-application.
    consumeRedacted :: String -> String
    consumeRedacted leftover =
        fromMaybe leftover (stripPrefix "<REDACTED>" leftover)

    -- \| Case-insensitive 'stripPrefix' over the prefix only.
    stripPrefixCI :: String -> String -> Maybe String
    stripPrefixCI prefix s = stripPrefix (map toLower prefix) (lowerN (length prefix) s)
      where
        lowerN _ [] = []
        lowerN 0 cs = cs
        lowerN n (c : cs) = toLower c : lowerN (n - 1) cs

    -- \| After 'request ip: ' consume an IPv4 or IPv6-ish run.
    dropIpish :: String -> String
    dropIpish = dropWhile isIpChar
      where
        isIpChar c =
            isDigit c
                || c == '.'
                || c == ':'
                || (c >= 'a' && c <= 'f')
                || (c >= 'A' && c <= 'F')

    isListenKeyChar c = isHexDigit c || isAlphaNum' c || c == '-' || c == '_'
    isUuidChar c = isHexDigit c || c == '-' || c == '_'
    isAlphaNum' c = isAsciiLower c || isAsciiUpper c || isDigit c

collapseSpaces :: String -> String
collapseSpaces = go False
  where
    go _ [] = []
    go prevWs (c : rest)
        | c == '\n' || c == '\r' || c == '\t' = go True rest
        | c == ' ' = if prevWs then go True rest else ' ' : go True rest
        | otherwise = c : go False rest
