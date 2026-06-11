{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.AutoStartBackoff (
    autoStartBackoffSuite,
) where

import qualified Data.HashMap.Strict as HM

import Trader.App.AutoStartBackoff (
    BackoffPolicy (..),
    CircuitPolicy (..),
    ErrorClass (..),
    SymbolBackoff (..),
    classifyAutoStartError,
    defaultBackoffPolicy,
    defaultCircuitPolicy,
    initialBackoff,
    nextBackoff,
    normalizeAutoStartErrorMessage,
    shouldAttempt,
    shouldOpenCircuit,
    summarizeAuthSymbols,
 )

autoStartBackoffSuite :: [(String, IO ())]
autoStartBackoffSuite =
    [ ("classify -2015 wrapped HTTP 401 as ErrAuth", testClassifyAuth)
    , ("classify getAddrInfo DNS failure as ErrTransient", testClassifyDns)
    , ("classify HTTP 502/503 as ErrTransient", testClassifyTransientHttp)
    , ("classify LOT_SIZE rejection as ErrPermanent", testClassifyPermanentLotSize)
    , ("classify unknown noise as ErrUnknown", testClassifyUnknown)
    , ("transient backoff is monotone up to cap", testTransientMonotone)
    , ("auth backoff jumps to circuit-open delay on first failure", testAuthInitialDelay)
    , ("auth backoff stays >= circuit-open on subsequent failures", testAuthStayHigh)
    , ("shouldAttempt is true at exactly nextAllowedAt", testShouldAttemptBoundary)
    , ("shouldAttempt is false before nextAllowedAt", testShouldAttemptBefore)
    , ("Nothing backoff record permits immediate retry", testShouldAttemptNothing)
    , ("global circuit opens above threshold of auth symbols", testCircuitOpens)
    , ("global circuit stays closed below threshold", testCircuitClosed)
    , ("transient symbols do not feed global circuit", testTransientNoCircuit)
    , ("summarizeAuthSymbols returns sorted auth-only set", testSummarize)
    , ("classification is deterministic across reorders", testDeterministic)
    , ("transient delay clamps at bpMaxDelaySec for high consecutive counts", testTransientCap)
    , ("fingerprint is stable across request-ip rotation", testFingerprintIpRotation)
    , ("fingerprint is stable across signature/timestamp rotation", testFingerprintSigRotation)
    , ("fingerprint is stable across listenKey rotation", testFingerprintListenKey)
    , ("fingerprint normalization is idempotent", testFingerprintIdempotent)
    , ("fingerprint preserves ErrorClass classification", testFingerprintPreservesClass)
    , ("sbLastErrorMessage stores the fingerprint, not the raw message", testBackoffStoresFingerprint)
    , ("backoff stores the same fingerprint across IP rotation", testBackoffStableAcrossIps)
    ]

policy :: BackoffPolicy
policy = defaultBackoffPolicy

circuit :: CircuitPolicy
circuit = defaultCircuitPolicy

now0 :: Integer -> Integer
now0 = id

now :: Integer
now = 1_700_000_000_000

testClassifyAuth :: IO ()
testClassifyAuth =
    expectEq
        "wrapped HTTP 401 / -2015"
        ErrAuth
        ( classifyAutoStartError
            "futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action, request ip: 157.100.191.150"
        )

testClassifyDns :: IO ()
testClassifyDns =
    expectEq
        "getAddrInfo failure"
        ErrTransient
        ( classifyAutoStartError
            "HttpExceptionRequest Request { host = \"fapi.binance.com\" ... } (ConnectionFailure Network.Socket.getAddrInfo (called with preferred socket type/protocol: AddrInfo {...}, host name: \"fapi.binance.com\", service name: \"443\"): does not exist (nodename nor servname provided, or not known))"
        )

testClassifyTransientHttp :: IO ()
testClassifyTransientHttp = do
    expectEq "HTTP 502" ErrTransient (classifyAutoStartError "order/test HTTP 502: Bad Gateway")
    expectEq "HTTP 503" ErrTransient (classifyAutoStartError "order/test HTTP 503: Service Unavailable")
    expectEq "HTTP 429" ErrTransient (classifyAutoStartError "order/test HTTP 429: Too Many Requests")

testClassifyPermanentLotSize :: IO ()
testClassifyPermanentLotSize =
    expectEq
        "LOT_SIZE"
        ErrPermanent
        (classifyAutoStartError "order/test HTTP 400: Binance code -1013: Filter failure: LOT_SIZE")

testClassifyUnknown :: IO ()
testClassifyUnknown =
    expectEq
        "no useful structure"
        ErrUnknown
        (classifyAutoStartError "something went sideways but who knows")

testTransientMonotone :: IO ()
testTransientMonotone = do
    let mkChain n =
            let go 0 acc = acc
                go k acc =
                    let last' = head acc
                        sb = nextBackoff policy (fromInteger now) (Just last') ErrTransient "DNS"
                     in go (k - 1) (sb : acc)
                seed = initialBackoff (fromInteger now) ErrTransient "DNS" policy
             in reverse (go n [seed])
        chain = mkChain 6
        delays = zipWith (\a b -> sbNextAllowedAtMs b - sbLastErrorAtMs b) chain (tail chain ++ [last chain])
    -- All delays must be >= initial and <= cap, and non-decreasing until the cap.
    let initialMs = fromIntegral (bpInitialDelaySec policy) * 1000
        capMs = fromIntegral (bpMaxDelaySec policy) * 1000
    mapM_
        ( \d -> do
            expectTrue ("delay >= initial: " ++ show d) (d >= initialMs)
            expectTrue ("delay <= cap: " ++ show d) (d <= capMs)
        )
        delays
    -- Strictly non-decreasing (modulo cap).
    expectTrue
        ("monotone non-decreasing modulo cap " ++ show delays)
        (and (zipWith (\a b -> b >= a || b == capMs) delays (tail delays)))

testAuthInitialDelay :: IO ()
testAuthInitialDelay = do
    let sb = initialBackoff (fromInteger now) ErrAuth "401" policy
        expected = fromInteger now + fromIntegral (bpAuthCircuitOpenSec policy) * 1000
    expectEq "auth initial nextAllowedAt" expected (sbNextAllowedAtMs sb)
    expectEq "auth class recorded" ErrAuth (sbLastErrorClass sb)

testAuthStayHigh :: IO ()
testAuthStayHigh = do
    let sb0 = initialBackoff (fromInteger now) ErrAuth "401" policy
        sb1 = nextBackoff policy (fromInteger (now + 1000)) (Just sb0) ErrAuth "401"
        gap1 = sbNextAllowedAtMs sb1 - sbLastErrorAtMs sb1
    expectTrue
        "second auth failure also yields >= circuit open"
        (gap1 >= fromIntegral (bpAuthCircuitOpenSec policy) * 1000)

testShouldAttemptBoundary :: IO ()
testShouldAttemptBoundary = do
    let sb = initialBackoff (fromInteger now) ErrTransient "DNS" policy
        nextTs = sbNextAllowedAtMs sb
    expectTrue "at boundary -> attempt" (shouldAttempt nextTs (Just sb))

testShouldAttemptBefore :: IO ()
testShouldAttemptBefore = do
    let sb = initialBackoff (fromInteger now) ErrTransient "DNS" policy
        beforeTs = sbNextAllowedAtMs sb - 1
    expectTrue "before boundary -> no attempt" (not (shouldAttempt beforeTs (Just sb)))

testShouldAttemptNothing :: IO ()
testShouldAttemptNothing =
    expectTrue "Nothing -> always attempt" (shouldAttempt (fromInteger now) Nothing)

testCircuitOpens :: IO ()
testCircuitOpens = do
    let mkAuth s =
            ( s
            , initialBackoff
                (fromInteger now)
                ErrAuth
                "Invalid API-key, IP, or permissions for action"
                policy
            )
        m =
            HM.fromList
                [ mkAuth "XRPUSDT"
                , mkAuth "SOLUSDT"
                , mkAuth "FILUSDT"
                ]
    expectTrue "3 auth >= threshold opens circuit" (shouldOpenCircuit circuit m)

testCircuitClosed :: IO ()
testCircuitClosed = do
    let mkAuth s = (s, initialBackoff (fromInteger now) ErrAuth "401" policy)
        m = HM.fromList [mkAuth "XRPUSDT"]
    expectTrue "1 auth < threshold leaves circuit closed" (not (shouldOpenCircuit circuit m))

testTransientNoCircuit :: IO ()
testTransientNoCircuit = do
    let mkTrans s = (s, initialBackoff (fromInteger now) ErrTransient "DNS" policy)
        m =
            HM.fromList
                [ mkTrans "XRPUSDT"
                , mkTrans "SOLUSDT"
                , mkTrans "FILUSDT"
                , mkTrans "DOGEUSDT"
                , mkTrans "ADAUSDT"
                ]
    expectTrue
        "transient errors never open the auth circuit"
        (not (shouldOpenCircuit circuit m))

testSummarize :: IO ()
testSummarize = do
    let mkAuth s = (s, initialBackoff (fromInteger now) ErrAuth "401" policy)
        mkTrans s = (s, initialBackoff (fromInteger now) ErrTransient "DNS" policy)
        m =
            HM.fromList
                [ mkAuth "SOLUSDT"
                , mkAuth "ADAUSDT"
                , mkTrans "BTCUSDT"
                ]
    expectEq "auth-only sorted" ["ADAUSDT", "SOLUSDT"] (summarizeAuthSymbols m)

testDeterministic :: IO ()
testDeterministic = do
    let msg = "futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action, request ip: 1.2.3.4"
    let a = classifyAutoStartError msg
        b = classifyAutoStartError msg
        c = classifyAutoStartError msg
    expectEq "stable a==b" a b
    expectEq "stable b==c" b c

testTransientCap :: IO ()
testTransientCap = do
    -- Drive the chain to a large consecutive count and check the gap stays
    -- bounded by the cap (no overflow, no NaN/Infinity from the exponential).
    let go 0 sb = sb
        go k sb = go (k - 1) (nextBackoff policy (fromInteger now) (Just sb) ErrTransient "DNS")
        sbBig = go 64 (initialBackoff (fromInteger now) ErrTransient "DNS" policy)
        gapMs = sbNextAllowedAtMs sbBig - sbLastErrorAtMs sbBig
        capMs = fromIntegral (bpMaxDelaySec policy) * 1000
    expectTrue ("gap clamped at cap: gap=" ++ show gapMs ++ " cap=" ++ show capMs) (gapMs == capMs)

_unused :: Integer -> Integer
_unused = now0

{- | Two -2015 messages identical except for the rotating egress IP
(exactly the 2026-06-10 launchd-log pattern, 5 distinct IPs in one day).
-}
testFingerprintIpRotation :: IO ()
testFingerprintIpRotation = do
    let m1 = "futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action, request ip: 157.100.191.150"
        m2 = "futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action, request ip: 148.227.107.253"
        m3 = "futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action, request ip: 148.227.107.97"
    expectEq "m1==m2 fingerprint" (normalizeAutoStartErrorMessage m1) (normalizeAutoStartErrorMessage m2)
    expectEq "m2==m3 fingerprint" (normalizeAutoStartErrorMessage m2) (normalizeAutoStartErrorMessage m3)

-- | Auth replies with rotating signature/timestamp must fingerprint the same.
testFingerprintSigRotation :: IO ()
testFingerprintSigRotation = do
    let m1 = "GET /fapi/v2/positionRisk?timestamp=1780170205667&signature=00cbe708fc8b55db89ac17cab HTTP 401"
        m2 = "GET /fapi/v2/positionRisk?timestamp=1780199990001&signature=015fbbf22c3abcada54b756ad HTTP 401"
    expectEq
        "timestamp/signature noise -> same fingerprint"
        (normalizeAutoStartErrorMessage m1)
        (normalizeAutoStartErrorMessage m2)

-- | listenKey-bearing failures must fingerprint the same regardless of key.
testFingerprintListenKey :: IO ()
testFingerprintListenKey = do
    let m1 = "listenKey=uimwOyInCSXkV7OhKuZ61MCqZZAJx1UCzQFfekkYM9sKCiea6VWEUHkdhMQxYmo0 HTTP 401"
        m2 = "listenKey=oxXAEpqUg2vXuELdT8gJsZcAzeKqWENikLgsfdf3qLP4tHSmaQEFYrKZqdL9aPa9 HTTP 401"
    expectEq
        "listenKey noise -> same fingerprint"
        (normalizeAutoStartErrorMessage m1)
        (normalizeAutoStartErrorMessage m2)

-- | Normalization must be idempotent: f . f = f.
testFingerprintIdempotent :: IO ()
testFingerprintIdempotent = do
    let inputs =
            [ "futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action, request ip: 157.100.191.150"
            , "GET /fapi/v2/positionRisk?timestamp=1780170205667&signature=00cbe708fc8b55db89ac17cab HTTP 401"
            , "order/test HTTP 502: Bad Gateway"
            , ""
            , "   "
            ]
    mapM_
        ( \s ->
            let once = normalizeAutoStartErrorMessage s
                twice = normalizeAutoStartErrorMessage once
             in expectEq ("idempotent on: " ++ s) once twice
        )
        inputs

{- | Normalization must not change the classified ErrorClass; otherwise we
would shift a permanent failure into the transient bucket on accident.
-}
testFingerprintPreservesClass :: IO ()
testFingerprintPreservesClass = do
    let pairs =
            [ (ErrAuth, "futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action, request ip: 1.2.3.4")
            , (ErrTransient, "order/test HTTP 502: Bad Gateway timestamp=1780170205667")
            , (ErrPermanent, "order/test HTTP 400: Binance code -1013: Filter failure: LOT_SIZE signature=abc123")
            , (ErrTransient, "HttpExceptionRequest Request { host=fapi.binance.com } (ConnectionFailure ... getaddrinfo : does not exist (nodename nor servname provided, or not known))")
            ]
    mapM_
        ( \(expected, raw) -> do
            expectEq ("class on raw: " ++ raw) expected (classifyAutoStartError raw)
            let normalized = normalizeAutoStartErrorMessage raw
            expectEq ("class on normalized: " ++ normalized) expected (classifyAutoStartError normalized)
        )
        pairs

{- | After 'initialBackoff', sbLastErrorMessage must equal the fingerprint,
not the raw message. This is the invariant that closes the 2026-06-10 bug
where 'sbLastErrorMessage' rotated whenever the egress IP rotated.
-}
testBackoffStoresFingerprint :: IO ()
testBackoffStoresFingerprint = do
    let raw = "futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action, request ip: 148.227.107.253"
        sb = initialBackoff (fromInteger now) ErrAuth raw policy
    expectEq
        "sbLastErrorMessage uses fingerprint"
        (normalizeAutoStartErrorMessage raw)
        (sbLastErrorMessage sb)
    expectTrue
        "sbLastErrorMessage no longer contains the raw IP"
        ("148.227.107.253" `notElem` words (sbLastErrorMessage sb))

{- | After two failures whose only difference is the egress IP,
both backoff records must carry the same fingerprint. Without this
invariant the 2026-06-10 log re-flood is allowed back in.
-}
testBackoffStableAcrossIps :: IO ()
testBackoffStableAcrossIps = do
    let raw1 = "futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action, request ip: 157.100.191.150"
        raw2 = "futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action, request ip: 148.227.107.253"
        sb1 = initialBackoff (fromInteger now) ErrAuth raw1 policy
        sb2 = nextBackoff policy (fromInteger (now + 1000)) (Just sb1) ErrAuth raw2
    expectEq
        "backoff fingerprint stable across IP rotation"
        (sbLastErrorMessage sb1)
        (sbLastErrorMessage sb2)

expectEq :: (Eq a, Show a) => String -> a -> a -> IO ()
expectEq label expected actual =
    if expected == actual
        then pure ()
        else error (label ++ ": expected " ++ show expected ++ ", got " ++ show actual)

expectTrue :: String -> Bool -> IO ()
expectTrue label cond =
    if cond
        then pure ()
        else error (label ++ ": condition failed")
