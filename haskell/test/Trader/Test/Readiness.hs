module Trader.Test.Readiness (readinessSuite) where

import Trader.App.Readiness

readinessSuite :: [(String, IO ())]
readinessSuite =
    [ ("readiness holds only at the scanned runtime epoch", epochBound)
    , ("readiness expires after the freshness bound", freshnessBound)
    , ("servers without live recovery stay ready; revoked readiness never holds", fixedStates)
    , ("freshness bound is ten scan intervals and at least five minutes", maxAge)
    ]

expect :: (Eq a, Show a) => String -> a -> a -> IO ()
expect label wanted actual =
    if wanted == actual then pure () else ioError (userError (label ++ ": expected " ++ show wanted ++ ", got " ++ show actual))

epochBound :: IO ()
epochBound = do
    let maxAgeMs = readinessMaxAgeMs 30
    -- A stop, start, promotion or manual trade after the scan bumps the epoch, so the old scan no longer holds.
    expect "same epoch" True (readinessHolds 4 10000 maxAgeMs (ReadyAt 4 9000))
    expect "epoch moved on" False (readinessHolds 5 10000 maxAgeMs (ReadyAt 4 9000))

freshnessBound :: IO ()
freshnessBound = do
    let maxAgeMs = readinessMaxAgeMs 30
    expect "at the bound" True (readinessHolds 1 (1000 + maxAgeMs) maxAgeMs (ReadyAt 1 1000))
    expect "past the bound" False (readinessHolds 1 (1001 + maxAgeMs) maxAgeMs (ReadyAt 1 1000))
    expect "scan from the future" False (readinessHolds 1 999 maxAgeMs (ReadyAt 1 1000))

fixedStates :: IO ()
fixedStates = do
    expect "not required" True (readinessHolds 9 0 0 ReadinessNotRequired)
    expect "revoked" False (readinessHolds 0 0 maxBound NotReady)

maxAge :: IO ()
maxAge = expect "bounds" [300000, 300000, 600000] (map readinessMaxAgeMs [5, 30, 60])
