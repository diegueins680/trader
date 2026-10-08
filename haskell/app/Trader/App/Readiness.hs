{- | Live bot readiness that cannot outlive the reconciliation it reports.

The auto-start loop scans venue inventory against a snapshot of the bot
runtime map and publishes the result. Without more, a bot stopped or replaced
after the snapshot (or a manual trade that changes inventory) left readiness
reporting a reconciliation that no longer held until the next scan. Readiness
is therefore tied to the runtime epoch its snapshot was read under: every
runtime mutation bumps the epoch under the runtime lock, so a reader sees a
published scan only while nothing has changed since its snapshot, and only
while the scan is fresh enough to bound exchange-side drift.
-}
module Trader.App.Readiness (
    Readiness (..),
    readinessHolds,
    readinessMaxAgeMs,
) where

import Data.Int (Int64)

data Readiness
    = -- | No live recovery is required on this server (e.g. bot trading disabled).
      ReadinessNotRequired
    | NotReady
    | -- | A reconciled scan whose runtime snapshot was read at this epoch, at this time (ms).
      ReadyAt !Int !Int64
    deriving (Eq, Show)

{- | Whether readiness holds now, given the number of live manual trades in
flight (read before the epoch), the current runtime epoch, the time (ms) and
the freshness bound (ms). An in-flight trade may already have changed venue
inventory, so it suspends readiness until it completes and a later scan runs.
-}
readinessHolds :: Int -> Int -> Int64 -> Int64 -> Readiness -> Bool
readinessHolds manualInFlight epochNow nowMs maxAgeMs readiness =
    case readiness of
        ReadinessNotRequired -> True
        NotReady -> False
        ReadyAt epoch scannedAtMs ->
            manualInFlight == 0 && epoch == epochNow && scannedAtMs <= nowMs && nowMs - scannedAtMs <= maxAgeMs

-- | Freshness bound for a reconciled scan: ten scan intervals, and never under five minutes.
readinessMaxAgeMs :: Int -> Int64
readinessMaxAgeMs pollSec = 1000 * fromIntegral (max 300 (10 * max 1 pollSec))
