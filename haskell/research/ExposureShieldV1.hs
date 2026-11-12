{- | Deterministic exposure shield (exposure_shield_v1), integer arithmetic only.

Exposure is a signed integer in units of 1e-4 of account equity. A learned
policy is untrusted: its raw output is decoded, validated and projected here.
This module holds no order capability. 'ResearchIntent' and 'Snapshot' have no
exported constructors, so an intent can only be produced by 'shield' from a
validated snapshot, and nothing here can construct or authorize a live order.

The module is research-only: it is compiled by the formal target and is not a
component of any executable. It mirrors scripts/research/exposure_rl/shield.py.
-}
module ExposureShieldV1 (
    Snapshot,
    mkSnapshot,
    snapshotExposure,
    ResearchIntent,
    intentTarget,
    Outcome (..),
    capUnits,
    minTradeUnits,
    drawdownLimitUnits,
    snapshotBound,
    scaleUnits,
    decodeProposal,
    project,
    shield,
    reserve,
    orderAuthorized,
) where

-- | Exposure units per unit of equity.
scaleUnits :: Integer
scaleUnits = 10000

-- | Maximum admitted absolute target exposure.
capUnits :: Integer
capUnits = 1666

-- | Smallest admitted absolute change of exposure.
minTradeUnits :: Integer
minTradeUnits = 25

-- | Drawdown at which only reductions are admitted.
drawdownLimitUnits :: Integer
drawdownLimitUnits = 1000

-- | Absolute exposure beyond which a snapshot is not valid.
snapshotBound :: Integer
snapshotBound = 100000

-- | A validated account snapshot: exposure, drawdown, freshness, headroom.
data Snapshot = Snapshot Integer Integer Bool Integer
    deriving (Eq, Show)

-- | Validate a snapshot. Out-of-range fields yield 'Nothing'.
mkSnapshot :: Integer -> Integer -> Bool -> Integer -> Maybe Snapshot
mkSnapshot exposure drawdown fresh headroom
    | abs exposure > snapshotBound = Nothing
    | drawdown < 0 || drawdown > scaleUnits = Nothing
    | headroom < 0 || headroom > scaleUnits = Nothing
    | otherwise = Just (Snapshot exposure drawdown fresh headroom)

snapshotExposure :: Snapshot -> Integer
snapshotExposure (Snapshot exposure _ _ _) = exposure

-- | A shield-admitted simulated intent. It carries no order authority.
newtype ResearchIntent = ResearchIntent Integer
    deriving (Eq, Show)

intentTarget :: ResearchIntent -> Integer
intentTarget (ResearchIntent target) = target

-- | No research intent ever authorizes a live order.
orderAuthorized :: ResearchIntent -> Bool
orderAuthorized _ = False

data Outcome
    = -- | A valid proposal admitted as a research intent.
      Permitted ResearchIntent
    | -- | A valid proposal whose change is below the minimum size.
      Abstain
    | -- | An invalid proposal; a state-dependent fallback reduction, if any.
      Rejected (Maybe ResearchIntent)
    | -- | An invalid snapshot; no intent may be formed.
      RecoveryRequired
    deriving (Eq, Show)

clamp :: Integer -> Integer -> Integer -> Integer
clamp low high value
    | value < low = low
    | value > high = high
    | otherwise = value

{- | Decode a raw policy output in [-1, 1] to units, truncating toward zero.
Non-finite and out-of-range values are rejected.
-}
decodeProposal :: Double -> Maybe Integer
decodeProposal raw
    | isNaN raw || isInfinite raw = Nothing
    | raw < -1 || raw > 1 = Nothing
    | otherwise = Just (truncate (raw * fromInteger capUnits))

-- | Project an integer proposal onto the admitted set.
project :: Snapshot -> Integer -> Integer
project (Snapshot current drawdown fresh headroom) proposal =
    let capped = clamp (negate capUnits) capUnits proposal
        limit = min capUnits (min (abs current) capUnits + headroom)
        limited = clamp (negate limit) limit capped
        riskOff = not fresh || drawdown >= drawdownLimitUnits
     in if riskOff then reduceOnly current limited else limited

-- Never increase the absolute exposure and never cross through zero.
reduceOnly :: Integer -> Integer -> Integer
reduceOnly current target
    | current > 0 = clamp 0 current target
    | current < 0 = clamp current 0 target
    | otherwise = 0

{- | Total shield. The first argument is the result of 'mkSnapshot', so an
invalid snapshot is represented explicitly and always requires recovery.
-}
shield :: Maybe Snapshot -> Double -> Outcome
shield Nothing _ = RecoveryRequired
shield (Just snapshot) raw =
    case decodeProposal raw of
        Just proposal -> admit Permitted Abstain (project snapshot proposal)
        Nothing -> admit (Rejected . Just) (Rejected Nothing) (project snapshot current)
  where
    current = snapshotExposure snapshot
    admit accept decline target
        | abs (target - current) < minTradeUnits = decline
        | otherwise = accept (ResearchIntent target)

{- | Atomic compare-and-swap reservation of absolute exposure against a shared
budget. A stale version, or an increase that would exceed the budget, leaves
the state unchanged. Reductions are always granted.
-}
reserve :: Integer -> [Integer] -> Integer -> Integer -> Int -> Integer -> ([Integer], Integer, Bool)
reserve budget held version seenVersion index wanted
    | seenVersion /= version || wanted < 0 = unchanged
    | index < 0 || index >= length held = unchanged
    | total > budget && wanted > previous = unchanged
    | otherwise = (take index held ++ [wanted] ++ drop (index + 1) held, version + 1, True)
  where
    unchanged = (held, version, False)
    previous = held !! index
    total = sum held - previous + wanted
