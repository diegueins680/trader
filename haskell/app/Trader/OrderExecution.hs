module Trader.OrderExecution (
    OrderExecutionEvidence (..),
    orderAppliedQuantity,
    orderAppliedFraction,
    applyExecutedQuantity,
    applyReduceOnlyExecutedQuantity,
    applySplitReversalExecutedQuantities,
) where

import Control.Exception (assert)
import Trader.Text (normalizeKey, trim)

data OrderExecutionEvidence = OrderExecutionEvidence
    { oeeSent :: !Bool
    , oeeLive :: !Bool
    , oeeStatus :: !(Maybe String)
    , oeeExecutedQty :: !(Maybe Double)
    }
    deriving (Eq, Show)

orderAppliedQuantity :: OrderExecutionEvidence -> Double -> Maybe Double
orderAppliedQuantity ev fallbackQty =
    let fallback = positiveFinite fallbackQty
        status = normalizedStatus (oeeStatus ev)
        executed = oeeExecutedQty ev >>= positiveFinite
     in if not (oeeSent ev)
            then Nothing
            else
                if not (oeeLive ev)
                    then fallback
                    else case executed of
                        -- Trust explicit fill evidence first, even on canceled/expired statuses,
                        -- because exchanges can report partial fills alongside terminal statuses.
                        Just q -> Just q
                        Nothing ->
                            case status of
                                Just s | statusHasNoFill s -> Nothing
                                Just s | statusImpliesFilled s -> fallback
                                _ -> Nothing

{- | Order evidence quantities (requested\/executed) are denominated in
base-asset units, while bot position state tracks fractions of equity.
Applying base units to fraction state books a 300-contract fill as a 300x
equity position. Convert instead by scaling the intended fraction by the
filled proportion of the requested base quantity; without a base-unit
denominator (test mode, fill-implied statuses with no quantities) fall back
to all-or-nothing on the intended fraction.
-}
orderAppliedFraction :: OrderExecutionEvidence -> Maybe Double -> Double -> Maybe Double
orderAppliedFraction ev mRequestedBase intendedFraction = do
    intended <- positiveFinite intendedFraction
    case mRequestedBase >>= positiveFinite of
        -- Without a base-unit denominator, fill evidence is all-or-nothing:
        -- its magnitude cannot safely be interpreted as an equity fraction.
        Nothing -> intended <$ orderAppliedQuantity ev intended
        Just reqBase -> do
            appliedBase <- orderAppliedQuantity ev reqBase
            -- Exchange evidence must never make the applied fraction exceed
            -- the requested exposure.  Overfills are malformed evidence for
            -- position accounting, so conservatively cap them at the intent.
            positiveFinite (intended * min 1 (appliedBase / reqBase))

applyExecutedQuantity :: Int -> Double -> Bool -> Double -> (Int, Double, Double, Double)
applyExecutedQuantity prevPos prevSize isBuy qtyRaw =
    let eps = 1e-9
        qty = sanitizeExecutedQuantity eps qtyRaw
        currentSigned = sanitizeSignedExposure prevPos prevSize
        deltaSigned = if isBuy then qty else negate qty
        newSigned = currentSigned + deltaSigned
        posNew
            | newSigned > eps = 1
            | newSigned < negate eps = -1
            | otherwise = 0
        sizeNew =
            if posNew == 0
                then 0
                else abs newSigned
        closeQtyRaw =
            if qty <= 0 || abs currentSigned <= eps || currentSigned * deltaSigned >= 0
                then 0
                else min (abs currentSigned) qty
        closeQty =
            if closeQtyRaw <= eps
                then 0
                else closeQtyRaw
        openQtyRaw = max 0 (qty - closeQty)
        openQty =
            if openQtyRaw <= eps || posNew == 0
                then 0
                else openQtyRaw
        -- Runtime invariant: fill quantities are conserved (within epsilon)
        qtyConserved = abs ((closeQty + openQty) - qty) <= max eps (abs qty * 1e-12)
     in assert qtyConserved (posNew, sizeNew, closeQty, openQty)

applyReduceOnlyExecutedQuantity :: Int -> Double -> Double -> (Int, Double, Double, Double)
applyReduceOnlyExecutedQuantity prevPos prevSize qtyRaw =
    let eps = 1e-9
        qty = sanitizeExecutedQuantity eps qtyRaw
        prevSign = sanitizePosition prevPos
        currentSize =
            if prevSign == 0
                then 0
                else sanitizeMagnitude prevSize
        closeQtyRaw =
            if qty <= 0 || currentSize <= eps
                then 0
                else min currentSize qty
        closeQty =
            if closeQtyRaw <= eps
                then 0
                else closeQtyRaw
        sizeNewRaw = max 0 (currentSize - closeQty)
        sizeNew =
            if sizeNewRaw <= eps
                then 0
                else sizeNewRaw
        posNew
            | sizeNew <= 0 = 0
            | otherwise = prevSign
        -- Runtime invariants: reduce-only never increases exposure or flips sign
        neverIncreases = sizeNew <= currentSize || currentSize <= eps
        neverFlips = posNew == 0 || posNew == prevSign
     in assert (neverIncreases && neverFlips) (posNew, sizeNew, closeQty, 0)

{- | Apply independently observed close and entry fills for a split reversal.
The reduce-only leg is accounted first, so an absent or rejected entry leaves
the position reduced or flat instead of fabricating a completed flip.
-}
applySplitReversalExecutedQuantities :: Int -> Double -> Bool -> Double -> Double -> (Int, Double, Double, Double)
applySplitReversalExecutedQuantities prevPos prevSize isBuy closeQty entryQty =
    let (posAfterClose, sizeAfterClose, closeApplied, _) =
            applyReduceOnlyExecutedQuantity prevPos prevSize closeQty
        (posNew, sizeNew, entryCloseApplied, openApplied) =
            applyExecutedQuantity posAfterClose sizeAfterClose isBuy entryQty
     in (posNew, sizeNew, closeApplied + entryCloseApplied, openApplied)

sanitizeSignedExposure :: Int -> Double -> Double
sanitizeSignedExposure prevPos prevSize =
    let prevSign = sanitizePosition prevPos
        size = sanitizeMagnitude prevSize
     in if prevSign == 0
            then 0
            else fromIntegral prevSign * size

sanitizePosition :: Int -> Int
sanitizePosition = signum

sanitizeExecutedQuantity :: Double -> Double -> Double
sanitizeExecutedQuantity eps qtyRaw =
    let qty = sanitizeMagnitude qtyRaw
     in if qty <= eps
            then 0
            else qty

sanitizeMagnitude :: Double -> Double
sanitizeMagnitude x =
    case positiveFinite x of
        Just y -> min maxSanitizedMagnitude y
        Nothing -> 0

maxSanitizedMagnitude :: Double
-- Keep each operand comfortably below maxBound so signed exposure updates
-- stay finite even when one sanitized position and one sanitized fill combine.
maxSanitizedMagnitude = 1.7976931348623157e308 / 4

normalizedStatus :: Maybe String -> Maybe String
normalizedStatus mRaw =
    case fmap (normalizeKey . trim) mRaw of
        Just "" -> Nothing
        other -> other

statusHasNoFill :: String -> Bool
statusHasNoFill s =
    s == "new"
        || s == "canceled"
        || s == "cancelled"
        || s == "pendingcancel"
        || s == "rejected"
        || s == "expired"
        || s == "expiredinmatch"
        || s == "failed"
        || s == "error"
        || s == "notsent"

statusImpliesFilled :: String -> Bool
statusImpliesFilled s =
    s == "filled"
        || s == "done"
        || s == "closed"
        || s == "complete"
        || s == "completed"
        || s == "success"

positiveFinite :: Double -> Maybe Double
positiveFinite x =
    if isNaN x || isInfinite x || x <= 0
        then Nothing
        else Just x
