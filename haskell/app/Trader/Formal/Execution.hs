module Trader.Formal.Execution (
    ExecutionVerificationReport (..),
    applyExecutedQuantitySpec,
    applyReduceOnlyExecutedQuantitySpec,
    orderAppliedQuantitySpec,
    verifyFormalExecution,
) where

import Data.Char (toLower)
import Data.Maybe (isNothing)
import Trader.OrderExecution (
    OrderExecutionEvidence (..),
    applyExecutedQuantity,
    applyReduceOnlyExecutedQuantity,
    orderAppliedQuantity,
 )
import Trader.Types.Safe (NonNegative (..), Quantity (..), sanitizeNonNegative)

-- ---------------------------------------------------------------------------
-- Spec-level definitions (naive, obviously correct)
-- ---------------------------------------------------------------------------

{- | Spec version of 'applyExecutedQuantity'.

Invariant: position after fill is the algebraic sum of previous signed
exposure and signed fill quantity.  Close quantity is the amount that
reduces the existing position; open quantity is whatever remains.
-}
applyExecutedQuantitySpec ::
    Int -> Double -> Bool -> Double -> (Int, Double, Double, Double)
applyExecutedQuantitySpec prevPos prevSize isBuy qtyRaw =
    let eps = 1e-9
        qty = max 0 (if isNaN qtyRaw || isInfinite qtyRaw then 0 else qtyRaw)
        qtySanitized = if qty <= eps then 0 else qty
        prevSign = signum prevPos
        currentSigned = fromIntegral prevSign * max 0 (if isNaN prevSize || isInfinite prevSize then 0 else prevSize)
        deltaSigned = if isBuy then qtySanitized else negate qtySanitized
        newSigned = currentSigned + deltaSigned
        posNew
            | newSigned > eps = 1
            | newSigned < negate eps = -1
            | otherwise = 0
        sizeNew = if posNew == 0 then 0 else abs newSigned
        closeQtyRaw
            | qtySanitized <= 0 = 0
            | abs currentSigned <= eps = 0
            | currentSigned * deltaSigned >= 0 = 0
            | otherwise = min (abs currentSigned) qtySanitized
        closeQty = if closeQtyRaw <= eps then 0 else closeQtyRaw
        openQtyRaw = max 0 (qtySanitized - closeQty)
        openQty = if openQtyRaw <= eps || posNew == 0 then 0 else openQtyRaw
     in (posNew, sizeNew, closeQty, openQty)

{- | Spec version of 'applyReduceOnlyExecutedQuantity'.

Invariant: reduce-only orders may only decrease the magnitude of an
existing position.  They can never increase it, flip it, or reopen it.
-}
applyReduceOnlyExecutedQuantitySpec ::
    Int -> Double -> Double -> (Int, Double, Double, Double)
applyReduceOnlyExecutedQuantitySpec prevPos prevSize qtyRaw =
    let eps = 1e-9
        qty = max 0 (if isNaN qtyRaw || isInfinite qtyRaw then 0 else qtyRaw)
        qtySanitized = if qty <= eps then 0 else qty
        prevSign = signum prevPos
        currentSize =
            if prevSign == 0
                then 0
                else max 0 (if isNaN prevSize || isInfinite prevSize then 0 else prevSize)
        closeQtyRaw
            | qtySanitized <= 0 = 0
            | currentSize <= eps = 0
            | otherwise = min currentSize qtySanitized
        closeQty = if closeQtyRaw <= eps then 0 else closeQtyRaw
        sizeNewRaw = max 0 (currentSize - closeQty)
        sizeNew = if sizeNewRaw <= eps then 0 else sizeNewRaw
        posNew
            | sizeNew <= 0 = 0
            | otherwise = prevSign
     in (posNew, sizeNew, closeQty, 0)

{- | Spec version of 'orderAppliedQuantity'.

Invariant: if the order was not sent, no quantity is applied.
If it is not live, use the fallback.
Otherwise trust executed quantity first, then infer from status.
-}
orderAppliedQuantitySpec :: OrderExecutionEvidence -> Double -> Maybe Double
orderAppliedQuantitySpec ev fallbackQty =
    let fallback
            | isNaN fallbackQty || isInfinite fallbackQty = Nothing
            | fallbackQty > 0 = Just fallbackQty
            | otherwise = Nothing
        executed = oeeExecutedQty ev >>= \q -> if isNaN q || isInfinite q || q <= 0 then Nothing else Just q
        status = oeeStatus ev >>= normalize
        normalize s =
            let s' = filter (/= ' ') (map toLower s)
             in if null s' then Nothing else Just s'
        statusHasNoFill' s = s `elem` noFillList
        noFillList =
            [ "new"
            , "canceled"
            , "cancelled"
            , "pendingcancel"
            , "rejected"
            , "expired"
            , "expiredinmatch"
            , "failed"
            , "error"
            , "notsent"
            ]
        statusImpliesFilled' s = s `elem` fillList
        fillList = ["filled", "done", "closed", "complete", "completed", "success"]
     in if not (oeeSent ev)
            then Nothing
            else
                if not (oeeLive ev)
                    then fallback
                    else case executed of
                        Just q -> Just q
                        Nothing ->
                            case status of
                                Just s' | statusHasNoFill' s' -> Nothing
                                Just s' | statusImpliesFilled' s' -> fallback
                                _ -> Nothing

-- ---------------------------------------------------------------------------
-- Verification report
-- ---------------------------------------------------------------------------

data ExecutionVerificationReport = ExecutionVerificationReport
    { fvrExecImplMatchesSpec :: !Bool
    , fvrExecReduceOnlyImplMatchesSpec :: !Bool
    , fvrExecOrderAppliedImplMatchesSpec :: !Bool
    , fvrExecReduceOnlyNeverIncreases :: !Bool
    , fvrExecReduceOnlyNeverFlips :: !Bool
    , fvrExecQtyConservation :: !Bool
    , fvrExecCloseQtyMonotone :: !Bool
    , fvrExecOpenQtyMonotone :: !Bool
    }
    deriving (Eq, Show)

{- | Exhaustive enumeration over a bounded grid of inputs.
The implementation in 'Trader.OrderExecution' uses epsilon-aware
arithmetic; the spec uses naive guards.  We verify equivalence on
a dense enough grid to catch drift.
-}
verifyFormalExecution :: ExecutionVerificationReport
verifyFormalExecution =
    let positions = [-2, -1, 0, 1, 2]
        sizes = [0, 1e-12, 1e-9, 1e-6, 0.5, 1, 10, 1e3]
        qtys = [negate (1 / 0), -1, 0, 1e-12, 1e-9, 0.5, 1, 3, 10, 1 / 0]
        sides = [False, True]

        implMatchSpec =
            all
                ( \(prevPos, prevSize, isBuy, qty) ->
                    let spec = applyExecutedQuantitySpec prevPos prevSize isBuy qty
                        impl = applyExecutedQuantity prevPos prevSize isBuy qty
                     in positionsClose spec impl
                )
                [ (p, s, b, q) | p <- positions, s <- sizes, b <- sides, q <- qtys
                ]

        reduceOnlyImplMatchSpec =
            all
                ( \(prevPos, prevSize, qty) ->
                    let spec = applyReduceOnlyExecutedQuantitySpec prevPos prevSize qty
                        impl = applyReduceOnlyExecutedQuantity prevPos prevSize qty
                     in positionsClose spec impl
                )
                [ (p, s, q) | p <- positions, s <- sizes, q <- qtys
                ]

        orderAppliedImplMatchSpec =
            all
                ( \(ev, fb) ->
                    orderAppliedQuantity ev fb == orderAppliedQuantitySpec ev fb
                )
                [ (ev, fb)
                | sent <- [False, True]
                , live <- [False, True]
                , status <- [Nothing, Just "filled", Just "canceled", Just "", Just "done", Just "error"]
                , executed <- [Nothing, Just 0, Just (-1), Just (0 / 0), Just 0.5, Just 10]
                , let ev =
                        OrderExecutionEvidence
                            { oeeSent = sent
                            , oeeLive = live
                            , oeeStatus = status
                            , oeeExecutedQty = executed
                            }
                , fb <- [negate (1 / 0), 0, 0.5, 1 / 0]
                ]

        reduceOnlyNeverIncreases =
            all
                ( \(prevPos, prevSize, qty) ->
                    let (_, sizeNew, _, _) = applyReduceOnlyExecutedQuantity prevPos prevSize qty
                        prevSizeSanitized = max 0 (if isNaN prevSize || isInfinite prevSize then 0 else prevSize)
                     in sizeNew >= 0 && sizeNew <= prevSizeSanitized
                )
                [ (p, s, q) | p <- positions, s <- sizes, q <- qtys
                ]

        reduceOnlyNeverFlips =
            all
                ( \(prevPos, prevSize, qty) ->
                    let (posNew, _, _, _) = applyReduceOnlyExecutedQuantity prevPos prevSize qty
                        prevSign = signum prevPos
                     in posNew == 0 || posNew == prevSign
                )
                [ (p, s, q) | p <- positions, s <- sizes, q <- qtys
                ]

        qtyConservation =
            all
                ( \(prevPos, prevSize, isBuy, qty) ->
                    let (_, _, closeQty, openQty) = applyExecutedQuantity prevPos prevSize isBuy qty
                        qtySanitized =
                            let q = max 0 (if isNaN qty || isInfinite qty then 0 else qty)
                                eps = 1e-9
                             in if q <= eps then 0 else q
                        total = closeQty + openQty
                     in -- Allow epsilon slippage because of multiple rounding steps
                        abs (total - qtySanitized) <= 1e-8
                )
                [ (p, s, b, q) | p <- positions, s <- sizes, b <- sides, q <- qtys
                ]

        closeQtyMonotone =
            all
                ( \(prevPos, prevSize, isBuy) ->
                    let qtysAsc = [0, 1e-12, 1e-9, 0.001, 0.5, 1, 2]
                        results = map (\q -> let (_, _, c, _) = applyExecutedQuantity prevPos prevSize isBuy q in c) qtysAsc
                     in and (zipWith (<=) results (drop 1 results))
                )
                [ (p, s, b) | p <- positions, s <- sizes, b <- sides
                ]

        openQtyMonotone =
            all
                ( \(prevPos, prevSize, isBuy) ->
                    let qtysAsc = [0, 1e-12, 1e-9, 0.001, 0.5, 1, 2]
                        results = map (\q -> let (_, _, _, o) = applyExecutedQuantity prevPos prevSize isBuy q in o) qtysAsc
                     in and (zipWith (<=) results (drop 1 results))
                )
                [ (p, s, b) | p <- positions, s <- sizes, b <- sides
                ]
     in ExecutionVerificationReport
            { fvrExecImplMatchesSpec = implMatchSpec
            , fvrExecReduceOnlyImplMatchesSpec = reduceOnlyImplMatchSpec
            , fvrExecOrderAppliedImplMatchesSpec = orderAppliedImplMatchSpec
            , fvrExecReduceOnlyNeverIncreases = reduceOnlyNeverIncreases
            , fvrExecReduceOnlyNeverFlips = reduceOnlyNeverFlips
            , fvrExecQtyConservation = qtyConservation
            , fvrExecCloseQtyMonotone = closeQtyMonotone
            , fvrExecOpenQtyMonotone = openQtyMonotone
            }

-- | Compare two position tuples up to epsilon.
positionsClose :: (Int, Double, Double, Double) -> (Int, Double, Double, Double) -> Bool
positionsClose (p1, s1, c1, o1) (p2, s2, c2, o2) =
    p1 == p2
        && closeEnough s1 s2
        && closeEnough c1 c2
        && closeEnough o1 o2

closeEnough :: Double -> Double -> Bool
closeEnough a b
    | isNaN a && isNaN b = True
    | otherwise = abs (a - b) <= max 1e-9 (abs a * 1e-12)
