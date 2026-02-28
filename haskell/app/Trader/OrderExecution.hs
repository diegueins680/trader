module Trader.OrderExecution (
    OrderExecutionEvidence (..),
    orderAppliedQuantity,
    applyExecutedQuantity,
) where

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
                    else case status of
                        Just s | statusHasNoFill s -> Nothing
                        _ ->
                            case executed of
                                Just q -> Just q
                                Nothing ->
                                    case status of
                                        Just s | statusImpliesFilled s -> fallback
                                        _ -> Nothing

applyExecutedQuantity :: Int -> Double -> Bool -> Double -> (Int, Double, Double, Double)
applyExecutedQuantity prevPos prevSize isBuy qtyRaw =
    let qty = maybe 0 id (positiveFinite qtyRaw)
        prevSign = signum prevPos
        currentSigned = fromIntegral prevSign * max 0 prevSize
        deltaSigned = if isBuy then qty else negate qty
        newSigned = currentSigned + deltaSigned
        eps = 1e-9
        posNew
            | newSigned > eps = 1
            | newSigned < negate eps = -1
            | otherwise = 0
        sizeNew =
            if posNew == 0
                then 0
                else abs newSigned
        closeQty =
            if qty <= 0 || abs currentSigned <= eps || currentSigned * deltaSigned >= 0
                then 0
                else min (abs currentSigned) qty
        openQty = max 0 (qty - closeQty)
     in (posNew, sizeNew, closeQty, openQty)

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
