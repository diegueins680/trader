module Trader.SignalGates (
    signalMetaLabelOk,
    signalMtfConsensusCheck,
    signalCrossAssetCheck,
    signalRegimeEdgeOk,
    signalFundingOiCheck,
    signalRunPostDirectionGates,
) where

import Data.Maybe (catMaybes)

signalMetaLabelOk ::
    Bool ->
    Double ->
    Maybe Double ->
    Double ->
    Maybe Double ->
    Bool ->
    Bool ->
    Bool
signalMetaLabelOk enabled minEdge edgeForMethod minConfidence methodConfidence requireBand bandAgree =
    not enabled
        || let edgeOk = minEdge <= 0 || maybe False (>= minEdge) edgeForMethod
               confidenceOk = minConfidence <= 0 || maybe False (>= minConfidence) methodConfidence
               bandOk = not requireBand || bandAgree
            in edgeOk && confidenceOk && bandOk

signalMtfConsensusCheck :: Bool -> [Maybe Int] -> Int -> Int -> (Bool, Maybe String)
signalMtfConsensusCheck enabled mtfDirs minAgree dir =
    if not enabled
        then (True, Nothing)
        else
            let available = catMaybes mtfDirs
                agree = length (filter (== dir) available)
             in if length available < minAgree
                    then (False, Just "MTF_WARMUP")
                    else
                        if agree >= minAgree
                            then (True, Nothing)
                            else (False, Just "MTF_CONSENSUS")

signalCrossAssetCheck :: Bool -> Maybe Int -> Int -> (Bool, Maybe String)
signalCrossAssetCheck enabled crossAssetDir dir =
    if not enabled
        then (True, Nothing)
        else case crossAssetDir of
            Nothing -> (False, Just "CROSS_ASSET")
            Just d ->
                if d == dir
                    then (True, Nothing)
                    else (False, Just "CROSS_ASSET")

signalRegimeEdgeOk :: Bool -> Double -> Maybe Double -> Bool
signalRegimeEdgeOk enabled minEdgeRegime edgeForMethod =
    not enabled
        || minEdgeRegime <= 0
        || maybe False (>= minEdgeRegime) edgeForMethod

signalFundingOiCheck ::
    Bool ->
    Maybe Double ->
    Maybe Double ->
    Double ->
    Double ->
    Maybe Double ->
    (Bool, Double)
signalFundingOiCheck enabled fundingCap volCap sizeMult funding oiVolProxy =
    if not enabled
        then (True, 1.0)
        else
            let fundingOk =
                    case fundingCap of
                        Nothing -> True
                        Just cap -> funding <= cap
                volProxyOk =
                    case volCap of
                        Nothing -> True
                        Just cap ->
                            case oiVolProxy of
                                Nothing -> False
                                Just v -> v <= cap
                fundingPenalty =
                    case fundingCap of
                        Just cap
                            | cap > 0 ->
                                max 0 ((funding - cap) / cap)
                        _ -> 0
                volPenalty =
                    case (volCap, oiVolProxy) of
                        (Just cap, Just v)
                            | cap > 0 ->
                                max 0 ((v - cap) / cap)
                        _ -> 0
                dampRaw = 1 / (1 + fundingPenalty + volPenalty)
                damp = max sizeMult (min 1 dampRaw)
             in (fundingOk && volProxyOk, damp)

signalRunPostDirectionGates ::
    Maybe Int ->
    Maybe String ->
    Bool ->
    Bool ->
    (Int -> Bool) ->
    (Int -> Bool) ->
    (Int -> Bool) ->
    Bool ->
    Bool ->
    (Int -> (Bool, Maybe String)) ->
    (Int -> (Bool, Maybe String)) ->
    (Int -> Bool) ->
    (Int -> (Bool, Double)) ->
    (Maybe Int, Maybe String)
signalRunPostDirectionGates chosenDir mPairsOverlayReason volOk volTargetReady trendOk cloudOk priceActionOk signalToNoiseOk regimeEdgeOk mtfConsensusCheck crossAssetCheck metaLabelOk fundingOiCheck =
    case chosenDir of
        Nothing -> (Nothing, mPairsOverlayReason)
        Just dir ->
            if not volOk
                then (Nothing, Just "MAX_VOLATILITY")
                else
                    if not volTargetReady
                        then (Nothing, Just "VOL_TARGET_WARMUP")
                        else
                            if not (trendOk dir)
                                then (Nothing, Just "TREND_FILTER")
                                else
                                    if not (cloudOk dir)
                                        then (Nothing, Just "KALMAN_CLOUD")
                                        else
                                            if not (priceActionOk dir)
                                                then (Nothing, Just "PRICE_ACTION")
                                                else
                                                    if not signalToNoiseOk
                                                        then (Nothing, Just "SIGNAL_TO_NOISE")
                                                        else
                                                            if not regimeEdgeOk
                                                                then (Nothing, Just "REGIME_BANK")
                                                                else
                                                                    let (mtfOk, mMtfReason) = mtfConsensusCheck dir
                                                                     in if not mtfOk
                                                                            then (Nothing, mMtfReason)
                                                                            else
                                                                                let (crossOk, mCrossReason) = crossAssetCheck dir
                                                                                 in if not crossOk
                                                                                        then (Nothing, mCrossReason)
                                                                                        else
                                                                                            if not (metaLabelOk dir)
                                                                                                then (Nothing, Just "META_LABEL")
                                                                                                else
                                                                                                    let (fundingOiOk, _) = fundingOiCheck dir
                                                                                                     in if not fundingOiOk
                                                                                                            then (Nothing, Just "FUNDING_OI")
                                                                                                            else (Just dir, Nothing)
