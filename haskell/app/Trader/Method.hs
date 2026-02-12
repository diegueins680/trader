module Trader.Method (
    Method (..),
    methodCode,
    parseMethod,
    selectPredictions,
) where

import Data.Char (isSpace, toLower)

data Method
    = MethodBoth
    | MethodKalmanOnly
    | MethodLstmOnly
    | MethodBlend
    | MethodConfBlend
    | MethodConfPick
    | MethodCostPick
    | MethodHarmonicBlend
    | MethodDisagreementGuard
    | MethodMedianBlend
    | MethodNeutralGuard
    | MethodEdgeBlend
    | MethodEdgePick
    | MethodGeoBlend
    | MethodRegimeSwitch
    | MethodRouter
    | MethodBanditRouter
    deriving (Eq, Show)

methodCode :: Method -> String
methodCode m =
    case m of
        MethodBoth -> "11"
        MethodKalmanOnly -> "10"
        MethodLstmOnly -> "01"
        MethodBlend -> "blend"
        MethodConfBlend -> "conf_blend"
        MethodConfPick -> "conf_pick"
        MethodCostPick -> "cost_pick"
        MethodHarmonicBlend -> "harmonic_blend"
        MethodDisagreementGuard -> "disagreement_guard"
        MethodMedianBlend -> "median_blend"
        MethodNeutralGuard -> "neutral_guard"
        MethodEdgeBlend -> "edge_blend"
        MethodEdgePick -> "edge_pick"
        MethodGeoBlend -> "geo_blend"
        MethodRegimeSwitch -> "regime_switch"
        MethodRouter -> "router"
        MethodBanditRouter -> "bandit_router"

parseMethod :: String -> Either String Method
parseMethod raw =
    case map toLower (trim raw) of
        "11" -> Right MethodBoth
        "both" -> Right MethodBoth
        "ensemble" -> Right MethodBoth
        "agreement" -> Right MethodBoth
        "gated" -> Right MethodBoth
        "kalman+lstm" -> Right MethodBoth
        "lstm+kalman" -> Right MethodBoth
        "10" -> Right MethodKalmanOnly
        "kalman" -> Right MethodKalmanOnly
        "kalman-only" -> Right MethodKalmanOnly
        "kalman_only" -> Right MethodKalmanOnly
        "kalmanonly" -> Right MethodKalmanOnly
        "01" -> Right MethodLstmOnly
        "lstm" -> Right MethodLstmOnly
        "lstm-only" -> Right MethodLstmOnly
        "lstm_only" -> Right MethodLstmOnly
        "lstmonly" -> Right MethodLstmOnly
        "blend" -> Right MethodBlend
        "avg" -> Right MethodBlend
        "average" -> Right MethodBlend
        "mix" -> Right MethodBlend
        "weighted" -> Right MethodBlend
        "12" -> Right MethodBlend
        "conf_blend" -> Right MethodConfBlend
        "conf-blend" -> Right MethodConfBlend
        "confblend" -> Right MethodConfBlend
        "adaptive_blend" -> Right MethodConfBlend
        "adaptive-blend" -> Right MethodConfBlend
        "adaptiveblend" -> Right MethodConfBlend
        "conf_pick" -> Right MethodConfPick
        "conf-pick" -> Right MethodConfPick
        "confpick" -> Right MethodConfPick
        "confidence_pick" -> Right MethodConfPick
        "confidence-pick" -> Right MethodConfPick
        "confidencepick" -> Right MethodConfPick
        "cost_pick" -> Right MethodCostPick
        "cost-pick" -> Right MethodCostPick
        "costpick" -> Right MethodCostPick
        "netedge_pick" -> Right MethodCostPick
        "netedge-pick" -> Right MethodCostPick
        "netedgepick" -> Right MethodCostPick
        "harmonic_blend" -> Right MethodHarmonicBlend
        "harmonic-blend" -> Right MethodHarmonicBlend
        "harmonicblend" -> Right MethodHarmonicBlend
        "harmonic_mix" -> Right MethodHarmonicBlend
        "harmonic-mix" -> Right MethodHarmonicBlend
        "harmonicmix" -> Right MethodHarmonicBlend
        "disagreement_guard" -> Right MethodDisagreementGuard
        "disagreement-guard" -> Right MethodDisagreementGuard
        "disagreementguard" -> Right MethodDisagreementGuard
        "conflict_guard" -> Right MethodDisagreementGuard
        "conflict-guard" -> Right MethodDisagreementGuard
        "conflictguard" -> Right MethodDisagreementGuard
        "median_blend" -> Right MethodMedianBlend
        "median-blend" -> Right MethodMedianBlend
        "medianblend" -> Right MethodMedianBlend
        "robust_blend" -> Right MethodMedianBlend
        "robust-blend" -> Right MethodMedianBlend
        "robustblend" -> Right MethodMedianBlend
        "neutral_guard" -> Right MethodNeutralGuard
        "neutral-guard" -> Right MethodNeutralGuard
        "neutralguard" -> Right MethodNeutralGuard
        "conflict_neutral" -> Right MethodNeutralGuard
        "conflict-neutral" -> Right MethodNeutralGuard
        "conflictneutral" -> Right MethodNeutralGuard
        "edge_blend" -> Right MethodEdgeBlend
        "edge-blend" -> Right MethodEdgeBlend
        "edgeblend" -> Right MethodEdgeBlend
        "edge_mix" -> Right MethodEdgeBlend
        "edge-mix" -> Right MethodEdgeBlend
        "edgemix" -> Right MethodEdgeBlend
        "edge_pick" -> Right MethodEdgePick
        "edge-pick" -> Right MethodEdgePick
        "edgepick" -> Right MethodEdgePick
        "edge_select" -> Right MethodEdgePick
        "edge-select" -> Right MethodEdgePick
        "edgeselect" -> Right MethodEdgePick
        "geo_blend" -> Right MethodGeoBlend
        "geo-blend" -> Right MethodGeoBlend
        "geoblend" -> Right MethodGeoBlend
        "geometric_blend" -> Right MethodGeoBlend
        "geometric-blend" -> Right MethodGeoBlend
        "geometricblend" -> Right MethodGeoBlend
        "regime_switch" -> Right MethodRegimeSwitch
        "regime-switch" -> Right MethodRegimeSwitch
        "regimeswitch" -> Right MethodRegimeSwitch
        "regime_router" -> Right MethodRegimeSwitch
        "regime-router" -> Right MethodRegimeSwitch
        "regimerouter" -> Right MethodRegimeSwitch
        "router" -> Right MethodRouter
        "route" -> Right MethodRouter
        "adaptive" -> Right MethodRouter
        "auto" -> Right MethodRouter
        "bandit_router" -> Right MethodBanditRouter
        "bandit-router" -> Right MethodBanditRouter
        "banditrouter" -> Right MethodBanditRouter
        "ucb_router" -> Right MethodBanditRouter
        "ucb-router" -> Right MethodBanditRouter
        "ucbrouter" -> Right MethodBanditRouter
        other ->
            Left
                ( "Invalid --method: "
                    ++ show other
                    ++ " (expected 11|both, 10|kalman, 01|lstm, blend, conf_blend, conf_pick, cost_pick, harmonic_blend, disagreement_guard, median_blend, neutral_guard, edge_blend, edge_pick, geo_blend, regime_switch, router, bandit_router)"
                )

selectPredictions :: Method -> Double -> [Double] -> [Double] -> ([Double], [Double])
selectPredictions m blendWeight kalPred lstmPred =
    case m of
        MethodBoth -> (kalPred, lstmPred)
        MethodKalmanOnly -> (kalPred, kalPred)
        MethodLstmOnly -> (lstmPred, lstmPred)
        MethodBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodConfBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodConfPick ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodCostPick ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodHarmonicBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodDisagreementGuard ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodMedianBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodNeutralGuard ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodEdgeBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodEdgePick ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodGeoBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodRegimeSwitch ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodRouter -> (kalPred, lstmPred)
        MethodBanditRouter -> (kalPred, lstmPred)
  where
    clamp01 x = max 0 (min 1 x)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse
