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
    | MethodRiskParityBlend
    | MethodConsensusBoost
    | MethodAnchorBlend
    | MethodTensionGate
    | MethodEntropyBlend
    | MethodCoherenceGate
    | MethodFractalBlend
    | MethodPhaseCancel
    | MethodSoftmaxBlend
    | MethodNetSoftmaxBlend
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
        MethodRiskParityBlend -> "risk_parity_blend"
        MethodConsensusBoost -> "consensus_boost"
        MethodAnchorBlend -> "anchor_blend"
        MethodTensionGate -> "tension_gate"
        MethodEntropyBlend -> "entropy_blend"
        MethodCoherenceGate -> "coherence_gate"
        MethodFractalBlend -> "fractal_blend"
        MethodPhaseCancel -> "phase_cancel"
        MethodSoftmaxBlend -> "softmax_blend"
        MethodNetSoftmaxBlend -> "net_softmax_blend"
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
        "risk_parity_blend" -> Right MethodRiskParityBlend
        "risk-parity-blend" -> Right MethodRiskParityBlend
        "riskparityblend" -> Right MethodRiskParityBlend
        "inverse_edge_blend" -> Right MethodRiskParityBlend
        "inverse-edge-blend" -> Right MethodRiskParityBlend
        "inverseedgeblend" -> Right MethodRiskParityBlend
        "consensus_boost" -> Right MethodConsensusBoost
        "consensus-boost" -> Right MethodConsensusBoost
        "consensusboost" -> Right MethodConsensusBoost
        "agreement_boost" -> Right MethodConsensusBoost
        "agreement-boost" -> Right MethodConsensusBoost
        "agreementboost" -> Right MethodConsensusBoost
        "anchor_blend" -> Right MethodAnchorBlend
        "anchor-blend" -> Right MethodAnchorBlend
        "anchorblend" -> Right MethodAnchorBlend
        "shrink_blend" -> Right MethodAnchorBlend
        "shrink-blend" -> Right MethodAnchorBlend
        "shrinkblend" -> Right MethodAnchorBlend
        "tension_gate" -> Right MethodTensionGate
        "tension-gate" -> Right MethodTensionGate
        "tensiongate" -> Right MethodTensionGate
        "conflict_damp" -> Right MethodTensionGate
        "conflict-damp" -> Right MethodTensionGate
        "conflictdamp" -> Right MethodTensionGate
        "entropy_blend" -> Right MethodEntropyBlend
        "entropy-blend" -> Right MethodEntropyBlend
        "entropyblend" -> Right MethodEntropyBlend
        "info_blend" -> Right MethodEntropyBlend
        "info-blend" -> Right MethodEntropyBlend
        "infoblend" -> Right MethodEntropyBlend
        "coherence_gate" -> Right MethodCoherenceGate
        "coherence-gate" -> Right MethodCoherenceGate
        "coherencegate" -> Right MethodCoherenceGate
        "phase_lock" -> Right MethodCoherenceGate
        "phase-lock" -> Right MethodCoherenceGate
        "phaselock" -> Right MethodCoherenceGate
        "fractal_blend" -> Right MethodFractalBlend
        "fractal-blend" -> Right MethodFractalBlend
        "fractalblend" -> Right MethodFractalBlend
        "root_blend" -> Right MethodFractalBlend
        "root-blend" -> Right MethodFractalBlend
        "rootblend" -> Right MethodFractalBlend
        "phase_cancel" -> Right MethodPhaseCancel
        "phase-cancel" -> Right MethodPhaseCancel
        "phasecancel" -> Right MethodPhaseCancel
        "wave_cancel" -> Right MethodPhaseCancel
        "wave-cancel" -> Right MethodPhaseCancel
        "wavecancel" -> Right MethodPhaseCancel
        "softmax_blend" -> Right MethodSoftmaxBlend
        "softmax-blend" -> Right MethodSoftmaxBlend
        "softmaxblend" -> Right MethodSoftmaxBlend
        "exp_blend" -> Right MethodSoftmaxBlend
        "exp-blend" -> Right MethodSoftmaxBlend
        "expblend" -> Right MethodSoftmaxBlend
        "net_softmax_blend" -> Right MethodNetSoftmaxBlend
        "net-softmax-blend" -> Right MethodNetSoftmaxBlend
        "netsoftmaxblend" -> Right MethodNetSoftmaxBlend
        "cost_softmax_blend" -> Right MethodNetSoftmaxBlend
        "cost-softmax-blend" -> Right MethodNetSoftmaxBlend
        "costsoftmaxblend" -> Right MethodNetSoftmaxBlend
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
                    ++ " (expected 11|both, 10|kalman, 01|lstm, blend, conf_blend, conf_pick, cost_pick, harmonic_blend, disagreement_guard, median_blend, neutral_guard, risk_parity_blend, consensus_boost, anchor_blend, tension_gate, entropy_blend, coherence_gate, fractal_blend, phase_cancel, softmax_blend, net_softmax_blend, edge_blend, edge_pick, geo_blend, regime_switch, router, bandit_router)"
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
        MethodRiskParityBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodConsensusBoost ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodAnchorBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodTensionGate ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodEntropyBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodCoherenceGate ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodFractalBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodPhaseCancel ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodSoftmaxBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodNetSoftmaxBlend ->
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
