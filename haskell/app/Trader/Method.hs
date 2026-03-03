module Trader.Method (
    Method (..),
    methodCode,
    parseMethod,
    runtimeMethod,
    selectPredictions,
) where

import Data.Char (isSpace, toLower)

data Method
    = MethodBoth
    | MethodKalmanOnly
    | MethodKalmanPhysicsError
    | MethodLstmOnly
    | MethodBlend
    | MethodConfBlend
    | MethodConfPick
    | MethodConformalClip
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
    | MethodDivergenceGate
    | MethodFractalBlend
    | MethodPhaseCancel
    | MethodSoftmaxBlend
    | MethodSmoothSoftmaxBlend
    | MethodHedgeBlend
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
        MethodKalmanPhysicsError -> "kalman_physics_error"
        MethodLstmOnly -> "01"
        MethodBlend -> "blend"
        MethodConfBlend -> "conf_blend"
        MethodConfPick -> "conf_pick"
        MethodConformalClip -> "conformal_clip"
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
        MethodDivergenceGate -> "divergence_gate"
        MethodFractalBlend -> "fractal_blend"
        MethodPhaseCancel -> "phase_cancel"
        MethodSoftmaxBlend -> "softmax_blend"
        MethodSmoothSoftmaxBlend -> "smooth_softmax_blend"
        MethodHedgeBlend -> "hedge_blend"
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
        "kalman_physics_error" -> Right MethodKalmanPhysicsError
        "kalman-physics-error" -> Right MethodKalmanPhysicsError
        "kalmanphysicserror" -> Right MethodKalmanPhysicsError
        "physics_error" -> Right MethodKalmanPhysicsError
        "physics-error" -> Right MethodKalmanPhysicsError
        "physicserror" -> Right MethodKalmanPhysicsError
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
        "conformal_clip" -> Right MethodConformalClip
        "conformal-clip" -> Right MethodConformalClip
        "conformalclip" -> Right MethodConformalClip
        "interval_clip" -> Right MethodConformalClip
        "interval-clip" -> Right MethodConformalClip
        "intervalclip" -> Right MethodConformalClip
        "uncertainty_clip" -> Right MethodConformalClip
        "uncertainty-clip" -> Right MethodConformalClip
        "uncertaintyclip" -> Right MethodConformalClip
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
        "divergence_gate" -> Right MethodDivergenceGate
        "divergence-gate" -> Right MethodDivergenceGate
        "divergencegate" -> Right MethodDivergenceGate
        "dispersion_gate" -> Right MethodDivergenceGate
        "dispersion-gate" -> Right MethodDivergenceGate
        "dispersiongate" -> Right MethodDivergenceGate
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
        "smooth_softmax_blend" -> Right MethodSmoothSoftmaxBlend
        "smooth-softmax-blend" -> Right MethodSmoothSoftmaxBlend
        "smoothsoftmaxblend" -> Right MethodSmoothSoftmaxBlend
        "ema_softmax_blend" -> Right MethodSmoothSoftmaxBlend
        "ema-softmax-blend" -> Right MethodSmoothSoftmaxBlend
        "emasoftmaxblend" -> Right MethodSmoothSoftmaxBlend
        "hedge_blend" -> Right MethodHedgeBlend
        "hedge-blend" -> Right MethodHedgeBlend
        "hedgeblend" -> Right MethodHedgeBlend
        "exp_weights_blend" -> Right MethodHedgeBlend
        "exp-weights-blend" -> Right MethodHedgeBlend
        "expweightsblend" -> Right MethodHedgeBlend
        "online_blend" -> Right MethodHedgeBlend
        "online-blend" -> Right MethodHedgeBlend
        "onlineblend" -> Right MethodHedgeBlend
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
                    ++ " (expected 11|both, 10|kalman, kalman_physics_error, 01|lstm, blend, conf_blend, conf_pick, conformal_clip, cost_pick, harmonic_blend, disagreement_guard, median_blend, neutral_guard, risk_parity_blend, consensus_boost, anchor_blend, tension_gate, entropy_blend, coherence_gate, divergence_gate, fractal_blend, phase_cancel, softmax_blend, smooth_softmax_blend, hedge_blend, net_softmax_blend, edge_blend, edge_pick, geo_blend, regime_switch, router, bandit_router)"
                )

runtimeMethod :: Method -> Method
runtimeMethod m =
    case m of
        MethodKalmanPhysicsError -> MethodKalmanOnly
        _ -> m

selectPredictions :: Method -> Double -> [Double] -> [Double] -> ([Double], [Double])
selectPredictions m blendWeight kalPred lstmPred =
    case runtimeMethod m of
        MethodBoth -> (kalPred, lstmPred)
        MethodKalmanOnly -> (kalPred, kalPred)
        MethodLstmOnly -> (lstmPred, lstmPred)
        MethodBlend -> blendedPredictions
        MethodConfBlend -> blendedPredictions
        MethodConfPick -> blendedPredictions
        MethodConformalClip -> blendedPredictions
        MethodCostPick -> blendedPredictions
        MethodHarmonicBlend -> blendedPredictions
        MethodDisagreementGuard -> blendedPredictions
        MethodMedianBlend -> blendedPredictions
        MethodNeutralGuard -> blendedPredictions
        MethodRiskParityBlend -> blendedPredictions
        MethodConsensusBoost -> blendedPredictions
        MethodAnchorBlend -> blendedPredictions
        MethodTensionGate -> blendedPredictions
        MethodEntropyBlend -> blendedPredictions
        MethodCoherenceGate -> blendedPredictions
        MethodDivergenceGate -> blendedPredictions
        MethodFractalBlend -> blendedPredictions
        MethodPhaseCancel -> blendedPredictions
        MethodSoftmaxBlend -> blendedPredictions
        MethodSmoothSoftmaxBlend -> blendedPredictions
        MethodHedgeBlend -> blendedPredictions
        MethodNetSoftmaxBlend -> blendedPredictions
        MethodEdgeBlend -> blendedPredictions
        MethodEdgePick -> blendedPredictions
        MethodGeoBlend -> blendedPredictions
        MethodRegimeSwitch -> regimeSwitchedPredictions
        MethodRouter -> (kalPred, lstmPred)
        MethodBanditRouter -> (kalPred, lstmPred)
  where
    w = clamp01 blendWeight
    blendedPredictions = (blendPredictions w kalPred lstmPred, blendPredictions w kalPred lstmPred)
    regimeSwitchedPredictions = (regimeSwitchPredictions w kalPred lstmPred, regimeSwitchPredictions w kalPred lstmPred)

clamp01 :: Double -> Double
clamp01 x = max 0 (min 1 x)

blendPredictions :: Double -> [Double] -> [Double] -> [Double]
blendPredictions w = zipWith (\k l -> w * k + (1 - w) * l)

regimeSwitchPredictions :: Double -> [Double] -> [Double] -> [Double]
regimeSwitchPredictions w kalPred lstmPred =
    case zip kalPred lstmPred of
        [] -> []
        _ : pairs ->
            let baseBlend = blendPredictions w kalPred lstmPred
                stepped = zip3 pairs (zip kalPred (drop 1 kalPred)) (zip lstmPred (drop 1 lstmPred))
             in case baseBlend of
                    [] -> []
                    b0 : rest ->
                        b0
                            : zipWith applyRegime rest stepped
  where
    applyRegime :: Double -> ((Double, Double), (Double, Double), (Double, Double)) -> Double
    applyRegime blended ((k, l), (kPrev, _), (lPrev, _))
        | sameMomentumDirection kMomentum lMomentum = trendBlend
        | isStrongDivergence k l = (k + l) / 2
        | otherwise = blended
      where
        kMomentum = k - kPrev
        lMomentum = l - lPrev
        trendBlend = weightedByMagnitude (abs kMomentum) (abs lMomentum) k l

    weightedByMagnitude :: Double -> Double -> Double -> Double -> Double
    weightedByMagnitude kMag lMag k l
        | kMag + lMag <= 1.0e-12 = (k + l) / 2
        | otherwise =
            let kalWeight = kMag / (kMag + lMag)
             in kalWeight * k + (1 - kalWeight) * l

    sameMomentumDirection :: Double -> Double -> Bool
    sameMomentumDirection a b = signWithTolerance a == signWithTolerance b && signWithTolerance a /= 0

    signWithTolerance :: Double -> Int
    signWithTolerance x
        | x > 1.0e-9 = 1
        | x < -1.0e-9 = -1
        | otherwise = 0

    isStrongDivergence :: Double -> Double -> Bool
    isStrongDivergence k l =
        let denom = max 1 (max (abs k) (abs l))
         in abs (k - l) / denom >= 0.02

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse
