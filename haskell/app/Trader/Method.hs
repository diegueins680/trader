module Trader.Method (
    Method (..),
    methodCode,
    methodIsTechnicalAnalysis,
    parseMethod,
    runtimeMethod,
    selectPredictions,
) where

import Data.Char (isSpace, toLower)

{- | Floor for the sum of Kalman and LSTM momentum magnitudes below which both
  are treated as zero (dimensionless; avoids division by near-zero values).
-}
momentumMagnitudeFloor :: Double
momentumMagnitudeFloor = 1.0e-12

{- | Minimum absolute momentum (price-units/step) for a reading to register as
  a non-zero direction in the sign test.
-}
momentumSignTolerance :: Double
momentumSignTolerance = 1.0e-9

{- | Relative divergence fraction above which the two predictors are considered
  to be in "strong divergence" (0.02 == 2 %).
-}
regimeDivergenceThreshold :: Double
regimeDivergenceThreshold = 0.02

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
    | MethodCrossSectionalMomentum
    | MethodOnlineNeural
    | MethodTaTrend
    | MethodTaReversion
    | MethodTaBreakout
    | MethodTaBest
    | MethodTaRegimeSwitch
    | MethodSmaCross
    | MethodSmaCrossRegime
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
        MethodCrossSectionalMomentum -> "cross_sectional_momentum"
        MethodOnlineNeural -> "online_nn"
        MethodTaTrend -> "ta_trend"
        MethodTaReversion -> "ta_reversion"
        MethodTaBreakout -> "ta_breakout"
        MethodTaBest -> "ta_best"
        MethodTaRegimeSwitch -> "ta_regime_switch"
        MethodSmaCross -> "sma_cross"
        MethodSmaCrossRegime -> "sma_cross_regime"

methodIsTechnicalAnalysis :: Method -> Bool
methodIsTechnicalAnalysis m =
    case runtimeMethod m of
        MethodTaTrend -> True
        MethodTaReversion -> True
        MethodTaBreakout -> True
        MethodTaBest -> True
        MethodTaRegimeSwitch -> True
        MethodSmaCross -> True
        MethodSmaCrossRegime -> True
        _ -> False

parseMethod :: String -> Either String Method
parseMethod raw =
    case map toLower (trim raw) of
        "11" -> Left "Method '11' (Kalman+LSTM ensemble) is disabled: LSTM initialization hangs indefinitely. Use '01' (LSTM-only), '10' (Kalman-only), or 'blend'."
        "both" -> Left "Method 'both' (Kalman+LSTM ensemble) is disabled: LSTM initialization hangs indefinitely. Use '01' (LSTM-only), '10' (Kalman-only), or 'blend'."
        "ensemble" -> Left "Method 'ensemble' (Kalman+LSTM ensemble) is disabled: LSTM initialization hangs indefinitely. Use '01' (LSTM-only), '10' (Kalman-only), or 'blend'."
        "agreement" -> Left "Method 'agreement' (Kalman+LSTM ensemble) is disabled: LSTM initialization hangs indefinitely. Use '01' (LSTM-only), '10' (Kalman-only), or 'blend'."
        "gated" -> Left "Method 'gated' (Kalman+LSTM ensemble) is disabled: LSTM initialization hangs indefinitely. Use '01' (LSTM-only), '10' (Kalman-only), or 'blend'."
        "kalman+lstm" -> Left "Method 'kalman+lstm' (Kalman+LSTM ensemble) is disabled: LSTM initialization hangs indefinitely. Use '01' (LSTM-only), '10' (Kalman-only), or 'blend'."
        "lstm+kalman" -> Left "Method 'lstm+kalman' (Kalman+LSTM ensemble) is disabled: LSTM initialization hangs indefinitely. Use '01' (LSTM-only), '10' (Kalman-only), or 'blend'."
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
        "cross_sectional_momentum" -> Right MethodCrossSectionalMomentum
        "cross-sectional-momentum" -> Right MethodCrossSectionalMomentum
        "crosssectionalmomentum" -> Right MethodCrossSectionalMomentum
        "xsmom" -> Right MethodCrossSectionalMomentum
        "relative_momentum" -> Right MethodCrossSectionalMomentum
        "relative-momentum" -> Right MethodCrossSectionalMomentum
        "relmom" -> Right MethodCrossSectionalMomentum
        "funding_momentum" -> Right MethodCrossSectionalMomentum
        "funding-momentum" -> Right MethodCrossSectionalMomentum
        "online_nn" -> Right MethodOnlineNeural
        "online-nn" -> Right MethodOnlineNeural
        "onlinenn" -> Right MethodOnlineNeural
        "online_neural" -> Right MethodOnlineNeural
        "online-neural" -> Right MethodOnlineNeural
        "onlineneural" -> Right MethodOnlineNeural
        "online_mlp" -> Right MethodOnlineNeural
        "online-mlp" -> Right MethodOnlineNeural
        "onlinemlp" -> Right MethodOnlineNeural
        "deep_online" -> Right MethodOnlineNeural
        "deep-online" -> Right MethodOnlineNeural
        "deeponline" -> Right MethodOnlineNeural
        "ta_trend" -> Right MethodTaTrend
        "ta-trend" -> Right MethodTaTrend
        "tatrend" -> Right MethodTaTrend
        "technical_trend" -> Right MethodTaTrend
        "technical-trend" -> Right MethodTaTrend
        "trend_following" -> Right MethodTaTrend
        "trend-following" -> Right MethodTaTrend
        "ta_reversion" -> Right MethodTaReversion
        "ta-reversion" -> Right MethodTaReversion
        "tareversion" -> Right MethodTaReversion
        "technical_reversion" -> Right MethodTaReversion
        "technical-reversion" -> Right MethodTaReversion
        "momentum_reversion" -> Right MethodTaReversion
        "momentum-reversion" -> Right MethodTaReversion
        "mean_reversion" -> Right MethodTaReversion
        "mean-reversion" -> Right MethodTaReversion
        "ta_breakout" -> Right MethodTaBreakout
        "ta-breakout" -> Right MethodTaBreakout
        "tabreakout" -> Right MethodTaBreakout
        "technical_breakout" -> Right MethodTaBreakout
        "technical-breakout" -> Right MethodTaBreakout
        "volume_breakout" -> Right MethodTaBreakout
        "volume-breakout" -> Right MethodTaBreakout
        "ta_best" -> Right MethodTaBest
        "ta-best" -> Right MethodTaBest
        "tabest" -> Right MethodTaBest
        "ta" -> Right MethodTaBest
        "technical" -> Right MethodTaBest
        "technical_analysis" -> Right MethodTaBest
        "technical-analysis" -> Right MethodTaBest
        "ta_regime_switch" -> Right MethodTaRegimeSwitch
        "ta-regime-switch" -> Right MethodTaRegimeSwitch
        "taregimeswitch" -> Right MethodTaRegimeSwitch
        "technical_regime_switch" -> Right MethodTaRegimeSwitch
        "technical-regime-switch" -> Right MethodTaRegimeSwitch
        "technicalregimeswitch" -> Right MethodTaRegimeSwitch
        "sma_cross" -> Right MethodSmaCross
        "sma-cross" -> Right MethodSmaCross
        "smacross" -> Right MethodSmaCross
        "sma_cross_regime" -> Right MethodSmaCrossRegime
        "sma-cross-regime" -> Right MethodSmaCrossRegime
        "smacrossregime" -> Right MethodSmaCrossRegime
        other ->
            Left
                ( "Invalid --method: "
                    ++ show other
                    ++ " (expected 11|both, 10|kalman, kalman_physics_error, 01|lstm, blend, conf_blend, conf_pick, conformal_clip, cost_pick, harmonic_blend, disagreement_guard, median_blend, neutral_guard, risk_parity_blend, consensus_boost, anchor_blend, tension_gate, entropy_blend, coherence_gate, divergence_gate, fractal_blend, phase_cancel, softmax_blend, smooth_softmax_blend, hedge_blend, net_softmax_blend, edge_blend, edge_pick, geo_blend, regime_switch, router, bandit_router, cross_sectional_momentum|xsmom, online_nn, ta_trend, ta_reversion, ta_breakout, ta_best, ta_regime_switch, sma_cross)"
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
        MethodKalmanPhysicsError -> (kalPred, kalPred)
        MethodLstmOnly -> (lstmPred, lstmPred)
        MethodTaTrend -> (kalPred, kalPred)
        MethodTaReversion -> (kalPred, kalPred)
        MethodTaBreakout -> (kalPred, kalPred)
        MethodTaBest -> (kalPred, kalPred)
        MethodTaRegimeSwitch -> (kalPred, kalPred)
        MethodSmaCross -> (kalPred, kalPred)
        MethodSmaCrossRegime -> (kalPred, kalPred)
        MethodRegimeSwitch -> (regimeSwitched, regimeSwitched)
        MethodRouter -> (kalPred, lstmPred)
        MethodBanditRouter -> (kalPred, lstmPred)
        MethodCrossSectionalMomentum -> (kalPred, kalPred)
        MethodOnlineNeural -> (kalPred, kalPred)
        _ -> (blended, blended)
  where
    w = clamp01 blendWeight
    blended = zipWith (blendPair w) kalPred lstmPred
    regimeSwitched = regimeSwitchPredictions w kalPred lstmPred

    blendPair :: Double -> Double -> Double -> Double
    blendPair weight kal lstm =
        case (isFinite kal, isFinite lstm) of
            (True, True) ->
                let v = weight * kal + (1 - weight) * lstm
                 in if isFinite v then v else if weight >= 0.5 then kal else lstm
            (True, False) -> kal
            (False, True) -> lstm
            (False, False) -> 0

    clamp01 :: Double -> Double
    clamp01 x
        | not (isFinite x) = 0.5
        | otherwise = max 0 (min 1 x)

    regimeSwitchPredictions :: Double -> [Double] -> [Double] -> [Double]
    regimeSwitchPredictions weight = go Nothing
      where
        go :: Maybe (Double, Double) -> [Double] -> [Double] -> [Double]
        go _ [] _ = []
        go _ _ [] = []
        go Nothing (kal : kalRest) (lstm : lstmRest) =
            blendPair weight kal lstm : go (Just (kal, lstm)) kalRest lstmRest
        go (Just (kalPrev, lstmPrev)) (kal : kalRest) (lstm : lstmRest) =
            let blendedCurrent = blendPair weight kal lstm
             in applyRegime blendedCurrent kalPrev lstmPrev kal lstm
                    : go (Just (kal, lstm)) kalRest lstmRest

        applyRegime :: Double -> Double -> Double -> Double -> Double -> Double
        applyRegime blendedCurrent kalPrev lstmPrev kal lstm
            | not (all isFinite [kalPrev, lstmPrev, kal, lstm]) = blendedCurrent
            | sameMomentumDirection kalMomentum lstmMomentum = weightedByMagnitude (abs kalMomentum) (abs lstmMomentum) kal lstm
            | isStrongDivergence kal lstm = midpoint kal lstm
            | otherwise = blendedCurrent
          where
            kalMomentum = kal - kalPrev
            lstmMomentum = lstm - lstmPrev

        weightedByMagnitude :: Double -> Double -> Double -> Double -> Double
        weightedByMagnitude kalMagnitude lstmMagnitude kal lstm
            | kalMagnitude + lstmMagnitude <= momentumMagnitudeFloor = midpoint kal lstm
            | otherwise =
                let kalWeight = kalMagnitude / (kalMagnitude + lstmMagnitude)
                 in kalWeight * kal + (1 - kalWeight) * lstm

        sameMomentumDirection :: Double -> Double -> Bool
        sameMomentumDirection lhs rhs =
            let lhsSign = signWithTolerance lhs
                rhsSign = signWithTolerance rhs
             in lhsSign == rhsSign && lhsSign /= 0

        signWithTolerance :: Double -> Int
        signWithTolerance x
            | x > momentumSignTolerance = 1
            | x < -momentumSignTolerance = -1
            | otherwise = 0

        isStrongDivergence :: Double -> Double -> Bool
        isStrongDivergence kal lstm =
            let denom = max 1 (max (abs kal) (abs lstm))
             in abs (kal - lstm) / denom >= regimeDivergenceThreshold

        midpoint :: Double -> Double -> Double
        midpoint kal lstm = 0.5 * (kal + lstm)

    isFinite :: Double -> Bool
    isFinite x = not (isNaN x || isInfinite x)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse
