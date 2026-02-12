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
    | MethodEdgeBlend
    | MethodGeoBlend
    | MethodRouter
    deriving (Eq, Show)

methodCode :: Method -> String
methodCode m =
    case m of
        MethodBoth -> "11"
        MethodKalmanOnly -> "10"
        MethodLstmOnly -> "01"
        MethodBlend -> "blend"
        MethodConfBlend -> "conf_blend"
        MethodEdgeBlend -> "edge_blend"
        MethodGeoBlend -> "geo_blend"
        MethodRouter -> "router"

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
        "edge_blend" -> Right MethodEdgeBlend
        "edge-blend" -> Right MethodEdgeBlend
        "edgeblend" -> Right MethodEdgeBlend
        "edge_mix" -> Right MethodEdgeBlend
        "edge-mix" -> Right MethodEdgeBlend
        "edgemix" -> Right MethodEdgeBlend
        "geo_blend" -> Right MethodGeoBlend
        "geo-blend" -> Right MethodGeoBlend
        "geoblend" -> Right MethodGeoBlend
        "geometric_blend" -> Right MethodGeoBlend
        "geometric-blend" -> Right MethodGeoBlend
        "geometricblend" -> Right MethodGeoBlend
        "router" -> Right MethodRouter
        "route" -> Right MethodRouter
        "adaptive" -> Right MethodRouter
        "auto" -> Right MethodRouter
        other ->
            Left
                ( "Invalid --method: "
                    ++ show other
                    ++ " (expected 11|both, 10|kalman, 01|lstm, blend, conf_blend, edge_blend, geo_blend, router)"
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
        MethodEdgeBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodGeoBlend ->
            let w = clamp01 blendWeight
                blend = zipWith (\k l -> w * k + (1 - w) * l) kalPred lstmPred
             in (blend, blend)
        MethodRouter -> (kalPred, lstmPred)
  where
    clamp01 x = max 0 (min 1 x)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse
