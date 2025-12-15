module Trader.Method
  ( Method(..)
  , methodCode
  , parseMethod
  , selectPredictions
  ) where

import Data.Char (isSpace, toLower)

data Method
  = MethodBoth
  | MethodKalmanOnly
  | MethodLstmOnly
  deriving (Eq, Show)

methodCode :: Method -> String
methodCode m =
  case m of
    MethodBoth -> "11"
    MethodKalmanOnly -> "10"
    MethodLstmOnly -> "01"

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
    other ->
      Left
        ( "Invalid --method: "
            ++ show other
            ++ " (expected 11|both, 10|kalman, 01|lstm)"
        )

selectPredictions :: Method -> [Double] -> [Double] -> ([Double], [Double])
selectPredictions m kalPred lstmPred =
  case m of
    MethodBoth -> (kalPred, lstmPred)
    MethodKalmanOnly -> (kalPred, kalPred)
    MethodLstmOnly -> (lstmPred, lstmPred)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse
