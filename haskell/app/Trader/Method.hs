module Trader.Method
  ( Method(..)
  , methodCode
  , parseMethod
  , selectPredictions
  ) where

import Data.Char (isSpace)

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
  case trim raw of
    "11" -> Right MethodBoth
    "10" -> Right MethodKalmanOnly
    "01" -> Right MethodLstmOnly
    other ->
      Left
        ( "Invalid --method: "
            ++ show other
            ++ " (expected 11 for both, 10 for Kalman only, 01 for LSTM only)"
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
