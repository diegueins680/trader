module Trader.Predictors.Types
  ( SensorId(..)
  , PredictorSet
  , allPredictors
  , predictorCode
  , predictorEnabled
  , predictorSetFromString
  , predictorSetToCsv
  , predictorSetToList
  , RegimeProbs(..)
  , Quantiles(..)
  , Interval(..)
  , SensorOutput(..)
  ) where

import Data.Char (isSpace, toLower)
import Data.List (intercalate)
import qualified Data.Set as Set

data SensorId
  = SensorGBT
  | SensorTCN
  | SensorTransformer
  | SensorHMM
  | SensorQuantile
  | SensorConformal
  deriving (Eq, Ord, Show, Enum, Bounded)

type PredictorSet = Set.Set SensorId

allPredictors :: PredictorSet
allPredictors = Set.fromList [minBound .. maxBound]

predictorCode :: SensorId -> String
predictorCode sid =
  case sid of
    SensorGBT -> "gbdt"
    SensorTCN -> "tcn"
    SensorTransformer -> "transformer"
    SensorHMM -> "hmm"
    SensorQuantile -> "quantile"
    SensorConformal -> "conformal"

predictorEnabled :: PredictorSet -> SensorId -> Bool
predictorEnabled = Set.member

predictorSetToList :: PredictorSet -> [SensorId]
predictorSetToList = Set.toList

predictorSetToCsv :: PredictorSet -> String
predictorSetToCsv preds =
  if Set.null preds
    then "none"
    else intercalate "," (map predictorCode (Set.toList preds))

predictorSetFromString :: String -> Either String PredictorSet
predictorSetFromString raw =
  let tokens = splitTokens raw
      lowered = map normalizeToken tokens
      hasAll = any (== "all") lowered || any (== "default") lowered
      hasNone = any (== "none") lowered || any (== "off") lowered
      isSpecial tok = tok == "all" || tok == "default" || tok == "none" || tok == "off"
      parseOne tok =
        case tok of
          "gbdt" -> Right SensorGBT
          "gbt" -> Right SensorGBT
          "tcn" -> Right SensorTCN
          "transformer" -> Right SensorTransformer
          "hmm" -> Right SensorHMM
          "quantile" -> Right SensorQuantile
          "quantiles" -> Right SensorQuantile
          "conformal" -> Right SensorConformal
          _ -> Left tok
      bad =
        [ tok
        | tok <- lowered
        , not (isSpecial tok)
        , case parseOne tok of
            Left _ -> True
            Right _ -> False
        ]
   in if null lowered
        then Left "Predictors list is empty (expected gbdt,tcn,transformer,hmm,quantile,conformal, all, none)."
        else if not (null bad)
          then
            Left
              ( "Invalid predictors: "
                  ++ intercalate ", " bad
                  ++ " (expected gbdt,tcn,transformer,hmm,quantile,conformal, all, none)."
              )
        else if hasAll
          then Right allPredictors
        else if hasNone
          then
            if length lowered == 1
              then Right Set.empty
              else Left "Predictors list mixes 'none' with other entries."
        else
          let parsed = map parseOne lowered
           in Right (Set.fromList [sid | Right sid <- parsed])
  where
    splitTokens =
      filter (not . null) . words . map (\c -> if c == ',' then ' ' else c)
    normalizeToken =
      map toLower . filter (\c -> not (isSpace c) && c /= '-' && c /= '_')

data RegimeProbs = RegimeProbs
  { rpTrend :: !Double
  , rpMR :: !Double
  , rpHighVol :: !Double
  } deriving (Eq, Show)

data Quantiles = Quantiles
  { q10 :: !Double
  , q50 :: !Double
  , q90 :: !Double
  } deriving (Eq, Show)

data Interval = Interval
  { iLo :: !Double
  , iHi :: !Double
  } deriving (Eq, Show)

data SensorOutput = SensorOutput
  { soMu :: !Double
  , soSigma :: !(Maybe Double)
  , soRegimes :: !(Maybe RegimeProbs)
  , soQuantiles :: !(Maybe Quantiles)
  , soInterval :: !(Maybe Interval)
  } deriving (Eq, Show)
