{-# LANGUAGE OverloadedStrings #-}

module Trader.Optimizer.Optimize
  ( OptimizerArgs (..)
  , applyQualityPreset
  , runOptimizer
  ) where

import Control.Concurrent (forkIO)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)
import Control.Exception (SomeException, evaluate, try)
import Control.Monad (forM_, when)
import Data.Aeson (Value (..), object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.Char (isAlphaNum, isSpace, toLower, toUpper)
import Data.List (foldl', intercalate, sort, sortBy)
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Ord (comparing)
import qualified Data.Map.Strict as M
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.Encoding.Error as TEE
import qualified Data.Vector as V
import Data.Scientific (Scientific, toRealFloat)
import Data.Time.Clock.POSIX (getPOSIXTime)
import System.Directory
  ( canonicalizePath
  , createDirectoryIfMissing
  , doesFileExist
  , getCurrentDirectory
  , getHomeDirectory
  )
import System.Environment (getEnvironment)
import System.Exit (ExitCode (..))
import System.FilePath ((</>), takeDirectory)
import System.IO
  ( IOMode (..)
  , hClose
  , hFlush
  , hGetContents
  , hPutStrLn
  , hSetEncoding
  , openFile
  , stderr
  , stdout
  , utf8
  )
import System.Process
  ( CreateProcess (..)
  , StdStream (..)
  , createProcess
  , proc
  , terminateProcess
  , waitForProcess
  )
import System.Timeout (timeout)
import Text.Printf (printf)

import Trader.BinanceIntervals (binanceIntervals, binanceIntervalsCsv)
import Trader.Duration (lookbackBarsFrom)
import Trader.Optimizer.Json (encodePretty)
import Trader.Optimizer.Random
  ( Rng
  , nextChoice
  , nextDouble
  , nextIntRange
  , nextLogUniform
  , nextMaybe
  , nextUniform
  , seedRng
  )
import Trader.Platform (Platform (..), platformIntervals)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

clamp :: Double -> Double -> Double -> Double
clamp x lo hi = max lo (min hi x)

normalizeSymbol :: Maybe String -> Maybe String
normalizeSymbol raw =
  case raw of
    Nothing -> Nothing
    Just v ->
      let s = map toUpper (trim v)
       in if null s then Nothing else Just s

normalizeHeaderName :: String -> String
normalizeHeaderName raw =
  let s = map toLower (trim raw)
   in [c | c <- s, isAlphaNum c]

expandUser :: FilePath -> IO FilePath
expandUser path =
  case path of
    ('~' : '/' : rest) -> do
      home <- getHomeDirectory
      pure (home </> rest)
    "~" -> getHomeDirectory
    _ -> pure path

parseCsvLine :: String -> [String]
parseCsvLine input =
  let go acc field inQuotes chars =
        case chars of
          [] -> reverse (reverse field : acc)
          c : cs
            | c == '"' ->
                if inQuotes
                  then case cs of
                    '"' : rest -> go acc ('"' : field) True rest
                    _ -> go acc field False cs
                  else go acc field True cs
            | c == ',' && not inQuotes -> go (reverse field : acc) [] False cs
            | (c == '\n' || c == '\r') && not inQuotes -> reverse (reverse field : acc)
            | otherwise -> go acc (c : field) inQuotes cs
   in case input of
        [] -> []
        _ -> go [] [] False input

detectHighLowColumns :: FilePath -> IO (Maybe String, Maybe String)
detectHighLowColumns path = do
  res <- try (BS.readFile path) :: IO (Either SomeException BS.ByteString)
  case res of
    Left _ -> pure (Nothing, Nothing)
    Right bs ->
      let (line, _) = BS.break (== 10) bs
       in if BS.null line
            then pure (Nothing, Nothing)
            else do
              let headerText =
                    TE.decodeUtf8With TEE.lenientDecode line
                  header = parseCsvLine (T.unpack headerText)
              if null header
                then pure (Nothing, Nothing)
                else do
                  let normalized = foldl' insertHeaderKey M.empty header
                      high = firstMatch normalized ["high", "highprice", "highpx"]
                      low = firstMatch normalized ["low", "lowprice", "lowpx"]
                  pure (high, low)
  where
    insertHeaderKey acc col =
      let key = normalizeHeaderName col
       in if null key || M.member key acc
            then acc
            else M.insert key col acc
    firstMatch acc keys =
      case keys of
        [] -> Nothing
        k : ks ->
          case M.lookup k acc of
            Just v -> Just v
            Nothing -> firstMatch acc ks

countCsvRows :: FilePath -> IO Int
countCsvRows path = do
  bs <- BS.readFile path
  if BS.null bs
    then pure 0
    else do
      let newlines = BS.count 10 bs
          lastByte = BS.last bs
          extra = if lastByte == 10 then 0 else 1
      pure (fromIntegral newlines + extra)

metricFloat :: Maybe (KM.KeyMap Value) -> String -> Double -> Double
metricFloat m key def =
  case m of
    Nothing -> def
    Just metrics ->
      case KM.lookup (Key.fromString key) metrics of
        Just (Number n) -> fromMaybe def (scientificToDouble n)
        _ -> def

metricInt :: Maybe (KM.KeyMap Value) -> String -> Int -> Int
metricInt m key def =
  case m of
    Nothing -> def
    Just metrics ->
      case KM.lookup (Key.fromString key) metrics of
        Just (Bool v) -> if v then 1 else 0
        Just (Number n) ->
          case scientificToDouble n of
            Just d -> truncate d
            Nothing -> def
        _ -> def

metricProfitFactor :: Maybe (KM.KeyMap Value) -> Double
metricProfitFactor m =
  case m of
    Nothing -> 0
    Just metrics ->
      case KM.lookup (Key.fromString "profitFactor") metrics of
        Just (Number n) -> fromMaybe 0 (scientificToDouble n)
        _ ->
          let grossProfit = metricFloat m "grossProfit" 0
              grossLoss = metricFloat m "grossLoss" 0
           in if grossLoss > 0
                then grossProfit / grossLoss
                else if grossProfit > 0
                  then 1 / 0
                  else 0

scientificToDouble :: Scientific -> Maybe Double
scientificToDouble n =
  let d = toRealFloat n
   in if isInfinite d || isNaN d then Nothing else Just d

extractWalkForwardSummary :: Maybe Value -> Maybe (KM.KeyMap Value)
extractWalkForwardSummary raw = do
  v <- raw
  bt <- valueObjectAt v "backtest"
  wf <- valueObjectAt (Object bt) "walkForward"
  valueObjectAt (Object wf) "summary"

valueObjectAt :: Value -> String -> Maybe (KM.KeyMap Value)
valueObjectAt val key =
  case val of
    Object obj ->
      case KM.lookup (Key.fromString key) obj of
        Just (Object v) -> Just v
        _ -> Nothing
    _ -> Nothing

coerceFloatValue :: Value -> Maybe Double
coerceFloatValue value =
  case value of
    Null -> Nothing
    Bool v -> Just (if v then 1 else 0)
    Number n -> scientificToDouble n
    String s ->
      let trimmed = trim (T.unpack s)
       in if null trimmed
            then Nothing
            else case reads trimmed of
              [(v, "")] -> if isInfinite v || isNaN v then Nothing else Just v
              _ -> Nothing
    _ -> Nothing

coerceIntValue :: Value -> Maybe Int
coerceIntValue value =
  case value of
    Null -> Nothing
    Bool v -> Just (if v then 1 else 0)
    Number n ->
      case scientificToDouble n of
        Just d -> Just (truncate d)
        Nothing -> Nothing
    String s ->
      let trimmed = trim (T.unpack s)
       in if null trimmed
            then Nothing
            else case reads trimmed of
              [(v, "")] -> Just (truncate (v :: Double))
              _ -> Nothing
    _ -> Nothing

objectiveScore :: KM.KeyMap Value -> String -> Double -> Double -> Double
objectiveScore metrics objective penaltyMaxDd penaltyTurnover =
  let finalEq = metricFloat (Just metrics) "finalEquity" 0
      maxDd = metricFloat (Just metrics) "maxDrawdown" 0
      sharpe = metricFloat (Just metrics) "sharpe" 0
      annRet = metricFloat (Just metrics) "annualizedReturn" 0
      turnover = metricFloat (Just metrics) "turnover" 0
      obj = map toLower (trim objective)
   in if obj `elem` ["final-equity", "final_equity", "finalequity"]
        then finalEq
        else if obj == "sharpe"
          then sharpe
          else if obj == "calmar"
            then annRet / max 1e-12 maxDd
            else if obj `elem` ["equity-dd", "equity_maxdd", "equity-dd-only"]
              then finalEq - penaltyMaxDd * maxDd
              else if obj `elem` ["equity-dd-turnover", "equity-dd-ops", "equity-dd-turn"]
                then finalEq - penaltyMaxDd * maxDd - penaltyTurnover * turnover
                else error ("unknown objective: " ++ show objective)

extractOperations :: Maybe Value -> Maybe [Value]
extractOperations raw = do
  v <- raw
  bt <- valueObjectAt v "backtest"
  case KM.lookup (Key.fromString "trades") bt of
    Just (Array trades) ->
      let ops = mapMaybe tradeToOp (V.toList trades)
       in if null ops then Nothing else Just ops
    _ -> Nothing
  where
    tradeToOp value =
      case value of
        Object trade -> do
          entryIdx <- KM.lookup (Key.fromString "entryIndex") trade >>= coerceIntValue
          exitIdx <- KM.lookup (Key.fromString "exitIndex") trade >>= coerceIntValue
          let entryEquity = KM.lookup (Key.fromString "entryEquity") trade >>= coerceFloatValue
              exitEquity = KM.lookup (Key.fromString "exitEquity") trade >>= coerceFloatValue
              retVal = KM.lookup (Key.fromString "return") trade >>= coerceFloatValue
              holding = KM.lookup (Key.fromString "holdingPeriods") trade >>= coerceIntValue
              exitReason =
                case KM.lookup (Key.fromString "exitReason") trade of
                  Just (String s) -> Just (T.unpack s)
                  _ -> Nothing
          Just
            ( object
                [ "entryIndex" .= entryIdx
                , "exitIndex" .= exitIdx
                , "entryEquity" .= entryEquity
                , "exitEquity" .= exitEquity
                , "return" .= retVal
                , "holdingPeriods" .= holding
                , "exitReason" .= exitReason
                ]
            )
        _ -> Nothing

parsePlatforms :: String -> Either String [String]
parsePlatforms raw =
  let cleaned = map (\c -> if c == ',' then ' ' else c) raw
      parts = filter (not . null) (words (map toLower (trim cleaned)))
      allowed = ["binance", "coinbase", "kraken", "poloniex"]
   in if null parts
        then Left "no platforms provided"
        else collect allowed [] parts
  where
    collect _ acc [] = Right (reverse acc)
    collect allowed acc (p : ps)
      | p `elem` allowed =
          if p `elem` acc
            then collect allowed acc ps
            else collect allowed (p : acc) ps
      | otherwise =
          Left
            ( "invalid platform: "
                ++ p
                ++ " (expected "
                ++ intercalate ", " (sort allowed)
                ++ ")"
            )

resolveSourceLabel :: Maybe String -> String -> String -> String
resolveSourceLabel platform dataSource override
  | not (null override) = override
  | otherwise = fromMaybe dataSource platform

pickIntervals :: [String] -> String -> Int -> [String]
pickIntervals intervals lookbackWindow maxBarsCap =
  let keep itv =
        case lookbackBarsFrom itv lookbackWindow of
          Left _ -> False
          Right lb -> lb >= 2 && lb + 3 <= maxBarsCap
   in filter keep intervals

resolvePlatformIntervals :: [String] -> [String] -> String -> Int -> Either String [(String, [String])]
resolvePlatformIntervals platforms intervalsRaw lookbackWindow maxBarsCap =
  traverse resolveOne platforms
  where
    resolveOne platform =
      let supported = platformIntervals (platformFromString platform)
          filtered =
            if null intervalsRaw
              then supported
              else filter (`elem` supported) intervalsRaw
          intervals = pickIntervals filtered lookbackWindow maxBarsCap
       in if null intervals
            then
              let supportedStr = if null supported then "none" else intercalate ", " supported
               in Left ("No valid intervals for platform=" ++ platform ++ " (supported: " ++ supportedStr ++ ").")
            else Right (platform, intervals)

platformFromString :: String -> Platform
platformFromString platform =
  case platform of
    "coinbase" -> PlatformCoinbase
    "kraken" -> PlatformKraken
    "poloniex" -> PlatformPoloniex
    _ -> PlatformBinance

data OptimizerArgs = OptimizerArgs
  { oaData :: !(Maybe FilePath)
  , oaBinanceSymbol :: !(Maybe String)
  , oaSymbolLabel :: !String
  , oaSourceLabel :: !String
  , oaPriceColumn :: !String
  , oaHighColumn :: !String
  , oaLowColumn :: !String
  , oaLookbackWindow :: !String
  , oaBacktestRatio :: !Double
  , oaTuneRatio :: !Double
  , oaTrials :: !Int
  , oaSeed :: !Int
  , oaTimeoutSec :: !Double
  , oaOutput :: !String
  , oaAppend :: !Bool
  , oaBinary :: !String
  , oaNoSweepThreshold :: !Bool
  , oaDisableLstmPersistence :: !Bool
  , oaTopJson :: !String
  , oaQuality :: !Bool
  , oaAutoHighLow :: !Bool
  , oaObjective :: !String
  , oaPenaltyMaxDrawdown :: !Double
  , oaPenaltyTurnover :: !Double
  , oaMinRoundTrips :: !Int
  , oaMinWinRate :: !Double
  , oaMinProfitFactor :: !Double
  , oaMinExposure :: !Double
  , oaMinSharpe :: !Double
  , oaMinWfSharpeMean :: !Double
  , oaMaxWfSharpeStd :: !Double
  , oaTuneObjective :: !String
  , oaTunePenaltyMaxDrawdown :: !Double
  , oaTunePenaltyTurnover :: !Double
  , oaTuneStressVolMult :: !Double
  , oaTuneStressShock :: !Double
  , oaTuneStressWeight :: !Double
  , oaTuneStressVolMultMin :: !(Maybe Double)
  , oaTuneStressVolMultMax :: !(Maybe Double)
  , oaTuneStressShockMin :: !(Maybe Double)
  , oaTuneStressShockMax :: !(Maybe Double)
  , oaTuneStressWeightMin :: !(Maybe Double)
  , oaTuneStressWeightMax :: !(Maybe Double)
  , oaWalkForwardFoldsMin :: !Int
  , oaWalkForwardFoldsMax :: !Int
  , oaInterval :: !(Maybe String)
  , oaIntervals :: !(Maybe String)
  , oaPlatform :: !(Maybe String)
  , oaPlatforms :: !(Maybe String)
  , oaBarsMin :: !Int
  , oaBarsMax :: !Int
  , oaBarsAutoProb :: !Double
  , oaBarsDistribution :: !String
  , oaEpochsMin :: !Int
  , oaEpochsMax :: !Int
  , oaSlippageMax :: !Double
  , oaSpreadMax :: !Double
  , oaFeeMin :: !Double
  , oaFeeMax :: !Double
  , oaOpenThresholdMin :: !Double
  , oaOpenThresholdMax :: !Double
  , oaCloseThresholdMin :: !Double
  , oaCloseThresholdMax :: !Double
  , oaMinHoldBarsMin :: !Int
  , oaMinHoldBarsMax :: !Int
  , oaCooldownBarsMin :: !Int
  , oaCooldownBarsMax :: !Int
  , oaMaxHoldBarsMin :: !Int
  , oaMaxHoldBarsMax :: !Int
  , oaMinEdgeMin :: !Double
  , oaMinEdgeMax :: !Double
  , oaMinSignalToNoiseMin :: !Double
  , oaMinSignalToNoiseMax :: !Double
  , oaEdgeBufferMin :: !Double
  , oaEdgeBufferMax :: !Double
  , oaPCostAwareEdge :: !Double
  , oaTrendLookbackMin :: !Int
  , oaTrendLookbackMax :: !Int
  , oaPLongShort :: !Double
  , oaPIntrabarTakeProfitFirst :: !Double
  , oaKalmanDtMin :: !Double
  , oaKalmanDtMax :: !Double
  , oaKalmanProcessVarMin :: !Double
  , oaKalmanProcessVarMax :: !Double
  , oaKalmanMeasurementVarMin :: !Double
  , oaKalmanMeasurementVarMax :: !Double
  , oaKalmanZMinMin :: !Double
  , oaKalmanZMinMax :: !Double
  , oaKalmanZMaxMin :: !Double
  , oaKalmanZMaxMax :: !Double
  , oaKalmanMarketTopNMin :: !Int
  , oaKalmanMarketTopNMax :: !Int
  , oaPDisableMaxHighVolProb :: !Double
  , oaMaxHighVolProbMin :: !Double
  , oaMaxHighVolProbMax :: !Double
  , oaPDisableMaxConformalWidth :: !Double
  , oaMaxConformalWidthMin :: !Double
  , oaMaxConformalWidthMax :: !Double
  , oaPDisableMaxQuantileWidth :: !Double
  , oaMaxQuantileWidthMin :: !Double
  , oaMaxQuantileWidthMax :: !Double
  , oaPConfirmConformal :: !Double
  , oaPConfirmQuantiles :: !Double
  , oaPConfidenceSizing :: !Double
  , oaMinPositionSizeMin :: !Double
  , oaMinPositionSizeMax :: !Double
  , oaMaxPositionSizeMin :: !Double
  , oaMaxPositionSizeMax :: !Double
  , oaVolTargetMin :: !Double
  , oaVolTargetMax :: !Double
  , oaPDisableVolTarget :: !Double
  , oaVolLookbackMin :: !Int
  , oaVolLookbackMax :: !Int
  , oaVolEwmaAlphaMin :: !Double
  , oaVolEwmaAlphaMax :: !Double
  , oaPDisableVolEwmaAlpha :: !Double
  , oaVolFloorMin :: !Double
  , oaVolFloorMax :: !Double
  , oaVolScaleMaxMin :: !Double
  , oaVolScaleMaxMax :: !Double
  , oaMaxVolatilityMin :: !Double
  , oaMaxVolatilityMax :: !Double
  , oaPDisableMaxVolatility :: !Double
  , oaPeriodsPerYearMin :: !Double
  , oaPeriodsPerYearMax :: !Double
  , oaStopMin :: !Double
  , oaStopMax :: !Double
  , oaTpMin :: !Double
  , oaTpMax :: !Double
  , oaTrailMin :: !Double
  , oaTrailMax :: !Double
  , oaPDisableStop :: !Double
  , oaPDisableTp :: !Double
  , oaPDisableTrail :: !Double
  , oaStopVolMultMin :: !Double
  , oaStopVolMultMax :: !Double
  , oaTpVolMultMin :: !Double
  , oaTpVolMultMax :: !Double
  , oaTrailVolMultMin :: !Double
  , oaTrailVolMultMax :: !Double
  , oaPDisableStopVolMult :: !Double
  , oaPDisableTpVolMult :: !Double
  , oaPDisableTrailVolMult :: !Double
  , oaPDisableMaxDd :: !Double
  , oaPDisableMaxDl :: !Double
  , oaPDisableMaxOe :: !Double
  , oaMaxDdMin :: !Double
  , oaMaxDdMax :: !Double
  , oaMaxDlMin :: !Double
  , oaMaxDlMax :: !Double
  , oaMaxOeMin :: !Int
  , oaMaxOeMax :: !Int
  , oaMethodWeight11 :: !Double
  , oaMethodWeight10 :: !Double
  , oaMethodWeight01 :: !Double
  , oaMethodWeightBlend :: !Double
  , oaBlendWeightMin :: !Double
  , oaBlendWeightMax :: !Double
  , oaNormalizations :: !String
  , oaHiddenSizeMin :: !Int
  , oaHiddenSizeMax :: !Int
  , oaLrMin :: !Double
  , oaLrMax :: !Double
  , oaValRatioMin :: !Double
  , oaValRatioMax :: !Double
  , oaPatienceMax :: !Int
  , oaGradClipMin :: !Double
  , oaGradClipMax :: !Double
  , oaPDisableGradClip :: !Double
  }
  deriving (Eq, Show)

applyQualityPreset :: OptimizerArgs -> OptimizerArgs
applyQualityPreset args =
  let maxIf field val = max field val
      minIf field val = min field val
      objective = map toLower (trim (oaObjective args))
      objective' =
        if objective `elem` ["final-equity", "final_equity", "finalequity"]
          then "equity-dd-turnover"
          else oaObjective args
      intervalReset =
        case oaInterval args of
          Just v | not (null (trim v)) -> True
          _ -> False
      intervals' = if intervalReset then Just binanceIntervalsCsv else oaIntervals args
   in args
        { oaTrials = maxIf (oaTrials args) 500
        , oaMinRoundTrips = maxIf (oaMinRoundTrips args) 5
        , oaOpenThresholdMax = maxIf (oaOpenThresholdMax args) 5e-2
        , oaCloseThresholdMax = maxIf (oaCloseThresholdMax args) 5e-2
        , oaMinWinRate = maxIf (oaMinWinRate args) 0.45
        , oaMinProfitFactor = maxIf (oaMinProfitFactor args) 1.1
        , oaMinExposure = maxIf (oaMinExposure args) 0.05
        , oaMinSharpe = maxIf (oaMinSharpe args) 0.25
        , oaMinWfSharpeMean = maxIf (oaMinWfSharpeMean args) 0.2
        , oaMaxWfSharpeStd =
            if oaMaxWfSharpeStd args <= 0
              then 1.5
              else oaMaxWfSharpeStd args
        , oaMinSignalToNoiseMin = maxIf (oaMinSignalToNoiseMin args) 0.2
        , oaMinSignalToNoiseMax = maxIf (oaMinSignalToNoiseMax args) 1.0
        , oaEpochsMax = maxIf (oaEpochsMax args) 50
        , oaHiddenSizeMax = maxIf (oaHiddenSizeMax args) 128
        , oaLrMax = maxIf (oaLrMax args) 5e-2
        , oaBacktestRatio = minIf (oaBacktestRatio args) 0.10
        , oaTuneRatio = minIf (oaTuneRatio args) 0.15
        , oaObjective = objective'
        , oaPenaltyTurnover = maxIf (oaPenaltyTurnover args) 0.1
        , oaBarsMax = if oaBarsMax args > 0 then 0 else oaBarsMax args
        , oaAutoHighLow = True
        , oaWalkForwardFoldsMin = maxIf (oaWalkForwardFoldsMin args) 3
        , oaWalkForwardFoldsMax = maxIf (oaWalkForwardFoldsMax args) (oaWalkForwardFoldsMin args)
        , oaInterval = if intervalReset then Nothing else oaInterval args
        , oaIntervals = intervals'
        }

data TrialParams = TrialParams
  { tpPlatform :: !(Maybe String)
  , tpInterval :: !String
  , tpBars :: !Int
  , tpMethod :: !String
  , tpBlendWeight :: !Double
  , tpPositioning :: !String
  , tpNormalization :: !String
  , tpBaseOpenThreshold :: !Double
  , tpBaseCloseThreshold :: !Double
  , tpMinHoldBars :: !Int
  , tpCooldownBars :: !Int
  , tpMaxHoldBars :: !(Maybe Int)
  , tpMinEdge :: !Double
  , tpMinSignalToNoise :: !Double
  , tpEdgeBuffer :: !Double
  , tpCostAwareEdge :: !Bool
  , tpTrendLookback :: !Int
  , tpMaxPositionSize :: !Double
  , tpVolTarget :: !(Maybe Double)
  , tpVolLookback :: !Int
  , tpVolEwmaAlpha :: !(Maybe Double)
  , tpVolFloor :: !Double
  , tpVolScaleMax :: !Double
  , tpMaxVolatility :: !(Maybe Double)
  , tpPeriodsPerYear :: !(Maybe Double)
  , tpKalmanMarketTopN :: !Int
  , tpFee :: !Double
  , tpEpochs :: !Int
  , tpHiddenSize :: !Int
  , tpLearningRate :: !Double
  , tpValRatio :: !Double
  , tpPatience :: !Int
  , tpWalkForwardFolds :: !Int
  , tpTuneStressVolMult :: !Double
  , tpTuneStressShock :: !Double
  , tpTuneStressWeight :: !Double
  , tpGradClip :: !(Maybe Double)
  , tpSlippage :: !Double
  , tpSpread :: !Double
  , tpIntrabarFill :: !String
  , tpStopLoss :: !(Maybe Double)
  , tpTakeProfit :: !(Maybe Double)
  , tpTrailingStop :: !(Maybe Double)
  , tpStopLossVolMult :: !(Maybe Double)
  , tpTakeProfitVolMult :: !(Maybe Double)
  , tpTrailingStopVolMult :: !(Maybe Double)
  , tpMaxDrawdown :: !(Maybe Double)
  , tpMaxDailyLoss :: !(Maybe Double)
  , tpMaxOrderErrors :: !(Maybe Int)
  , tpKalmanDt :: !Double
  , tpKalmanProcessVar :: !Double
  , tpKalmanMeasurementVar :: !Double
  , tpKalmanZMin :: !Double
  , tpKalmanZMax :: !Double
  , tpMaxHighVolProb :: !(Maybe Double)
  , tpMaxConformalWidth :: !(Maybe Double)
  , tpMaxQuantileWidth :: !(Maybe Double)
  , tpConfirmConformal :: !Bool
  , tpConfirmQuantiles :: !Bool
  , tpConfidenceSizing :: !Bool
  , tpMinPositionSize :: !Double
  }
  deriving (Eq, Show)

data TrialResult = TrialResult
  { trOk :: !Bool
  , trReason :: !(Maybe String)
  , trElapsedSec :: !Double
  , trParams :: !TrialParams
  , trFinalEquity :: !(Maybe Double)
  , trMetrics :: !(Maybe (KM.KeyMap Value))
  , trOpenThreshold :: !(Maybe Double)
  , trCloseThreshold :: !(Maybe Double)
  , trStdoutJson :: !(Maybe Value)
  , trEligible :: !Bool
  , trFilterReason :: !(Maybe String)
  , trObjective :: !String
  , trScore :: !(Maybe Double)
  }
  deriving (Eq, Show)

fmtOptFloat :: Maybe Double -> String
fmtOptFloat v =
  case v of
    Nothing -> "null"
    Just x -> printf "%.8f" x

fmtOptInt :: Maybe Int -> String
fmtOptInt v =
  case v of
    Nothing -> "null"
    Just x -> show x

buildCommand :: FilePath -> [String] -> TrialParams -> Double -> Bool -> [String]
buildCommand traderBin baseArgs params tuneRatio useSweepThreshold =
  let cmd0 = [traderBin] ++ baseArgs
      cmd1 =
        case tpPlatform params of
          Just platform -> cmd0 ++ ["--platform", platform]
          Nothing -> cmd0
      cmd2 = cmd1 ++ ["--interval", tpInterval params]
      isBinance = tpPlatform params == Just "binance" && "--binance-symbol" `elem` baseArgs
      cmd3 =
        if tpBars params <= 0
          then cmd2 ++ ["--bars", if isBinance then "auto" else "0"]
          else cmd2 ++ ["--bars", show (tpBars params)]
      cmd4 =
        cmd3
          ++ [ "--positioning"
             , tpPositioning params
             , "--method"
             , tpMethod params
             , "--blend-weight"
             , printf "%.6f" (clamp (tpBlendWeight params) 0 1)
             , "--normalization"
             , tpNormalization params
             , "--open-threshold"
             , printf "%.12g" (max 0 (tpBaseOpenThreshold params))
             , "--close-threshold"
             , printf "%.12g" (max 0 (tpBaseCloseThreshold params))
             , "--min-hold-bars"
             , show (max 0 (tpMinHoldBars params))
             , "--cooldown-bars"
             , show (max 0 (tpCooldownBars params))
             ]
      cmd5 =
        case tpMaxHoldBars params of
          Just v -> cmd4 ++ ["--max-hold-bars", show (max 1 v)]
          Nothing -> cmd4
      cmd6 =
        cmd5
          ++ [ "--min-edge"
             , printf "%.12g" (max 0 (tpMinEdge params))
             , "--min-signal-to-noise"
             , printf "%.12g" (max 0 (tpMinSignalToNoise params))
             , "--edge-buffer"
             , printf "%.12g" (max 0 (tpEdgeBuffer params))
             ]
      cmd7 = if tpCostAwareEdge params then cmd6 ++ ["--cost-aware-edge"] else cmd6
      cmd8 =
        cmd7
          ++ [ "--trend-lookback"
             , show (max 0 (tpTrendLookback params))
             , "--max-position-size"
             , printf "%.12g" (max 0 (tpMaxPositionSize params))
             ]
      cmd9 =
        case tpVolTarget params of
          Just v -> cmd8 ++ ["--vol-target", printf "%.12g" (max 1e-12 v)]
          Nothing -> cmd8
      cmd10 =
        cmd9
          ++ [ "--vol-lookback"
             , show (max 0 (tpVolLookback params))
             ]
      cmd11 =
        case tpVolEwmaAlpha params of
          Just v -> cmd10 ++ ["--vol-ewma-alpha", printf "%.6f" (clamp v 0 1)]
          Nothing -> cmd10
      cmd12 =
        cmd11
          ++ [ "--vol-floor"
             , printf "%.12g" (max 0 (tpVolFloor params))
             , "--vol-scale-max"
             , printf "%.12g" (max 0 (tpVolScaleMax params))
             ]
      cmd13 =
        case tpMaxVolatility params of
          Just v -> cmd12 ++ ["--max-volatility", printf "%.12g" (max 1e-12 v)]
          Nothing -> cmd12
      cmd14 =
        case tpPeriodsPerYear params of
          Just v -> cmd13 ++ ["--periods-per-year", printf "%.6f" (max 1e-12 v)]
          Nothing -> cmd13
      cmd15 =
        cmd14
          ++ [ "--fee"
             , printf "%.12g" (max 0 (tpFee params))
             , "--epochs"
             , show (tpEpochs params)
             , "--hidden-size"
             , show (tpHiddenSize params)
             , "--lr"
             , printf "%.8f" (tpLearningRate params)
             , "--val-ratio"
             , printf "%.6f" (tpValRatio params)
             , "--patience"
             , show (tpPatience params)
             , "--walk-forward-folds"
             , show (max 1 (tpWalkForwardFolds params))
             , "--tune-stress-vol-mult"
             , printf "%.6f" (max 1e-12 (tpTuneStressVolMult params))
             , "--tune-stress-shock"
             , printf "%.6f" (tpTuneStressShock params)
             , "--tune-stress-weight"
             , printf "%.6f" (max 0 (tpTuneStressWeight params))
             ]
      cmd16 =
        case tpGradClip params of
          Just v -> cmd15 ++ ["--grad-clip", printf "%.8f" v]
          Nothing -> cmd15
      cmd17 =
        cmd16
          ++ [ "--slippage"
             , printf "%.8f" (tpSlippage params)
             , "--spread"
             , printf "%.8f" (tpSpread params)
             , "--intrabar-fill"
             , tpIntrabarFill params
             ]
      cmd18 =
        case tpStopLoss params of
          Just v -> cmd17 ++ ["--stop-loss", printf "%.8f" v]
          Nothing -> cmd17
      cmd19 =
        case tpTakeProfit params of
          Just v -> cmd18 ++ ["--take-profit", printf "%.8f" v]
          Nothing -> cmd18
      cmd20 =
        case tpTrailingStop params of
          Just v -> cmd19 ++ ["--trailing-stop", printf "%.8f" v]
          Nothing -> cmd19
      cmd21 =
        case tpStopLossVolMult params of
          Just v -> cmd20 ++ ["--stop-loss-vol-mult", printf "%.8f" v]
          Nothing -> cmd20
      cmd22 =
        case tpTakeProfitVolMult params of
          Just v -> cmd21 ++ ["--take-profit-vol-mult", printf "%.8f" v]
          Nothing -> cmd21
      cmd23 =
        case tpTrailingStopVolMult params of
          Just v -> cmd22 ++ ["--trailing-stop-vol-mult", printf "%.8f" v]
          Nothing -> cmd22
      cmd24 =
        case tpMaxDrawdown params of
          Just v -> cmd23 ++ ["--max-drawdown", printf "%.8f" v]
          Nothing -> cmd23
      cmd25 =
        case tpMaxDailyLoss params of
          Just v -> cmd24 ++ ["--max-daily-loss", printf "%.8f" v]
          Nothing -> cmd24
      cmd26 =
        case tpMaxOrderErrors params of
          Just v -> cmd25 ++ ["--max-order-errors", show v]
          Nothing -> cmd25
      cmd27 =
        cmd26
          ++ [ "--kalman-market-top-n"
             , show (max 0 (tpKalmanMarketTopN params))
             , "--kalman-dt"
             , printf "%.12g" (max 1e-12 (tpKalmanDt params))
             , "--kalman-process-var"
             , printf "%.12g" (max 1e-12 (tpKalmanProcessVar params))
             , "--kalman-measurement-var"
             , printf "%.12g" (max 1e-12 (tpKalmanMeasurementVar params))
             , "--kalman-z-min"
             , printf "%.12g" (max 0 (tpKalmanZMin params))
             , "--kalman-z-max"
             , printf "%.12g" (max (max 0 (tpKalmanZMin params)) (tpKalmanZMax params))
             ]
      cmd28 =
        case tpMaxHighVolProb params of
          Just v -> cmd27 ++ ["--max-high-vol-prob", printf "%.12g" (clamp v 0 1)]
          Nothing -> cmd27
      cmd29 =
        case tpMaxConformalWidth params of
          Just v -> cmd28 ++ ["--max-conformal-width", printf "%.12g" (max 0 v)]
          Nothing -> cmd28
      cmd30 =
        case tpMaxQuantileWidth params of
          Just v -> cmd29 ++ ["--max-quantile-width", printf "%.12g" (max 0 v)]
          Nothing -> cmd29
      cmd31 = if tpConfirmConformal params then cmd30 ++ ["--confirm-conformal"] else cmd30
      cmd32 = if tpConfirmQuantiles params then cmd31 ++ ["--confirm-quantiles"] else cmd31
      cmd33 = if tpConfidenceSizing params then cmd32 ++ ["--confidence-sizing"] else cmd32
      cmd34 = cmd33 ++ ["--min-position-size", printf "%.12g" (clamp (tpMinPositionSize params) 0 1)]
      cmd35 =
        if useSweepThreshold
          then cmd34 ++ ["--sweep-threshold", "--tune-ratio", printf "%.6f" tuneRatio]
          else cmd34
   in cmd35 ++ ["--json"]

runTrial :: FilePath -> [String] -> TrialParams -> Double -> Bool -> Double -> Bool -> IO TrialResult
runTrial traderBin baseArgs params tuneRatio useSweepThreshold timeoutSec disableLstm = do
  let cmd = buildCommand traderBin baseArgs params tuneRatio useSweepThreshold
  env0 <- getEnvironment
  let env = if disableLstm then setEnv "TRADER_LSTM_WEIGHTS_DIR" "" env0 else env0
      procSpec =
        (proc traderBin (tail cmd))
          { cwd = Just (takeDirectory traderBin)
          , env = Just env
          , std_out = CreatePipe
          , std_err = CreatePipe
          }
  t0 <- getPOSIXTime
  res <- runWithTimeout procSpec timeoutSec
  t1 <- getPOSIXTime
  let elapsed = realToFrac (t1 - t0)
  case res of
    Nothing ->
      pure
        TrialResult
          { trOk = False
          , trReason = Just ("timeout>" ++ show timeoutSec ++ "s")
          , trElapsedSec = elapsed
          , trParams = params
          , trFinalEquity = Nothing
          , trMetrics = Nothing
          , trOpenThreshold = Nothing
          , trCloseThreshold = Nothing
          , trStdoutJson = Nothing
          , trEligible = False
          , trFilterReason = Nothing
          , trObjective = "final-equity"
          , trScore = Nothing
          }
    Just (exitCode, out, err) ->
      case exitCode of
        ExitFailure code -> do
          let chosen = if null err then out else err
              trimmed = trim chosen
              short = if length trimmed > 300 then take 300 trimmed ++ "..." else trimmed
              reason = if null short then "exit=" ++ show code else short
          pure
            TrialResult
              { trOk = False
              , trReason = Just reason
              , trElapsedSec = elapsed
              , trParams = params
              , trFinalEquity = Nothing
              , trMetrics = Nothing
              , trOpenThreshold = Nothing
              , trCloseThreshold = Nothing
              , trStdoutJson = Nothing
              , trEligible = False
              , trFilterReason = Nothing
              , trObjective = "final-equity"
              , trScore = Nothing
              }
        ExitSuccess ->
          case Aeson.eitherDecode (BL.fromStrict (TE.encodeUtf8 (T.pack out))) of
            Left e ->
              pure
                TrialResult
                  { trOk = False
                  , trReason = Just ("json parse error: " ++ e)
                  , trElapsedSec = elapsed
                  , trParams = params
                  , trFinalEquity = Nothing
                  , trMetrics = Nothing
                  , trOpenThreshold = Nothing
                  , trCloseThreshold = Nothing
                  , trStdoutJson = Nothing
                  , trEligible = False
                  , trFilterReason = Nothing
                  , trObjective = "final-equity"
                  , trScore = Nothing
                  }
            Right outVal ->
              case extractBacktest outVal of
                Left e ->
                  pure
                    TrialResult
                      { trOk = False
                      , trReason = Just ("unexpected json schema: " ++ e)
                      , trElapsedSec = elapsed
                      , trParams = params
                      , trFinalEquity = Nothing
                      , trMetrics = Nothing
                      , trOpenThreshold = Nothing
                      , trCloseThreshold = Nothing
                      , trStdoutJson = Just outVal
                      , trEligible = False
                      , trFilterReason = Nothing
                      , trObjective = "final-equity"
                      , trScore = Nothing
                      }
                Right (metrics, finalEq, openThr, closeThr) ->
                  pure
                    TrialResult
                      { trOk = True
                      , trReason = Nothing
                      , trElapsedSec = elapsed
                      , trParams = params
                      , trFinalEquity = Just finalEq
                      , trMetrics = Just metrics
                      , trOpenThreshold = openThr
                      , trCloseThreshold = closeThr
                      , trStdoutJson = Just outVal
                      , trEligible = False
                      , trFilterReason = Nothing
                      , trObjective = "final-equity"
                      , trScore = Nothing
                      }

extractBacktest :: Value -> Either String (KM.KeyMap Value, Double, Maybe Double, Maybe Double)
extractBacktest val =
  case val of
    Object obj ->
      case KM.lookup (Key.fromString "backtest") obj of
        Just (Object bt) ->
          case KM.lookup (Key.fromString "metrics") bt of
            Just (Object metrics) -> do
              finalEq <-
                case KM.lookup (Key.fromString "finalEquity") metrics of
                  Just (Number n) ->
                    case scientificToDouble n of
                      Just d -> Right d
                      Nothing -> Left "finalEquity not finite"
                  _ -> Left "finalEquity missing"
              let openThr =
                    case KM.lookup (Key.fromString "openThreshold") bt of
                      Just (Number n) -> scientificToDouble n
                      _ -> Nothing
                  closeThr =
                    case KM.lookup (Key.fromString "closeThreshold") bt of
                      Just (Number n) -> scientificToDouble n
                      _ -> Nothing
              Right (metrics, finalEq, openThr, closeThr)
            _ -> Left "metrics missing"
        _ -> Left "backtest missing"
    _ -> Left "backtest missing"

runWithTimeout :: CreateProcess -> Double -> IO (Maybe (ExitCode, String, String))
runWithTimeout procSpec timeoutSec = do
  let timeoutMicros = max 0 (floor (timeoutSec * 1000000))
  (_, Just hout, Just herr, ph) <- createProcess procSpec
  hSetEncoding hout utf8
  hSetEncoding herr utf8
  outVar <- newEmptyMVar
  errVar <- newEmptyMVar
  _ <- forkIO $ do
    out <- hGetContents hout
    _ <- evaluate (length out)
    putMVar outVar out
  _ <- forkIO $ do
    err <- hGetContents herr
    _ <- evaluate (length err)
    putMVar errVar err
  mExit <- timeout timeoutMicros (waitForProcess ph)
  case mExit of
    Nothing -> do
      terminateProcess ph
      _ <- waitForProcess ph
      pure Nothing
    Just exitCode -> do
      out <- takeMVar outVar
      err <- takeMVar errVar
      pure (Just (exitCode, out, err))

setEnv :: String -> String -> [(String, String)] -> [(String, String)]
setEnv key val env =
  let filtered = filter (\(k, _) -> k /= key) env
   in (key, val) : filtered

trialToRecord :: TrialResult -> Maybe String -> Value
trialToRecord tr symbolLabel =
  let paramsPairs =
        [ "platform" .= tpPlatform (trParams tr)
        , "interval" .= tpInterval (trParams tr)
        , "bars" .= tpBars (trParams tr)
        , "method" .= tpMethod (trParams tr)
        , "blendWeight" .= tpBlendWeight (trParams tr)
        , "positioning" .= tpPositioning (trParams tr)
        , "normalization" .= tpNormalization (trParams tr)
        , "baseOpenThreshold" .= tpBaseOpenThreshold (trParams tr)
        , "baseCloseThreshold" .= tpBaseCloseThreshold (trParams tr)
        , "minHoldBars" .= tpMinHoldBars (trParams tr)
        , "cooldownBars" .= tpCooldownBars (trParams tr)
        , "maxHoldBars" .= tpMaxHoldBars (trParams tr)
        , "minEdge" .= tpMinEdge (trParams tr)
        , "minSignalToNoise" .= tpMinSignalToNoise (trParams tr)
        , "edgeBuffer" .= tpEdgeBuffer (trParams tr)
        , "costAwareEdge" .= tpCostAwareEdge (trParams tr)
        , "trendLookback" .= tpTrendLookback (trParams tr)
        , "maxPositionSize" .= tpMaxPositionSize (trParams tr)
        , "volTarget" .= tpVolTarget (trParams tr)
        , "volLookback" .= tpVolLookback (trParams tr)
        , "volEwmaAlpha" .= tpVolEwmaAlpha (trParams tr)
        , "volFloor" .= tpVolFloor (trParams tr)
        , "volScaleMax" .= tpVolScaleMax (trParams tr)
        , "maxVolatility" .= tpMaxVolatility (trParams tr)
        , "periodsPerYear" .= tpPeriodsPerYear (trParams tr)
        , "kalmanMarketTopN" .= tpKalmanMarketTopN (trParams tr)
        , "fee" .= tpFee (trParams tr)
        , "epochs" .= tpEpochs (trParams tr)
        , "hiddenSize" .= tpHiddenSize (trParams tr)
        , "learningRate" .= tpLearningRate (trParams tr)
        , "valRatio" .= tpValRatio (trParams tr)
        , "patience" .= tpPatience (trParams tr)
        , "walkForwardFolds" .= tpWalkForwardFolds (trParams tr)
        , "tuneStressVolMult" .= tpTuneStressVolMult (trParams tr)
        , "tuneStressShock" .= tpTuneStressShock (trParams tr)
        , "tuneStressWeight" .= tpTuneStressWeight (trParams tr)
        , "gradClip" .= tpGradClip (trParams tr)
        , "slippage" .= tpSlippage (trParams tr)
        , "spread" .= tpSpread (trParams tr)
        , "intrabarFill" .= tpIntrabarFill (trParams tr)
        , "stopLoss" .= tpStopLoss (trParams tr)
        , "takeProfit" .= tpTakeProfit (trParams tr)
        , "trailingStop" .= tpTrailingStop (trParams tr)
        , "stopLossVolMult" .= tpStopLossVolMult (trParams tr)
        , "takeProfitVolMult" .= tpTakeProfitVolMult (trParams tr)
        , "trailingStopVolMult" .= tpTrailingStopVolMult (trParams tr)
        , "maxDrawdown" .= tpMaxDrawdown (trParams tr)
        , "maxDailyLoss" .= tpMaxDailyLoss (trParams tr)
        , "maxOrderErrors" .= tpMaxOrderErrors (trParams tr)
        , "kalmanDt" .= tpKalmanDt (trParams tr)
        , "kalmanProcessVar" .= tpKalmanProcessVar (trParams tr)
        , "kalmanMeasurementVar" .= tpKalmanMeasurementVar (trParams tr)
        , "kalmanZMin" .= tpKalmanZMin (trParams tr)
        , "kalmanZMax" .= tpKalmanZMax (trParams tr)
        , "maxHighVolProb" .= tpMaxHighVolProb (trParams tr)
        , "maxConformalWidth" .= tpMaxConformalWidth (trParams tr)
        , "maxQuantileWidth" .= tpMaxQuantileWidth (trParams tr)
        , "confirmConformal" .= tpConfirmConformal (trParams tr)
        , "confirmQuantiles" .= tpConfirmQuantiles (trParams tr)
        , "confidenceSizing" .= tpConfidenceSizing (trParams tr)
        , "minPositionSize" .= tpMinPositionSize (trParams tr)
        ]
      paramsPairs' =
        case normalizeSymbol symbolLabel of
          Just sym -> paramsPairs ++ ["binanceSymbol" .= sym]
          Nothing -> paramsPairs
      baseFields =
        [ "ok" .= trOk tr
        , "eligible" .= trEligible tr
        , "objective" .= trObjective tr
        , "score" .= trScore tr
        , "filterReason" .= trFilterReason tr
        , "reason" .= trReason tr
        , "elapsedSec" .= trElapsedSec tr
        , "finalEquity" .= trFinalEquity tr
        , "openThreshold" .= trOpenThreshold tr
        , "closeThreshold" .= trCloseThreshold tr
        , "params" .= object paramsPairs'
        ]
      metricsField =
        case trMetrics tr of
          Just m -> ["metrics" .= Object m]
          Nothing -> []
      opsField =
        case extractOperations (trStdoutJson tr) of
          Just ops -> ["operations" .= ops]
          Nothing -> []
   in object (baseFields ++ metricsField ++ opsField)

sampleParams ::
  Rng ->
  [String] ->
  [(String, [String])] ->
  [String] ->
  Double ->
  Int ->
  Int ->
  String ->
  Double ->
  Double ->
  Double ->
  Double ->
  (Int, Int) ->
  (Int, Int) ->
  (Int, Int) ->
  (Double, Double) ->
  (Double, Double) ->
  (Double, Double) ->
  (Int, Int) ->
  (Double, Double) ->
  (Double, Double) ->
  (Int, Int) ->
  (Double, Double) ->
  Double ->
  (Double, Double) ->
  (Double, Double) ->
  (Double, Double) ->
  (Double, Double) ->
  (Int, Int) ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Int ->
  Int ->
  Double ->
  Double ->
  Double ->
  Int ->
  (Int, Int) ->
  (Double, Double) ->
  (Double, Double) ->
  (Double, Double) ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  (Double, Double) ->
  (Double, Double) ->
  (Double, Double) ->
  (Double, Double) ->
  (Double, Double) ->
  Double ->
  Double ->
  Double ->
  (Double, Double) ->
  (Double, Double) ->
  (Double, Double) ->
  (Double, Double) ->
  (Int, Int) ->
  (String, Double, Double, Double, Double) ->
  [String] ->
  (Double, Double) ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  Double ->
  (Double, Double) ->
  (Double, Double) ->
  (Int, Int) ->
  (TrialParams, Rng)
sampleParams
  rng0
  platforms
  platformIntervals
  intervals
  pAutoBars
  barsMin
  barsMax
  barsDistribution
  openThresholdMin
  openThresholdMax
  closeThresholdMin
  closeThresholdMax
  minHoldBarsRange
  cooldownBarsRange
  maxHoldBarsRange
  minEdgeRange
  minSignalToNoiseRange
  edgeBufferRange
  trendLookbackRange
  maxPositionSizeRange
  volTargetRange
  volLookbackRange
  volEwmaAlphaRange
  pDisableVolEwmaAlpha
  volFloorRange
  volScaleMaxRange
  maxVolatilityRange
  periodsPerYearRange
  kalmanMarketTopNRange
  pCostAwareEdge
  feeMin
  feeMax
  pLongShort
  pIntrabarTakeProfitFirst
  epochsMin
  epochsMax
  hiddenMin
  hiddenMax
  lrMin
  lrMax
  valMin
  valMax
  patienceMax
  walkForwardFoldsRange
  tuneStressVolMultRange
  tuneStressShockRange
  tuneStressWeightRange
  gradClipMin
  gradClipMax
  slippageMax
  spreadMax
  kalmanDtMin
  kalmanDtMax
  kalmanProcessVarMin
  kalmanProcessVarMax
  kalmanMeasurementVarMin
  kalmanMeasurementVarMax
  kalmanZMinMin
  kalmanZMinMax
  kalmanZMaxMin
  kalmanZMaxMax
  pDisableMaxHighVolProb
  maxHighVolProbRange
  pDisableMaxConformalWidth
  maxConformalWidthRange
  pDisableMaxQuantileWidth
  maxQuantileWidthRange
  pConfirmConformal
  pConfirmQuantiles
  pConfidenceSizing
  minPositionSizeRange
  stopRange
  takeRange
  trailRange
  stopVolMultRange
  takeVolMultRange
  trailVolMultRange
  (methodW11, methodW10, methodW01, methodWBlend)
  normalizationChoices
  blendWeightRange
  pDisableStop
  pDisableTp
  pDisableTrail
  pDisableStopVolMult
  pDisableTpVolMult
  pDisableTrailVolMult
  pDisableMaxDd
  pDisableMaxDl
  pDisableMaxOe
  pDisableGradClip
  pDisableVolTarget
  pDisableMaxVolatility
  maxDdRange
  maxDlRange
  maxOeRange =
    let (platform, rng1) =
          case platforms of
            [] -> (Nothing, rng0)
            _ ->
              let (p, rng') = nextChoice platforms rng0
               in (Just p, rng')
        intervalPool =
          case platform of
            Nothing -> intervals
            Just p -> fromMaybe intervals (lookup p platformIntervals)
        (interval, rng2) = nextChoice intervalPool rng1
        (bars, rng3) = sampleBars rng2
        methods = [("11", methodW11), ("10", methodW10), ("01", methodW01), ("blend", methodWBlend)]
        (method, rng4) = chooseWeighted methods rng3
        (blendWeight, rng5) =
          let (bwLo, bwHi) = ordered blendWeightRange
              (val, rng') = nextUniform bwLo bwHi rng4
           in (clamp val 0 1, rng')
        (normalization, rng6) = nextChoice normalizationChoices rng5
        (baseOpenThreshold, rng7) =
          nextLogUniform (max 1e-12 openThresholdMin) (max 1e-12 openThresholdMax) rng6
        (baseCloseThreshold, rng8) =
          nextLogUniform (max 1e-12 closeThresholdMin) (max 1e-12 closeThresholdMax) rng7
        (minHoldBars, rng9) = nextIntRange (fst minHoldBarsRange) (snd minHoldBarsRange) rng8
        (cooldownBars, rng10) = nextIntRange (fst cooldownBarsRange) (snd cooldownBarsRange) rng9
        (maxHoldBars, rng11) =
          if snd maxHoldBarsRange > 0
            then
              let (sample, rng') = nextIntRange (fst maxHoldBarsRange) (snd maxHoldBarsRange) rng10
               in if sample > 0 then (Just sample, rng') else (Nothing, rng')
            else (Nothing, rng10)
        (minEdge, rng12) = nextUniform (fst minEdgeRange) (snd minEdgeRange) rng11
        (minSignalToNoise, rng13) =
          let (lo, hi) = ordered minSignalToNoiseRange
           in nextUniform lo hi rng12
        (edgeBuffer, rng14) = nextUniform (fst edgeBufferRange) (snd edgeBufferRange) rng13
        (costAwareEdge, edgeBuffer', rng15) =
          if pCostAwareEdge < 0
            then (edgeBuffer > 0, edgeBuffer, rng14)
            else
              let (r, rng') = nextDouble rng14
                  enabled = r < clamp pCostAwareEdge 0 1
               in if enabled
                    then (True, edgeBuffer, rng')
                    else (False, 0, rng')
        (trendLookback, rng16) = nextIntRange (fst trendLookbackRange) (snd trendLookbackRange) rng15
        (maxPositionSize, rng17) =
          let (val, rng') = nextUniform (fst maxPositionSizeRange) (snd maxPositionSizeRange) rng16
           in (max 0 val, rng')
        (volTarget, rng18) =
          if max (fst volTargetRange) (snd volTargetRange) > 0
            then
              let (r, rng') = nextDouble rng17
               in if r >= clamp pDisableVolTarget 0 1
                    then
                      let vtLo = max 1e-12 (min (fst volTargetRange) (snd volTargetRange))
                          vtHi = max vtLo (max (fst volTargetRange) (snd volTargetRange))
                          (val, rng'') = nextUniform vtLo vtHi rng'
                       in (Just val, rng'')
                    else (Nothing, rng')
            else (Nothing, rng17)
        (volEwmaAlpha, rng19) =
          if max (fst volEwmaAlphaRange) (snd volEwmaAlphaRange) > 0
            then
              let (r, rng') = nextDouble rng18
               in if r >= clamp pDisableVolEwmaAlpha 0 1
                    then
                      let vaLo = max 1e-6 (min (fst volEwmaAlphaRange) (snd volEwmaAlphaRange))
                          vaHi = min 0.999 (max (fst volEwmaAlphaRange) (snd volEwmaAlphaRange))
                       in if vaHi >= vaLo
                            then
                              let (val, rng'') = nextUniform vaLo vaHi rng'
                               in (Just val, rng'')
                            else (Nothing, rng')
                    else (Nothing, rng')
            else (Nothing, rng18)
        (volLookback0, rng20) = nextIntRange (fst volLookbackRange) (snd volLookbackRange) rng19
        volLookback = if volTarget /= Nothing && volEwmaAlpha == Nothing then max 2 volLookback0 else volLookback0
        (volFloor, rng21) =
          let (val, rng') = nextUniform (fst volFloorRange) (snd volFloorRange) rng20
           in (max 0 val, rng')
        (volScaleMax, rng22) =
          let (val, rng') = nextUniform (fst volScaleMaxRange) (snd volScaleMaxRange) rng21
           in (max 0 val, rng')
        (periodsPerYear, rng23) =
          if max (fst periodsPerYearRange) (snd periodsPerYearRange) > 0
            then
              let ppyMin = max 1e-12 (min (fst periodsPerYearRange) (snd periodsPerYearRange))
                  ppyMax = max ppyMin (max (fst periodsPerYearRange) (snd periodsPerYearRange))
                  (val, rng') = nextUniform ppyMin ppyMax rng22
               in (Just val, rng')
            else (Nothing, rng22)
        (kalmanMarketTopN, rng24) = nextIntRange (fst kalmanMarketTopNRange) (snd kalmanMarketTopNRange) rng23
        (maxVolatility, rng25) =
          if max (fst maxVolatilityRange) (snd maxVolatilityRange) > 0
            then
              let (r, rng') = nextDouble rng24
               in if r >= clamp pDisableMaxVolatility 0 1
                    then
                      let mvLo = max 1e-12 (min (fst maxVolatilityRange) (snd maxVolatilityRange))
                          mvHi = max mvLo (max (fst maxVolatilityRange) (snd maxVolatilityRange))
                          (val, rng'') = nextUniform mvLo mvHi rng'
                       in (Just val, rng'')
                    else (Nothing, rng')
            else (Nothing, rng24)
        (fee, rng26) = nextUniform (max 0 feeMin) (max 0 feeMax) rng25
        (epochs, rng27) = nextIntRange epochsMin epochsMax rng26
        (hiddenSize, rng28) = nextIntRange hiddenMin hiddenMax rng27
        (learningRate, rng29) = nextLogUniform lrMin lrMax rng28
        (valRatio, rng30) = nextUniform valMin valMax rng29
        (patience, rng31) = nextIntRange 0 patienceMax rng30
        (walkForwardFolds, rng32) =
          let (lo, hi) = ordered walkForwardFoldsRange
           in nextIntRange (max 1 lo) (max 1 hi) rng31
        (tuneStressVolMult, rng33) =
          let (lo, hi) = ordered tuneStressVolMultRange
              lo' = max 1e-12 lo
              hi' = max lo' hi
           in nextUniform lo' hi' rng32
        (tuneStressShock, rng34) =
          let (lo, hi) = ordered tuneStressShockRange
           in nextUniform lo hi rng33
        (tuneStressWeight, rng35) =
          let (lo, hi) = ordered tuneStressWeightRange
              lo' = max 0 lo
              hi' = max lo' hi
           in nextUniform lo' hi' rng34
        (gradClip, rng36) = nextMaybe pDisableGradClip (nextLogUniform gradClipMin gradClipMax) rng35
        (slippage, rng37) = nextUniform 0 (max 0 slippageMax) rng36
        (spread, rng38) = nextUniform 0 (max 0 spreadMax) rng37
        (positioning, rng39) =
          let (r, rng') = nextDouble rng38
           in (if r < clamp pLongShort 0 1 then "long-short" else "long-flat", rng')
        (intrabarFill, rng40) =
          let (r, rng') = nextDouble rng39
           in (if r < clamp pIntrabarTakeProfitFirst 0 1 then "take-profit-first" else "stop-first", rng')
        (kalmanDt, rng41) = nextUniform (max 1e-12 kalmanDtMin) (max 1e-12 kalmanDtMax) rng40
        (kalmanProcessVar, rng42) =
          nextLogUniform (max 1e-12 kalmanProcessVarMin) (max 1e-12 kalmanProcessVarMax) rng41
        (kalmanMeasurementVar, rng43) =
          nextLogUniform (max 1e-12 kalmanMeasurementVarMin) (max 1e-12 kalmanMeasurementVarMax) rng42
        (kalmanZMin, rng44) = nextUniform (max 0 kalmanZMinMin) (max 0 kalmanZMinMax) rng43
        (kalmanZMax, rng45) =
          let zMaxLo = max kalmanZMin (max 0 kalmanZMaxMin)
              zMaxHi = max zMaxLo kalmanZMaxMax
           in nextUniform zMaxLo zMaxHi rng44
        (maxHighVolProb, rng46) =
          let (lo, hi) = ordered maxHighVolProbRange
              (val, rng') =
                nextMaybe pDisableMaxHighVolProb (nextUniform (min lo hi) (max lo hi)) rng45
           in (fmap (\v -> clamp v 0 1) val, rng')
        (maxConformalWidth, rng47) =
          let (lo, hi) = ordered maxConformalWidthRange
              lo' = max 1e-12 (min lo hi)
              hi' = max lo' (max lo hi)
           in nextMaybe pDisableMaxConformalWidth (nextLogUniform lo' hi') rng46
        (maxQuantileWidth, rng48) =
          let (lo, hi) = ordered maxQuantileWidthRange
              lo' = max 1e-12 (min lo hi)
              hi' = max lo' (max lo hi)
           in nextMaybe pDisableMaxQuantileWidth (nextLogUniform lo' hi') rng47
        (confirmConformal, rng49) =
          let (r, rng') = nextDouble rng48
           in (r < clamp pConfirmConformal 0 1, rng')
        (confirmQuantiles, rng50) =
          let (r, rng') = nextDouble rng49
           in (r < clamp pConfirmQuantiles 0 1, rng')
        (confidenceSizing, rng51) =
          let (r, rng') = nextDouble rng50
           in (r < clamp pConfidenceSizing 0 1, rng')
        (minPositionSize, rng52) =
          if confidenceSizing
            then
              let (lo, hi) = ordered minPositionSizeRange
                  (val, rng') = nextUniform lo hi rng51
               in (clamp val 0 1, rng')
            else (0, rng51)
        (stopLoss, rng53) = nextMaybe pDisableStop (nextLogUniform (fst stopRange) (snd stopRange)) rng52
        (takeProfit, rng54) = nextMaybe pDisableTp (nextLogUniform (fst takeRange) (snd takeRange)) rng53
        (trailingStop, rng55) = nextMaybe pDisableTrail (nextLogUniform (fst trailRange) (snd trailRange)) rng54
        (stopLossVolMult, rng56) =
          if max (fst stopVolMultRange) (snd stopVolMultRange) > 0
            then
              let lo = max 1e-6 (min (fst stopVolMultRange) (snd stopVolMultRange))
                  hi = max lo (max (fst stopVolMultRange) (snd stopVolMultRange))
               in nextMaybe pDisableStopVolMult (nextLogUniform lo hi) rng55
            else (Nothing, rng55)
        (takeProfitVolMult, rng57) =
          if max (fst takeVolMultRange) (snd takeVolMultRange) > 0
            then
              let lo = max 1e-6 (min (fst takeVolMultRange) (snd takeVolMultRange))
                  hi = max lo (max (fst takeVolMultRange) (snd takeVolMultRange))
               in nextMaybe pDisableTpVolMult (nextLogUniform lo hi) rng56
            else (Nothing, rng56)
        (trailingStopVolMult, rng58) =
          if max (fst trailVolMultRange) (snd trailVolMultRange) > 0
            then
              let lo = max 1e-6 (min (fst trailVolMultRange) (snd trailVolMultRange))
                  hi = max lo (max (fst trailVolMultRange) (snd trailVolMultRange))
               in nextMaybe pDisableTrailVolMult (nextLogUniform lo hi) rng57
            else (Nothing, rng57)
        (maxDrawdown, rng59) = nextMaybe pDisableMaxDd (nextUniform (fst maxDdRange) (snd maxDdRange)) rng58
        (maxDailyLoss, rng60) = nextMaybe pDisableMaxDl (nextUniform (fst maxDlRange) (snd maxDlRange)) rng59
        (maxOrderErrors, rng61) =
          nextMaybe
            pDisableMaxOe
            (\r ->
                let (val, r') = nextIntRange (fst maxOeRange) (snd maxOeRange) r
                 in (val, r')
            )
            rng60
     in ( TrialParams
            { tpPlatform = platform
            , tpInterval = interval
            , tpBars = bars
            , tpMethod = method
            , tpBlendWeight = blendWeight
            , tpPositioning = positioning
            , tpNormalization = normalization
            , tpBaseOpenThreshold = baseOpenThreshold
            , tpBaseCloseThreshold = baseCloseThreshold
            , tpMinHoldBars = minHoldBars
            , tpCooldownBars = cooldownBars
            , tpMaxHoldBars = maxHoldBars
            , tpMinEdge = minEdge
            , tpMinSignalToNoise = minSignalToNoise
            , tpEdgeBuffer = edgeBuffer'
            , tpCostAwareEdge = costAwareEdge
            , tpTrendLookback = trendLookback
            , tpMaxPositionSize = maxPositionSize
            , tpVolTarget = volTarget
            , tpVolLookback = volLookback
            , tpVolEwmaAlpha = volEwmaAlpha
            , tpVolFloor = volFloor
            , tpVolScaleMax = volScaleMax
            , tpMaxVolatility = maxVolatility
            , tpPeriodsPerYear = periodsPerYear
            , tpKalmanMarketTopN = kalmanMarketTopN
            , tpFee = fee
            , tpEpochs = epochs
            , tpHiddenSize = hiddenSize
            , tpLearningRate = learningRate
            , tpValRatio = valRatio
            , tpPatience = patience
            , tpWalkForwardFolds = walkForwardFolds
            , tpTuneStressVolMult = tuneStressVolMult
            , tpTuneStressShock = tuneStressShock
            , tpTuneStressWeight = tuneStressWeight
            , tpGradClip = gradClip
            , tpSlippage = slippage
            , tpSpread = spread
            , tpIntrabarFill = intrabarFill
            , tpStopLoss = stopLoss
            , tpTakeProfit = takeProfit
            , tpTrailingStop = trailingStop
            , tpStopLossVolMult = stopLossVolMult
            , tpTakeProfitVolMult = takeProfitVolMult
            , tpTrailingStopVolMult = trailingStopVolMult
            , tpMaxDrawdown = maxDrawdown
            , tpMaxDailyLoss = maxDailyLoss
            , tpMaxOrderErrors = maxOrderErrors
            , tpKalmanDt = kalmanDt
            , tpKalmanProcessVar = kalmanProcessVar
            , tpKalmanMeasurementVar = kalmanMeasurementVar
            , tpKalmanZMin = kalmanZMin
            , tpKalmanZMax = kalmanZMax
            , tpMaxHighVolProb = maxHighVolProb
            , tpMaxConformalWidth = maxConformalWidth
            , tpMaxQuantileWidth = maxQuantileWidth
            , tpConfirmConformal = confirmConformal
            , tpConfirmQuantiles = confirmQuantiles
            , tpConfidenceSizing = confidenceSizing
            , tpMinPositionSize = minPositionSize
            }
        , rng61
        )
  where
    ordered (a, b) = if a <= b then (a, b) else (b, a)
    sampleBars rng =
      let (r, rng') = nextDouble rng
       in if r < clamp pAutoBars 0 1
            then (0, rng')
            else
              if map toLower (trim barsDistribution) == "log"
                then
                  let lo = max 2 barsMin
                      hi = max lo barsMax
                      (val, rng'') = nextLogUniform (fromIntegral lo) (fromIntegral hi) rng'
                   in (round val, rng'')
                else nextIntRange barsMin barsMax rng'
    chooseWeighted options rng =
      let weights = map (max 0 . snd) options
          total = sum weights
       in if total <= 0
            then
              let (choice, rng') = nextChoice (map fst options) rng
               in (choice, rng')
            else
              let (r, rng') = nextDouble rng
                  target = r * total
                  pick acc ((name, w) : rest) =
                    let acc' = acc + max 0 w
                     in if target <= acc' then name else pick acc' rest
                  pick _ [] = fst (last options)
               in (pick 0 options, rng')

runOptimizer :: OptimizerArgs -> IO Int
runOptimizer args0 = do
  let args = if oaQuality args0 then applyQualityPreset args0 else args0
  traderBin <- resolveTraderBin args
  case traderBin of
    Left err -> do
      hPutStrLn stderr err
      pure 2
    Right traderBinPath -> do
      intervalsIn <- resolveIntervals args
      case intervalsIn of
        Left err -> do
          hPutStrLn stderr err
          pure 2
        Right intervalsRaw -> do
          csvInfo <-
            case oaData args of
              Nothing -> pure (Nothing, 1000, Nothing)
              Just raw -> do
                p <- expandUser raw
                exists <- doesFileExist p
                if not exists
                  then do
                    hPutStrLn stderr ("--data not found: " ++ p)
                    pure (Just (Left "missing"), 0, Nothing)
                  else do
                    let autoDetect = oaAutoHighLow args && null (oaHighColumn args) && null (oaLowColumn args)
                    (highCol, lowCol) <-
                      if autoDetect
                        then do
                          (h, l) <- detectHighLowColumns p
                          case (h, l) of
                            (Just hc, Just lc) -> do
                              hPutStrLn stderr ("Auto-detected high/low columns: " ++ hc ++ "/" ++ lc)
                              pure (Just hc, Just lc)
                            _ -> pure (Nothing, Nothing)
                        else pure (Nothing, Nothing)
                    rows <- countCsvRows p
                    let barsCap = max 2 (max 0 (rows - 1))
                        cols =
                          if autoDetect
                            then (highCol, lowCol)
                            else
                              let hCol = trim (oaHighColumn args)
                                  lCol = trim (oaLowColumn args)
                               in (if null hCol then Nothing else Just hCol, if null lCol then Nothing else Just lCol)
                    pure (Nothing, barsCap, Just (p, cols))
          case csvInfo of
            (Just (Left _), _, _) -> pure 2
            (_, maxBarsCap, csvCols) -> do
              let (platforms, platformIntervalsMap, intervalsResolved) =
                    if oaBinanceSymbol args == Nothing
                      then ([], [], pickIntervals intervalsRaw (oaLookbackWindow args) maxBarsCap)
                      else
                        let rawPlatforms =
                              case oaPlatform args of
                                Just v | not (null (trim v)) -> v
                                _ ->
                                  case oaPlatforms args of
                                    Just v | not (null (trim v)) -> v
                                    _ -> "binance"
                         in case parsePlatforms rawPlatforms of
                              Left e -> ([], [], Left e)
                              Right ps ->
                                case resolvePlatformIntervals ps intervalsRaw (oaLookbackWindow args) maxBarsCap of
                                  Left e -> ([], [], Left e)
                                  Right platformIntervalsMap' ->
                                    let allIntervals = sort (unique (concatMap snd platformIntervalsMap'))
                                     in (ps, platformIntervalsMap', Right allIntervals)
              case intervalsResolved of
                Left err -> do
                  hPutStrLn stderr err
                  pure 2
                Right intervals ->
                  if null intervals
                    then do
                      hPutStrLn stderr
                        ( "No feasible intervals for lookback-window="
                            ++ show (oaLookbackWindow args)
                            ++ " and max bars="
                            ++ show maxBarsCap
                            ++ "."
                        )
                      pure 2
                    else do
                      let useSweepThreshold = not (oaNoSweepThreshold args)
                      barsResult <- resolveBars args intervals maxBarsCap useSweepThreshold
                      case barsResult of
                        Left err -> do
                          hPutStrLn stderr err
                          pure 2
                        Right (barsMin, barsMax) -> do
                          let barsAutoProb = clamp (oaBarsAutoProb args) 0 1
                              barsDistribution = map toLower (trim (oaBarsDistribution args))
                              epochsMin = max 0 (oaEpochsMin args)
                              epochsMax = max epochsMin (oaEpochsMax args)
                              hiddenMin = max 1 (oaHiddenSizeMin args)
                              hiddenMax = max hiddenMin (oaHiddenSizeMax args)
                              lrMin = max 1e-9 (oaLrMin args)
                              lrMax = max lrMin (oaLrMax args)
                              valMin = clamp (oaValRatioMin args) 0 0.9
                              valMax = max valMin (oaValRatioMax args)
                              patienceMax = max 0 (oaPatienceMax args)
                              gradClipMin = max 1e-9 (oaGradClipMin args)
                              gradClipMax = max gradClipMin (oaGradClipMax args)
                              walkForwardFoldsMin = max 1 (oaWalkForwardFoldsMin args)
                              walkForwardFoldsMax = max walkForwardFoldsMin (oaWalkForwardFoldsMax args)
                              tuneStressVolMultBase = max 1e-12 (oaTuneStressVolMult args)
                              tuneStressShockBase = oaTuneStressShock args
                              tuneStressWeightBase = max 0 (oaTuneStressWeight args)
                              (tuneStressVolMultRange, tuneStressShockRange, tuneStressWeightRange) =
                                resolveStressRanges
                                  tuneStressVolMultBase
                                  tuneStressShockBase
                                  tuneStressWeightBase
                                  (oaTuneStressVolMultMin args, oaTuneStressVolMultMax args)
                                  (oaTuneStressShockMin args, oaTuneStressShockMax args)
                                  (oaTuneStressWeightMin args, oaTuneStressWeightMax args)
                              stopRange = (oaStopMin args, oaStopMax args)
                              takeRange = (oaTpMin args, oaTpMax args)
                              trailRange = (oaTrailMin args, oaTrailMax args)
                              stopVolMultRange = (max 0 (oaStopVolMultMin args), max 0 (oaStopVolMultMax args))
                              takeVolMultRange = (max 0 (oaTpVolMultMin args), max 0 (oaTpVolMultMax args))
                              trailVolMultRange = (max 0 (oaTrailVolMultMin args), max 0 (oaTrailVolMultMax args))
                              feeMin = max 0 (oaFeeMin args)
                              feeMax = max feeMin (oaFeeMax args)
                              openThresholdMin = max 1e-12 (oaOpenThresholdMin args)
                              openThresholdMax = max openThresholdMin (oaOpenThresholdMax args)
                              closeThresholdMin = max 1e-12 (oaCloseThresholdMin args)
                              closeThresholdMax = max closeThresholdMin (oaCloseThresholdMax args)
                              minHoldMin = max 0 (oaMinHoldBarsMin args)
                              minHoldMax = max minHoldMin (oaMinHoldBarsMax args)
                              cooldownMin = max 0 (oaCooldownBarsMin args)
                              cooldownMax = max cooldownMin (oaCooldownBarsMax args)
                              maxHoldMin = max 0 (oaMaxHoldBarsMin args)
                              maxHoldMax = max maxHoldMin (oaMaxHoldBarsMax args)
                              minEdgeMin = max 0 (oaMinEdgeMin args)
                              minEdgeMax = max minEdgeMin (oaMinEdgeMax args)
                              minSnMin = max 0 (oaMinSignalToNoiseMin args)
                              minSnMax = max minSnMin (oaMinSignalToNoiseMax args)
                              edgeBufferMin = max 0 (oaEdgeBufferMin args)
                              edgeBufferMax = max edgeBufferMin (oaEdgeBufferMax args)
                              trendLookbackMin = max 0 (oaTrendLookbackMin args)
                              trendLookbackMax = max trendLookbackMin (oaTrendLookbackMax args)
                              pLongShort = clamp (oaPLongShort args) 0 1
                              pIntrabarTakeProfitFirst = clamp (oaPIntrabarTakeProfitFirst args) 0 1
                              kalmanDtMin = max 1e-12 (oaKalmanDtMin args)
                              kalmanDtMax = max kalmanDtMin (oaKalmanDtMax args)
                              kalmanProcessVarMin = max 1e-12 (oaKalmanProcessVarMin args)
                              kalmanProcessVarMax = max kalmanProcessVarMin (oaKalmanProcessVarMax args)
                              kalmanMeasurementVarMin = max 1e-12 (oaKalmanMeasurementVarMin args)
                              kalmanMeasurementVarMax = max kalmanMeasurementVarMin (oaKalmanMeasurementVarMax args)
                              kalmanZMinMin = max 0 (oaKalmanZMinMin args)
                              kalmanZMinMax = max kalmanZMinMin (oaKalmanZMinMax args)
                              kalmanZMaxMin = max 0 (oaKalmanZMaxMin args)
                              kalmanZMaxMax = max kalmanZMaxMin (oaKalmanZMaxMax args)
                              kalmanMarketTopNMin = max 0 (oaKalmanMarketTopNMin args)
                              kalmanMarketTopNMax = max kalmanMarketTopNMin (oaKalmanMarketTopNMax args)
                              pDisableMaxHighVolProb = clamp (oaPDisableMaxHighVolProb args) 0 1
                              maxHighVolProbRange = (oaMaxHighVolProbMin args, oaMaxHighVolProbMax args)
                              pDisableMaxConformalWidth = clamp (oaPDisableMaxConformalWidth args) 0 1
                              maxConformalWidthRange = (oaMaxConformalWidthMin args, oaMaxConformalWidthMax args)
                              pDisableMaxQuantileWidth = clamp (oaPDisableMaxQuantileWidth args) 0 1
                              maxQuantileWidthRange = (oaMaxQuantileWidthMin args, oaMaxQuantileWidthMax args)
                              pConfirmConformal = clamp (oaPConfirmConformal args) 0 1
                              pConfirmQuantiles = clamp (oaPConfirmQuantiles args) 0 1
                              pConfidenceSizing = clamp (oaPConfidenceSizing args) 0 1
                              minPositionSizeRange = (oaMinPositionSizeMin args, oaMinPositionSizeMax args)
                              maxPositionSizeMin = max 0 (oaMaxPositionSizeMin args)
                              maxPositionSizeMax = max maxPositionSizeMin (oaMaxPositionSizeMax args)
                              maxPositionSizeRange = (maxPositionSizeMin, maxPositionSizeMax)
                              volTargetMin = max 0 (oaVolTargetMin args)
                              volTargetMax = max volTargetMin (oaVolTargetMax args)
                              volTargetRange = (volTargetMin, volTargetMax)
                              pDisableVolTarget = clamp (oaPDisableVolTarget args) 0 1
                              volLookbackMin = max 0 (oaVolLookbackMin args)
                              volLookbackMax = max volLookbackMin (oaVolLookbackMax args)
                              volLookbackRange = (volLookbackMin, volLookbackMax)
                              volEwmaAlphaMin = max 0 (oaVolEwmaAlphaMin args)
                              volEwmaAlphaMax = max volEwmaAlphaMin (oaVolEwmaAlphaMax args)
                              volEwmaAlphaRange = (volEwmaAlphaMin, volEwmaAlphaMax)
                              pDisableVolEwmaAlpha = clamp (oaPDisableVolEwmaAlpha args) 0 1
                              volFloorMin = max 0 (oaVolFloorMin args)
                              volFloorMax = max volFloorMin (oaVolFloorMax args)
                              volFloorRange = (volFloorMin, volFloorMax)
                              volScaleMaxMin = max 0 (oaVolScaleMaxMin args)
                              volScaleMaxMax = max volScaleMaxMin (oaVolScaleMaxMax args)
                              volScaleMaxRange = (volScaleMaxMin, volScaleMaxMax)
                              maxVolatilityMin = max 0 (oaMaxVolatilityMin args)
                              maxVolatilityMax = max maxVolatilityMin (oaMaxVolatilityMax args)
                              maxVolatilityRange = (maxVolatilityMin, maxVolatilityMax)
                              pDisableMaxVolatility = clamp (oaPDisableMaxVolatility args) 0 1
                              periodsPerYearMin = max 0 (oaPeriodsPerYearMin args)
                              periodsPerYearMax = max periodsPerYearMin (oaPeriodsPerYearMax args)
                              periodsPerYearRange = (periodsPerYearMin, periodsPerYearMax)
                              methodWeights =
                                ( oaMethodWeight11 args
                                , oaMethodWeight10 args
                                , oaMethodWeight01 args
                                , oaMethodWeightBlend args
                                )
                              blendWeightRange =
                                let lo = clamp (oaBlendWeightMin args) 0 1
                                    hi = clamp (oaBlendWeightMax args) 0 1
                                 in (lo, hi)
                              normalizationChoices =
                                [trim s | s <- splitCsv (oaNormalizations args), not (null (trim s))]
                          if null normalizationChoices
                            then do
                              hPutStrLn stderr "No normalizations provided."
                              pure 2
                            else do
                              baseArgsResult <- buildBaseArgs args csvCols
                              case baseArgsResult of
                                Left err -> do
                                  hPutStrLn stderr err
                                  pure 2
                                Right baseArgs -> do
                                  let rngStart = seedRng (oaSeed args)
                                      dataSource = if oaData args == Nothing then "binance" else "csv"
                                      sourceOverride = map toLower (trim (oaSourceLabel args))
                                      symbolLabel = normalizeSymbol (Just (oaSymbolLabel args))
                                      symbolFallback = normalizeSymbol (oaBinanceSymbol args)
                                      symbolFinal = case symbolLabel of
                                        Just _ -> symbolLabel
                                        Nothing -> symbolFallback
                                      trials = max 1 (oaTrials args)
                                      minRoundTrips = max 0 (oaMinRoundTrips args)
                                      minWinRate = max 0 (oaMinWinRate args)
                                      minProfitFactor = max 0 (oaMinProfitFactor args)
                                      minExposure = max 0 (oaMinExposure args)
                                      minSharpe = max 0 (oaMinSharpe args)
                                      minWfSharpeMean = max 0 (oaMinWfSharpeMean args)
                                      maxWfSharpeStd = max 0 (oaMaxWfSharpeStd args)
                                  outHandle <-
                                    if null (trim (oaOutput args))
                                      then pure Nothing
                                      else do
                                        p <- expandUser (oaOutput args)
                                        createDirectoryIfMissing True (takeDirectory p)
                                        h <- openFile p (if oaAppend args then AppendMode else WriteMode)
                                        hSetEncoding h utf8
                                        pure (Just h)
                                  let loop i rng best records
                                        | i > trials = pure (best, reverse records)
                                        | otherwise = do
                                            let (params, rng') =
                                                  sampleParams
                                                    rng
                                                    platforms
                                                    platformIntervalsMap
                                                    intervals
                                                    barsAutoProb
                                                    barsMin
                                                    barsMax
                                                    barsDistribution
                                                    openThresholdMin
                                                    openThresholdMax
                                                    closeThresholdMin
                                                    closeThresholdMax
                                                    (minHoldMin, minHoldMax)
                                                    (cooldownMin, cooldownMax)
                                                    (maxHoldMin, maxHoldMax)
                                                    (minEdgeMin, minEdgeMax)
                                                    (minSnMin, minSnMax)
                                                    (edgeBufferMin, edgeBufferMax)
                                                    (trendLookbackMin, trendLookbackMax)
                                                    maxPositionSizeRange
                                                    volTargetRange
                                                    volLookbackRange
                                                    volEwmaAlphaRange
                                                    pDisableVolEwmaAlpha
                                                    volFloorRange
                                                    volScaleMaxRange
                                                    maxVolatilityRange
                                                    periodsPerYearRange
                                                    (kalmanMarketTopNMin, kalmanMarketTopNMax)
                                                    (oaPCostAwareEdge args)
                                                    feeMin
                                                    feeMax
                                                    pLongShort
                                                    pIntrabarTakeProfitFirst
                                                    epochsMin
                                                    epochsMax
                                                    hiddenMin
                                                    hiddenMax
                                                    lrMin
                                                    lrMax
                                                    valMin
                                                    valMax
                                                    patienceMax
                                                    (walkForwardFoldsMin, walkForwardFoldsMax)
                                                    tuneStressVolMultRange
                                                    tuneStressShockRange
                                                    tuneStressWeightRange
                                                    gradClipMin
                                                    gradClipMax
                                                    (oaSlippageMax args)
                                                    (oaSpreadMax args)
                                                    kalmanDtMin
                                                    kalmanDtMax
                                                    kalmanProcessVarMin
                                                    kalmanProcessVarMax
                                                    kalmanMeasurementVarMin
                                                    kalmanMeasurementVarMax
                                                    kalmanZMinMin
                                                    kalmanZMinMax
                                                    kalmanZMaxMin
                                                    kalmanZMaxMax
                                                    pDisableMaxHighVolProb
                                                    maxHighVolProbRange
                                                    pDisableMaxConformalWidth
                                                    maxConformalWidthRange
                                                    pDisableMaxQuantileWidth
                                                    maxQuantileWidthRange
                                                    pConfirmConformal
                                                    pConfirmQuantiles
                                                    pConfidenceSizing
                                                    minPositionSizeRange
                                                    stopRange
                                                    takeRange
                                                    trailRange
                                                    stopVolMultRange
                                                    takeVolMultRange
                                                    trailVolMultRange
                                                    methodWeights
                                                    normalizationChoices
                                                    blendWeightRange
                                                    (clamp (oaPDisableStop args) 0 1)
                                                    (clamp (oaPDisableTp args) 0 1)
                                                    (clamp (oaPDisableTrail args) 0 1)
                                                    (clamp (oaPDisableStopVolMult args) 0 1)
                                                    (clamp (oaPDisableTpVolMult args) 0 1)
                                                    (clamp (oaPDisableTrailVolMult args) 0 1)
                                                    (clamp (oaPDisableMaxDd args) 0 1)
                                                    (clamp (oaPDisableMaxDl args) 0 1)
                                                    (clamp (oaPDisableMaxOe args) 0 1)
                                                    (clamp (oaPDisableGradClip args) 0 1)
                                                    pDisableVolTarget
                                                    pDisableMaxVolatility
                                                    (oaMaxDdMin args, oaMaxDdMax args)
                                                    (oaMaxDlMin args, oaMaxDlMax args)
                                                    (oaMaxOeMin args, oaMaxOeMax args)
                                            tr0 <-
                                              runTrial
                                                traderBinPath
                                                baseArgs
                                                params
                                                (oaTuneRatio args)
                                                useSweepThreshold
                                                (oaTimeoutSec args)
                                                (oaDisableLstmPersistence args)
                                            let objective = oaObjective args
                                                (eligible, filterReason, score) =
                                                  case (trOk tr0, trFinalEquity tr0, trMetrics tr0) of
                                                    (True, Just _, Just metrics) ->
                                                      let rts = metricInt (trMetrics tr0) "roundTrips" 0
                                                       in if minRoundTrips > 0 && rts < minRoundTrips
                                                            then (False, Just ("roundTrips<" ++ show minRoundTrips), Nothing)
                                                            else
                                                              let winRate = metricFloat (trMetrics tr0) "winRate" 0
                                                               in if minWinRate > 0 && winRate < minWinRate
                                                                    then (False, Just (printf "winRate<%.3f" minWinRate), Nothing)
                                                                    else
                                                                      let profitFactor = metricProfitFactor (trMetrics tr0)
                                                                       in if minProfitFactor > 0 && profitFactor < minProfitFactor
                                                                            then (False, Just (printf "profitFactor<%.3f" minProfitFactor), Nothing)
                                                                            else
                                                                              let exposure = metricFloat (trMetrics tr0) "exposure" 0
                                                                               in if minExposure > 0 && exposure < minExposure
                                                                                    then (False, Just (printf "exposure<%.3f" minExposure), Nothing)
                                                                                    else
                                                                                      let sharpe = metricFloat (trMetrics tr0) "sharpe" 0
                                                                                       in if minSharpe > 0 && sharpe < minSharpe
                                                                                            then (False, Just (printf "sharpe<%.3f" minSharpe), Nothing)
                                                                                            else
                                                                                              if minWfSharpeMean > 0 || maxWfSharpeStd > 0
                                                                                                then
                                                                                                  case extractWalkForwardSummary (trStdoutJson tr0) of
                                                                                                    Nothing -> (False, Just "walkForwardMissing", Nothing)
                                                                                                    Just wfSummary ->
                                                                                                      let wfSharpeMean = metricFloat (Just wfSummary) "sharpeMean" 0
                                                                                                          wfSharpeStd = metricFloat (Just wfSummary) "sharpeStd" 0
                                                                                                       in if minWfSharpeMean > 0 && wfSharpeMean < minWfSharpeMean
                                                                                                            then (False, Just (printf "wfSharpeMean<%.3f" minWfSharpeMean), Nothing)
                                                                                                            else if maxWfSharpeStd > 0 && wfSharpeStd > maxWfSharpeStd
                                                                                                              then (False, Just (printf "wfSharpeStd>%.3f" maxWfSharpeStd), Nothing)
                                                                                                              else
                                                                                                                ( True
                                                                                                                , Nothing
                                                                                                                , Just
                                                                                                                    ( objectiveScore
                                                                                                                        metrics
                                                                                                                        objective
                                                                                                                        (oaPenaltyMaxDrawdown args)
                                                                                                                        (oaPenaltyTurnover args)
                                                                                                                    )
                                                                                                                )
                                                                                                else
                                                                                                  ( True
                                                                                                  , Nothing
                                                                                                  , Just
                                                                                                      ( objectiveScore
                                                                                                          metrics
                                                                                                          objective
                                                                                                          (oaPenaltyMaxDrawdown args)
                                                                                                          (oaPenaltyTurnover args)
                                                                                                      )
                                                                                                  )
                                                    _ -> (False, Nothing, Nothing)
                                                tr =
                                                  tr0
                                                    { trEligible = eligible
                                                    , trFilterReason = filterReason
                                                    , trObjective = objective
                                                    , trScore = score
                                                    }
                                            case outHandle of
                                              Nothing -> pure ()
                                              Just h -> do
                                                let rec0 = trialToRecord tr symbolFinal
                                                    source = resolveSourceLabel (tpPlatform params) dataSource sourceOverride
                                                    rec = addField "source" (String (T.pack source)) rec0
                                                BL.hPutStr h (Aeson.encode rec)
                                                hPutStrLn h ""
                                                hFlush h
                                            let best' =
                                                  case (trEligible tr, trScore tr, best) of
                                                    (True, Just sc, Nothing) -> Just tr
                                                    (True, Just sc, Just b) ->
                                                      let bScore = fromMaybe (-1e18) (trScore b)
                                                       in if sc > bScore then Just tr else Just b
                                                    _ -> best
                                            printTrialStatus i trials tr
                                            loop (i + 1) rng' best' (tr : records)
                                  (best, records) <- loop 1 rngStart Nothing []
                                  case outHandle of
                                    Nothing -> pure ()
                                    Just h -> hClose h
                                  case best of
                                    Nothing -> do
                                      printNoEligible minRoundTrips minWinRate minProfitFactor minExposure minSharpe minWfSharpeMean maxWfSharpeStd
                                      pure 1
                                    Just b -> do
                                      printBest b
                                      printRepro traderBinPath baseArgs b (oaTuneRatio args) useSweepThreshold (oaDisableLstmPersistence args)
                                      when (not (null (trim (oaTopJson args)))) $ do
                                        writeTopJson
                                          (oaTopJson args)
                                          dataSource
                                          sourceOverride
                                          symbolFinal
                                          records
                                      pure 0
  where
    resolveStressRanges baseVol baseShock baseWeight volRange shockRange weightRange =
      let (vMin0, vMax0) = fillRange baseVol volRange
          vMin = max 1e-12 vMin0
          vMax = max 1e-12 vMax0
          (sMin, sMax) = fillRange baseShock shockRange
          (wMin0, wMax0) = fillRange baseWeight weightRange
          wMin = max 0 wMin0
          wMax = max 0 wMax0
       in ((min vMin vMax, max vMin vMax), (min sMin sMax, max sMin sMax), (min wMin wMax, max wMin wMax))
    fillRange base (mn, mx) =
      case (mn, mx) of
        (Nothing, Nothing) -> (base, base)
        (Just a, Nothing) -> (a, a)
        (Nothing, Just b) -> (b, b)
        (Just a, Just b) -> (a, b)
    unique = foldl' (\acc v -> if v `elem` acc then acc else acc ++ [v]) []

resolveTraderBin :: OptimizerArgs -> IO (Either String FilePath)
resolveTraderBin args =
  if not (null (trim (oaBinary args)))
    then do
      p <- expandUser (oaBinary args)
      pure (Right p)
    else do
      cwd <- getCurrentDirectory
      let procSpec = (proc "cabal" ["list-bin", "trader-hs"]) {cwd = Just cwd}
      r <- try (readProcessOutput procSpec) :: IO (Either SomeException (ExitCode, String, String))
      case r of
        Left e -> pure (Left ("failed to discover trader-hs binary via cabal: " ++ show e))
        Right (ExitFailure _, out, err) ->
          pure (Left ("failed to discover trader-hs binary via cabal: " ++ trim (if null err then out else err)))
        Right (ExitSuccess, out, _) ->
          let p = trim out
           in if null p
                then pure (Left "cabal returned empty binary path")
                else do
                  exists <- doesFileExist p
                  if exists
                    then pure (Right p)
                    else pure (Left ("cabal returned non-existent binary path: " ++ p))

readProcessOutput :: CreateProcess -> IO (ExitCode, String, String)
readProcessOutput procSpec = do
  (_, Just hout, Just herr, ph) <- createProcess procSpec
  hSetEncoding hout utf8
  hSetEncoding herr utf8
  out <- hGetContents hout
  err <- hGetContents herr
  _ <- evaluate (length out + length err)
  code <- waitForProcess ph
  pure (code, out, err)

resolveIntervals :: OptimizerArgs -> IO (Either String [String])
resolveIntervals args =
  let intervalsIn =
        case oaInterval args of
          Just v | not (null (trim v)) -> [trim v]
          _ ->
            let raw = fromMaybe binanceIntervalsCsv (oaIntervals args)
             in [trim s | s <- splitCsv raw, not (null (trim s))]
   in if null intervalsIn then pure (Left "No intervals provided.") else pure (Right intervalsIn)

splitCsv :: String -> [String]
splitCsv raw =
  case raw of
    [] -> []
    _ ->
      let go acc current [] = reverse (reverse current : acc)
          go acc current (c : cs)
            | c == ',' = go (reverse current : acc) [] cs
            | otherwise = go acc (c : current) cs
       in map trim (go [] [] raw)

resolveBars :: OptimizerArgs -> [String] -> Int -> Bool -> IO (Either String (Int, Int))
resolveBars args intervals maxBarsCap useSweepThreshold = do
  let barsMax0 = oaBarsMax args
      barsMax = if barsMax0 <= 0 then maxBarsCap else barsMax0
      barsMin0 = oaBarsMin args
  if barsMin0 > 0
    then do
      let barsMin = max 2 (min barsMin0 barsMax)
      pure (Right (barsMin, barsMax))
    else do
      let worstLb = maximum (map (either (const 0) id . lookbackBarsFrom' (oaLookbackWindow args)) intervals)
          minRequired0 = worstLb + 3
      if useSweepThreshold
        then do
          let br = oaBacktestRatio args
              tr = oaTuneRatio args
          if br <= 0 || br >= 1
            then pure (Left "--backtest-ratio must be between 0 and 1.")
            else
              if tr <= 0 || tr >= 1
                then pure (Left "--tune-ratio must be between 0 and 1 when sweep-threshold is enabled.")
                else do
                  let denom = max 1e-12 ((1 - br) * (1 - tr))
                      minRequired1 = max minRequired0 (ceiling ((fromIntegral worstLb + 1) / denom) + 2)
                      minTrain = ceiling (2 / tr)
                      minRequired2 =
                        max minRequired1 (ceiling (fromIntegral minTrain / max 1e-12 (1 - br)) + 2)
                      autoBars = if oaBinanceSymbol args == Nothing then maxBarsCap else 500
                  if minRequired2 > max barsMax autoBars
                    then
                      pure
                        ( Left
                            ( "Not enough bars for lookback="
                                ++ show worstLb
                                ++ " with backtest-ratio="
                                ++ show br
                                ++ " and tune-ratio="
                                ++ show tr
                                ++ ". Need bars >= "
                                ++ show minRequired2
                                ++ ". Increase --bars-max, reduce --tune-ratio/--backtest-ratio, reduce --lookback-window, or pass --no-sweep-threshold."
                            )
                        )
                    else do
                      let barsMin = min barsMax (max 10 minRequired2)
                      pure (Right (max 2 (min barsMin barsMax), barsMax))
        else do
          let barsMin = min barsMax (max 10 minRequired0)
          pure (Right (max 2 (min barsMin barsMax), barsMax))
  where
    lookbackBarsFrom' lookbackWindow itv =
      case lookbackBarsFrom itv lookbackWindow of
        Left _ -> Left "invalid"
        Right lb -> Right lb

buildBaseArgs :: OptimizerArgs -> Maybe (FilePath, (Maybe String, Maybe String)) -> IO (Either String [String])
buildBaseArgs args csvCols = do
  let base0 = []
  base1 <-
    case oaData args of
      Just _ ->
        case csvCols of
          Nothing -> pure (Left "CSV path not resolved")
          Just (csvPath, (mHigh, mLow)) -> do
            let base = base0 ++ ["--data", csvPath, "--price-column", oaPriceColumn args]
            case (mHigh, mLow) of
              (Nothing, Nothing) -> pure (Right base)
              (Just h, Just l) -> pure (Right (base ++ ["--high-column", h, "--low-column", l]))
              _ -> pure (Left "When using --high-column/--low-column, you must provide both.")
      Nothing ->
        case oaBinanceSymbol args of
          Just sym -> pure (Right (base0 ++ ["--binance-symbol", sym]))
          Nothing -> pure (Left "--symbol/--data is required")
  case base1 of
    Left err -> pure (Left err)
    Right baseArgs -> do
      let baseArgs' =
            baseArgs
              ++ [ "--lookback-window"
                 , oaLookbackWindow args
                 , "--backtest-ratio"
                 , printf "%.6f" (oaBacktestRatio args)
                 , "--tune-objective"
                 , oaTuneObjective args
                 , "--tune-penalty-max-drawdown"
                 , printf "%.6f" (max 0 (oaTunePenaltyMaxDrawdown args))
                 , "--tune-penalty-turnover"
                 , printf "%.6f" (max 0 (oaTunePenaltyTurnover args))
                 , "--seed"
                 , show (oaSeed args)
                 ]
      pure (Right baseArgs')

printTrialStatus :: Int -> Int -> TrialResult -> IO ()
printTrialStatus i trials tr = do
  let status = if trEligible tr then "OK" else if trOk tr then "SKIP" else "FAIL"
      eq = maybe "-" (\v -> printf "%.6fx" v) (trFinalEquity tr)
      scoreLabel = maybe "-" (\v -> printf "%.6f" v) (trScore tr)
      params = trParams tr
      msg =
        printf
          "[%4d/%d] %s score=%s eq=%s t=%.2fs interval=%s bars=%d method=%s norm=%s epochs=%d slip=%.6f spr=%.6f sl=%s tp=%s trail=%s maxDD=%s maxDL=%s maxOE=%s"
          i
          trials
          status
          scoreLabel
          eq
          (trElapsedSec tr)
          (tpInterval params)
          (tpBars params)
          (tpMethod params)
          (tpNormalization params)
          (tpEpochs params)
          (tpSlippage params)
          (tpSpread params)
          (fmtOptFloat (tpStopLoss params))
          (fmtOptFloat (tpTakeProfit params))
          (fmtOptFloat (tpTrailingStop params))
          (fmtOptFloat (tpMaxDrawdown params))
          (fmtOptFloat (tpMaxDailyLoss params))
          (fmtOptInt (tpMaxOrderErrors params))
      suffix =
        case (trFilterReason tr, trReason tr) of
          (Just reason, _) -> " (filter: " ++ reason ++ ")"
          (Nothing, Just reason) -> " (" ++ reason ++ ")"
          _ -> ""
  putStrLn (msg ++ suffix)
  hFlush stdout

printNoEligible :: Int -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> IO ()
printNoEligible minRoundTrips minWinRate minProfitFactor minExposure minSharpe minWfSharpeMean maxWfSharpeStd = do
  let hints =
        [ "--min-round-trips" | minRoundTrips > 0 ]
          ++ [ "--min-win-rate" | minWinRate > 0 ]
          ++ [ "--min-profit-factor" | minProfitFactor > 0 ]
          ++ [ "--min-exposure" | minExposure > 0 ]
          ++ [ "--min-sharpe" | minSharpe > 0 ]
          ++ [ "--min-wf-sharpe-mean" | minWfSharpeMean > 0 ]
          ++ [ "--max-wf-sharpe-std" | maxWfSharpeStd > 0 ]
      msg =
        if null hints
          then "No eligible trials."
          else "No eligible trials. (Try lowering " ++ intercalate ", " hints ++ ".)"
  hPutStrLn stderr msg

printBest :: TrialResult -> IO ()
printBest tr = do
  let p = trParams tr
  putStrLn "\nBest:"
  case trScore tr of
    Just sc -> putStrLn ("  objective:   " ++ trObjective tr ++ " (score=" ++ printf "%.8f" sc ++ ")")
    Nothing -> pure ()
  putStrLn ("  finalEquity: " ++ printf "%.8f" (fromMaybe 0 (trFinalEquity tr)) ++ "x")
  case tpPlatform p of
    Just platform -> putStrLn ("  platform:    " ++ platform)
    Nothing -> pure ()
  putStrLn ("  interval:    " ++ tpInterval p)
  putStrLn ("  bars:        " ++ show (tpBars p))
  putStrLn ("  method:      " ++ tpMethod p)
  putStrLn ("  positioning: " ++ tpPositioning p)
  putStrLn ("  thresholds:  open=" ++ showMaybe (trOpenThreshold tr) ++ " close=" ++ showMaybe (trCloseThreshold tr) ++ " (from sweep)")
  putStrLn ("  base thresholds: open=" ++ show (tpBaseOpenThreshold p) ++ " close=" ++ show (tpBaseCloseThreshold p))
  putStrLn ("  blendWeight:  " ++ show (tpBlendWeight p))
  putStrLn ("  minHoldBars:  " ++ show (tpMinHoldBars p))
  putStrLn ("  cooldownBars: " ++ show (tpCooldownBars p))
  putStrLn ("  maxHoldBars:  " ++ showMaybe (tpMaxHoldBars p))
  putStrLn ("  minEdge:      " ++ show (tpMinEdge p))
  putStrLn ("  minSignalToNoise: " ++ show (tpMinSignalToNoise p))
  putStrLn ("  edgeBuffer:   " ++ show (tpEdgeBuffer p))
  putStrLn ("  costAwareEdge:" ++ show (tpCostAwareEdge p))
  putStrLn ("  trendLookback:" ++ show (tpTrendLookback p))
  putStrLn ("  maxPositionSize:" ++ show (tpMaxPositionSize p))
  putStrLn ("  volTarget:    " ++ showMaybe (tpVolTarget p))
  putStrLn ("  volLookback:  " ++ show (tpVolLookback p))
  putStrLn ("  volEwmaAlpha: " ++ showMaybe (tpVolEwmaAlpha p))
  putStrLn ("  volFloor:     " ++ show (tpVolFloor p))
  putStrLn ("  volScaleMax:  " ++ show (tpVolScaleMax p))
  putStrLn ("  maxVolatility: " ++ showMaybe (tpMaxVolatility p))
  putStrLn ("  periodsPerYear:" ++ showMaybe (tpPeriodsPerYear p))
  putStrLn ("  kalmanMarketTopN:" ++ show (tpKalmanMarketTopN p))
  putStrLn ("  normalization: " ++ show (tpNormalization p))
  putStrLn ("  epochs:        " ++ show (tpEpochs p))
  putStrLn ("  hiddenSize:    " ++ show (tpHiddenSize p))
  putStrLn ("  lr:            " ++ show (tpLearningRate p))
  putStrLn ("  valRatio:      " ++ show (tpValRatio p))
  putStrLn ("  patience:      " ++ show (tpPatience p))
  putStrLn ("  walkForwardFolds:" ++ show (tpWalkForwardFolds p))
  putStrLn ("  tuneStressVolMult:" ++ show (tpTuneStressVolMult p))
  putStrLn ("  tuneStressShock: " ++ show (tpTuneStressShock p))
  putStrLn ("  tuneStressWeight:" ++ show (tpTuneStressWeight p))
  putStrLn ("  gradClip:      " ++ showMaybe (tpGradClip p))
  putStrLn ("  fee:           " ++ show (tpFee p))
  putStrLn ("  slippage:      " ++ show (tpSlippage p))
  putStrLn ("  spread:        " ++ show (tpSpread p))
  putStrLn ("  intrabarFill:  " ++ show (tpIntrabarFill p))
  putStrLn ("  stopLoss:      " ++ showMaybe (tpStopLoss p))
  putStrLn ("  takeProfit:    " ++ showMaybe (tpTakeProfit p))
  putStrLn ("  trailingStop:  " ++ showMaybe (tpTrailingStop p))
  putStrLn ("  stopLossVolMult:    " ++ showMaybe (tpStopLossVolMult p))
  putStrLn ("  takeProfitVolMult:  " ++ showMaybe (tpTakeProfitVolMult p))
  putStrLn ("  trailingStopVolMult:" ++ showMaybe (tpTrailingStopVolMult p))
  putStrLn ("  maxDrawdown:   " ++ showMaybe (tpMaxDrawdown p))
  putStrLn ("  maxDailyLoss:  " ++ showMaybe (tpMaxDailyLoss p))
  putStrLn ("  maxOrderErrors:" ++ showMaybe (tpMaxOrderErrors p))
  putStrLn ("  kalmanDt:            " ++ show (tpKalmanDt p))
  putStrLn ("  kalmanProcessVar:    " ++ show (tpKalmanProcessVar p))
  putStrLn ("  kalmanMeasurementVar:" ++ show (tpKalmanMeasurementVar p))
  putStrLn ("  kalmanZMin:          " ++ show (tpKalmanZMin p))
  putStrLn ("  kalmanZMax:          " ++ show (tpKalmanZMax p))
  putStrLn ("  maxHighVolProb:      " ++ showMaybe (tpMaxHighVolProb p))
  putStrLn ("  maxConformalWidth:   " ++ showMaybe (tpMaxConformalWidth p))
  putStrLn ("  maxQuantileWidth:    " ++ showMaybe (tpMaxQuantileWidth p))
  putStrLn ("  confirmConformal:    " ++ show (tpConfirmConformal p))
  putStrLn ("  confirmQuantiles:    " ++ show (tpConfirmQuantiles p))
  putStrLn ("  confidenceSizing:    " ++ show (tpConfidenceSizing p))
  putStrLn ("  minPositionSize:     " ++ show (tpMinPositionSize p))

showMaybe :: Show a => Maybe a -> String
showMaybe v =
  case v of
    Nothing -> "None"
    Just x -> show x

printRepro :: FilePath -> [String] -> TrialResult -> Double -> Bool -> Bool -> IO ()
printRepro traderBin baseArgs tr tuneRatio useSweepThreshold disableLstm = do
  putStrLn "\nRepro command:"
  let envPrefix = if disableLstm then "TRADER_LSTM_WEIGHTS_DIR='' " else ""
      cmd = buildCommand traderBin baseArgs (trParams tr) tuneRatio useSweepThreshold
  putStrLn ("  " ++ envPrefix ++ unwords cmd)

writeTopJson :: String -> String -> String -> Maybe String -> [TrialResult] -> IO ()
writeTopJson topPath dataSource sourceOverride symbolLabel records = do
  let successful =
        [ tr
        | tr <- records
        , trEligible tr
        , trFinalEquity tr /= Nothing
        , trScore tr /= Nothing
        ]
      sorted = sortBy (flip (comparing trScore)) successful
      combos = zipWith (comboFromTrial dataSource sourceOverride symbolLabel) [1 ..] (take 10 sorted)
  path <- expandUser topPath
  createDirectoryIfMissing True (takeDirectory path)
  nowMs <- fmap (floor . (* 1000)) getPOSIXTime
  let export' =
        object
          [ "generatedAtMs" .= nowMs
          , "source" .= ("optimize_equity.py" :: String)
          , "combos" .= combos
          ]
  BL.writeFile path (encodePretty export')
  putStrLn ("Wrote top combos JSON: " ++ path)

comboFromTrial :: String -> String -> Maybe String -> Int -> TrialResult -> Value
comboFromTrial dataSource sourceOverride symbolLabel rank tr =
  let metrics = trMetrics tr
      sharpe = metricFloat metrics "sharpe" 0
      maxDd = metricFloat metrics "maxDrawdown" 0
      turnover = metricFloat metrics "turnover" 0
      roundTrips = metricInt metrics "roundTrips" 0
      symbol = normalizeSymbol symbolLabel
      source = resolveSourceLabel (tpPlatform (trParams tr)) dataSource sourceOverride
      params = trParams tr
      paramsValue =
        object
          [ "platform" .= tpPlatform params
          , "interval" .= tpInterval params
          , "bars" .= tpBars params
          , "method" .= tpMethod params
          , "blendWeight" .= tpBlendWeight params
          , "positioning" .= tpPositioning params
          , "normalization" .= tpNormalization params
          , "baseOpenThreshold" .= tpBaseOpenThreshold params
          , "baseCloseThreshold" .= tpBaseCloseThreshold params
          , "minHoldBars" .= tpMinHoldBars params
          , "cooldownBars" .= tpCooldownBars params
          , "maxHoldBars" .= tpMaxHoldBars params
          , "minEdge" .= tpMinEdge params
          , "minSignalToNoise" .= tpMinSignalToNoise params
          , "edgeBuffer" .= tpEdgeBuffer params
          , "costAwareEdge" .= tpCostAwareEdge params
          , "trendLookback" .= tpTrendLookback params
          , "maxPositionSize" .= tpMaxPositionSize params
          , "volTarget" .= tpVolTarget params
          , "volLookback" .= tpVolLookback params
          , "volEwmaAlpha" .= tpVolEwmaAlpha params
          , "volFloor" .= tpVolFloor params
          , "volScaleMax" .= tpVolScaleMax params
          , "maxVolatility" .= tpMaxVolatility params
          , "periodsPerYear" .= tpPeriodsPerYear params
          , "kalmanMarketTopN" .= tpKalmanMarketTopN params
          , "fee" .= tpFee params
          , "epochs" .= tpEpochs params
          , "hiddenSize" .= tpHiddenSize params
          , "learningRate" .= tpLearningRate params
          , "valRatio" .= tpValRatio params
          , "patience" .= tpPatience params
          , "walkForwardFolds" .= tpWalkForwardFolds params
          , "tuneStressVolMult" .= tpTuneStressVolMult params
          , "tuneStressShock" .= tpTuneStressShock params
          , "tuneStressWeight" .= tpTuneStressWeight params
          , "gradClip" .= tpGradClip params
          , "slippage" .= tpSlippage params
          , "spread" .= tpSpread params
          , "intrabarFill" .= tpIntrabarFill params
          , "stopLoss" .= tpStopLoss params
          , "takeProfit" .= tpTakeProfit params
          , "trailingStop" .= tpTrailingStop params
          , "stopLossVolMult" .= tpStopLossVolMult params
          , "takeProfitVolMult" .= tpTakeProfitVolMult params
          , "trailingStopVolMult" .= tpTrailingStopVolMult params
          , "maxDrawdown" .= tpMaxDrawdown params
          , "maxDailyLoss" .= tpMaxDailyLoss params
          , "maxOrderErrors" .= tpMaxOrderErrors params
          , "kalmanDt" .= tpKalmanDt params
          , "kalmanProcessVar" .= tpKalmanProcessVar params
          , "kalmanMeasurementVar" .= tpKalmanMeasurementVar params
          , "kalmanZMin" .= tpKalmanZMin params
          , "kalmanZMax" .= tpKalmanZMax params
          , "maxHighVolProb" .= tpMaxHighVolProb params
          , "maxConformalWidth" .= tpMaxConformalWidth params
          , "maxQuantileWidth" .= tpMaxQuantileWidth params
          , "confirmConformal" .= tpConfirmConformal params
          , "confirmQuantiles" .= tpConfirmQuantiles params
          , "confidenceSizing" .= tpConfidenceSizing params
          , "minPositionSize" .= tpMinPositionSize params
          , "binanceSymbol" .= symbol
          ]
      combo =
        object
          [ "rank" .= rank
          , "finalEquity" .= trFinalEquity tr
          , "objective" .= trObjective tr
          , "score" .= trScore tr
          , "openThreshold" .= trOpenThreshold tr
          , "closeThreshold" .= trCloseThreshold tr
          , "source" .= source
          , "metrics" .= object
              [ "sharpe" .= sharpe
              , "maxDrawdown" .= maxDd
              , "turnover" .= turnover
              , "roundTrips" .= roundTrips
              ]
          , "params" .= paramsValue
          ]
   in case extractOperations (trStdoutJson tr) of
        Just ops -> addField "operations" (Array (V.fromList ops)) combo
        Nothing -> combo

addField :: String -> Value -> Value -> Value
addField key value val =
  case val of
    Object obj -> Object (KM.insert (Key.fromString key) value obj)
    _ -> val
