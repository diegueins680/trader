{-# LANGUAGE OverloadedStrings #-}

module Trader.Optimizer.Merge
  ( MergeArgs (..)
  , runMerge
  ) where

import Control.Applicative ((<|>))
import Control.Exception (SomeException, try)
import Control.Monad (when)
import Data.Aeson (Value (..), object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isSpace, toLower, toUpper)
import Data.List (foldl', isPrefixOf, isSuffixOf, sort, sortBy)
import qualified Data.Map.Strict as M
import Data.Maybe (fromMaybe, isJust, mapMaybe)
import Data.Ord (comparing)
import Data.Scientific (FPFormat (..), Scientific, formatScientific, fromFloatDigits, toRealFloat)
import Data.Time.Clock.POSIX (POSIXTime, getPOSIXTime)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Vector as V
import System.Directory
  ( copyFile
  , createDirectoryIfMissing
  , doesDirectoryExist
  , doesFileExist
  , getHomeDirectory
  , listDirectory
  )
import System.FilePath ((</>), takeDirectory)
import System.IO (hPutStrLn, stderr)

import Trader.Optimizer.Json (encodePretty)

data MergeArgs = MergeArgs
  { maTopJson :: !FilePath
  , maFromJsonl :: ![FilePath]
  , maFromTopJson :: ![FilePath]
  , maOut :: !FilePath
  , maMax :: !Int
  , maHistoryDir :: !(Maybe FilePath)
  , maCopyToDist :: !Bool
  }
  deriving (Eq, Show)

data Combo = Combo
  { comboFinalEquity :: !Double
  , comboObjective :: !(Maybe String)
  , comboScore :: !(Maybe Double)
  , comboOpenThreshold :: !(Maybe Double)
  , comboCloseThreshold :: !(Maybe Double)
  , comboSource :: !(Maybe String)
  , comboMetrics :: !(Maybe (KM.KeyMap Value))
  , comboOperations :: !(Maybe [Value])
  , comboParams :: !(KM.KeyMap Value)
  }
  deriving (Eq, Show)

runMerge :: MergeArgs -> IO Int
runMerge args = do
  topJsonPath <- expandUser (maTopJson args)
  outPath <-
    if null (trim (maOut args))
      then pure topJsonPath
      else expandUser (maOut args)
  jsonlPaths <-
    if null (maFromJsonl args)
      then discoverDefaultJsonlPaths
      else traverse expandUser (maFromJsonl args)
  otherTopJsonPaths <- traverse expandUser (maFromTopJson args)
  historyDir <- traverse expandUser (maHistoryDir args)
  sourcesResult <- loadSources topJsonPath otherTopJsonPaths jsonlPaths
  case sourcesResult of
    Left err -> do
      hPutStrLn stderr err
      pure 1
    Right sources -> do
      let merged = mergeCombos sources
          maxItems = max 0 (maMax args)
          sourceCount = sum (map length sources)
      writeTopJson outPath merged maxItems
      archiveTopJson historyDir outPath
      when (maCopyToDist args) (copyToDist outPath)
      putStrLn ("Merged " ++ show sourceCount ++ " candidates into " ++ show (length merged) ++ " unique combos.")
      putStrLn ("Wrote: " ++ outPath)
      pure 0

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

expandUser :: FilePath -> IO FilePath
expandUser path =
  case path of
    ('~' : '/' : rest) -> do
      home <- getHomeDirectory
      pure (home </> rest)
    "~" -> getHomeDirectory
    _ -> pure path

loadSources ::
  FilePath ->
  [FilePath] ->
  [FilePath] ->
  IO (Either String [[Value]])
loadSources topJsonPath otherTopJsonPaths jsonlPaths = do
  topResult <- loadTopCombos topJsonPath
  case topResult of
    Left err -> pure (Left err)
    Right topCombos -> do
      otherResults <- traverse loadTopCombos otherTopJsonPaths
      case sequence otherResults of
        Left err -> pure (Left err)
        Right otherCombos -> do
          jsonlCombos <- traverse loadCombosFromJsonl jsonlPaths
          pure (Right (topCombos : otherCombos ++ jsonlCombos))

loadTopCombos :: FilePath -> IO (Either String [Value])
loadTopCombos path = do
  exists <- doesFileExist path
  if not exists
    then pure (Right [])
    else do
      raw <- try (BL.readFile path) :: IO (Either SomeException BL.ByteString)
      case raw of
        Left e -> pure (Left ("Failed to read " ++ path ++ ": " ++ show e))
        Right contents ->
          case Aeson.eitherDecode contents of
            Left err -> pure (Left ("Failed to parse " ++ path ++ ": " ++ err))
            Right (Object obj) ->
              case KM.lookup (Key.fromString "combos") obj of
                Just (Array combos) ->
                  pure (Right [Object c | Object c <- V.toList combos])
                _ -> pure (Right [])
            Right _ -> pure (Right [])

loadCombosFromJsonl :: FilePath -> IO [Value]
loadCombosFromJsonl path = do
  exists <- doesFileExist path
  if not exists
    then pure []
    else do
      raw <- try (BL.readFile path) :: IO (Either SomeException BL.ByteString)
      case raw of
        Left _ -> pure []
        Right contents ->
          let lines' = filter (not . T.null . T.strip) (T.lines (TE.decodeUtf8 (BL.toStrict contents)))
           in pure (mapMaybe parseJsonlLine lines')
  where
    parseJsonlLine line =
      case Aeson.decode (BL.fromStrict (TE.encodeUtf8 line)) of
        Just (Object rec) ->
          case KM.lookup (Key.fromString "ok") rec of
            Just okVal | valueTruthy okVal ->
              case KM.lookup (Key.fromString "finalEquity") rec >>= coerceFloatValue of
                Nothing -> Nothing
                Just finalEq ->
                  let source =
                        case KM.lookup (Key.fromString "source") rec of
                          Just (String s) -> Just (T.unpack s)
                          _ -> Nothing
                      params =
                        case KM.lookup (Key.fromString "params") rec of
                          Just (Object p) -> p
                          _ -> KM.empty
                      objective = KM.lookup (Key.fromString "objective") rec >>= normalizeObjectiveValue
                      score = KM.lookup (Key.fromString "score") rec >>= coerceFloatValue
                      openThr = KM.lookup (Key.fromString "openThreshold") rec >>= coerceFloatValue
                      closeThr = KM.lookup (Key.fromString "closeThreshold") rec >>= coerceFloatValue
                      metrics =
                        case KM.lookup (Key.fromString "metrics") rec of
                          Just (Object m) -> Just m
                          _ -> Nothing
                      operations = KM.lookup (Key.fromString "operations") rec >>= coerceOperations
                      combo =
                        object
                          [ "finalEquity" .= finalEq
                          , "objective" .= objective
                          , "score" .= score
                          , "openThreshold" .= openThr
                          , "closeThreshold" .= closeThr
                          , "source" .= source
                          , "metrics" .= metrics
                          , "operations" .= operations
                          , "params" .= Object params
                          ]
                   in Just combo
            _ -> Nothing
        _ -> Nothing

discoverDefaultJsonlPaths :: IO [FilePath]
discoverDefaultJsonlPaths = do
  tmp <- discoverInDir ".tmp"
  hsTmp <- discoverInDir ("haskell" </> ".tmp")
  pure (tmp ++ hsTmp)
  where
    discoverInDir dir = do
      exists <- doesDirectoryExist dir
      if not exists
        then pure []
        else do
          entries <- listDirectory dir
          let matches =
                sort
                  [ dir </> entry
                  | entry <- entries
                  , "optimize_equity" `isPrefixOf` entry
                  , ".jsonl" `isSuffixOf` entry
                  ]
          pure matches

mergeCombos :: [[Value]] -> [Combo]
mergeCombos sources =
  let merged = foldl' mergeSource M.empty sources
   in M.elems merged
  where
    mergeSource acc combos = foldl' mergeCombo acc combos
    mergeCombo acc comboVal =
      case normalizeCombo comboVal of
        Nothing -> acc
        Just combo ->
          let key = signatureKey combo
           in case M.lookup key acc of
                Nothing -> M.insert key combo acc
                Just prev -> M.insert key (pickBest combo prev) acc
    pickBest newer prev =
      let objNew = comboObjective newer
          objPrev = comboObjective prev
          scoreNew = comboScore newer
          scorePrev = comboScore prev
          scoreVal = fromMaybe (-1 / 0)
       in if objNew == objPrev && (isJust scoreNew || isJust scorePrev)
            then if scoreVal scoreNew > scoreVal scorePrev then newer else prev
            else if comboFinalEquity newer > comboFinalEquity prev then newer else prev

signatureKey :: Combo -> BS.ByteString
signatureKey combo =
  let params = comboParams combo
      p name = fromMaybe Null (KM.lookup (Key.fromString name) params)
      values =
        [ maybe Null (String . T.pack) (comboSource combo)
        , p "platform"
        , p "interval"
        , p "bars"
        , p "binanceSymbol"
        , p "method"
        , p "blendWeight"
        , p "normalization"
        , p "positioning"
        , p "baseOpenThreshold"
        , p "baseCloseThreshold"
        , p "minHoldBars"
        , p "cooldownBars"
        , p "maxHoldBars"
        , p "minEdge"
        , p "edgeBuffer"
        , p "costAwareEdge"
        , p "trendLookback"
        , p "maxPositionSize"
        , p "volTarget"
        , p "volLookback"
        , p "volEwmaAlpha"
        , p "volFloor"
        , p "volScaleMax"
        , p "maxVolatility"
        , p "periodsPerYear"
        , p "walkForwardFolds"
        , p "tuneStressVolMult"
        , p "tuneStressShock"
        , p "tuneStressWeight"
        , p "fee"
        , p "epochs"
        , p "hiddenSize"
        , p "learningRate"
        , p "valRatio"
        , p "patience"
        , p "gradClip"
        , p "slippage"
        , p "spread"
        , p "intrabarFill"
        , p "triLayer"
        , p "triLayerFastMult"
        , p "triLayerSlowMult"
        , p "triLayerCloudPadding"
        , p "triLayerCloudSlope"
        , p "triLayerCloudWidth"
        , p "triLayerTouchLookback"
        , p "triLayerPriceAction"
        , p "triLayerPriceActionBody"
        , p "lstmExitFlipBars"
        , p "lstmExitFlipGraceBars"
        , p "lstmConfidenceSoft"
        , p "lstmConfidenceHard"
        , p "stopLoss"
        , p "takeProfit"
        , p "trailingStop"
        , p "maxDrawdown"
        , p "maxDailyLoss"
        , p "maxOrderErrors"
        , p "kalmanDt"
        , p "kalmanProcessVar"
        , p "kalmanMeasurementVar"
        , p "kalmanZMin"
        , p "kalmanZMax"
        , p "kalmanMarketTopN"
        , p "maxHighVolProb"
        , p "maxConformalWidth"
        , p "maxQuantileWidth"
        , p "confirmConformal"
        , p "confirmQuantiles"
        , p "confidenceSizing"
        , p "minPositionSize"
        , maybe Null (Number . fromFloatDigits) (comboOpenThreshold combo)
        , maybe Null (Number . fromFloatDigits) (comboCloseThreshold combo)
        ]
   in BL.toStrict (Aeson.encode values)

writeTopJson :: FilePath -> [Combo] -> Int -> IO ()
writeTopJson path combos maxItems = do
  let sorted = take maxItems (sortBy compareCombos combos)
  nowMs <- fmap (floor . (* 1000) :: POSIXTime -> Int) getPOSIXTime
  let comboValues = zipWith comboToValue [1 ..] sorted
      exportVal =
        object
          [ "generatedAtMs" .= nowMs
          , "source" .= ("merge_top_combos.py" :: String)
          , "combos" .= comboValues
          ]
  createDirectoryIfMissing True (takeDirectory path)
  BL.writeFile path (encodePretty exportVal <> "\n")

compareCombos :: Combo -> Combo -> Ordering
compareCombos a b =
  let objA = comboObjective a
      objB = comboObjective b
      scoreA = fromMaybe (-1 / 0) (comboScore a)
      scoreB = fromMaybe (-1 / 0) (comboScore b)
      eqA = comboFinalEquity a
      eqB = comboFinalEquity b
   in if objA == objB
        then
          case compareDesc scoreA scoreB of
            EQ -> compareDesc eqA eqB
            ord -> ord
        else compareDesc eqA eqB

compareDesc :: Ord a => a -> a -> Ordering
compareDesc a b
  | a > b = LT
  | a < b = GT
  | otherwise = EQ

comboToValue :: Int -> Combo -> Value
comboToValue rank combo =
  let metricsVal =
        case comboMetrics combo of
          Just m -> Object m
          Nothing -> Null
      base =
        object
          [ "rank" .= rank
          , "finalEquity" .= comboFinalEquity combo
          , "objective" .= comboObjective combo
          , "score" .= comboScore combo
          , "openThreshold" .= comboOpenThreshold combo
          , "closeThreshold" .= comboCloseThreshold combo
          , "source" .= comboSource combo
          , "metrics" .= metricsVal
          , "params" .= Object (comboParams combo)
          ]
   in case comboOperations combo of
        Just ops -> addField "operations" (Array (V.fromList ops)) base
        Nothing -> base

archiveTopJson :: Maybe FilePath -> FilePath -> IO ()
archiveTopJson historyDir outPath =
  case historyDir of
    Nothing -> pure ()
    Just dir -> do
      dirResult <- try (createDirectoryIfMissing True dir) :: IO (Either SomeException ())
      case dirResult of
        Left _ -> pure ()
        Right _ -> do
          nowMs <- fmap (floor . (* 1000)) getPOSIXTime
          let archivePath = dir </> ("top-combos-" ++ show (nowMs :: Integer) ++ ".json")
          _ <- try (copyFile outPath archivePath) :: IO (Either SomeException ())
          pure ()

copyToDist :: FilePath -> IO ()
copyToDist outPath = do
  let distDir = "haskell" </> "web" </> "dist"
  exists <- doesDirectoryExist distDir
  when exists $ do
    let target = distDir </> "top-combos.json"
    _ <- try (copyFile outPath target) :: IO (Either SomeException ())
    pure ()

normalizeCombo :: Value -> Maybe Combo
normalizeCombo value =
  case value of
    Object obj -> do
      finalEq <- KM.lookup (Key.fromString "finalEquity") obj >>= coerceFloatValue
      let source = normalizeSource (KM.lookup (Key.fromString "source") obj)
          paramsRaw =
            case KM.lookup (Key.fromString "params") obj of
              Just (Object p) -> p
              _ -> KM.empty
          platform = normalizePlatform paramsRaw source
          objective = KM.lookup (Key.fromString "objective") obj >>= normalizeObjectiveValue
          score = KM.lookup (Key.fromString "score") obj >>= coerceFloatValue
          metrics =
            case KM.lookup (Key.fromString "metrics") obj of
              Just (Object m) -> Just m
              _ -> Nothing
          operations = KM.lookup (Key.fromString "operations") obj >>= coerceOperations
          interval = valueToStringMaybe (KM.lookup (Key.fromString "interval") paramsRaw)
          bars = fromMaybe 0 (KM.lookup (Key.fromString "bars") paramsRaw >>= coerceIntValue)
          method = valueToStringMaybe (KM.lookup (Key.fromString "method") paramsRaw)
          normalization = valueToStringMaybe (KM.lookup (Key.fromString "normalization") paramsRaw)
          epochs = fromMaybe 0 (KM.lookup (Key.fromString "epochs") paramsRaw >>= coerceIntValue)
          hiddenSize = KM.lookup (Key.fromString "hiddenSize") paramsRaw >>= coerceIntValue
          learningRate = KM.lookup (Key.fromString "learningRate") paramsRaw >>= coerceFloatValue
          valRatio = KM.lookup (Key.fromString "valRatio") paramsRaw >>= coerceFloatValue
          patience = KM.lookup (Key.fromString "patience") paramsRaw >>= coerceIntValue
          gradClip = KM.lookup (Key.fromString "gradClip") paramsRaw >>= coerceFloatValue
          positioning = valueToStringMaybe (KM.lookup (Key.fromString "positioning") paramsRaw)
          slippage = KM.lookup (Key.fromString "slippage") paramsRaw >>= coerceFloatValue
          spread = KM.lookup (Key.fromString "spread") paramsRaw >>= coerceFloatValue
          costAwareEdge = fromMaybe False (KM.lookup (Key.fromString "costAwareEdge") paramsRaw >>= coerceBoolValue)
          triLayer = fromMaybe False (KM.lookup (Key.fromString "triLayer") paramsRaw >>= coerceBoolValue)
          confirmConformal = fromMaybe False (KM.lookup (Key.fromString "confirmConformal") paramsRaw >>= coerceBoolValue)
          confirmQuantiles = fromMaybe False (KM.lookup (Key.fromString "confirmQuantiles") paramsRaw >>= coerceBoolValue)
          confidenceSizing = fromMaybe False (KM.lookup (Key.fromString "confidenceSizing") paramsRaw >>= coerceBoolValue)
          symbol =
            normalizeSymbolValue
              ( fromMaybe
                  Null
                  ( KM.lookup (Key.fromString "binanceSymbol") paramsRaw
                      <|> KM.lookup (Key.fromString "symbol") paramsRaw
                  )
              )
          normalizedParams =
            KM.union
              ( KM.fromList
                  [ (Key.fromString "platform", maybe Null (String . T.pack) platform)
                  , (Key.fromString "interval", String (T.pack interval))
                  , (Key.fromString "bars", Number (fromIntegral bars))
                  , (Key.fromString "method", String (T.pack method))
                  , (Key.fromString "normalization", String (T.pack normalization))
                  , (Key.fromString "positioning", String (T.pack positioning))
                  , (Key.fromString "baseOpenThreshold", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "baseOpenThreshold") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "baseCloseThreshold", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "baseCloseThreshold") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "blendWeight", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "blendWeight") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "minHoldBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "minHoldBars") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "cooldownBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "cooldownBars") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "maxHoldBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "maxHoldBars") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "minEdge", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "minEdge") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "edgeBuffer", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "edgeBuffer") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "costAwareEdge", Bool costAwareEdge)
                  , (Key.fromString "trendLookback", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "trendLookback") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "maxPositionSize", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxPositionSize") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "volTarget", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "volTarget") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "volLookback", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "volLookback") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "volEwmaAlpha", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "volEwmaAlpha") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "volFloor", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "volFloor") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "volScaleMax", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "volScaleMax") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "maxVolatility", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxVolatility") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "periodsPerYear", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "periodsPerYear") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "kalmanMarketTopN", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "kalmanMarketTopN") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "fee", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "fee") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "epochs", Number (fromIntegral epochs))
                  , (Key.fromString "hiddenSize", maybe Null (Number . fromIntegral) hiddenSize)
                  , (Key.fromString "learningRate", maybe Null (Number . fromFloatDigits) learningRate)
                  , (Key.fromString "valRatio", maybe Null (Number . fromFloatDigits) valRatio)
                  , (Key.fromString "patience", maybe Null (Number . fromIntegral) patience)
                  , (Key.fromString "walkForwardFolds", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "walkForwardFolds") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "tuneStressVolMult", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "tuneStressVolMult") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "tuneStressShock", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "tuneStressShock") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "tuneStressWeight", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "tuneStressWeight") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "gradClip", maybe Null (Number . fromFloatDigits) gradClip)
                  , (Key.fromString "slippage", maybe Null (Number . fromFloatDigits) slippage)
                  , (Key.fromString "spread", maybe Null (Number . fromFloatDigits) spread)
                  , (Key.fromString "intrabarFill", maybe Null (String . T.pack) (valueToStringMaybeMaybe (KM.lookup (Key.fromString "intrabarFill") paramsRaw)))
                  , (Key.fromString "triLayer", Bool triLayer)
                  , (Key.fromString "triLayerFastMult", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerFastMult") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "triLayerSlowMult", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerSlowMult") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "triLayerCloudPadding", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerCloudPadding") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "triLayerCloudSlope", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerCloudSlope") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "triLayerCloudWidth", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerCloudWidth") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "triLayerTouchLookback", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "triLayerTouchLookback") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "triLayerPriceAction", maybe Null Bool (KM.lookup (Key.fromString "triLayerPriceAction") paramsRaw >>= coerceBoolValue))
                  , (Key.fromString "triLayerPriceActionBody", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerPriceActionBody") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "lstmExitFlipBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "lstmExitFlipBars") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "lstmExitFlipGraceBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "lstmExitFlipGraceBars") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "lstmConfidenceSoft", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "lstmConfidenceSoft") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "lstmConfidenceHard", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "lstmConfidenceHard") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "stopLoss", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "stopLoss") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "takeProfit", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "takeProfit") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "trailingStop", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "trailingStop") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "maxDrawdown", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxDrawdown") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "maxDailyLoss", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxDailyLoss") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "maxOrderErrors", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "maxOrderErrors") paramsRaw >>= coerceIntValue))
                  , (Key.fromString "kalmanDt", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanDt") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "kalmanProcessVar", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanProcessVar") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "kalmanMeasurementVar", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanMeasurementVar") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "kalmanZMin", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanZMin") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "kalmanZMax", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanZMax") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "maxHighVolProb", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxHighVolProb") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "maxConformalWidth", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxConformalWidth") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "maxQuantileWidth", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxQuantileWidth") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "confirmConformal", Bool confirmConformal)
                  , (Key.fromString "confirmQuantiles", Bool confirmQuantiles)
                  , (Key.fromString "confidenceSizing", Bool confidenceSizing)
                  , (Key.fromString "minPositionSize", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "minPositionSize") paramsRaw >>= coerceFloatValue))
                  , (Key.fromString "binanceSymbol", maybe Null (String . T.pack) symbol)
                  ]
              )
              paramsRaw
          openThreshold = KM.lookup (Key.fromString "openThreshold") obj >>= coerceFloatValue
          closeThreshold = KM.lookup (Key.fromString "closeThreshold") obj >>= coerceFloatValue
      pure
        Combo
          { comboFinalEquity = finalEq
          , comboObjective = objective
          , comboScore = score
          , comboOpenThreshold = openThreshold
          , comboCloseThreshold = closeThreshold
          , comboSource = source
          , comboMetrics = metrics
          , comboOperations = operations
          , comboParams = normalizedParams
          }
    _ -> Nothing

normalizeSource :: Maybe Value -> Maybe String
normalizeSource value =
  case value of
    Just (String s) ->
      case map toLower (trim (T.unpack s)) of
        "binance" -> Just "binance"
        "coinbase" -> Just "coinbase"
        "kraken" -> Just "kraken"
        "poloniex" -> Just "poloniex"
        "csv" -> Just "csv"
        _ -> Nothing
    _ -> Nothing

normalizePlatform :: KM.KeyMap Value -> Maybe String -> Maybe String
normalizePlatform params source =
  let raw =
        case KM.lookup (Key.fromString "platform") params of
          Just (String s) -> map toLower (trim (T.unpack s))
          _ -> ""
   in if raw `elem` ["binance", "coinbase", "kraken", "poloniex"]
        then Just raw
        else
          case source of
            Just s | s `elem` ["binance", "coinbase", "kraken", "poloniex"] -> Just s
            _ -> Nothing

normalizeObjectiveValue :: Value -> Maybe String
normalizeObjectiveValue value =
  let s = map toLower (trim (valueToString value))
   in if null s then Nothing else Just s

normalizeSymbolValue :: Value -> Maybe String
normalizeSymbolValue value =
  let s = map toUpper (trim (valueToString value))
   in if null s then Nothing else Just s

valueToStringMaybe :: Maybe Value -> String
valueToStringMaybe value =
  case value of
    Nothing -> ""
    Just v -> valueToString v

valueToStringMaybeMaybe :: Maybe Value -> Maybe String
valueToStringMaybeMaybe value =
  case value of
    Nothing -> Nothing
    Just Null -> Nothing
    Just v -> Just (valueToString v)

valueToString :: Value -> String
valueToString value =
  case value of
    String s -> T.unpack s
    Number n -> formatScientific Generic Nothing n
    Bool True -> "True"
    Bool False -> "False"
    Null -> ""
    _ -> T.unpack (TE.decodeUtf8 (BL.toStrict (Aeson.encode value)))

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

coerceBoolValue :: Value -> Maybe Bool
coerceBoolValue value =
  case value of
    Null -> Nothing
    Bool v -> Just v
    Number n ->
      case scientificToDouble n of
        Just d -> Just (d /= 0)
        Nothing -> Nothing
    String s ->
      let trimmed = map toLower (trim (T.unpack s))
       in if trimmed `elem` ["1", "true", "t", "yes", "y", "on"]
            then Just True
            else if trimmed `elem` ["0", "false", "f", "no", "n", "off"]
              then Just False
              else Nothing
    _ -> Nothing

coerceOperations :: Value -> Maybe [Value]
coerceOperations value =
  case value of
    Array arr ->
      let ops = [Object o | Object o <- V.toList arr]
       in if null ops then Nothing else Just ops
    _ -> Nothing

valueTruthy :: Value -> Bool
valueTruthy value =
  case value of
    Null -> False
    Bool v -> v
    Number n ->
      case scientificToDouble n of
        Just d -> d /= 0
        Nothing -> False
    String s -> not (T.null s)
    Array arr -> not (V.null arr)
    Object obj -> not (KM.null obj)

scientificToDouble :: Scientific -> Maybe Double
scientificToDouble n =
  let d = toRealFloat n
   in if isInfinite d || isNaN d then Nothing else Just d

addField :: String -> Value -> Value -> Value
addField key value val =
  case val of
    Object obj -> Object (KM.insert (Key.fromString key) value obj)
    _ -> val
