module Main (main) where

import Control.Monad (when)
import Data.Char (isSpace)
import Data.List (intercalate)
import Data.Maybe (fromMaybe, isJust)
import Options.Applicative
import System.Environment (getArgs, getProgName)
import System.Exit (ExitCode (..), exitSuccess, exitWith)
import System.IO (hPutStrLn, stderr)

import Trader.Optimizer.Optimize (OptimizerArgs (..), runOptimizer)

main :: IO ()
main = do
  argv <- getArgs
  progName <- getProgName
  args <- parseArgs progName argv
  case validateArgs args of
    Left err -> do
      hPutStrLn stderr err
      exitWith (ExitFailure 2)
    Right args' -> do
      code <- runOptimizer args'
      exitWith (if code == 0 then ExitSuccess else ExitFailure code)

parseArgs :: String -> [String] -> IO OptimizerArgs
parseArgs progName argv =
  handleParseResult $
    execParserPure
      (prefs showHelpOnError)
      (info (optimizerArgsParser <**> helper) (fullDesc <> progDesc "Random-search optimizer for trader-hs cumulative equity (finalEquity)."))
      argv
  where
    handleParseResult result =
      case result of
        Success opts -> pure opts
        Failure failure -> do
          let (msg, _) = renderFailure failure progName
          hPutStrLn stderr (stripTrailingSpace msg)
          exitWith (ExitFailure 2)
        CompletionInvoked compl -> do
          msg <- execCompletion compl progName
          putStr msg
          exitSuccess

optimizerArgsParser :: Parser OptimizerArgs
optimizerArgsParser =
  OptimizerArgs
    <$> optional (strOption (long "data" <> metavar "PATH"))
    <*> optional (strOption (long "symbol" <> long "binance-symbol" <> metavar "SYMBOL"))
    <*> strOption (long "symbol-label" <> value "" <> metavar "TEXT")
    <*> strOption (long "source-label" <> value "" <> metavar "TEXT")
    <*> strOption (long "price-column" <> value "close" <> metavar "NAME")
    <*> strOption (long "high-column" <> value "" <> metavar "NAME")
    <*> strOption (long "low-column" <> value "" <> metavar "NAME")
    <*> strOption (long "lookback-window" <> value "7d" <> metavar "DURATION")
    <*> option auto (long "backtest-ratio" <> value 0.2 <> metavar "FLOAT")
    <*> option auto (long "tune-ratio" <> value 0.25 <> metavar "FLOAT")
    <*> option auto (long "trials" <> value 50 <> metavar "INT")
    <*> option auto (long "seed" <> value 42 <> metavar "INT")
    <*> option auto (long "timeout-sec" <> value 60.0 <> metavar "FLOAT")
    <*> strOption (long "output" <> value "" <> metavar "PATH")
    <*> switch (long "append")
    <*> strOption (long "binary" <> value "" <> metavar "PATH")
    <*> switch (long "no-sweep-threshold")
    <*> switch (long "disable-lstm-persistence")
    <*> strOption (long "top-json" <> value "" <> metavar "PATH")
    <*> switch (long "quality")
    <*> switch (long "auto-high-low")
    <*> strOption (long "objective" <> value "equity-dd-turnover" <> metavar "NAME")
    <*> option auto (long "penalty-max-drawdown" <> value 1.5 <> metavar "FLOAT")
    <*> option auto (long "penalty-turnover" <> value 0.2 <> metavar "FLOAT")
    <*> option auto (long "min-round-trips" <> value 0 <> metavar "INT")
    <*> option auto (long "min-win-rate" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "min-profit-factor" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "min-exposure" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "min-sharpe" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "min-wf-sharpe-mean" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "max-wf-sharpe-std" <> value 0.0 <> metavar "FLOAT")
    <*> strOption (long "tune-objective" <> value "equity-dd-turnover" <> metavar "NAME")
    <*> option auto (long "tune-penalty-max-drawdown" <> value 1.5 <> metavar "FLOAT")
    <*> option auto (long "tune-penalty-turnover" <> value 0.2 <> metavar "FLOAT")
    <*> option auto (long "tune-stress-vol-mult" <> value 1.0 <> metavar "FLOAT")
    <*> option auto (long "tune-stress-shock" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tune-stress-weight" <> value 0.0 <> metavar "FLOAT")
    <*> optional (option auto (long "tune-stress-vol-mult-min" <> metavar "FLOAT"))
    <*> optional (option auto (long "tune-stress-vol-mult-max" <> metavar "FLOAT"))
    <*> optional (option auto (long "tune-stress-shock-min" <> metavar "FLOAT"))
    <*> optional (option auto (long "tune-stress-shock-max" <> metavar "FLOAT"))
    <*> optional (option auto (long "tune-stress-weight-min" <> metavar "FLOAT"))
    <*> optional (option auto (long "tune-stress-weight-max" <> metavar "FLOAT"))
    <*> option auto (long "walk-forward-folds-min" <> value 7 <> metavar "INT")
    <*> option auto (long "walk-forward-folds-max" <> value 7 <> metavar "INT")
    <*> optional (strOption (long "interval" <> metavar "INTERVAL"))
    <*> optional (strOption (long "intervals" <> metavar "LIST"))
    <*> optional (strOption (long "platform" <> metavar "NAME"))
    <*> optional (strOption (long "platforms" <> metavar "LIST"))
    <*> option auto (long "bars-min" <> value 0 <> metavar "INT")
    <*> option auto (long "bars-max" <> value 0 <> metavar "INT")
    <*> option auto (long "bars-auto-prob" <> value 0.25 <> metavar "FLOAT")
    <*> strOption (long "bars-distribution" <> value "uniform" <> metavar "NAME")
    <*> option auto (long "epochs-min" <> value 0 <> metavar "INT")
    <*> option auto (long "epochs-max" <> value 10 <> metavar "INT")
    <*> option auto (long "slippage-max" <> value 0.0005 <> metavar "FLOAT")
    <*> option auto (long "spread-max" <> value 0.0005 <> metavar "FLOAT")
    <*> option auto (long "fee-min" <> value 0.0004 <> metavar "FLOAT")
    <*> option auto (long "fee-max" <> value 0.001 <> metavar "FLOAT")
    <*> option auto (long "open-threshold-min" <> value 5e-4 <> metavar "FLOAT")
    <*> option auto (long "open-threshold-max" <> value 2e-2 <> metavar "FLOAT")
    <*> option auto (long "close-threshold-min" <> value 5e-4 <> metavar "FLOAT")
    <*> option auto (long "close-threshold-max" <> value 2e-2 <> metavar "FLOAT")
    <*> option auto (long "min-hold-bars-min" <> value 2 <> metavar "INT")
    <*> option auto (long "min-hold-bars-max" <> value 8 <> metavar "INT")
    <*> option auto (long "cooldown-bars-min" <> value 1 <> metavar "INT")
    <*> option auto (long "cooldown-bars-max" <> value 3 <> metavar "INT")
    <*> option auto (long "max-hold-bars-min" <> value 24 <> metavar "INT")
    <*> option auto (long "max-hold-bars-max" <> value 72 <> metavar "INT")
    <*> option auto (long "min-edge-min" <> value 0.0002 <> metavar "FLOAT")
    <*> option auto (long "min-edge-max" <> value 0.001 <> metavar "FLOAT")
    <*> option auto (long "min-signal-to-noise-min" <> value 0.5 <> metavar "FLOAT")
    <*> option auto (long "min-signal-to-noise-max" <> value 1.2 <> metavar "FLOAT")
    <*> option auto (long "edge-buffer-min" <> value 0.0001 <> metavar "FLOAT")
    <*> option auto (long "edge-buffer-max" <> value 0.0005 <> metavar "FLOAT")
    <*> option auto (long "p-cost-aware-edge" <> value (-1.0) <> metavar "FLOAT")
    <*> option auto (long "trend-lookback-min" <> value 20 <> metavar "INT")
    <*> option auto (long "trend-lookback-max" <> value 60 <> metavar "INT")
    <*> option auto (long "p-long-short" <> value 0.2 <> metavar "FLOAT")
    <*> option auto (long "p-intrabar-take-profit-first" <> value 0.2 <> metavar "FLOAT")
    <*> option auto (long "p-tri-layer" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-fast-mult-min" <> value 0.5 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-fast-mult-max" <> value 0.5 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-slow-mult-min" <> value 2.0 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-slow-mult-max" <> value 2.0 <> metavar "FLOAT")
    <*> option auto (long "p-tri-layer-price-action" <> value 1.0 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-cloud-padding-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-cloud-padding-max" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-cloud-slope-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-cloud-slope-max" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-cloud-width-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-cloud-width-max" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-touch-lookback-min" <> value 1 <> metavar "INT")
    <*> option auto (long "tri-layer-touch-lookback-max" <> value 1 <> metavar "INT")
    <*> option auto (long "tri-layer-price-action-body-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tri-layer-price-action-body-max" <> value 0.0 <> metavar "FLOAT")
    <*> switch (long "tri-layer-exit-on-slow")
    <*> option auto (long "kalman-band-lookback-min" <> value 0 <> metavar "INT")
    <*> option auto (long "kalman-band-lookback-max" <> value 0 <> metavar "INT")
    <*> option auto (long "kalman-band-std-mult-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "kalman-band-std-mult-max" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "lstm-exit-flip-bars-min" <> value 0 <> metavar "INT")
    <*> option auto (long "lstm-exit-flip-bars-max" <> value 0 <> metavar "INT")
    <*> option auto (long "lstm-exit-flip-grace-bars-min" <> value 0 <> metavar "INT")
    <*> option auto (long "lstm-exit-flip-grace-bars-max" <> value 0 <> metavar "INT")
    <*> switch (long "lstm-exit-flip-strong")
    <*> option auto (long "lstm-confidence-soft-min" <> value 0.6 <> metavar "FLOAT")
    <*> option auto (long "lstm-confidence-soft-max" <> value 0.6 <> metavar "FLOAT")
    <*> option auto (long "lstm-confidence-hard-min" <> value 0.8 <> metavar "FLOAT")
    <*> option auto (long "lstm-confidence-hard-max" <> value 0.8 <> metavar "FLOAT")
    <*> option auto (long "kalman-dt-min" <> value 0.5 <> metavar "FLOAT")
    <*> option auto (long "kalman-dt-max" <> value 2.0 <> metavar "FLOAT")
    <*> option auto (long "kalman-process-var-min" <> value 1e-7 <> metavar "FLOAT")
    <*> option auto (long "kalman-process-var-max" <> value 1e-3 <> metavar "FLOAT")
    <*> option auto (long "kalman-measurement-var-min" <> value 1e-6 <> metavar "FLOAT")
    <*> option auto (long "kalman-measurement-var-max" <> value 1e-1 <> metavar "FLOAT")
    <*> option auto (long "kalman-z-min-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "kalman-z-min-max" <> value 2.0 <> metavar "FLOAT")
    <*> option auto (long "kalman-z-max-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "kalman-z-max-max" <> value 6.0 <> metavar "FLOAT")
    <*> option auto (long "kalman-market-top-n-min" <> value 50 <> metavar "INT")
    <*> option auto (long "kalman-market-top-n-max" <> value 50 <> metavar "INT")
    <*> option auto (long "p-disable-max-high-vol-prob" <> value 0.9 <> metavar "FLOAT")
    <*> option auto (long "max-high-vol-prob-min" <> value 0.2 <> metavar "FLOAT")
    <*> option auto (long "max-high-vol-prob-max" <> value 0.95 <> metavar "FLOAT")
    <*> option auto (long "p-disable-max-conformal-width" <> value 0.95 <> metavar "FLOAT")
    <*> option auto (long "max-conformal-width-min" <> value 0.002 <> metavar "FLOAT")
    <*> option auto (long "max-conformal-width-max" <> value 0.20 <> metavar "FLOAT")
    <*> option auto (long "p-disable-max-quantile-width" <> value 0.95 <> metavar "FLOAT")
    <*> option auto (long "max-quantile-width-min" <> value 0.002 <> metavar "FLOAT")
    <*> option auto (long "max-quantile-width-max" <> value 0.20 <> metavar "FLOAT")
    <*> option auto (long "p-confirm-conformal" <> value 0.6 <> metavar "FLOAT")
    <*> option auto (long "p-confirm-quantiles" <> value 0.6 <> metavar "FLOAT")
    <*> option auto (long "p-confidence-sizing" <> value 0.6 <> metavar "FLOAT")
    <*> option auto (long "min-position-size-min" <> value 0.1 <> metavar "FLOAT")
    <*> option auto (long "min-position-size-max" <> value 0.3 <> metavar "FLOAT")
    <*> option auto (long "max-position-size-min" <> value 0.6 <> metavar "FLOAT")
    <*> option auto (long "max-position-size-max" <> value 0.9 <> metavar "FLOAT")
    <*> option auto (long "vol-target-min" <> value 0.5 <> metavar "FLOAT")
    <*> option auto (long "vol-target-max" <> value 0.9 <> metavar "FLOAT")
    <*> option auto (long "p-disable-vol-target" <> value 0.2 <> metavar "FLOAT")
    <*> option auto (long "vol-lookback-min" <> value 20 <> metavar "INT")
    <*> option auto (long "vol-lookback-max" <> value 60 <> metavar "INT")
    <*> option auto (long "vol-ewma-alpha-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "vol-ewma-alpha-max" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "p-disable-vol-ewma-alpha" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "vol-floor-min" <> value 0.1 <> metavar "FLOAT")
    <*> option auto (long "vol-floor-max" <> value 0.2 <> metavar "FLOAT")
    <*> option auto (long "vol-scale-max-min" <> value 1.0 <> metavar "FLOAT")
    <*> option auto (long "vol-scale-max-max" <> value 1.0 <> metavar "FLOAT")
    <*> option auto (long "max-volatility-min" <> value 1.2 <> metavar "FLOAT")
    <*> option auto (long "max-volatility-max" <> value 2.0 <> metavar "FLOAT")
    <*> option auto (long "p-disable-max-volatility" <> value 0.2 <> metavar "FLOAT")
    <*> option auto (long "periods-per-year-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "periods-per-year-max" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "stop-min" <> value 0.002 <> metavar "FLOAT")
    <*> option auto (long "stop-max" <> value 0.20 <> metavar "FLOAT")
    <*> option auto (long "tp-min" <> value 0.002 <> metavar "FLOAT")
    <*> option auto (long "tp-max" <> value 0.20 <> metavar "FLOAT")
    <*> option auto (long "trail-min" <> value 0.002 <> metavar "FLOAT")
    <*> option auto (long "trail-max" <> value 0.20 <> metavar "FLOAT")
    <*> option auto (long "p-disable-stop" <> value 0.5 <> metavar "FLOAT")
    <*> option auto (long "p-disable-tp" <> value 0.5 <> metavar "FLOAT")
    <*> option auto (long "p-disable-trail" <> value 0.6 <> metavar "FLOAT")
    <*> option auto (long "stop-vol-mult-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "stop-vol-mult-max" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tp-vol-mult-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "tp-vol-mult-max" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "trail-vol-mult-min" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "trail-vol-mult-max" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "p-disable-stop-vol-mult" <> value 0.5 <> metavar "FLOAT")
    <*> option auto (long "p-disable-tp-vol-mult" <> value 0.5 <> metavar "FLOAT")
    <*> option auto (long "p-disable-trail-vol-mult" <> value 0.6 <> metavar "FLOAT")
    <*> option auto (long "p-disable-max-dd" <> value 0.9 <> metavar "FLOAT")
    <*> option auto (long "p-disable-max-dl" <> value 0.9 <> metavar "FLOAT")
    <*> option auto (long "p-disable-max-oe" <> value 0.95 <> metavar "FLOAT")
    <*> option auto (long "max-dd-min" <> value 0.05 <> metavar "FLOAT")
    <*> option auto (long "max-dd-max" <> value 0.50 <> metavar "FLOAT")
    <*> option auto (long "max-dl-min" <> value 0.02 <> metavar "FLOAT")
    <*> option auto (long "max-dl-max" <> value 0.30 <> metavar "FLOAT")
    <*> option auto (long "max-oe-min" <> value 1 <> metavar "INT")
    <*> option auto (long "max-oe-max" <> value 10 <> metavar "INT")
    <*> option auto (long "method-weight-11" <> value 1.0 <> metavar "FLOAT")
    <*> option auto (long "method-weight-10" <> value 2.0 <> metavar "FLOAT")
    <*> option auto (long "method-weight-01" <> value 1.0 <> metavar "FLOAT")
    <*> option auto (long "method-weight-blend" <> value 0.0 <> metavar "FLOAT")
    <*> option auto (long "blend-weight-min" <> value 0.5 <> metavar "FLOAT")
    <*> option auto (long "blend-weight-max" <> value 0.5 <> metavar "FLOAT")
    <*> strOption (long "normalizations" <> value "none,minmax,standard,log" <> metavar "LIST")
    <*> option auto (long "hidden-size-min" <> value 8 <> metavar "INT")
    <*> option auto (long "hidden-size-max" <> value 64 <> metavar "INT")
    <*> option auto (long "lr-min" <> value 1e-4 <> metavar "FLOAT")
    <*> option auto (long "lr-max" <> value 1e-2 <> metavar "FLOAT")
    <*> option auto (long "val-ratio-min" <> value 0.1 <> metavar "FLOAT")
    <*> option auto (long "val-ratio-max" <> value 0.4 <> metavar "FLOAT")
    <*> option auto (long "patience-max" <> value 20 <> metavar "INT")
    <*> option auto (long "grad-clip-min" <> value 0.001 <> metavar "FLOAT")
    <*> option auto (long "grad-clip-max" <> value 1.0 <> metavar "FLOAT")
    <*> option auto (long "p-disable-grad-clip" <> value 0.7 <> metavar "FLOAT")

validateArgs :: OptimizerArgs -> Either String OptimizerArgs
validateArgs args = do
  let dataVal = normalizeMaybe (oaData args)
      symbolVal = normalizeMaybe (oaBinanceSymbol args)
      intervalVal = normalizeMaybe (oaInterval args)
      intervalsVal = oaIntervals args
      platformVal = normalizeMaybe (oaPlatform args)
      platformsVal = normalizeMaybe (oaPlatforms args)
      intervalsFinal = Just (fromMaybe defaultIntervals intervalsVal)
      args' =
        args
          { oaData = dataVal
          , oaBinanceSymbol = symbolVal
          , oaInterval = intervalVal
          , oaIntervals = intervalsFinal
          , oaPlatform = platformVal
          , oaPlatforms = platformsVal
          }
  when (isJust dataVal == isJust symbolVal) $
    Left "Provide exactly one of --data or --symbol/--binance-symbol."
  when (isJust intervalVal && isJust intervalsVal) $
    Left "Provide only one of --interval or --intervals."
  when (isJust platformVal && isJust platformsVal) $
    Left "Provide only one of --platform or --platforms."
  when (oaObjective args `notElem` objectiveChoices) $
    Left ("Invalid objective: " ++ show (oaObjective args) ++ " (expected one of: " ++ intercalate ", " objectiveChoices ++ ")")
  when (oaTuneObjective args `notElem` objectiveChoices) $
    Left ("Invalid tune objective: " ++ show (oaTuneObjective args) ++ " (expected one of: " ++ intercalate ", " objectiveChoices ++ ")")
  when (oaBarsDistribution args `notElem` barsDistributionChoices) $
    Left
      ( "Invalid bars distribution: "
          ++ show (oaBarsDistribution args)
          ++ " (expected one of: "
          ++ intercalate ", " barsDistributionChoices
          ++ ")"
      )
  pure args'

objectiveChoices :: [String]
objectiveChoices =
  [ "final-equity"
  , "sharpe"
  , "calmar"
  , "equity-dd"
  , "equity-dd-turnover"
  ]

barsDistributionChoices :: [String]
barsDistributionChoices = ["uniform", "log"]

defaultIntervals :: String
defaultIntervals = intercalate "," ["1h", "2h", "4h", "6h", "12h", "1d"]

normalizeMaybe :: Maybe String -> Maybe String
normalizeMaybe value =
  case value of
    Nothing -> Nothing
    Just raw ->
      let trimmed = trim raw
       in if null trimmed then Nothing else Just trimmed

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

stripTrailingSpace :: String -> String
stripTrailingSpace = reverse . dropWhile isSpace . reverse
