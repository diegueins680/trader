module Main (main) where

import Control.Monad (unless, when)
import Data.Char (isSpace, toLower)
import Data.List (intercalate)
import Data.Maybe (fromMaybe, isJust)
import Options.Applicative
import System.Environment (getArgs, getProgName)
import System.Exit (ExitCode (..), exitSuccess, exitWith)
import System.IO (hPutStrLn, stderr)

import Trader.Optimization (TuneObjective (..), tuneObjectiveCode)
import Trader.Optimizer.Optimize (
    OptimizerArgs (..),
    TechnicalOptimizerRanges (..),
    normalizeObjectiveCode,
    optimizerOptionPresent,
    runOptimizer,
 )

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
            (info (optimizerArgsParser <**> helper) (fullDesc <> progDesc "Random-search optimizer for trader-hs ROI (risk-adjusted annualized return)."))
            argv
  where
    handleParseResult result =
        case result of
            Success opts ->
                pure
                    opts
                        { oaOpenThresholdMaxExplicit = optimizerOptionPresent "open-threshold-max" argv
                        , oaCloseThresholdMaxExplicit = optimizerOptionPresent "close-threshold-max" argv
                        }
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
        <*> option auto (long "trials" <> value 300 <> metavar "INT")
        <*> option auto (long "seed" <> value 42 <> metavar "INT")
        <*> optional (option auto (long "seed-trials" <> metavar "INT"))
        <*> optional (option auto (long "seed-ratio" <> metavar "FLOAT"))
        <*> option auto (long "survivor-fraction" <> value 0.5 <> metavar "FLOAT")
        <*> option auto (long "perturb-scale-double" <> value 0.1 <> metavar "FLOAT")
        <*> option auto (long "perturb-scale-int" <> value 2 <> metavar "INT")
        <*> option auto (long "early-stop-no-improve" <> value 0 <> metavar "INT")
        <*> option auto (long "timeout-sec" <> value 90.0 <> metavar "FLOAT")
        <*> strOption (long "output" <> value "" <> metavar "PATH")
        <*> switch (long "append")
        <*> strOption (long "binary" <> value "" <> metavar "PATH")
        <*> switch (long "no-sweep-threshold")
        <*> switch (long "disable-lstm-persistence")
        <*> strOption (long "top-json" <> value "" <> metavar "PATH")
        <*> switch (long "quality")
        <*> option auto (long "quality-min-trials" <> value 500 <> metavar "INT")
        <*> option auto (long "quality-max-epochs" <> value 50 <> metavar "INT")
        <*> option auto (long "quality-max-hidden-size" <> value 128 <> metavar "INT")
        <*> switch (long "auto-high-low")
        <*> strOption (long "objective" <> value "roi" <> metavar "NAME")
        <*> option auto (long "penalty-max-drawdown" <> value 1.5 <> metavar "FLOAT")
        <*> option auto (long "penalty-turnover" <> value 0.2 <> metavar "FLOAT")
        <*> option auto (long "min-annualized-return" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "min-calmar" <> value 0.8 <> metavar "FLOAT")
        <*> option auto (long "max-turnover" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "min-round-trips" <> value 20 <> metavar "INT")
        <*> option auto (long "min-win-rate" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "min-profit-factor" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "min-exposure" <> value 0.10 <> metavar "FLOAT")
        <*> option auto (long "min-kelly-lite-exposure-reduction" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "max-kelly-lite-exposure-ratio" <> value 0.95 <> metavar "FLOAT")
        <*> option auto (long "min-sharpe" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "min-wf-sharpe-mean" <> value 0.8 <> metavar "FLOAT")
        <*> option auto (long "max-wf-sharpe-std" <> value 1.0 <> metavar "FLOAT")
        <*> strOption (long "tune-objective" <> value "roi" <> metavar "NAME")
        <*> option auto (long "tune-penalty-max-drawdown" <> value 1.5 <> metavar "FLOAT")
        <*> option auto (long "tune-penalty-turnover" <> value 0.2 <> metavar "FLOAT")
        <*> option auto (long "tune-stress-vol-mult" <> value 1.25 <> metavar "FLOAT")
        <*> option auto (long "tune-stress-shock" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "tune-stress-weight" <> value 0.2 <> metavar "FLOAT")
        <*> optional (option auto (long "tune-stress-vol-mult-min" <> metavar "FLOAT"))
        <*> optional (option auto (long "tune-stress-vol-mult-max" <> metavar "FLOAT"))
        <*> optional (option auto (long "tune-stress-shock-min" <> metavar "FLOAT"))
        <*> optional (option auto (long "tune-stress-shock-max" <> metavar "FLOAT"))
        <*> optional (option auto (long "tune-stress-weight-min" <> metavar "FLOAT"))
        <*> optional (option auto (long "tune-stress-weight-max" <> metavar "FLOAT"))
        <*> option auto (long "walk-forward-folds-min" <> value 7 <> metavar "INT")
        <*> option auto (long "walk-forward-folds-max" <> value 7 <> metavar "INT")
        <*> option auto (long "walk-forward-embargo-bars-min" <> value 1 <> metavar "INT")
        <*> option auto (long "walk-forward-embargo-bars-max" <> value 3 <> metavar "INT")
        <*> optional (strOption (long "interval" <> metavar "INTERVAL"))
        <*> optional (strOption (long "intervals" <> metavar "LIST"))
        <*> optional (strOption (long "platform" <> metavar "NAME"))
        <*> optional (strOption (long "platforms" <> metavar "LIST"))
        <*> switch (long "futures" <> help "Use Binance USDT-M futures endpoints for data (Binance only)")
        <*> option auto (long "bars-min" <> value 0 <> metavar "INT")
        <*> option auto (long "bars-max" <> value 0 <> metavar "INT")
        <*> option auto (long "bars-auto-prob" <> value 0.25 <> metavar "FLOAT")
        <*> strOption (long "bars-distribution" <> value "uniform" <> metavar "NAME")
        <*> option auto (long "epochs-min" <> value 0 <> metavar "INT")
        <*> option auto (long "epochs-max" <> value 2 <> metavar "INT")
        <*> option auto (long "slippage-max" <> value 0.0005 <> metavar "FLOAT")
        <*> option auto (long "spread-max" <> value 0.0005 <> metavar "FLOAT")
        -- Degenerate fee range on purpose: every trial scores under the same real
        -- Binance taker rate, so cost draws can never differentiate leaderboard rank.
        <*> option auto (long "fee-min" <> value 0.0005 <> metavar "FLOAT")
        <*> option auto (long "fee-max" <> value 0.0005 <> metavar "FLOAT")
        <*> option auto (long "funding-rate-min" <> value 0.1 <> metavar "FLOAT")
        <*> option auto (long "funding-rate-max" <> value 0.1 <> metavar "FLOAT")
        <*> option auto (long "p-funding-by-side" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "p-funding-on-open" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "rebalance-bars-min" <> value 24 <> metavar "INT")
        <*> option auto (long "rebalance-bars-max" <> value 24 <> metavar "INT")
        <*> option auto (long "rebalance-threshold-min" <> value 0.05 <> metavar "FLOAT")
        <*> option auto (long "rebalance-threshold-max" <> value 0.05 <> metavar "FLOAT")
        <*> option auto (long "rebalance-cost-mult-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "rebalance-cost-mult-max" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "p-rebalance-global" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "p-rebalance-reset-on-signal" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "open-threshold-min" <> value 5e-4 <> metavar "FLOAT")
        <*> option auto (long "open-threshold-max" <> value 2e-2 <> metavar "FLOAT")
        <*> pure False
        <*> option auto (long "close-threshold-min" <> value 5e-4 <> metavar "FLOAT")
        <*> option auto (long "close-threshold-max" <> value 2e-2 <> metavar "FLOAT")
        <*> pure False
        <*> option auto (long "min-hold-bars-min" <> value 2 <> metavar "INT")
        <*> option auto (long "min-hold-bars-max" <> value 8 <> metavar "INT")
        <*> option auto (long "cooldown-bars-min" <> value 1 <> metavar "INT")
        <*> option auto (long "cooldown-bars-max" <> value 3 <> metavar "INT")
        <*> option auto (long "max-hold-bars-min" <> value 24 <> metavar "INT")
        <*> option auto (long "max-hold-bars-max" <> value 72 <> metavar "INT")
        -- Defaults must clear `venueMinEdgeFloor` (1.5x the venue round-trip
        -- cost). Sampling below it produces negative-expectancy combos.
        <*> option auto (long "min-edge-min" <> value 0.002 <> metavar "FLOAT")
        <*> option auto (long "min-edge-max" <> value 0.006 <> metavar "FLOAT")
        <*> option auto (long "min-signal-to-noise-min" <> value 0.5 <> metavar "FLOAT")
        <*> option auto (long "min-signal-to-noise-max" <> value 1.2 <> metavar "FLOAT")
        <*> option auto (long "snr-size-weight-min" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "snr-size-weight-max" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "p-threshold-factor" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "threshold-factor-alpha-min" <> value 0.1 <> metavar "FLOAT")
        <*> option auto (long "threshold-factor-alpha-max" <> value 0.4 <> metavar "FLOAT")
        <*> option auto (long "threshold-factor-min-min" <> value 0.5 <> metavar "FLOAT")
        <*> option auto (long "threshold-factor-min-max" <> value 0.9 <> metavar "FLOAT")
        <*> option auto (long "threshold-factor-max-min" <> value 1.1 <> metavar "FLOAT")
        <*> option auto (long "threshold-factor-max-max" <> value 2.0 <> metavar "FLOAT")
        <*> option auto (long "threshold-factor-floor-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "threshold-factor-floor-max" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "threshold-factor-weight-min" <> value (-1.0) <> metavar "FLOAT")
        <*> option auto (long "threshold-factor-weight-max" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "edge-buffer-min" <> value 0.0001 <> metavar "FLOAT")
        <*> option auto (long "edge-buffer-max" <> value 0.0005 <> metavar "FLOAT")
        <*> option auto (long "p-cost-aware-edge" <> value 1.0 <> metavar "FLOAT")
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
        <*> option auto (long "protection-min-confidence-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "protection-min-confidence-max" <> value 0.0 <> metavar "FLOAT")
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
        <*> option auto (long "p-confidence-sizing" <> value 0.85 <> metavar "FLOAT")
        <*> option auto (long "min-position-size-min" <> value 0.1 <> metavar "FLOAT")
        <*> option auto (long "min-position-size-max" <> value 0.3 <> metavar "FLOAT")
        -- Position size caps must be modest until the system shows >=30 live
        -- orders with positive expectancy under the new cost-floor guards. At
        -- 10-20x leverage on perps a single 2% adverse move on a 0.9-sized
        -- position is a 18-36% account hit (the 2026-06-13 cliff shape).
        <*> option auto (long "max-position-size-min" <> value 0.15 <> metavar "FLOAT")
        <*> option auto (long "max-position-size-max" <> value 0.25 <> metavar "FLOAT")
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
        <*> option auto (long "p-disable-stop" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "p-disable-tp" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "p-disable-trail" <> value 0.6 <> metavar "FLOAT")
        <*> option auto (long "stop-vol-mult-min" <> value 0.8 <> metavar "FLOAT")
        <*> option auto (long "stop-vol-mult-max" <> value 3.0 <> metavar "FLOAT")
        <*> option auto (long "tp-vol-mult-min" <> value 1.2 <> metavar "FLOAT")
        <*> option auto (long "tp-vol-mult-max" <> value 4.0 <> metavar "FLOAT")
        <*> option auto (long "trail-vol-mult-min" <> value 0.8 <> metavar "FLOAT")
        <*> option auto (long "trail-vol-mult-max" <> value 3.0 <> metavar "FLOAT")
        <*> option auto (long "p-disable-stop-vol-mult" <> value 0.35 <> metavar "FLOAT")
        <*> option auto (long "p-disable-tp-vol-mult" <> value 0.35 <> metavar "FLOAT")
        <*> option auto (long "p-disable-trail-vol-mult" <> value 0.5 <> metavar "FLOAT")
        <*> option auto (long "risk-per-trade-min" <> value 0.005 <> metavar "FLOAT")
        <*> option auto (long "risk-per-trade-max" <> value 0.02 <> metavar "FLOAT")
        <*> option auto (long "p-disable-risk-per-trade" <> value 0.3 <> metavar "FLOAT")
        <*> option auto (long "p-disable-max-dd" <> value 0.9 <> metavar "FLOAT")
        <*> option auto (long "p-disable-max-dl" <> value 0.9 <> metavar "FLOAT")
        <*> option auto (long "p-disable-max-wl" <> value 0.9 <> metavar "FLOAT")
        <*> option auto (long "p-disable-max-oe" <> value 0.95 <> metavar "FLOAT")
        <*> option auto (long "max-dd-min" <> value 0.05 <> metavar "FLOAT")
        <*> option auto (long "max-dd-max" <> value 0.50 <> metavar "FLOAT")
        <*> option auto (long "max-dl-min" <> value 0.02 <> metavar "FLOAT")
        <*> option auto (long "max-dl-max" <> value 0.30 <> metavar "FLOAT")
        <*> option auto (long "max-wl-min" <> value 0.03 <> metavar "FLOAT")
        <*> option auto (long "max-wl-max" <> value 0.40 <> metavar "FLOAT")
        <*> option auto (long "max-oe-min" <> value 1 <> metavar "INT")
        <*> option auto (long "max-oe-max" <> value 10 <> metavar "INT")
        <*> option auto (long "method-weight-11" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-10" <> value 4.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-01" <> value 0.1 <> metavar "FLOAT")
        <*> option auto (long "method-weight-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-conf-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-conf-pick" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-conformal-clip" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-cost-pick" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-harmonic-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-disagreement-guard" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-median-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-neutral-guard" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-risk-parity-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-consensus-boost" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-anchor-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-tension-gate" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-entropy-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-coherence-gate" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-divergence-gate" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-fractal-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-phase-cancel" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-softmax-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-smooth-softmax-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-hedge-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-net-softmax-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-edge-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-edge-pick" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-geo-blend" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-regime-switch" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "method-weight-bandit-router" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "blend-weight-min" <> value 0.5 <> metavar "FLOAT")
        <*> option auto (long "blend-weight-max" <> value 0.5 <> metavar "FLOAT")
        <*> option auto (long "router-score-pnl-weight-min" <> value 0.25 <> metavar "FLOAT")
        <*> option auto (long "router-score-pnl-weight-max" <> value 0.85 <> metavar "FLOAT")
        <*> strOption (long "normalizations" <> value "none,minmax,standard,log" <> metavar "LIST")
        <*> option auto (long "hidden-size-min" <> value 8 <> metavar "INT")
        <*> option auto (long "hidden-size-max" <> value 16 <> metavar "INT")
        <*> option auto (long "lr-min" <> value 1e-4 <> metavar "FLOAT")
        <*> option auto (long "lr-max" <> value 1e-2 <> metavar "FLOAT")
        <*> option auto (long "val-ratio-min" <> value 0.1 <> metavar "FLOAT")
        <*> option auto (long "val-ratio-max" <> value 0.4 <> metavar "FLOAT")
        <*> option auto (long "patience-max" <> value 20 <> metavar "INT")
        <*> option auto (long "grad-clip-min" <> value 0.001 <> metavar "FLOAT")
        <*> option auto (long "grad-clip-max" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "p-disable-grad-clip" <> value 0.7 <> metavar "FLOAT")
        <*> strOption (long "predictors" <> value "all" <> metavar "LIST")
        <*> option auto (long "router-lookback-min" <> value 20 <> metavar "INT")
        <*> option auto (long "router-lookback-max" <> value 180 <> metavar "INT")
        <*> option auto (long "router-min-score-min" <> value 0.05 <> metavar "FLOAT")
        <*> option auto (long "router-min-score-max" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "fee-fixed-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "fee-fixed-max" <> value 0.002 <> metavar "FLOAT")
        <*> option auto (long "slippage-impact-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "slippage-impact-max" <> value 0.01 <> metavar "FLOAT")
        <*> option auto (long "slippage-impact-power-min" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "slippage-impact-power-max" <> value 2.0 <> metavar "FLOAT")
        <*> option auto (long "slippage-vol-mult-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "slippage-vol-mult-max" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "spread-vol-mult-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "spread-vol-mult-max" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "take-profit-partial-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "take-profit-partial-max" <> value 0.75 <> metavar "FLOAT")
        <*> option auto (long "p-disable-take-profit-partial" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "max-trades-per-day-min" <> value 0 <> metavar "INT")
        <*> option auto (long "max-trades-per-day-max" <> value 8 <> metavar "INT")
        <*> option auto (long "p-disable-max-trades-per-day" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "expectancy-lookback-min" <> value 5 <> metavar "INT")
        <*> option auto (long "expectancy-lookback-max" <> value 80 <> metavar "INT")
        <*> option auto (long "min-expectancy-min" <> value (-0.01) <> metavar "FLOAT")
        <*> option auto (long "min-expectancy-max" <> value 0.01 <> metavar "FLOAT")
        <*> option auto (long "p-disable-min-expectancy" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "loss-streak-max-min" <> value 0 <> metavar "INT")
        <*> option auto (long "loss-streak-max-max" <> value 8 <> metavar "INT")
        <*> option auto (long "p-disable-loss-streak-max" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "loss-streak-cooldown-bars-min" <> value 0 <> metavar "INT")
        <*> option auto (long "loss-streak-cooldown-bars-max" <> value 40 <> metavar "INT")
        <*> option auto (long "p-disable-loss-streak-cooldown-bars" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "max-open-positions-min" <> value 0 <> metavar "INT")
        <*> option auto (long "max-open-positions-max" <> value 8 <> metavar "INT")
        <*> option auto (long "p-disable-max-open-positions" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "max-gross-exposure-min" <> value 0.5 <> metavar "FLOAT")
        <*> option auto (long "max-gross-exposure-max" <> value 3.0 <> metavar "FLOAT")
        <*> option auto (long "p-disable-max-gross-exposure" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "max-net-exposure-min" <> value 0.25 <> metavar "FLOAT")
        <*> option auto (long "max-net-exposure-max" <> value 2.0 <> metavar "FLOAT")
        <*> option auto (long "p-disable-max-net-exposure" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "max-exposure-per-base-min" <> value 0.25 <> metavar "FLOAT")
        <*> option auto (long "max-exposure-per-base-max" <> value 2.0 <> metavar "FLOAT")
        <*> option auto (long "p-disable-max-exposure-per-base" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "max-open-per-base-min" <> value 0 <> metavar "INT")
        <*> option auto (long "max-open-per-base-max" <> value 4 <> metavar "INT")
        <*> option auto (long "p-disable-max-open-per-base" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "adaptive-edge-buffer-max-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "adaptive-edge-buffer-max-max" <> value 0.02 <> metavar "FLOAT")
        <*> option auto (long "adaptive-min-signal-to-noise-max-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "adaptive-min-signal-to-noise-max-max" <> value 2.0 <> metavar "FLOAT")
        <*> option auto (long "adaptive-trend-lookback-max-min" <> value 2 <> metavar "INT")
        <*> option auto (long "adaptive-trend-lookback-max-max" <> value 120 <> metavar "INT")
        <*> option auto (long "adaptive-kalman-z-min-max-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "adaptive-kalman-z-min-max-max" <> value 3.0 <> metavar "FLOAT")
        <*> option auto (long "p-adaptive-filters" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "perf-lookback-min" <> value 5 <> metavar "INT")
        <*> option auto (long "perf-lookback-max" <> value 60 <> metavar "INT")
        <*> option auto (long "perf-min-win-rate-min" <> value 0.45 <> metavar "FLOAT")
        <*> option auto (long "perf-min-win-rate-max" <> value 0.65 <> metavar "FLOAT")
        <*> option auto (long "p-disable-perf-min-win-rate" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "perf-min-profit-factor-min" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "perf-min-profit-factor-max" <> value 2.5 <> metavar "FLOAT")
        <*> option auto (long "p-disable-perf-min-profit-factor" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "p-meta-label-filter" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "meta-label-min-edge-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "meta-label-min-edge-max" <> value 0.001 <> metavar "FLOAT")
        <*> option auto (long "meta-label-min-confidence-min" <> value 0.4 <> metavar "FLOAT")
        <*> option auto (long "meta-label-min-confidence-max" <> value 0.8 <> metavar "FLOAT")
        <*> option auto (long "p-meta-label-require-band" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "p-regime-parameter-bank" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "regime-bank-hysteresis-min" <> value 0.05 <> metavar "FLOAT")
        <*> option auto (long "regime-bank-hysteresis-max" <> value 0.05 <> metavar "FLOAT")
        <*> option auto (long "regime-trend-open-mult-min" <> value 0.9 <> metavar "FLOAT")
        <*> option auto (long "regime-trend-open-mult-max" <> value 0.9 <> metavar "FLOAT")
        <*> option auto (long "regime-mr-open-mult-min" <> value 1.1 <> metavar "FLOAT")
        <*> option auto (long "regime-mr-open-mult-max" <> value 1.1 <> metavar "FLOAT")
        <*> option auto (long "regime-high-vol-open-mult-min" <> value 1.3 <> metavar "FLOAT")
        <*> option auto (long "regime-high-vol-open-mult-max" <> value 1.3 <> metavar "FLOAT")
        <*> option auto (long "regime-trend-size-mult-min" <> value 1.1 <> metavar "FLOAT")
        <*> option auto (long "regime-trend-size-mult-max" <> value 1.1 <> metavar "FLOAT")
        <*> option auto (long "regime-mr-size-mult-min" <> value 0.9 <> metavar "FLOAT")
        <*> option auto (long "regime-mr-size-mult-max" <> value 0.9 <> metavar "FLOAT")
        <*> option auto (long "regime-high-vol-size-mult-min" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "regime-high-vol-size-mult-max" <> value 0.7 <> metavar "FLOAT")
        <*> technicalOptimizerRangesParser
        <*> option auto (long "p-multi-timeframe-consensus" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "mtf-fast-bars-min" <> value 5 <> metavar "INT")
        <*> option auto (long "mtf-fast-bars-max" <> value 5 <> metavar "INT")
        <*> option auto (long "mtf-mid-bars-min" <> value 20 <> metavar "INT")
        <*> option auto (long "mtf-mid-bars-max" <> value 20 <> metavar "INT")
        <*> option auto (long "mtf-slow-bars-min" <> value 60 <> metavar "INT")
        <*> option auto (long "mtf-slow-bars-max" <> value 60 <> metavar "INT")
        <*> option auto (long "mtf-min-agree-min" <> value 2 <> metavar "INT")
        <*> option auto (long "mtf-min-agree-max" <> value 2 <> metavar "INT")
        <*> option auto (long "p-cross-asset-confirmation" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "cross-asset-min-beta-min" <> value 0.05 <> metavar "FLOAT")
        <*> option auto (long "cross-asset-min-beta-max" <> value 0.05 <> metavar "FLOAT")
        <*> option auto (long "cross-asset-min-edge-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "cross-asset-min-edge-max" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "p-pairs-stat-arb" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "pairs-stat-arb-lookback-min" <> value 120 <> metavar "INT")
        <*> option auto (long "pairs-stat-arb-lookback-max" <> value 120 <> metavar "INT")
        <*> option auto (long "pairs-stat-arb-z-entry-min" <> value 2.0 <> metavar "FLOAT")
        <*> option auto (long "pairs-stat-arb-z-entry-max" <> value 2.0 <> metavar "FLOAT")
        <*> option auto (long "pairs-stat-arb-size-mult-min" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "pairs-stat-arb-size-mult-max" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "p-funding-oi-aware" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "funding-oi-funding-cap-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "funding-oi-funding-cap-max" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "p-disable-funding-oi-funding-cap" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "funding-oi-vol-lookback-min" <> value 48 <> metavar "INT")
        <*> option auto (long "funding-oi-vol-lookback-max" <> value 48 <> metavar "INT")
        <*> option auto (long "funding-oi-vol-cap-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "funding-oi-vol-cap-max" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "p-disable-funding-oi-vol-cap" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "funding-oi-size-mult-min" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "funding-oi-size-mult-max" <> value 0.7 <> metavar "FLOAT")
        <*> option auto (long "p-kelly-lite-sizing" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "kelly-lite-fraction-min" <> value 0.5 <> metavar "FLOAT")
        <*> option auto (long "kelly-lite-fraction-max" <> value 0.5 <> metavar "FLOAT")
        <*> option auto (long "kelly-lite-floor-min" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "kelly-lite-floor-max" <> value 0.0 <> metavar "FLOAT")
        <*> option auto (long "kelly-lite-cap-min" <> value 1.0 <> metavar "FLOAT")
        <*> option auto (long "kelly-lite-cap-max" <> value 1.0 <> metavar "FLOAT")
        <*> switch (long "cross-exchange-coinbase" <> help "Pass --cross-exchange-coinbase to each trial so it enriches the LSTM/Kalman with same-asset Coinbase data (CSV symbol/interval must be Coinbase-eligible; fail-open otherwise)")

technicalOptimizerRangesParser :: Parser TechnicalOptimizerRanges
technicalOptimizerRangesParser =
    TechnicalOptimizerRanges
        <$> doubleRangeOption "ta-entry-open-threshold" 5.0e-4 3.0e-3
        <*> doubleRangeOption "ta-trend-adx-threshold" 15 30
        <*> doubleRangeOption "ta-trend-stop-atr-mult" 1.0 3.5
        <*> doubleRangeOption "ta-trend-tp-atr-mult" 1.5 5.0
        <*> doubleRangeOption "ta-reversion-rsi-lower" 25 45
        <*> doubleRangeOption "ta-reversion-rsi-upper" 55 75
        <*> doubleRangeOption "ta-reversion-stoch-lower" 10 35
        <*> doubleRangeOption "ta-reversion-stoch-upper" 65 90
        <*> doubleRangeOption "ta-reversion-envelope-lower-mult" 1.0 1.04
        <*> doubleRangeOption "ta-reversion-envelope-upper-mult" 0.96 1.0
        <*> doubleRangeOption "ta-reversion-stop-atr-mult" 0.8 3.0
        <*> doubleRangeOption "ta-breakout-mfi-threshold" 40 60
        <*> doubleRangeOption "ta-breakout-stop-atr-mult" 1.0 3.5
        <*> doubleRangeOption "ta-breakout-tp-atr-mult" 2.0 6.0
        <*> doubleRangeOption "ta-sma-cross-stop-atr-mult" 1.0 3.5
        <*> doubleRangeOption "ta-sma-cross-tp-atr-mult" 1.5 5.0
        <*> doubleRangeOption "ta-regime-switch-adx-threshold" 18 35
        <*> doubleRangeOption "ta-regime-switch-bb-width-threshold" 5.0e-4 8.0e-3
        <*> doubleRangeOption "ta-best-min-confidence" 0.2 0.6
        <*> doubleRangeOption "ta-trend-confidence-adx-offset" 10 30
        <*> doubleRangeOption "ta-trend-confidence-adx-scale" 10 40
        <*> doubleRangeOption "ta-trend-confidence-ma-spread-mult" 3 15
        <*> doubleRangeOption "ta-trend-confidence-aroon-gap-mult" 0.005 0.02
        <*> doubleRangeOption "ta-reversion-confidence-roc-mult" 0.05 0.5
        <*> doubleRangeOption "ta-breakout-flow-disagreement-confidence" 0.2 0.8
        <*> doubleRangeOption "ta-breakout-confidence-distance-mult" 4 20
        <*> doubleRangeOption "ta-sma-cross-confidence-spread-mult" 8 40
        <*> doubleRangeOption "blend-softmax-scale" 100 1500
        <*> doubleRangeOption "blend-net-softmax-scale" 1000 12000
        <*> doubleRangeOption "blend-smooth-alpha" 0.05 0.5
        <*> doubleRangeOption "blend-hedge-eta" 1 12
        <*> doubleRangeOption "blend-hedge-max-error" 0.02 0.25
        <*> doubleRangeOption "blend-divergence-k" 1 8
        <*> doubleRangeOption "blend-regime-high-vol-cutoff" 0.4 0.8
        <*> doubleRangeOption "blend-regime-kalman-z-cutoff" 0.5 2.0
        <*> doubleRangeOption "blend-bandit-explore-scale" 0.0 0.5
        <*> doubleRangeOption "blend-fractal-return-clamp" 0.25 1.5
        <*> doubleRangeOption "blend-fractal-aligned-gain" 0.8 1.5
        <*> doubleRangeOption "blend-fractal-conflict-gain" 0.4 1.1
        <*> doubleRangeOption "blend-coherence-conflict-floor" 0.0 0.6
        <*> doubleRangeOption "blend-coherence-conflict-scale" 0.0 1.0
        <*> doubleRangeOption "blend-coherence-boost-threshold" 0.3 0.9
        <*> doubleRangeOption "blend-coherence-boost-gain" 0.0 0.8
        <*> doubleRangeOption "blend-coherence-boost-span" 0.1 0.8
        <*> doubleRangeOption "blend-anchor-conflict-base" 0.2 0.9
        <*> doubleRangeOption "blend-anchor-conflict-scale" 0.0 0.8
        <*> doubleRangeOption "blend-anchor-aligned-scale" 0.0 0.5
        <*> doubleRangeOption "blend-tension-conflict-shrink" 0.0 0.6
        <*> doubleRangeOption "blend-tension-neutral-shrink" 0.0 0.9
        <*> doubleRangeOption "blend-entropy-conflict-floor" 0.0 0.8
        <*> doubleRangeOption "blend-entropy-conflict-scale" 0.0 1.0
        <*> doubleRangeOption "blend-entropy-aligned-base" 0.6 1.0
        <*> doubleRangeOption "blend-entropy-aligned-entropy-scale" 0.0 0.6
        <*> doubleRangeOption "blend-phase-cancel-return-clamp" 0.25 1.5
        <*> doubleRangeOption "blend-phase-cancel-conflict-floor" 0.0 0.5
        <*> doubleRangeOption "blend-phase-cancel-conflict-scale" 0.0 1.2
        <*> doubleRangeOption "blend-phase-cancel-alignment-scale" 0.0 1.0
        <*> doubleRangeOption "signal-entry-edge-headroom-mult" 1.1 2.5
        <*> doubleRangeOption "signal-entry-edge-spike-mult" 200 2000
        <*> doubleRangeOption "signal-entry-edge-spike-cap" 1.0 8.0
        <*> intRangeOption "signal-entry-edge-spike-consecutive-limit" 1 5
        <*> doubleRangeOption "signal-trend-slack-mult" 0.0 1.0
        <*> doubleRangeOption "signal-trend-slack-cap" 2.0e-3 2.0e-2
        <*> intRangeOption "signal-directionality-lookback-bars" 12 48
        <*> doubleRangeOption "signal-directionality-chop-efficiency-max" 0.03 0.15
        <*> doubleRangeOption "signal-directionality-mr-efficiency-max" 0.20 0.50
        <*> doubleRangeOption "signal-directionality-weak-z-min" 0.20 1.00
        <*> doubleRangeOption "signal-predictor-tracking-floor" 0.01 0.15

doubleRangeOption :: String -> Double -> Double -> Parser (Double, Double)
doubleRangeOption name lo hi =
    (,)
        <$> option auto (long (name ++ "-min") <> value lo <> metavar "FLOAT")
        <*> option auto (long (name ++ "-max") <> value hi <> metavar "FLOAT")

intRangeOption :: String -> Int -> Int -> Parser (Int, Int)
intRangeOption name lo hi =
    (,)
        <$> option auto (long (name ++ "-min") <> value lo <> metavar "INT")
        <*> option auto (long (name ++ "-max") <> value hi <> metavar "INT")

validateArgs :: OptimizerArgs -> Either String OptimizerArgs
validateArgs args = do
    let dataVal = normalizeMaybe (oaData args)
        symbolVal = normalizeMaybe (oaBinanceSymbol args)
        intervalVal = normalizeMaybe (oaInterval args)
        intervalsVal = normalizeMaybe (oaIntervals args)
        platformVal = normalizeMaybe (oaPlatform args)
        platformsVal = normalizeMaybe (oaPlatforms args)
        intervalsFinal =
            case intervalVal of
                Just _ -> Nothing
                Nothing -> Just (fromMaybe defaultIntervals intervalsVal)
    objectiveCode <- normalizeCliObjective "objective" (oaObjective args)
    tuneObjective <- normalizeCliObjective "tune objective" (oaTuneObjective args)
    let args' =
            args
                { oaData = dataVal
                , oaBinanceSymbol = symbolVal
                , oaInterval = intervalVal
                , oaIntervals = intervalsFinal
                , oaPlatform = platformVal
                , oaPlatforms = platformsVal
                , oaObjective = objectiveCode
                , oaTuneObjective = tuneObjective
                }
    when (isJust dataVal == isJust symbolVal) $
        Left "Provide exactly one of --data or --symbol/--binance-symbol."
    when (isJust intervalVal && isJust intervalsVal) $
        Left "Provide only one of --interval or --intervals."
    when (isJust platformVal && isJust platformsVal) $
        Left "Provide only one of --platform or --platforms."
    when (oaBinanceFutures args' && maybe False (\p -> map toLower (trim p) /= "binance") platformVal) $
        Left "--futures requires --platform binance (or omit --platform to use the default)."
    when (maybe False (< 0) (oaSeedTrials args)) $
        Left "--seed-trials must be >= 0."
    when (maybe False (\r -> r < 0 || r > 1) (oaSeedRatio args)) $
        Left "--seed-ratio must be between 0 and 1."
    when (oaSurvivorFraction args < 0 || oaSurvivorFraction args > 1) $
        Left "--survivor-fraction must be between 0 and 1."
    when (oaPerturbScaleDouble args < 0) $
        Left "--perturb-scale-double must be >= 0."
    when (oaPerturbScaleInt args < 0) $
        Left "--perturb-scale-int must be >= 0."
    when (oaEarlyStopNoImprove args < 0) $
        Left "--early-stop-no-improve must be >= 0."
    when (oaQualityMinTrials args < 1) $
        Left "--quality-min-trials must be >= 1."
    when (oaQualityMaxEpochs args < 1) $
        Left "--quality-max-epochs must be >= 1."
    when (oaQualityMaxHiddenSize args < 1) $
        Left "--quality-max-hidden-size must be >= 1."
    unless (oaBarsDistribution args `elem` barsDistributionChoices) $
        Left
            ( "Invalid bars distribution: "
                ++ show (oaBarsDistribution args)
                ++ " (expected one of: "
                ++ intercalate ", " barsDistributionChoices
                ++ ")"
            )
    when (oaPThresholdFactor args < 0 || oaPThresholdFactor args > 1) $
        Left "--p-threshold-factor must be between 0 and 1."
    when
        ( oaThresholdFactorAlphaMin args < 0
            || oaThresholdFactorAlphaMin args > 1
            || oaThresholdFactorAlphaMax args < 0
            || oaThresholdFactorAlphaMax args > 1
        )
        $ Left "--threshold-factor-alpha-min/max must be between 0 and 1."
    when (oaThresholdFactorMinMin args < 0 || oaThresholdFactorMinMax args < 0) $
        Left "--threshold-factor-min-min/max must be >= 0."
    when (oaThresholdFactorMaxMin args < 0 || oaThresholdFactorMaxMax args < 0) $
        Left "--threshold-factor-max-min/max must be >= 0."
    when (oaThresholdFactorFloorMin args < 0 || oaThresholdFactorFloorMax args < 0) $
        Left "--threshold-factor-floor-min/max must be >= 0."
    when (oaRouterLookbackMin args < 2 || oaRouterLookbackMax args < 2) $
        Left "--router-lookback-min/max must be >= 2."
    when
        ( oaTakeProfitPartialMin args < 0
            || oaTakeProfitPartialMin args >= 1
            || oaTakeProfitPartialMax args < 0
            || oaTakeProfitPartialMax args >= 1
        )
        $ Left "--take-profit-partial-min/max must be >= 0 and < 1."
    when (oaMaxTradesPerDayMin args < 0 || oaMaxTradesPerDayMax args < 0) $
        Left "--max-trades-per-day-min/max must be >= 0."
    when (oaExpectancyLookbackMin args < 1 || oaExpectancyLookbackMax args < 1) $
        Left "--expectancy-lookback-min/max must be >= 1."
    when (oaLossStreakMaxMin args < 0 || oaLossStreakMaxMax args < 0) $
        Left "--loss-streak-max-min/max must be >= 0."
    when (oaLossStreakCooldownBarsMin args < 0 || oaLossStreakCooldownBarsMax args < 0) $
        Left "--loss-streak-cooldown-bars-min/max must be >= 0."
    when (oaMaxOpenPositionsMin args < 0 || oaMaxOpenPositionsMax args < 0) $
        Left "--max-open-positions-min/max must be >= 0."
    when (oaMaxOpenPerBaseMin args < 0 || oaMaxOpenPerBaseMax args < 0) $
        Left "--max-open-per-base-min/max must be >= 0."
    when (oaPDisableMaxWl args < 0 || oaPDisableMaxWl args > 1) $
        Left "--p-disable-max-wl must be between 0 and 1."
    when (oaMaxWlMin args < 0 || oaMaxWlMax args < 0) $
        Left "--max-wl-min/max must be >= 0."
    when (oaAdaptiveTrendLookbackMaxMin args < 1 || oaAdaptiveTrendLookbackMaxMax args < 1) $
        Left "--adaptive-trend-lookback-max-min/max must be >= 1."
    when (oaPAdaptiveFilters args < 0 || oaPAdaptiveFilters args > 1) $
        Left "--p-adaptive-filters must be between 0 and 1."
    when (oaPerfLookbackMin args < 1 || oaPerfLookbackMax args < 1) $
        Left "--perf-lookback-min/max must be >= 1."
    when
        ( oaPerfMinWinRateMin args < 0
            || oaPerfMinWinRateMin args > 1
            || oaPerfMinWinRateMax args < 0
            || oaPerfMinWinRateMax args > 1
        )
        $ Left "--perf-min-win-rate-min/max must be between 0 and 1."
    when (oaPDisablePerfMinWinRate args < 0 || oaPDisablePerfMinWinRate args > 1) $
        Left "--p-disable-perf-min-win-rate must be between 0 and 1."
    when (oaPerfMinProfitFactorMin args < 0 || oaPerfMinProfitFactorMax args < 0) $
        Left "--perf-min-profit-factor-min/max must be >= 0."
    when (oaPDisablePerfMinProfitFactor args < 0 || oaPDisablePerfMinProfitFactor args > 1) $
        Left "--p-disable-perf-min-profit-factor must be between 0 and 1."
    when (oaPMetaLabelFilter args < 0 || oaPMetaLabelFilter args > 1) $
        Left "--p-meta-label-filter must be between 0 and 1."
    when (oaMetaLabelMinEdgeMin args < 0 || oaMetaLabelMinEdgeMax args < 0) $
        Left "--meta-label-min-edge-min/max must be >= 0."
    when
        ( oaMetaLabelMinConfidenceMin args < 0
            || oaMetaLabelMinConfidenceMin args > 1
            || oaMetaLabelMinConfidenceMax args < 0
            || oaMetaLabelMinConfidenceMax args > 1
        )
        $ Left "--meta-label-min-confidence-min/max must be between 0 and 1."
    when (oaPMetaLabelRequireBand args < 0 || oaPMetaLabelRequireBand args > 1) $
        Left "--p-meta-label-require-band must be between 0 and 1."
    when (oaPRegimeParameterBank args < 0 || oaPRegimeParameterBank args > 1) $
        Left "--p-regime-parameter-bank must be between 0 and 1."
    when
        ( oaRegimeBankHysteresisMin args < 0
            || oaRegimeBankHysteresisMin args > 1
            || oaRegimeBankHysteresisMax args < 0
            || oaRegimeBankHysteresisMax args > 1
        )
        $ Left "--regime-bank-hysteresis-min/max must be between 0 and 1."
    when
        ( oaRegimeTrendOpenMultMin args < 0
            || oaRegimeTrendOpenMultMax args < 0
            || oaRegimeMrOpenMultMin args < 0
            || oaRegimeMrOpenMultMax args < 0
            || oaRegimeHighVolOpenMultMin args < 0
            || oaRegimeHighVolOpenMultMax args < 0
            || oaRegimeTrendSizeMultMin args < 0
            || oaRegimeTrendSizeMultMax args < 0
            || oaRegimeMrSizeMultMin args < 0
            || oaRegimeMrSizeMultMax args < 0
            || oaRegimeHighVolSizeMultMin args < 0
            || oaRegimeHighVolSizeMultMax args < 0
        )
        $ Left "--regime-*-mult-min/max must be >= 0."
    validateTechnicalOptimizerRanges (oaTechnicalRanges args)
    when (oaPMultiTimeframeConsensus args < 0 || oaPMultiTimeframeConsensus args > 1) $
        Left "--p-multi-timeframe-consensus must be between 0 and 1."
    when
        ( oaMtfFastBarsMin args < 1
            || oaMtfFastBarsMax args < 1
            || oaMtfMidBarsMin args < 1
            || oaMtfMidBarsMax args < 1
            || oaMtfSlowBarsMin args < 1
            || oaMtfSlowBarsMax args < 1
        )
        $ Left "--mtf-*-bars-min/max must be >= 1."
    when
        ( oaMtfMinAgreeMin args < 1
            || oaMtfMinAgreeMin args > 3
            || oaMtfMinAgreeMax args < 1
            || oaMtfMinAgreeMax args > 3
        )
        $ Left "--mtf-min-agree-min/max must be between 1 and 3."
    when (oaPCrossAssetConfirmation args < 0 || oaPCrossAssetConfirmation args > 1) $
        Left "--p-cross-asset-confirmation must be between 0 and 1."
    when
        ( oaCrossAssetMinBetaMin args < 0
            || oaCrossAssetMinBetaMax args < 0
            || oaCrossAssetMinEdgeMin args < 0
            || oaCrossAssetMinEdgeMax args < 0
        )
        $ Left "--cross-asset-min-beta/edge-min/max must be >= 0."
    when (oaPPairsStatArb args < 0 || oaPPairsStatArb args > 1) $
        Left "--p-pairs-stat-arb must be between 0 and 1."
    when (oaPairsStatArbLookbackMin args < 2 || oaPairsStatArbLookbackMax args < 2) $
        Left "--pairs-stat-arb-lookback-min/max must be >= 2."
    when (oaPairsStatArbZEntryMin args <= 0 || oaPairsStatArbZEntryMax args <= 0) $
        Left "--pairs-stat-arb-z-entry-min/max must be > 0."
    when
        ( oaPairsStatArbSizeMultMin args < 0
            || oaPairsStatArbSizeMultMin args > 1
            || oaPairsStatArbSizeMultMax args < 0
            || oaPairsStatArbSizeMultMax args > 1
        )
        $ Left "--pairs-stat-arb-size-mult-min/max must be between 0 and 1."
    when (oaPFundingOiAware args < 0 || oaPFundingOiAware args > 1) $
        Left "--p-funding-oi-aware must be between 0 and 1."
    when (oaFundingOiFundingCapMin args < 0 || oaFundingOiFundingCapMax args < 0) $
        Left "--funding-oi-funding-cap-min/max must be >= 0."
    when (oaPDisableFundingOiFundingCap args < 0 || oaPDisableFundingOiFundingCap args > 1) $
        Left "--p-disable-funding-oi-funding-cap must be between 0 and 1."
    when (oaFundingOiVolLookbackMin args < 2 || oaFundingOiVolLookbackMax args < 2) $
        Left "--funding-oi-vol-lookback-min/max must be >= 2."
    when (oaFundingOiVolCapMin args < 0 || oaFundingOiVolCapMax args < 0) $
        Left "--funding-oi-vol-cap-min/max must be >= 0."
    when (oaPDisableFundingOiVolCap args < 0 || oaPDisableFundingOiVolCap args > 1) $
        Left "--p-disable-funding-oi-vol-cap must be between 0 and 1."
    when
        ( oaFundingOiSizeMultMin args < 0
            || oaFundingOiSizeMultMin args > 1
            || oaFundingOiSizeMultMax args < 0
            || oaFundingOiSizeMultMax args > 1
        )
        $ Left "--funding-oi-size-mult-min/max must be between 0 and 1."
    when (oaPKellyLiteSizing args < 0 || oaPKellyLiteSizing args > 1) $
        Left "--p-kelly-lite-sizing must be between 0 and 1."
    when (oaMinKellyLiteExposureReduction args < 0) $
        Left "--min-kelly-lite-exposure-reduction must be >= 0."
    when (oaMaxKellyLiteExposureRatio args < 0) $
        Left "--max-kelly-lite-exposure-ratio must be >= 0."
    when (oaKellyLiteFractionMin args < 0 || oaKellyLiteFractionMax args < 0) $
        Left "--kelly-lite-fraction-min/max must be >= 0."
    when (oaKellyLiteFloorMin args < 0 || oaKellyLiteFloorMax args < 0) $
        Left "--kelly-lite-floor-min/max must be >= 0."
    when (oaKellyLiteCapMin args < 0 || oaKellyLiteCapMax args < 0) $
        Left "--kelly-lite-cap-min/max must be >= 0."
    pure args'

validateTechnicalOptimizerRanges :: TechnicalOptimizerRanges -> Either String ()
validateTechnicalOptimizerRanges ranges = do
    nonNegativeRange "ta-entry-open-threshold" (torTaEntryOpenThreshold ranges)
    unit100Range "ta-trend-adx-threshold" (torTaTrendAdxThreshold ranges)
    positiveRange "ta-trend-stop-atr-mult" (torTaTrendStopAtrMult ranges)
    positiveRange "ta-trend-tp-atr-mult" (torTaTrendTakeProfitAtrMult ranges)
    unit100Range "ta-reversion-rsi-lower" (torTaReversionRsiLower ranges)
    unit100Range "ta-reversion-rsi-upper" (torTaReversionRsiUpper ranges)
    unit100Range "ta-reversion-stoch-lower" (torTaReversionStochLower ranges)
    unit100Range "ta-reversion-stoch-upper" (torTaReversionStochUpper ranges)
    positiveRange "ta-reversion-envelope-lower-mult" (torTaReversionEnvelopeLowerMult ranges)
    positiveRange "ta-reversion-envelope-upper-mult" (torTaReversionEnvelopeUpperMult ranges)
    positiveRange "ta-reversion-stop-atr-mult" (torTaReversionStopAtrMult ranges)
    unit100Range "ta-breakout-mfi-threshold" (torTaBreakoutMfiThreshold ranges)
    positiveRange "ta-breakout-stop-atr-mult" (torTaBreakoutStopAtrMult ranges)
    positiveRange "ta-breakout-tp-atr-mult" (torTaBreakoutTakeProfitAtrMult ranges)
    positiveRange "ta-sma-cross-stop-atr-mult" (torTaSmaCrossStopAtrMult ranges)
    positiveRange "ta-sma-cross-tp-atr-mult" (torTaSmaCrossTakeProfitAtrMult ranges)
    unit100Range "ta-regime-switch-adx-threshold" (torTaRegimeSwitchAdxThreshold ranges)
    nonNegativeRange "ta-regime-switch-bb-width-threshold" (torTaRegimeSwitchBollingerBandwidthThreshold ranges)
    unitRange "ta-best-min-confidence" (torTaBestCandidateMinConfidence ranges)
    unit100Range "ta-trend-confidence-adx-offset" (torTaTrendConfidenceAdxOffset ranges)
    positiveRange "ta-trend-confidence-adx-scale" (torTaTrendConfidenceAdxScale ranges)
    nonNegativeRange "ta-trend-confidence-ma-spread-mult" (torTaTrendConfidenceMaSpreadMult ranges)
    nonNegativeRange "ta-trend-confidence-aroon-gap-mult" (torTaTrendConfidenceAroonGapMult ranges)
    nonNegativeRange "ta-reversion-confidence-roc-mult" (torTaReversionConfidenceRocMult ranges)
    unitRange "ta-breakout-flow-disagreement-confidence" (torTaBreakoutFlowDisagreementConfidence ranges)
    nonNegativeRange "ta-breakout-confidence-distance-mult" (torTaBreakoutConfidenceDistanceMult ranges)
    nonNegativeRange "ta-sma-cross-confidence-spread-mult" (torTaSmaCrossConfidenceSpreadMult ranges)
    positiveRange "blend-softmax-scale" (torBlendSoftmaxScale ranges)
    positiveRange "blend-net-softmax-scale" (torBlendNetSoftmaxScale ranges)
    unitRange "blend-smooth-alpha" (torBlendSmoothAlpha ranges)
    nonNegativeRange "blend-hedge-eta" (torBlendHedgeEta ranges)
    positiveRange "blend-hedge-max-error" (torBlendHedgeMaxError ranges)
    positiveRange "blend-divergence-k" (torBlendDivergenceK ranges)
    unitRange "blend-regime-high-vol-cutoff" (torBlendRegimeHighVolCutoff ranges)
    nonNegativeRange "blend-regime-kalman-z-cutoff" (torBlendRegimeKalmanZCutoff ranges)
    nonNegativeRange "blend-bandit-explore-scale" (torBlendBanditExploreScale ranges)
    positiveRange "blend-fractal-return-clamp" (torBlendFractalReturnClamp ranges)
    nonNegativeRange "blend-fractal-aligned-gain" (torBlendFractalAlignedGain ranges)
    nonNegativeRange "blend-fractal-conflict-gain" (torBlendFractalConflictGain ranges)
    nonNegativeRange "blend-coherence-conflict-floor" (torBlendCoherenceConflictFloor ranges)
    nonNegativeRange "blend-coherence-conflict-scale" (torBlendCoherenceConflictScale ranges)
    unitRange "blend-coherence-boost-threshold" (torBlendCoherenceBoostThreshold ranges)
    nonNegativeRange "blend-coherence-boost-gain" (torBlendCoherenceBoostGain ranges)
    positiveRange "blend-coherence-boost-span" (torBlendCoherenceBoostSpan ranges)
    unitRange "blend-anchor-conflict-base" (torBlendAnchorConflictBase ranges)
    unitRange "blend-anchor-conflict-scale" (torBlendAnchorConflictScale ranges)
    unitRange "blend-anchor-aligned-scale" (torBlendAnchorAlignedScale ranges)
    unitRange "blend-tension-conflict-shrink" (torBlendTensionConflictShrink ranges)
    unitRange "blend-tension-neutral-shrink" (torBlendTensionNeutralShrink ranges)
    unitRange "blend-entropy-conflict-floor" (torBlendEntropyConflictFloor ranges)
    unitRange "blend-entropy-conflict-scale" (torBlendEntropyConflictScale ranges)
    unitRange "blend-entropy-aligned-base" (torBlendEntropyAlignedBase ranges)
    unitRange "blend-entropy-aligned-entropy-scale" (torBlendEntropyAlignedEntropyScale ranges)
    positiveRange "blend-phase-cancel-return-clamp" (torBlendPhaseCancelReturnClamp ranges)
    nonNegativeRange "blend-phase-cancel-conflict-floor" (torBlendPhaseCancelConflictFloor ranges)
    nonNegativeRange "blend-phase-cancel-conflict-scale" (torBlendPhaseCancelConflictScale ranges)
    nonNegativeRange "blend-phase-cancel-alignment-scale" (torBlendPhaseCancelAlignmentScale ranges)
    positiveRange "signal-entry-edge-headroom-mult" (torSignalEntryEdgeHeadroomMult ranges)
    nonNegativeRange "signal-entry-edge-spike-mult" (torSignalEntryEdgeSpikeMult ranges)
    positiveRange "signal-entry-edge-spike-cap" (torSignalEntryEdgeSpikeCap ranges)
    nonNegativeIntRange "signal-entry-edge-spike-consecutive-limit" (torSignalEntryEdgeSpikeConsecutiveLimit ranges)
    nonNegativeRange "signal-trend-slack-mult" (torSignalTrendSlackMult ranges)
    nonNegativeRange "signal-trend-slack-cap" (torSignalTrendSlackCap ranges)
    positiveIntRange "signal-directionality-lookback-bars" (torSignalDirectionalityLookbackBars ranges)
    unitRange "signal-directionality-chop-efficiency-max" (torSignalDirectionalityChopEfficiencyMax ranges)
    unitRange "signal-directionality-mr-efficiency-max" (torSignalDirectionalityMrEfficiencyMax ranges)
    nonNegativeRange "signal-directionality-weak-z-min" (torSignalDirectionalityWeakZMin ranges)
    nonNegativeRange "signal-predictor-tracking-floor" (torSignalPredictorTrackingFloor ranges)
  where
    finiteRange label (lo, hi) =
        when (isNaN lo || isInfinite lo || isNaN hi || isInfinite hi) $
            Left ("--" ++ label ++ "-min/max must be finite.")
    nonNegativeRange label range@(lo, hi) = do
        finiteRange label range
        when (lo < 0 || hi < 0) $
            Left ("--" ++ label ++ "-min/max must be >= 0.")
    positiveRange label range@(lo, hi) = do
        finiteRange label range
        when (lo <= 0 || hi <= 0) $
            Left ("--" ++ label ++ "-min/max must be > 0.")
    unitRange label range@(lo, hi) = do
        finiteRange label range
        when (lo < 0 || lo > 1 || hi < 0 || hi > 1) $
            Left ("--" ++ label ++ "-min/max must be between 0 and 1.")
    unit100Range label range@(lo, hi) = do
        finiteRange label range
        when (lo < 0 || lo > 100 || hi < 0 || hi > 100) $
            Left ("--" ++ label ++ "-min/max must be between 0 and 100.")
    nonNegativeIntRange label (lo, hi) =
        when (lo < 0 || hi < 0) $
            Left ("--" ++ label ++ "-min/max must be >= 0.")
    positiveIntRange label (lo, hi) =
        when (lo < 1 || hi < 1) $
            Left ("--" ++ label ++ "-min/max must be >= 1.")

objectiveChoices :: [String]
objectiveChoices =
    map tuneObjectiveCode [TuneAnnualizedEquity, TuneRoi, TuneFinalEquity, TuneSharpe, TuneCalmar, TuneEquityDd, TuneEquityDdTurnover]

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

normalizeCliObjective :: String -> String -> Either String String
normalizeCliObjective label raw =
    case normalizeObjectiveCode raw of
        Right objectiveCode -> Right objectiveCode
        Left _ ->
            Left
                ( "Invalid "
                    ++ label
                    ++ ": "
                    ++ show raw
                    ++ " (expected one of: "
                    ++ intercalate ", " objectiveChoices
                    ++ ")"
                )

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

stripTrailingSpace :: String -> String
stripTrailingSpace = reverse . dropWhile isSpace . reverse
