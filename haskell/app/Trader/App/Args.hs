{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE RecordWildCards #-}
module Trader.App.Args
  ( Args(..)
  , resolveBarsForCsv
  , resolveBarsForBinance
  , resolveBarsForPlatform
  , positioningCode
  , parsePositioning
  , intrabarFillCode
  , parseIntrabarFill
  , opts
  , argBinanceMarket
  , argLookback
  , validateArgs
  ) where

import Control.Applicative ((<|>))
import Data.Char (isAlphaNum, toLower, toUpper)
import Data.Maybe (fromMaybe, isJust)
import Text.Read (readMaybe)

import Options.Applicative

import Trader.Binance (BinanceMarket(..))
import Trader.Duration (TimeWindow, lookbackBarsFrom, parseIntervalSeconds, parseTimeWindow)
import Trader.Method (Method(..), methodCode, parseMethod)
import Trader.Normalization (NormType(..), parseNormType)
import Trader.Optimization (TuneObjective(..), tuneObjectiveCode, parseTuneObjective)
import Trader.Platform
  ( Platform(..)
  , isPlatformInterval
  , parsePlatform
  , platformCode
  , platformDefaultBars
  , platformIntervalsCsv
  )
import Trader.Predictors.Types
  ( PredictorSet
  , allPredictors
  , predictorSetFromString
  , predictorSetToCsv
  )
import Trader.Text (normalizeKey, trim)
import Trader.Trading (IntrabarFill(..), Positioning(..))

data Args = Args
  { argData :: Maybe FilePath
  , argPriceCol :: String
  , argHighCol :: Maybe String
  , argLowCol :: Maybe String
  , argBinanceSymbol :: Maybe String
  , argPlatform :: Platform
  , argBinanceFutures :: Bool
  , argBinanceMargin :: Bool
  , argInterval :: String
  , argBars :: Maybe Int
  , argLookbackWindow :: String
  , argLookbackBars :: Maybe Int
  , argBinanceTestnet :: Bool
  , argBinanceApiKey :: Maybe String
  , argBinanceApiSecret :: Maybe String
  , argCoinbaseApiKey :: Maybe String
  , argCoinbaseApiSecret :: Maybe String
  , argCoinbaseApiPassphrase :: Maybe String
  , argBinanceTrade :: Bool
  , argBinanceLive :: Bool
  , argOrderQuote :: Maybe Double
  , argOrderQuantity :: Maybe Double
  , argOrderQuoteFraction :: Maybe Double
  , argMaxOrderQuote :: Maybe Double
  , argIdempotencyKey :: Maybe String
  , argNormalization :: NormType
  , argHiddenSize :: Int
  , argEpochs :: Int
  , argLr :: Double
  , argValRatio :: Double
  , argBacktestRatio :: Double
  , argTuneRatio :: Double
  , argTuneObjective :: TuneObjective
  , argTunePenaltyMaxDrawdown :: Double
  , argTunePenaltyTurnover :: Double
  , argMinRoundTrips :: Int
  , argWalkForwardFolds :: Int
  , argWalkForwardEmbargoBars :: Int
  , argPatience :: Int
  , argGradClip :: Maybe Double
  , argSeed :: Int
  , argKalmanDt :: Double
  , argKalmanProcessVar :: Double
  , argKalmanMeasurementVar :: Double
  , argPredictors :: PredictorSet
  , argKalmanMarketTopN :: Int
  , argOpenThreshold :: Double
  , argCloseThreshold :: Double
  , argMethod :: Method
  , argPositioning :: Positioning
  , argOptimizeOperations :: Bool
  , argSweepThreshold :: Bool
  , argTradeOnly :: Bool
  , argFee :: Double
  , argSlippage :: Double
  , argSpread :: Double
  , argIntrabarFill :: IntrabarFill
  , argStopLoss :: Maybe Double
  , argTakeProfit :: Maybe Double
  , argTrailingStop :: Maybe Double
  , argStopLossVolMult :: Double
  , argTakeProfitVolMult :: Double
  , argTrailingStopVolMult :: Double
  , argTakeProfitPartial :: Double
  , argMinHoldBars :: Int
  , argCooldownBars :: Int
  , argMaxHoldBars :: Maybe Int
  , argMaxDrawdown :: Maybe Double
  , argMaxDailyLoss :: Maybe Double
  , argMaxWeeklyLoss :: Maybe Double
  , argRiskPerTrade :: Maybe Double
  , argMaxTradesPerDay :: Maybe Int
  , argExpectancyLookback :: Int
  , argMinExpectancy :: Maybe Double
  , argPerfLookback :: Int
  , argPerfMinWinRate :: Maybe Double
  , argPerfMinProfitFactor :: Maybe Double
  , argLossStreakMax :: Int
  , argLossStreakCooldownBars :: Int
  , argNoTradeWindows :: [TimeWindow]
  , argMaxOpenPositions :: Maybe Int
  , argMaxOpenPerBase :: Maybe Int
  , argMinEdge :: Double
  , argMinSignalToNoise :: Double
  , argSnrSizeWeight :: Double
  , argAdaptiveFilters :: Bool
  , argAdaptiveEdgeBufferMax :: Double
  , argAdaptiveMinSignalToNoiseMax :: Double
  , argAdaptiveKalmanZMinMax :: Double
  , argAdaptiveTrendLookbackMax :: Int
  , argThresholdFactorEnabled :: Bool
  , argThresholdFactorAlpha :: Double
  , argThresholdFactorMin :: Double
  , argThresholdFactorMax :: Double
  , argThresholdFactorFloor :: Double
  , argThresholdFactorEdgeKalWeight :: Double
  , argThresholdFactorEdgeLstmWeight :: Double
  , argThresholdFactorKalmanZWeight :: Double
  , argThresholdFactorHighVolWeight :: Double
  , argThresholdFactorConformalWeight :: Double
  , argThresholdFactorQuantileWeight :: Double
  , argThresholdFactorLstmConfWeight :: Double
  , argThresholdFactorLstmHealthWeight :: Double
  , argCostAwareEdge :: Bool
  , argEdgeBuffer :: Double
  , argTrendLookback :: Int
  , argMaxPositionSize :: Double
  , argVolTarget :: Maybe Double
  , argVolLookback :: Int
  , argVolEwmaAlpha :: Maybe Double
  , argVolFloor :: Double
  , argVolScaleMax :: Double
  , argMaxVolatility :: Maybe Double
  , argRebalanceBars :: Int
  , argRebalanceThreshold :: Double
  , argRebalanceCostMult :: Double
  , argRebalanceGlobal :: Bool
  , argRebalanceResetOnSignal :: Bool
  , argFundingRate :: Double
  , argFundingBySide :: Bool
  , argFundingOnOpen :: Bool
  , argBlendWeight :: Double
  , argRouterLookback :: Int
  , argRouterMinScore :: Double
  , argRouterScorePnlWeight :: Double
  , argTriLayer :: Bool
  , argTriLayerFastMult :: Double
  , argTriLayerSlowMult :: Double
  , argTriLayerCloudPadding :: Double
  , argTriLayerCloudSlope :: Double
  , argTriLayerCloudWidth :: Double
  , argTriLayerTouchLookback :: Int
  , argTriLayerRequirePriceAction :: Bool
  , argTriLayerPriceActionBody :: Double
  , argTriLayerExitOnSlow :: Bool
  , argKalmanBandLookback :: Int
  , argKalmanBandStdMult :: Double
  , argLstmExitFlipBars :: Int
  , argLstmExitFlipGraceBars :: Int
  , argLstmExitFlipStrong :: Bool
  , argMaxOrderErrors :: Maybe Int
  , argPeriodsPerYear :: Maybe Double
  , argJson :: Bool
  , argServe :: Bool
  , argOpsBackfillCommits :: Bool
  , argPort :: Int
  -- Confidence gating/sizing (Kalman sensors + HMM/intervals)
  , argKalmanZMin :: Double
  , argKalmanZMax :: Double
  , argMaxHighVolProb :: Maybe Double
  , argMaxConformalWidth :: Maybe Double
  , argMaxQuantileWidth :: Maybe Double
  , argConfirmConformal :: Bool
  , argConfirmQuantiles :: Bool
  , argConfidenceSizing :: Bool
  , argLstmConfidenceSoft :: Double
  , argLstmConfidenceHard :: Double
  , argMinPositionSize :: Double
  , argTuneStressVolMult :: Double
  , argTuneStressShock :: Double
  , argTuneStressWeight :: Double
  } deriving (Eq, Show)

defaultBinanceBars :: Int
defaultBinanceBars = 500

resolveBarsForCsv :: Args -> Int
resolveBarsForCsv args =
  case argBars args of
    Nothing -> 0
    Just n -> n

resolveBarsForBinance :: Args -> Int
resolveBarsForBinance args =
  case argBars args of
    Nothing -> defaultBinanceBars
    Just 0 -> defaultBinanceBars
    Just n -> n

resolveBarsForPlatform :: Args -> Int
resolveBarsForPlatform args =
  case argPlatform args of
    PlatformBinance -> resolveBarsForBinance args
    _ ->
      case argBars args of
        Nothing -> platformDefaultBars (argPlatform args)
        Just 0 -> platformDefaultBars (argPlatform args)
        Just n -> n

parseBarsArg :: String -> Either String (Maybe Int)
parseBarsArg raw =
  let s = map toLower (trim raw)
   in if s == "auto"
        then Right Nothing
        else
          case (readMaybe s :: Maybe Int) of
            Just n -> Right (Just n)
            Nothing -> Left "Expected an integer (e.g. 500) or 'auto'."

positioningCode :: Positioning -> String
positioningCode p =
  case p of
    LongFlat -> "long-flat"
    LongShort -> "long-short"

parsePositioning :: String -> Either String Positioning
parsePositioning raw =
  case normalizeKey raw of
    "longflat" -> Right LongFlat
    "lf" -> Right LongFlat
    "flat" -> Right LongFlat
    "longonly" -> Right LongFlat
    "long" -> Right LongFlat
    "longshort" -> Right LongShort
    "ls" -> Right LongShort
    "short" -> Right LongShort
    _ -> Left "Invalid positioning (expected long-flat|long-only|long-short)"

intrabarFillCode :: IntrabarFill -> String
intrabarFillCode f =
  case f of
    StopFirst -> "stop-first"
    TakeProfitFirst -> "take-profit-first"

parseIntrabarFill :: String -> Either String IntrabarFill
parseIntrabarFill raw =
  case normalizeKey raw of
    "stopfirst" -> Right StopFirst
    "slfirst" -> Right StopFirst
    "stop" -> Right StopFirst
    "worst" -> Right StopFirst
    "takeprofitfirst" -> Right TakeProfitFirst
    "tpfirst" -> Right TakeProfitFirst
    "takeprofit" -> Right TakeProfitFirst
    "tp" -> Right TakeProfitFirst
    "best" -> Right TakeProfitFirst
    _ -> Left "Invalid intrabar fill (expected stop-first|take-profit-first)"

opts :: Parser Args
opts = do
  let defaultOnSwitch onFlag offFlag helpOn helpOff =
        flag' True (long onFlag <> help helpOn)
          <|> flag' False (long offFlag <> help helpOff)
          <|> pure True
      defaultOffSwitch onFlag offFlag helpOn helpOff =
        flag' True (long onFlag <> help helpOn)
          <|> flag' False (long offFlag <> help helpOff)
          <|> pure False
  argData <- optional (strOption (long "data" <> metavar "PATH" <> help "CSV file containing prices"))
  argPriceCol <- strOption (long "price-column" <> value "close" <> help "CSV column name for price (case-insensitive; prints suggestions on error)")
  argHighCol <- optional (strOption (long "high-column" <> help "CSV column name for high (requires --low-column; enables intrabar stops/TP/trailing)"))
  argLowCol <- optional (strOption (long "low-column" <> help "CSV column name for low (requires --high-column; enables intrabar stops/TP/trailing)"))
  argBinanceSymbol <- optional (strOption (long "binance-symbol" <> long "symbol" <> metavar "SYMBOL" <> help "Exchange symbol to fetch klines (platform via --platform)"))
  argPlatform <-
    option
      (eitherReader parsePlatform)
      ( long "platform"
          <> value PlatformBinance
          <> showDefaultWith platformCode
          <> help "Exchange platform for --binance-symbol (binance|coinbase|kraken|poloniex)"
      )
  argBinanceFutures <- switch (long "futures" <> help "Use Binance USDT-M futures endpoints for data/orders (Binance only)")
  argBinanceMargin <- switch (long "margin" <> help "Use Binance margin account endpoints for orders/balance (Binance only)")
  argInterval <- strOption (long "interval" <> long "binance-interval" <> value "1h" <> help "Bar interval / Binance kline interval (e.g., 1m, 5m, 1h, 1d)")
  argBars <-
    option
      (eitherReader parseBarsArg)
      ( long "bars"
          <> long "binance-limit"
          <> metavar "N|auto"
          <> value Nothing
          <> showDefaultWith (\mb -> maybe "auto" show mb)
          <> help "Number of bars/klines to use (auto/0=all CSV, exchange default=500; Binance supports 2..1000)"
      )
  argLookbackWindow <- strOption (long "lookback-window" <> value "7d" <> help "Lookback window duration (e.g., 90m, 24h, 7d)")
  argLookbackBars <- optional (option auto (long "lookback-bars" <> long "lookback" <> help "Override lookback bars (disables --lookback-window conversion)"))
  argBinanceTestnet <- switch (long "binance-testnet" <> help "Use Binance testnet base URL (public + signed endpoints; Binance only)")
  argBinanceApiKey <- optional (strOption (long "binance-api-key" <> help "Binance API key (or env BINANCE_API_KEY; Binance only)"))
  argBinanceApiSecret <- optional (strOption (long "binance-api-secret" <> help "Binance API secret (or env BINANCE_API_SECRET; Binance only)"))
  argCoinbaseApiKey <- optional (strOption (long "coinbase-api-key" <> help "Coinbase API key (or env COINBASE_API_KEY; Coinbase only)"))
  argCoinbaseApiSecret <- optional (strOption (long "coinbase-api-secret" <> help "Coinbase API secret (or env COINBASE_API_SECRET; Coinbase only)"))
  argCoinbaseApiPassphrase <- optional (strOption (long "coinbase-api-passphrase" <> help "Coinbase API passphrase (or env COINBASE_API_PASSPHRASE; Coinbase only)"))
  argBinanceTrade <- switch (long "binance-trade" <> help "If set, place a market order for the latest signal (Binance/Coinbase spot)")
  argBinanceLive <-
    defaultOnSwitch
      "binance-live"
      "no-binance-live"
      "Send LIVE orders (Binance/Coinbase; Coinbase has no test endpoint)."
      "Send TEST orders (Binance uses /order/test; Coinbase has no test endpoint)."
  argOrderQuote <- optional (option auto (long "order-quote" <> help "Quote amount to spend on BUY (quoteOrderQty)"))
  argOrderQuantity <- optional (option auto (long "order-quantity" <> help "Base quantity to trade (quantity)"))
  argOrderQuoteFraction <- optional (option auto (long "order-quote-fraction" <> help "Size BUY orders as a fraction of quote balance (0 < F <= 1) when --order-quote/--order-quantity not set"))
  argMaxOrderQuote <- optional (option auto (long "max-order-quote" <> help "Cap the computed quote amount when using --order-quote-fraction"))
  argIdempotencyKey <- optional (strOption (long "idempotency-key" <> metavar "ID" <> help "Optional Binance newClientOrderId for idempotent orders"))
  argNormalization <- option (maybeReader parseNormType) (long "normalization" <> value NormStandard <> help "none|minmax|standard|log")
  argHiddenSize <- option auto (long "hidden-size" <> value 16 <> help "LSTM hidden size")
  argEpochs <- option auto (long "epochs" <> value 30 <> help "LSTM training epochs (Adam)")
  argLr <- option auto (long "lr" <> value 1e-3 <> help "LSTM learning rate")
  argValRatio <- option auto (long "val-ratio" <> value 0.3 <> help "Validation split ratio (within training set)")
  argBacktestRatio <- option auto (long "backtest-ratio" <> value 0.2 <> help "Backtest holdout ratio (last portion of series)")
  argTuneRatio <- option auto (long "tune-ratio" <> value 0.25 <> help "When optimizing operations/threshold: tune on the last portion of the training split (avoids lookahead on the backtest split)")
  argTuneObjective <-
    option
      (eitherReader parseTuneObjective)
      ( long "tune-objective"
          <> value TuneEquityDdTurnover
          <> showDefaultWith tuneObjectiveCode
          <> help "Objective for --optimize-operations/--sweep-threshold: annualized-equity|final-equity|sharpe|calmar|equity-dd|equity-dd-turnover"
      )
  argTunePenaltyMaxDrawdown <- option auto (long "tune-penalty-max-drawdown" <> value 1.5 <> help "Penalty weight for max drawdown (used by equity-dd objectives)")
  argTunePenaltyTurnover <- option auto (long "tune-penalty-turnover" <> value 0.2 <> help "Penalty weight for turnover (used by equity-dd-turnover)")
  argMinRoundTrips <-
    option
      auto
      ( long "min-round-trips"
          <> value 0
          <> help "When optimizing/sweeping, require at least N round trips in the tune split (0 disables; helps avoid 'no-trade' winners)"
      )
  argWalkForwardFolds <- option auto (long "walk-forward-folds" <> value 7 <> help "Compute fold stats on tune/backtest windows (1 disables)")
  argWalkForwardEmbargoBars <-
    option
      auto
      ( long "walk-forward-embargo-bars"
          <> value 0
          <> help "Bars to drop from each walk-forward fold edge to reduce leakage (0 disables)."
      )
  argPatience <- option auto (long "patience" <> value 10 <> help "Early stopping patience (0 disables)")
  argGradClip <- optional (option auto (long "grad-clip" <> help "Gradient clipping max L2 norm"))
  argSeed <- option auto (long "seed" <> value 42 <> help "Random seed for LSTM init")
  argKalmanDt <- option auto (long "kalman-dt" <> value 1.0 <> help "Kalman dt")
  argKalmanProcessVar <- option auto (long "kalman-process-var" <> value 1e-5 <> help "Kalman process noise variance (white-noise jerk)")
  argKalmanMeasurementVar <- option auto (long "kalman-measurement-var" <> value 1e-3 <> help "Kalman measurement noise variance")
  argPredictors <-
    option
      (eitherReader predictorSetFromString)
      ( long "predictors"
          <> value allPredictors
          <> showDefaultWith predictorSetToCsv
          <> help "Comma-separated predictors to train/use (gbdt,tcn,transformer,hmm,quantile,conformal, all, none)"
      )
  argKalmanMarketTopN <-
    option
      auto
      ( long "kalman-market-top-n"
          <> value 50
          <> help "Use the top-N symbols by 24h quote volume as an extra Kalman measurement (0 disables; Binance only)."
      )
  argOpenThreshold <- option auto (long "open-threshold" <> long "threshold" <> value 0.002 <> help "Entry/open direction threshold (fractional deadband)")
  mCloseThreshold <- optional (option auto (long "close-threshold" <> help "Close-direction threshold (fractional deadband; defaults to open-threshold when omitted)"))
  argThresholdFactorEnabled <-
    defaultOffSwitch
      "threshold-factor"
      "no-threshold-factor"
      "Enable dynamic threshold factor learning."
      "Disable dynamic threshold factor learning."
  argThresholdFactorAlpha <- option auto (long "threshold-factor-alpha" <> value 0.2 <> help "EMA alpha for threshold factor updates (0 disables updates)")
  argThresholdFactorMin <- option auto (long "threshold-factor-min" <> value 0.5 <> help "Minimum multiplier for the threshold factor")
  argThresholdFactorMax <- option auto (long "threshold-factor-max" <> value 2.0 <> help "Maximum multiplier for the threshold factor")
  argThresholdFactorFloor <- option auto (long "threshold-factor-floor" <> value 0.0 <> help "Floor for adjusted thresholds after applying the factor")
  argThresholdFactorEdgeKalWeight <- option auto (long "threshold-factor-edge-kal-weight" <> value 0 <> help "Weight for Kalman edge feature in threshold factor")
  argThresholdFactorEdgeLstmWeight <- option auto (long "threshold-factor-edge-lstm-weight" <> value 0 <> help "Weight for LSTM edge feature in threshold factor")
  argThresholdFactorKalmanZWeight <- option auto (long "threshold-factor-kalman-z-weight" <> value 0 <> help "Weight for Kalman z-score feature in threshold factor")
  argThresholdFactorHighVolWeight <- option auto (long "threshold-factor-high-vol-weight" <> value 0 <> help "Weight for HMM high-vol probability feature in threshold factor")
  argThresholdFactorConformalWeight <- option auto (long "threshold-factor-conformal-weight" <> value 0 <> help "Weight for conformal width feature in threshold factor")
  argThresholdFactorQuantileWeight <- option auto (long "threshold-factor-quantile-weight" <> value 0 <> help "Weight for quantile width feature in threshold factor")
  argThresholdFactorLstmConfWeight <- option auto (long "threshold-factor-lstm-conf-weight" <> value 0 <> help "Weight for LSTM confidence feature in threshold factor")
  argThresholdFactorLstmHealthWeight <- option auto (long "threshold-factor-lstm-health-weight" <> value 0 <> help "Weight for LSTM training health feature in threshold factor")
  argMethod <-
    option
      (eitherReader parseMethod)
      ( long "method"
          <> value MethodBoth
          <> showDefaultWith methodCode
          <> help "Method: 11|both=Kalman+LSTM (direction-agreement gated), blend=weighted avg, router=adaptive model selection, 10|kalman=Kalman only, 01|lstm=LSTM only"
      )
  argPositioning <-
    option
      (eitherReader parsePositioning)
      ( long "positioning"
          <> value LongFlat
          <> showDefaultWith positioningCode
          <> help "Positioning: long-flat (default), long-only/long (alias), or long-short (futures-only when trading)"
      )
  argOptimizeOperations <- switch (long "optimize-operations" <> help "Optimize method (11/10/01), open-threshold, and close-threshold on a tune split (avoids lookahead on the backtest split)")
  argSweepThreshold <- switch (long "sweep-threshold" <> help "Sweep open/close thresholds on a tune split and print the best final equity (avoids lookahead on the backtest split)")
  argTradeOnly <- switch (long "trade-only" <> help "Skip backtest/metrics; only compute the latest signal (and optionally place an order)")
  argFee <- option auto (long "fee" <> value 0.0008 <> help "Fee applied when switching position")
  argSlippage <- option auto (long "slippage" <> value 0.0002 <> help "Slippage per side (fractional, e.g. 0.0002)")
  argSpread <- option auto (long "spread" <> value 0.0002 <> help "Bid-ask spread (fractional total; half applied per side)")
  argIntrabarFill <-
    option
      (eitherReader parseIntrabarFill)
      ( long "intrabar-fill"
          <> value StopFirst
          <> showDefaultWith intrabarFillCode
          <> help "If take-profit and stop are both hit within a bar: stop-first (conservative) or take-profit-first"
      )
  argStopLoss <- optional (option auto (long "stop-loss" <> help "Stop loss fraction for a bracket exit (e.g., 0.02 for 2%)"))
  argTakeProfit <- optional (option auto (long "take-profit" <> help "Take profit fraction for a bracket exit (e.g., 0.03 for 3%)"))
  argTrailingStop <- optional (option auto (long "trailing-stop" <> help "Trailing stop fraction for a bracket exit (e.g., 0.01 for 1%)"))
  argStopLossVolMult <- option auto (long "stop-loss-vol-mult" <> value 0 <> help "Stop loss in per-bar sigma multiples (0 disables; overrides --stop-loss when vol estimate available)")
  argTakeProfitVolMult <- option auto (long "take-profit-vol-mult" <> value 0 <> help "Take profit in per-bar sigma multiples (0 disables; overrides --take-profit when vol estimate available)")
  argTrailingStopVolMult <- option auto (long "trailing-stop-vol-mult" <> value 0 <> help "Trailing stop in per-bar sigma multiples (0 disables; overrides --trailing-stop when vol estimate available)")
  argTakeProfitPartial <-
    option
      auto
      ( long "take-profit-partial"
          <> value 0
          <> showDefault
          <> help "Scale out this fraction at take-profit before keeping the remainder open (0 disables; 0<frac<1)"
      )
  argMinHoldBars <- option auto (long "min-hold-bars" <> value 4 <> help "Minimum holding periods (bars) before allowing a signal-based exit (0 disables)")
  argCooldownBars <- option auto (long "cooldown-bars" <> value 2 <> help "When flat after an exit, wait this many bars before allowing a new entry (0 disables)")
  argMaxHoldBars <-
    optional
          ( option
              auto
              ( long "max-hold-bars"
              <> value 36
              <> showDefault
              <> help "Force exit after holding for this many bars (0 disables; enforces 1-bar cooldown)"
          )
      )
  argMaxDrawdown <- optional (option auto (long "max-drawdown" <> help "Halt the live bot if peak-to-trough drawdown exceeds this fraction (0..1)"))
  argMaxDailyLoss <- optional (option auto (long "max-daily-loss" <> help "Halt the live bot if daily loss exceeds this fraction (0..1), based on UTC day (requires bar timestamps)"))
  argMaxWeeklyLoss <- optional (option auto (long "max-weekly-loss" <> help "Halt the live bot if weekly loss exceeds this fraction (0..1), based on UTC weeks (requires bar timestamps)"))
  argRiskPerTrade <- optional (option auto (long "risk-per-trade" <> help "Fraction of equity to risk per trade (requires --stop-loss or --stop-loss-vol-mult)"))
  argMaxTradesPerDay <- optional (option auto (long "max-trades-per-day" <> help "Block new entries after N entries per UTC day (0 disables; requires bar timestamps)"))
  argExpectancyLookback <- option auto (long "expectancy-lookback" <> value 20 <> help "Lookback trades for expectancy gating (0 disables; requires --min-expectancy)")
  argMinExpectancy <- optional (option auto (long "min-expectancy" <> help "Halt trading when avg return of the last N trades falls below this"))
  argPerfLookback <- option auto (long "perf-lookback" <> value 0 <> showDefault <> help "Lookback trades for performance gates/adaptive filters (0 disables)")
  argPerfMinWinRate <- optional (option auto (long "perf-min-win-rate" <> help "Minimum rolling win rate for entry gating/adaptive filters (0 disables)"))
  argPerfMinProfitFactor <- optional (option auto (long "perf-min-profit-factor" <> help "Minimum rolling profit factor for entry gating/adaptive filters (0 disables)"))
  argLossStreakMax <- option auto (long "loss-streak-max" <> value 0 <> showDefault <> help "Trigger a cooldown after this many consecutive losing trades (0 disables)")
  argLossStreakCooldownBars <- option auto (long "loss-streak-cooldown-bars" <> value 0 <> showDefault <> help "Cooldown bars applied after hitting the loss streak threshold (0 disables)")
  argNoTradeWindows <-
    many
      ( option
          (eitherReader parseTimeWindow)
          ( long "no-trade-window"
              <> metavar "HH:MM-HH:MM"
              <> help "UTC time window to block new entries (repeatable; wraps across midnight; requires bar timestamps)."
          )
      )
  argMaxOpenPositions <- optional (option auto (long "max-open-positions" <> help "Max open positions across all running bots (0 disables)"))
  argMaxOpenPerBase <- optional (option auto (long "max-open-per-base" <> help "Max open positions per base asset across running bots (0 disables)"))
  argMinEdge <- option auto (long "min-edge" <> value 0.0004 <> help "Minimum predicted return magnitude required to enter (0 disables)")
  argMinSignalToNoise <- option auto (long "min-signal-to-noise" <> value 0.8 <> help "Minimum edge/vol (per-bar sigma) required to enter (0 disables)")
  argSnrSizeWeight <-
    option
      auto
      ( long "snr-size-weight"
          <> value 1.0
          <> help "Weight to scale position size by signal-to-noise (0=hard gate only, 1=fully scaled)"
      )
  argAdaptiveFilters <-
    defaultOffSwitch
      "adaptive-filters"
      "no-adaptive-filters"
      "Enable adaptive filter tightening based on rolling performance."
      "Disable adaptive filter tightening."
  argAdaptiveEdgeBufferMax <-
    option
      auto
      ( long "adaptive-edge-buffer-max"
          <> value 0
          <> showDefault
          <> help "Max additive edge-buffer when adaptive filters are fully tightened"
      )
  argAdaptiveMinSignalToNoiseMax <-
    option
      auto
      ( long "adaptive-min-signal-to-noise-max"
          <> value 0
          <> showDefault
          <> help "Max additive min-signal-to-noise when adaptive filters are fully tightened"
      )
  argAdaptiveKalmanZMinMax <-
    option
      auto
      ( long "adaptive-kalman-z-min-max"
          <> value 0
          <> showDefault
          <> help "Max additive Kalman z-min when adaptive filters are fully tightened"
      )
  argAdaptiveTrendLookbackMax <-
    option
      auto
      ( long "adaptive-trend-lookback-max"
          <> value 0
          <> showDefault
          <> help "Max additive trend lookback when adaptive filters are fully tightened"
      )
  argCostAwareEdge <-
    defaultOnSwitch
      "cost-aware-edge"
      "no-cost-aware-edge"
      "Enable cost-aware min-edge gating (default on)."
      "Disable cost-aware min-edge gating."
  argEdgeBuffer <- option auto (long "edge-buffer" <> value 0.0002 <> help "Extra buffer added to cost-aware min-edge")
  argTrendLookback <- option auto (long "trend-lookback" <> value 30 <> help "SMA lookback for trend filter (0 disables)")
  argMaxPositionSize <- option auto (long "max-position-size" <> value 0.8 <> help "Cap position size/leverage (1 = full size)")
  argVolTarget <-
    optional
          ( option
              auto
              ( long "vol-target"
              <> value 0.7
              <> showDefault
              <> help "Target annualized volatility for position sizing (0 disables)"
          )
      )
  argVolLookback <- option auto (long "vol-lookback" <> value 30 <> help "Lookback window for realized vol sizing (bars)")
  argVolEwmaAlpha <- optional (option auto (long "vol-ewma-alpha" <> help "EWMA alpha for vol sizing (overrides vol-lookback)"))
  argVolFloor <- option auto (long "vol-floor" <> value 0.15 <> help "Annualized vol floor for sizing")
  argVolScaleMax <- option auto (long "vol-scale-max" <> value 1 <> help "Max volatility scaling (caps leverage)")
  argMaxVolatility <-
    optional
          ( option
              auto
              ( long "max-volatility"
              <> value 1.5
              <> showDefault
              <> help "Block entries when annualized vol exceeds this (0 disables)"
          )
      )
  argRebalanceBars <- option auto (long "rebalance-bars" <> value 24 <> showDefault <> help "Rebalance position size every N bars when size targets change (0 disables; default anchors to entry age)")
  argRebalanceThreshold <- option auto (long "rebalance-threshold" <> value 0.05 <> showDefault <> help "Minimum abs size delta required to rebalance (0 disables)")
  argRebalanceCostMult <- option auto (long "rebalance-cost-mult" <> value 0.0 <> showDefault <> help "Extra rebalance threshold as a multiple of per-side cost (0 disables)")
  argRebalanceGlobal <- switch (long "rebalance-global" <> help "Anchor rebalance cadence to global bars instead of entry age")
  argRebalanceResetOnSignal <- switch (long "rebalance-reset-on-signal" <> help "Reset rebalance cadence when a same-side open signal updates size")
  argFundingRate <- option auto (long "funding-rate" <> long "financing-rate" <> value 0.1 <> showDefault <> help "Annualized funding/borrow rate applied per bar in backtests (fraction; negative allowed; side-agnostic unless --funding-by-side)")
  argFundingBySide <- switch (long "funding-by-side" <> help "Apply funding sign by side (long pays positive, short receives)")
  argFundingOnOpen <- switch (long "funding-on-open" <> help "Charge funding for bars opened with a position (even if exited intrabar)")
  argBlendWeight <- option auto (long "blend-weight" <> value 0.5 <> help "Kalman weight for --method blend (0..1)")
  argRouterLookback <- option auto (long "router-lookback" <> value 30 <> help "Lookback bars for --method router scoring (>= 2)")
  argRouterMinScore <- option auto (long "router-min-score" <> value 0.25 <> help "Minimum router score (blend of accuracy*coverage and return) to accept a model (0..1)")
  argRouterScorePnlWeight <-
    option
      auto
      ( long "router-score-pnl-weight"
          <> value 0.5
          <> help "Weight for return-aware router scoring (0=accuracy*coverage only, 1=return only)"
      )
  argTriLayer <- switch (long "tri-layer" <> help "Enable tri-layer entry gating (Kalman cloud + price action trigger)")
  argTriLayerFastMult <- option auto (long "tri-layer-fast-mult" <> value 0.5 <> help "Measurement variance multiplier for the fast Kalman cloud line (requires --tri-layer)")
  argTriLayerSlowMult <- option auto (long "tri-layer-slow-mult" <> value 2.0 <> help "Measurement variance multiplier for the slow Kalman cloud line (requires --tri-layer)")
  argTriLayerCloudPadding <- option auto (long "tri-layer-cloud-padding" <> value 0.0 <> showDefault <> help "Expand the Kalman cloud by this fraction of price when checking touches (0 = strict)")
  argTriLayerCloudSlope <- option auto (long "tri-layer-cloud-slope" <> value 0.0 <> showDefault <> help "Require Kalman cloud slope to exceed this fraction of price (0 = sign-only)")
  argTriLayerCloudWidth <- option auto (long "tri-layer-cloud-width" <> value 0.0 <> showDefault <> help "Block tri-layer entries when cloud width exceeds this fraction of price (0 disables)")
  argTriLayerTouchLookback <- option auto (long "tri-layer-touch-lookback" <> value 1 <> showDefault <> help "Allow cloud touches up to N bars back (1 = current bar)")
  argTriLayerRequirePriceAction <-
    defaultOnSwitch
      "tri-layer-price-action"
      "no-tri-layer-price-action"
      "Require price-action triggers when tri-layer is enabled (default on)."
      "Disable price-action triggers for tri-layer."
  argTriLayerPriceActionBody <- option auto (long "tri-layer-price-action-body" <> value 0.0 <> showDefault <> help "Override min candle body fraction for tri-layer price-action patterns (0 = default)")
  argTriLayerExitOnSlow <- switch (long "tri-layer-exit-on-slow" <> help "Exit when price closes across the slow Kalman line (requires --tri-layer)")
  argKalmanBandLookback <- option auto (long "kalman-band-lookback" <> value 0 <> showDefault <> help "Lookback bars for Kalman-band exits (0 disables; must be >= 2)")
  argKalmanBandStdMult <- option auto (long "kalman-band-std-mult" <> value 0 <> showDefault <> help "Std-dev multiple for Kalman-band exits (0 disables)")
  argLstmExitFlipBars <- option auto (long "lstm-exit-flip-bars" <> value 0 <> showDefault <> help "Exit after N consecutive LSTM bars flip against the position (0 disables)")
  argLstmExitFlipGraceBars <- option auto (long "lstm-exit-flip-grace-bars" <> value 0 <> showDefault <> help "Ignore LSTM flip exits during the first N bars of a trade")
  argLstmExitFlipStrong <- switch (long "lstm-exit-flip-strong" <> help "Require strong LSTM confidence for flip exits (uses --lstm-confidence-hard)")
  argMaxOrderErrors <- optional (option auto (long "max-order-errors" <> help "Halt the live bot after N consecutive order failures"))
  argPeriodsPerYear <- optional (option auto (long "periods-per-year" <> help "For annualized metrics (e.g., 365 for 1d, 8760 for 1h)"))
  argJson <- switch (long "json" <> help "Output JSON to stdout (CLI mode only)")
  argServe <- switch (long "serve" <> help "Run REST API server on localhost instead of running the CLI workflow")
  argOpsBackfillCommits <- switch (long "ops-backfill-commits" <> help "Backfill git_commits from repo history and link ops by commit time (requires TRADER_DB_URL)")
  argPort <- option auto (long "port" <> value 8080 <> help "REST API port (when --serve)")
  argKalmanZMin <- option auto (long "kalman-z-min" <> value 0.5 <> help "Min |Kalman mean|/std (z-score) required to treat Kalman as directional (0 disables)")
  argKalmanZMax <- option auto (long "kalman-z-max" <> value 3 <> help "Z-score mapped to position size=1 when --confidence-sizing is enabled")
  argMaxHighVolProb <- optional (option auto (long "max-high-vol-prob" <> help "If set, block trades when HMM high-vol regime prob exceeds this (0..1)"))
  argMaxConformalWidth <- optional (option auto (long "max-conformal-width" <> help "If set, block trades when conformal interval width exceeds this (return units)"))
  argMaxQuantileWidth <- optional (option auto (long "max-quantile-width" <> help "If set, block trades when quantile (q90-q10) width exceeds this (return units)"))
  argConfirmConformal <-
    defaultOnSwitch
      "confirm-conformal"
      "no-confirm-conformal"
      "Require conformal interval to agree with the chosen direction (default on)."
      "Disable conformal confirmation."
  argConfirmQuantiles <-
    defaultOnSwitch
      "confirm-quantiles"
      "no-confirm-quantiles"
      "Require quantiles to agree with the chosen direction (default on)."
      "Disable quantile confirmation."
  argConfidenceSizing <-
    defaultOnSwitch
      "confidence-sizing"
      "no-confidence-sizing"
      "Scale entries by confidence (Kalman z-score / interval widths); leaves exits unscaled (default on)."
      "Disable confidence sizing for entries."
  argLstmConfidenceSoft <- option auto (long "lstm-confidence-soft" <> value 0.6 <> showDefault <> help "Soft LSTM confidence threshold for sizing (linear ramp to --lstm-confidence-hard; requires --confidence-sizing)")
  argLstmConfidenceHard <- option auto (long "lstm-confidence-hard" <> value 0.8 <> showDefault <> help "Hard LSTM confidence threshold for sizing (0 disables; requires --confidence-sizing)")
  argMinPositionSize <- option auto (long "min-position-size" <> value 0.15 <> help "Minimum entry size after sizing/vol scaling; skip if below this (0..1)")
  argTuneStressVolMult <- option auto (long "tune-stress-vol-mult" <> value 1.0 <> help "Stress volatility multiplier for tune scoring (1 disables)")
  argTuneStressShock <- option auto (long "tune-stress-shock" <> value 0.0 <> help "Stress shock added to returns for tune scoring (0 disables)")
  argTuneStressWeight <- option auto (long "tune-stress-weight" <> value 0.0 <> help "Penalty weight for stress scenario in tune scoring (0 disables)")

  pure Args { argCloseThreshold = fromMaybe argOpenThreshold mCloseThreshold, .. }

argBinanceMarket :: Args -> BinanceMarket
argBinanceMarket args =
  case (argBinanceFutures args, argBinanceMargin args) of
    (True, True) -> error "Choose only one of --futures or --margin"
    (True, False) -> MarketFutures
    (False, True) -> MarketMargin
    (False, False) -> MarketSpot

argLookback :: Args -> Int
argLookback args =
  case argLookbackBars args of
    Just n ->
      if n < 2
        then error "--lookback-bars must be >= 2"
        else n
    Nothing ->
      case lookbackBarsFrom (argInterval args) (argLookbackWindow args) of
        Left err -> error err
        Right n ->
          if n < 2
            then
              error
                ( "Lookback window too small: "
                    ++ show (argLookbackWindow args)
                    ++ " at interval "
                    ++ show (argInterval args)
                    ++ " yields "
                    ++ show n
                    ++ " bars; need at least 2 bars."
                )
            else n

validateArgs :: Args -> Either String Args
validateArgs args0 = do
  let args =
        args0
          { argData = fmap trim (argData args0)
          , argBinanceSymbol = fmap (map toUpper . trim) (argBinanceSymbol args0)
          , argInterval = trim (argInterval args0)
          , argPriceCol = trim (argPriceCol args0)
          , argHighCol = fmap trim (argHighCol args0)
          , argLowCol = fmap trim (argLowCol args0)
          }
      present = maybe False (not . null)
  case argData args of
    Just "" -> Left "--data cannot be empty"
    _ -> pure ()
  case argBinanceSymbol args of
    Just "" -> Left "--binance-symbol cannot be empty"
    _ -> pure ()
  ensure "Provide only one of --data or --binance-symbol" (not (present (argData args) && present (argBinanceSymbol args)))
  ensure
    "Provide a data source: --data or --binance-symbol (unless using --serve or --ops-backfill-commits)"
    (argServe args || argOpsBackfillCommits args || present (argData args) || present (argBinanceSymbol args))
  ensure "--json cannot be used with --serve or --ops-backfill-commits" (not (argJson args && (argServe args || argOpsBackfillCommits args)))
  ensure "--ops-backfill-commits cannot be used with --serve" (not (argOpsBackfillCommits args && argServe args))
  ensure "Choose only one of --futures or --margin" (not (argBinanceFutures args && argBinanceMargin args))
  ensure "--min-round-trips must be >= 0" (argMinRoundTrips args >= 0)
  let isBinance = argPlatform args == PlatformBinance
      isCoinbase = argPlatform args == PlatformCoinbase
      supportsTrading = isBinance || isCoinbase
  ensure "--futures/--margin are only supported on Binance" (isBinance || not (argBinanceFutures args || argBinanceMargin args))
  ensure "--binance-testnet is only supported on Binance" (isBinance || not (argBinanceTestnet args))
  ensure "--binance-live is only supported on Binance/Coinbase" (supportsTrading || not (argBinanceLive args))
  ensure "--binance-trade is only supported on Binance/Coinbase" (supportsTrading || not (argBinanceTrade args))

  case argHighCol args of
    Nothing -> pure ()
    Just "" -> Left "--high-column cannot be empty"
    Just _ -> pure ()
  case argLowCol args of
    Nothing -> pure ()
    Just "" -> Left "--low-column cannot be empty"
    Just _ -> pure ()
  case (argHighCol args, argLowCol args) of
    (Nothing, Nothing) -> pure ()
    (Just _, Just _) ->
      ensure "--high-column/--low-column require --data" (isJust (argData args))
    _ -> Left "Provide both --high-column and --low-column (or omit both)."

  let intervalStr = argInterval args
  ensure "--interval is required" (not (null intervalStr))
  case parseIntervalSeconds intervalStr of
    Nothing -> Left ("Invalid interval: " ++ show intervalStr ++ " (expected like 5m, 1h, 1d)")
    Just sec -> ensure "--interval must be > 0" (sec > 0)

  case argBinanceSymbol args of
    Nothing -> pure ()
    Just _ ->
      ensure
        ("--interval must be supported for " ++ platformCode (argPlatform args) ++ ": " ++ platformIntervalsCsv (argPlatform args))
        (isPlatformInterval (argPlatform args) intervalStr)

  let barsRaw = argBars args
      barsCsv = resolveBarsForCsv args
      barsPlatform = resolveBarsForPlatform args
  ensure
    "--bars must be auto, 0 (all CSV), or >= 2"
    ( case barsRaw of
        Nothing -> True
        Just n -> n == 0 || n >= 2
    )
  ensure "--bars cannot be 1" (barsRaw /= Just 1)
  case argBinanceSymbol args of
    Nothing -> pure ()
    Just _ ->
      if argPlatform args == PlatformBinance
        then ensure "--bars must be between 2 and 1000 for Binance data" (barsPlatform >= 2 && barsPlatform <= 1000)
        else ensure "--bars must be >= 2 for exchange data" (barsPlatform >= 2)

  lookback <-
    case argLookbackBars args of
      Just n -> do
        ensure "--lookback-bars must be >= 2" (n >= 2)
        pure n
      Nothing ->
        case lookbackBarsFrom intervalStr (argLookbackWindow args) of
          Left e -> Left e
          Right n -> do
            ensure
              ("Lookback window too small: " ++ show (argLookbackWindow args) ++ " at interval " ++ show intervalStr ++ " yields " ++ show n ++ " bars; need at least 2 bars.")
              (n >= 2)
            pure n

  let hasDataSource = present (argData args) || present (argBinanceSymbol args)
      barsForLookback =
        case argBinanceSymbol args of
          Just _ -> barsPlatform
          Nothing -> barsCsv
  if hasDataSource && barsForLookback > 0
    then
      ensure
        ( "--bars must be >= lookback+1 (need at least "
            ++ show (lookback + 1)
            ++ " bars for lookback="
            ++ show lookback
            ++ ")"
        )
        (barsForLookback > lookback)
    else pure ()

  ensure "--hidden-size must be >= 1" (argHiddenSize args >= 1)
  ensure "--epochs must be >= 0" (argEpochs args >= 0)
  ensure "--lr must be > 0" (argLr args > 0)
  ensure "--val-ratio must be >= 0 and < 1" (argValRatio args >= 0 && argValRatio args < 1)
  ensure "--backtest-ratio must be between 0 and 1" (argBacktestRatio args > 0 && argBacktestRatio args < 1)
  ensure "--tune-ratio must be >= 0 and < 1" (argTuneRatio args >= 0 && argTuneRatio args < 1)
  ensure "--tune-penalty-max-drawdown must be >= 0" (argTunePenaltyMaxDrawdown args >= 0)
  ensure "--tune-penalty-turnover must be >= 0" (argTunePenaltyTurnover args >= 0)
  ensure "--walk-forward-folds must be >= 1" (argWalkForwardFolds args >= 1)
  ensure "--walk-forward-embargo-bars must be >= 0" (argWalkForwardEmbargoBars args >= 0)
  ensure "--patience must be >= 0" (argPatience args >= 0)
  case argGradClip args of
    Nothing -> pure ()
    Just g -> ensure "--grad-clip must be > 0" (g > 0)
  ensure "--kalman-dt must be > 0" (argKalmanDt args > 0)
  ensure "--kalman-process-var must be > 0" (argKalmanProcessVar args > 0)
  ensure "--kalman-measurement-var must be > 0" (argKalmanMeasurementVar args > 0)
  ensure "--kalman-market-top-n must be >= 0" (argKalmanMarketTopN args >= 0)
  ensure "--open-threshold/--threshold must be >= 0" (argOpenThreshold args >= 0)
  ensure "--close-threshold must be >= 0" (argCloseThreshold args >= 0)
  ensure "--router-lookback must be >= 2" (argRouterLookback args >= 2)
  ensure "--router-min-score must be between 0 and 1" (argRouterMinScore args >= 0 && argRouterMinScore args <= 1)
  ensure "--router-score-pnl-weight must be between 0 and 1" (argRouterScorePnlWeight args >= 0 && argRouterScorePnlWeight args <= 1)
  ensure "--method router cannot be used with --optimize-operations/--sweep-threshold" $
    not
      ( argMethod args == MethodRouter
          && (argOptimizeOperations args || argSweepThreshold args)
      )
  ensure "--fee must be >= 0" (argFee args >= 0)
  ensure "--slippage must be >= 0" (argSlippage args >= 0)
  ensure "--spread must be >= 0" (argSpread args >= 0)
  case argStopLoss args of
    Nothing -> pure ()
    Just v -> ensure "--stop-loss must be > 0 and < 1" (v > 0 && v < 1)
  case argTakeProfit args of
    Nothing -> pure ()
    Just v -> ensure "--take-profit must be > 0 and < 1" (v > 0 && v < 1)
  case argTrailingStop args of
    Nothing -> pure ()
    Just v -> ensure "--trailing-stop must be > 0 and < 1" (v > 0 && v < 1)
  ensure "--stop-loss-vol-mult must be >= 0" (argStopLossVolMult args >= 0)
  ensure "--take-profit-vol-mult must be >= 0" (argTakeProfitVolMult args >= 0)
  ensure "--trailing-stop-vol-mult must be >= 0" (argTrailingStopVolMult args >= 0)
  ensure "--take-profit-partial must be >= 0 and < 1" (argTakeProfitPartial args >= 0 && argTakeProfitPartial args < 1)
  ensure "--min-hold-bars must be >= 0" (argMinHoldBars args >= 0)
  ensure "--cooldown-bars must be >= 0" (argCooldownBars args >= 0)
  case argMaxHoldBars args of
    Nothing -> pure ()
    Just n -> ensure "--max-hold-bars must be >= 0" (n >= 0)
  case argMaxDrawdown args of
    Nothing -> pure ()
    Just v -> ensure "--max-drawdown must be > 0 and < 1" (v > 0 && v < 1)
  case argMaxDailyLoss args of
    Nothing -> pure ()
    Just v -> ensure "--max-daily-loss must be > 0 and < 1" (v > 0 && v < 1)
  case argMaxWeeklyLoss args of
    Nothing -> pure ()
    Just v -> ensure "--max-weekly-loss must be > 0 and < 1" (v > 0 && v < 1)
  case argRiskPerTrade args of
    Nothing -> pure ()
    Just v -> ensure "--risk-per-trade must be > 0 and < 1" (v > 0 && v < 1)
  case argRiskPerTrade args of
    Nothing -> pure ()
    Just _ ->
      let stopLossOn = maybe False (> 0) (argStopLoss args)
          stopLossVolOn = argStopLossVolMult args > 0
       in ensure "--risk-per-trade requires --stop-loss or --stop-loss-vol-mult" (stopLossOn || stopLossVolOn)
  case argMaxTradesPerDay args of
    Nothing -> pure ()
    Just n -> ensure "--max-trades-per-day must be >= 0" (n >= 0)
  ensure "--expectancy-lookback must be >= 0" (argExpectancyLookback args >= 0)
  case argMinExpectancy args of
    Nothing -> pure ()
    Just v ->
      ensure "--min-expectancy must be finite" (not (isNaN v || isInfinite v))
        >> ensure "--expectancy-lookback must be >= 1 when --min-expectancy is set" (argExpectancyLookback args >= 1)
  ensure "--perf-lookback must be >= 0" (argPerfLookback args >= 0)
  case argPerfMinWinRate args of
    Nothing -> pure ()
    Just v -> ensure "--perf-min-win-rate must be between 0 and 1" (v >= 0 && v <= 1)
  case argPerfMinProfitFactor args of
    Nothing -> pure ()
    Just v -> ensure "--perf-min-profit-factor must be >= 0" (v >= 0)
  ensure "--loss-streak-max must be >= 0" (argLossStreakMax args >= 0)
  ensure "--loss-streak-cooldown-bars must be >= 0" (argLossStreakCooldownBars args >= 0)
  case argMaxOpenPositions args of
    Nothing -> pure ()
    Just n -> ensure "--max-open-positions must be >= 0" (n >= 0)
  case argMaxOpenPerBase args of
    Nothing -> pure ()
    Just n -> ensure "--max-open-per-base must be >= 0" (n >= 0)
  ensure "--min-edge must be >= 0" (argMinEdge args >= 0)
  ensure "--min-signal-to-noise must be >= 0" (argMinSignalToNoise args >= 0)
  ensure "--snr-size-weight must be between 0 and 1" (argSnrSizeWeight args >= 0 && argSnrSizeWeight args <= 1)
  ensure "--adaptive-edge-buffer-max must be >= 0" (argAdaptiveEdgeBufferMax args >= 0)
  ensure "--adaptive-min-signal-to-noise-max must be >= 0" (argAdaptiveMinSignalToNoiseMax args >= 0)
  ensure "--adaptive-kalman-z-min-max must be >= 0" (argAdaptiveKalmanZMinMax args >= 0)
  ensure "--adaptive-trend-lookback-max must be >= 0" (argAdaptiveTrendLookbackMax args >= 0)
  ensure "--threshold-factor-alpha must be between 0 and 1" (argThresholdFactorAlpha args >= 0 && argThresholdFactorAlpha args <= 1)
  ensure "--threshold-factor-min must be >= 0" (argThresholdFactorMin args >= 0)
  ensure "--threshold-factor-max must be >= --threshold-factor-min" (argThresholdFactorMax args >= argThresholdFactorMin args)
  ensure "--threshold-factor-floor must be >= 0" (argThresholdFactorFloor args >= 0)
  ensure "--edge-buffer must be >= 0" (argEdgeBuffer args >= 0)
  ensure "--trend-lookback must be >= 0" (argTrendLookback args >= 0)
  ensure "--max-position-size must be >= 0" (argMaxPositionSize args >= 0)
  let volTargetEnabled = maybe False (> 0) (argVolTarget args)
  case argVolTarget args of
    Nothing -> pure ()
    Just v -> ensure "--vol-target must be >= 0" (v >= 0)
  ensure "--vol-lookback must be >= 0" (argVolLookback args >= 0)
  case (volTargetEnabled, argVolEwmaAlpha args) of
    (True, Nothing) ->
      ensure "--vol-lookback must be >= 2 when --vol-target is set (or use --vol-ewma-alpha)" (argVolLookback args >= 2)
    _ -> pure ()
  case argVolEwmaAlpha args of
    Nothing -> pure ()
    Just a -> ensure "--vol-ewma-alpha must be > 0 and < 1" (a > 0 && a < 1)
  ensure "--vol-floor must be >= 0" (argVolFloor args >= 0)
  ensure "--vol-scale-max must be >= 0" (argVolScaleMax args >= 0)
  case argMaxVolatility args of
    Nothing -> pure ()
    Just v -> ensure "--max-volatility must be >= 0" (v >= 0)
  ensure "--rebalance-bars must be >= 0" (argRebalanceBars args >= 0)
  ensure "--rebalance-threshold must be >= 0" (argRebalanceThreshold args >= 0)
  ensure "--rebalance-cost-mult must be >= 0" (argRebalanceCostMult args >= 0)
  let fundingRate = argFundingRate args
  ensure "--funding-rate must be finite" (not (isNaN fundingRate || isInfinite fundingRate))
  ensure "--blend-weight must be between 0 and 1" (argBlendWeight args >= 0 && argBlendWeight args <= 1)
  ensure "--tri-layer-fast-mult must be > 0" (argTriLayerFastMult args > 0)
  ensure "--tri-layer-slow-mult must be > 0" (argTriLayerSlowMult args > 0)
  ensure "--tri-layer-cloud-padding must be >= 0" (argTriLayerCloudPadding args >= 0)
  ensure "--tri-layer-cloud-slope must be >= 0" (argTriLayerCloudSlope args >= 0)
  ensure "--tri-layer-cloud-width must be >= 0" (argTriLayerCloudWidth args >= 0)
  ensure "--tri-layer-touch-lookback must be >= 1" (argTriLayerTouchLookback args >= 1)
  ensure "--tri-layer-price-action-body must be >= 0" (argTriLayerPriceActionBody args >= 0)
  ensure "--kalman-band-lookback must be >= 0" (argKalmanBandLookback args >= 0)
  ensure "--kalman-band-std-mult must be >= 0" (argKalmanBandStdMult args >= 0)
  case argKalmanBandStdMult args of
    v | v > 0 ->
      ensure "--kalman-band-lookback must be >= 2 when --kalman-band-std-mult is enabled" (argKalmanBandLookback args >= 2)
    _ -> pure ()
  ensure "--lstm-exit-flip-bars must be >= 0" (argLstmExitFlipBars args >= 0)
  ensure "--lstm-exit-flip-grace-bars must be >= 0" (argLstmExitFlipGraceBars args >= 0)
  ensure "--lstm-confidence-soft must be between 0 and 1" (argLstmConfidenceSoft args >= 0 && argLstmConfidenceSoft args <= 1)
  ensure "--lstm-confidence-hard must be between 0 and 1" (argLstmConfidenceHard args >= 0 && argLstmConfidenceHard args <= 1)
  ensure
    "--lstm-confidence-soft must be <= --lstm-confidence-hard (unless hard=0 to disable)"
    (argLstmConfidenceHard args <= 0 || argLstmConfidenceSoft args <= argLstmConfidenceHard args)
  case argMaxOrderErrors args of
    Nothing -> pure ()
    Just n -> ensure "--max-order-errors must be >= 1" (n >= 1)
  case argPeriodsPerYear args of
    Nothing -> pure ()
    Just v -> ensure "--periods-per-year must be > 0" (v > 0)
  case argOrderQuote args of
    Nothing -> pure ()
    Just q -> ensure "--order-quote must be >= 0" (q >= 0)
  case argOrderQuantity args of
    Nothing -> pure ()
    Just q -> ensure "--order-quantity must be >= 0" (q >= 0)

  let qtyOn = maybe False (> 0) (argOrderQuantity args)
      quoteOn = maybe False (> 0) (argOrderQuote args)
      fracOn = maybe False (> 0) (argOrderQuoteFraction args)
      sizingCount = fromEnum qtyOn + fromEnum quoteOn + fromEnum fracOn
  ensure "Provide only one of --order-quantity, --order-quote, --order-quote-fraction" (sizingCount <= 1)

  case argOrderQuoteFraction args of
    Nothing -> pure ()
    Just f -> ensure "--order-quote-fraction must be > 0 and <= 1" (f > 0 && f <= 1)
  case argMaxOrderQuote args of
    Nothing -> pure ()
    Just q -> ensure "--max-order-quote must be >= 0" (q >= 0)

  case argMaxOrderQuote args of
    Just q | q > 0 -> ensure "--max-order-quote requires --order-quote-fraction" fracOn
    _ -> pure ()

  ensure "--binance-trade requires --binance-symbol" (not (argBinanceTrade args && argBinanceSymbol args == Nothing))
  let market = argBinanceMarket args
  ensure "--positioning long-short requires --futures when trading" (not (argBinanceTrade args && argPositioning args == LongShort && market /= MarketFutures))
  ensure
    "--positioning long-short requires Binance futures when using exchange data"
    (not (argPositioning args == LongShort && isJust (argBinanceSymbol args) && (argPlatform args /= PlatformBinance || market /= MarketFutures)))
  ensure "--margin requires --binance-live for trading" (not (argBinanceMargin args && argBinanceTrade args && not (argBinanceLive args)))
  case argIdempotencyKey args of
    Nothing -> pure ()
    Just raw ->
      let k = trim raw
          okLen = not (null k) && length k <= 36
          okChars = all (\c -> isAlphaNum c || c == '-' || c == '_') k
       in ensure "--idempotency-key must be 1..36 chars of [A-Za-z0-9_-]" (okLen && okChars)

  ensure "--kalman-z-min must be >= 0" (argKalmanZMin args >= 0)
  ensure "--kalman-z-max must be >= --kalman-z-min" (argKalmanZMax args >= argKalmanZMin args)
  case argMaxHighVolProb args of
    Nothing -> pure ()
    Just v -> ensure "--max-high-vol-prob must be between 0 and 1" (v >= 0 && v <= 1)
  case argMaxConformalWidth args of
    Nothing -> pure ()
    Just v -> ensure "--max-conformal-width must be >= 0" (v >= 0)
  case argMaxQuantileWidth args of
    Nothing -> pure ()
    Just v -> ensure "--max-quantile-width must be >= 0" (v >= 0)
  ensure "--min-position-size must be between 0 and 1" (argMinPositionSize args >= 0 && argMinPositionSize args <= 1)
  ensure "--min-position-size must be <= --max-position-size" (argMinPositionSize args <= argMaxPositionSize args)
  ensure "--tune-stress-vol-mult must be > 0" (argTuneStressVolMult args > 0)
  ensure "--tune-stress-weight must be >= 0" (argTuneStressWeight args >= 0)

  pure args
  where
    ensure :: String -> Bool -> Either String ()
    ensure msg cond = if cond then Right () else Left msg
