module Trader.Trading
  ( Positioning(..)
  , EnsembleConfig(..)
  , IntrabarFill(..)
  , PositionSide(..)
  , ExitReason(..)
  , exitReasonCode
  , exitReasonFromCode
  , StepMeta(..)
  , Trade(..)
  , BacktestResult(..)
  , simulateEnsemble
  , simulateEnsembleV
  , simulateEnsembleWithHL
  , simulateEnsembleVWithHL
  , simulateEnsembleLongFlat
  , simulateEnsembleLongFlatV
  , simulateEnsembleLongFlatWithHL
  , simulateEnsembleLongFlatVWithHL
  ) where

import Data.Aeson (FromJSON(..), ToJSON(..), withText)
import qualified Data.Aeson as Aeson
import Data.List (foldl')
import qualified Data.Text as T
import qualified Data.Vector as V

data Positioning
  = LongFlat
  | LongShort
  deriving (Eq, Show)

data EnsembleConfig = EnsembleConfig
  { ecOpenThreshold :: !Double
  , ecCloseThreshold :: !Double
  , ecFee :: !Double
  , ecSlippage :: !Double          -- fractional per side, e.g. 0.0002
  , ecSpread :: !Double            -- fractional total spread, e.g. 0.0005 (half per side)
  , ecStopLoss :: !(Maybe Double)      -- fractional, e.g. 0.02
  , ecTakeProfit :: !(Maybe Double)    -- fractional, e.g. 0.03
  , ecTrailingStop :: !(Maybe Double)  -- fractional, e.g. 0.01
  , ecMinHoldBars :: !Int              -- bars; 0 disables (signal exits allowed immediately)
  , ecCooldownBars :: !Int             -- bars; 0 disables (wait after exiting before re-entering)
  , ecMaxDrawdown :: !(Maybe Double)   -- fraction, e.g. 0.2 (20%); halts and exits to flat
  , ecMaxDailyLoss :: !(Maybe Double)  -- fraction, e.g. 0.05 (5%); halts and exits to flat
  , ecIntervalSeconds :: !(Maybe Int)  -- required for daily-loss; inferred from CLI interval
  , ecPositioning :: !Positioning
  , ecIntrabarFill :: !IntrabarFill
  -- Confidence gating/sizing (Kalman sensors + HMM/intervals)
  , ecKalmanZMin :: !Double
  , ecKalmanZMax :: !Double
  , ecMaxHighVolProb :: !(Maybe Double)
  , ecMaxConformalWidth :: !(Maybe Double)
  , ecMaxQuantileWidth :: !(Maybe Double)
  , ecConfirmConformal :: !Bool
  , ecConfirmQuantiles :: !Bool
  , ecConfidenceSizing :: !Bool
  , ecMinPositionSize :: !Double
  } deriving (Eq, Show)

data IntrabarFill
  = StopFirst
  | TakeProfitFirst
  deriving (Eq, Show)

data PositionSide
  = SideLong
  | SideShort
  deriving (Eq, Show)

sideSign :: PositionSide -> Double
sideSign side =
  case side of
    SideLong -> 1
    SideShort -> -1

data ExitReason
  = ExitSignal
  | ExitStopLoss
  | ExitTrailingStop
  | ExitTakeProfit
  | ExitMaxDrawdown
  | ExitMaxDailyLoss
  | ExitLiquidation
  | ExitEod
  | ExitOther !String
  deriving (Eq, Show)

exitReasonCode :: ExitReason -> String
exitReasonCode why =
  case why of
    ExitSignal -> "SIGNAL"
    ExitStopLoss -> "STOP_LOSS"
    ExitTrailingStop -> "TRAILING_STOP"
    ExitTakeProfit -> "TAKE_PROFIT"
    ExitMaxDrawdown -> "MAX_DRAWDOWN"
    ExitMaxDailyLoss -> "MAX_DAILY_LOSS"
    ExitLiquidation -> "LIQUIDATION"
    ExitEod -> "EOD"
    ExitOther s -> s

exitReasonFromCode :: String -> ExitReason
exitReasonFromCode code =
  case code of
    "SIGNAL" -> ExitSignal
    "STOP_LOSS" -> ExitStopLoss
    "TRAILING_STOP" -> ExitTrailingStop
    "TAKE_PROFIT" -> ExitTakeProfit
    "MAX_DRAWDOWN" -> ExitMaxDrawdown
    "MAX_DAILY_LOSS" -> ExitMaxDailyLoss
    "LIQUIDATION" -> ExitLiquidation
    "EOD" -> ExitEod
    other -> ExitOther other

instance ToJSON ExitReason where
  toJSON = Aeson.String . T.pack . exitReasonCode

instance FromJSON ExitReason where
  parseJSON = withText "ExitReason" (pure . exitReasonFromCode . T.unpack)

data StepMeta = StepMeta
  { smKalmanMean :: !Double
  , smKalmanVar :: !Double
  , smHighVolProb :: !(Maybe Double)
  , smQuantile10 :: !(Maybe Double)
  , smQuantile90 :: !(Maybe Double)
  , smConformalLo :: !(Maybe Double)
  , smConformalHi :: !(Maybe Double)
  } deriving (Eq, Show)

data Trade = Trade
  { trEntryIndex :: !Int
  , trExitIndex :: !Int
  , trEntryEquity :: !Double
  , trExitEquity :: !Double
  , trReturn :: !Double
  , trHoldingPeriods :: !Int
  , trExitReason :: !(Maybe ExitReason)
  } deriving (Eq, Show)

data OpenTrade = OpenTrade
  { otEntryIndex :: !Int
  , otEntryEquity :: !Double
  , otHoldingPeriods :: !Int
  , otEntryPrice :: !Double
  , otTrail :: !Double
  , otSide :: !PositionSide
  } deriving (Eq, Show)

data BacktestResult = BacktestResult
  { brEquityCurve :: [Double]     -- length n
  , brPositions :: [Double]       -- length n-1 (signed position size held for return t->t+1, -1..1)
  , brAgreementOk :: [Bool]       -- length n-1 (only meaningful where both preds available)
  , brPositionChanges :: !Int
  , brTrades :: [Trade]
  } deriving (Eq, Show)

simulateEnsemble
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> [Double]       -- prices length n
  -> [Double]       -- kalman predicted next prices length n-1 (for t=0..n-2)
  -> [Double]       -- lstm predicted next prices length n-lookback (for t=lookback-1..n-2)
  -> Maybe [StepMeta] -- optional per-step confidence meta (length n-1)
  -> BacktestResult
simulateEnsemble = simulateEnsembleLongFlat

simulateEnsembleWithHL
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> [Double]       -- closes length n
  -> [Double]       -- highs length n (aligned to closes; bar i high is for close[i-1]..close[i])
  -> [Double]       -- lows length n
  -> [Double]       -- kalman predicted next prices length n-1 (for t=0..n-2)
  -> [Double]       -- lstm predicted next prices length n-lookback (for t=lookback-1..n-2)
  -> Maybe [StepMeta] -- optional per-step confidence meta (length n-1)
  -> BacktestResult
simulateEnsembleWithHL = simulateEnsembleLongFlatWithHL

simulateEnsembleV
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> V.Vector Double
  -> V.Vector Double
  -> V.Vector Double
  -> Maybe (V.Vector StepMeta) -- optional per-step confidence meta
  -> BacktestResult
simulateEnsembleV = simulateEnsembleLongFlatV

simulateEnsembleVWithHL
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> V.Vector Double -- closes
  -> V.Vector Double -- highs
  -> V.Vector Double -- lows
  -> V.Vector Double
  -> V.Vector Double
  -> Maybe (V.Vector StepMeta) -- optional per-step confidence meta
  -> BacktestResult
simulateEnsembleVWithHL = simulateEnsembleLongFlatVWithHL

{-# DEPRECATED simulateEnsembleLongFlat "Use simulateEnsemble" #-}
{-# DEPRECATED simulateEnsembleLongFlatWithHL "Use simulateEnsembleWithHL" #-}
{-# DEPRECATED simulateEnsembleLongFlatV "Use simulateEnsembleV" #-}
{-# DEPRECATED simulateEnsembleLongFlatVWithHL "Use simulateEnsembleVWithHL" #-}

simulateEnsembleLongFlat
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> [Double]       -- prices length n
  -> [Double]       -- kalman predicted next prices length n-1 (for t=0..n-2)
  -> [Double]       -- lstm predicted next prices length n-lookback (for t=lookback-1..n-2)
  -> Maybe [StepMeta] -- optional per-step confidence meta (length n-1)
  -> BacktestResult
simulateEnsembleLongFlat cfg lookback prices kalPredNext lstmPredNext mMeta =
  simulateEnsembleLongFlatV
    cfg
    lookback
    (V.fromList prices)
    (V.fromList kalPredNext)
    (V.fromList lstmPredNext)
    (V.fromList <$> mMeta)

simulateEnsembleLongFlatWithHL
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> [Double]       -- closes length n
  -> [Double]       -- highs length n (aligned to closes; bar i high is for close[i-1]..close[i])
  -> [Double]       -- lows length n
  -> [Double]       -- kalman predicted next prices length n-1 (for t=0..n-2)
  -> [Double]       -- lstm predicted next prices length n-lookback (for t=lookback-1..n-2)
  -> Maybe [StepMeta] -- optional per-step confidence meta (length n-1)
  -> BacktestResult
simulateEnsembleLongFlatWithHL cfg lookback closes highs lows kalPredNext lstmPredNext mMeta =
  simulateEnsembleLongFlatVWithHL
    cfg
    lookback
    (V.fromList closes)
    (V.fromList highs)
    (V.fromList lows)
    (V.fromList kalPredNext)
    (V.fromList lstmPredNext)
    (V.fromList <$> mMeta)

simulateEnsembleLongFlatV
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> V.Vector Double
  -> V.Vector Double
  -> V.Vector Double
  -> Maybe (V.Vector StepMeta) -- optional per-step confidence meta
  -> BacktestResult
simulateEnsembleLongFlatV cfg lookback pricesV kalPredNextV lstmPredNextV mMetaV =
  simulateEnsembleLongFlatVWithHL cfg lookback pricesV pricesV pricesV kalPredNextV lstmPredNextV mMetaV

simulateEnsembleLongFlatVWithHL
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> V.Vector Double -- closes
  -> V.Vector Double -- highs
  -> V.Vector Double -- lows
  -> V.Vector Double
  -> V.Vector Double
  -> Maybe (V.Vector StepMeta) -- optional per-step confidence meta
  -> BacktestResult
simulateEnsembleLongFlatVWithHL cfg lookback pricesV highsV lowsV kalPredNextV lstmPredNextV mMetaV =
  let n = V.length pricesV
   in if n < 2
        then error "Need at least 2 prices to simulate"
        else
          let startT = max 0 (lookback - 1)
              openThr = max 0 (ecOpenThreshold cfg)
              closeThr = max 0 (ecCloseThreshold cfg)
              minHoldBars = max 0 (ecMinHoldBars cfg)
              cooldownBars = max 0 (ecCooldownBars cfg)
              maxDrawdownLim =
                case ecMaxDrawdown cfg of
                  Just v | v > 0 && v < 1 && not (isNaN v || isInfinite v) -> Just v
                  _ -> Nothing
              intervalSeconds =
                case ecIntervalSeconds cfg of
                  Just s | s > 0 -> Just s
                  _ -> Nothing
              maxDailyLossLim =
                case (ecMaxDailyLoss cfg, intervalSeconds) of
                  (Just v, Just _) | v > 0 && v < 1 && not (isNaN v || isInfinite v) -> Just v
                  _ -> Nothing
              dayKeyAt :: Int -> Int
              dayKeyAt i =
                case intervalSeconds of
                  Nothing -> 0
                  Just sec ->
                    let tSec = fromIntegral i * fromIntegral sec :: Integer
                     in fromIntegral (tSec `div` 86400)
              direction thr prev pred =
                let upEdge = prev * (1 + thr)
                    downEdge = prev * (1 - thr)
                 in if pred > upEdge
                      then Just SideLong
                      else if pred < downEdge then Just SideShort else Nothing
              desiredSideFromDir d =
                case d of
                  SideLong -> Just SideLong
                  SideShort ->
                    case ecPositioning cfg of
                      LongFlat -> Nothing
                      LongShort -> Just SideShort
              stepCount = n - 1

              kalNeed = if startT < stepCount then stepCount else 0
              lstmNeed = max 0 (stepCount - startT)

              () =
                if V.length highsV /= n || V.length lowsV /= n
                  then error "high/low vectors must match closes length"
                  else
                    if maybe False (\mv -> V.length mv < stepCount) mMetaV
                      then error "meta vector too short for simulateEnsembleLongFlatVWithHL"
                      else
                        if V.length kalPredNextV < kalNeed
                          then
                            error
                              ( "kalPredNext too short: need at least "
                                  ++ show kalNeed
                                  ++ ", got "
                                  ++ show (V.length kalPredNextV)
                              )
                          else
                            if V.length lstmPredNextV < lstmNeed
                              then
                                error
                                  ( "lstmPredNext too short: need at least "
                                      ++ show lstmNeed
                                      ++ ", got "
                                      ++ show (V.length lstmPredNextV)
                                  )
                              else ()

              perSideCost =
                let fee = max 0 (ecFee cfg)
                    slip = max 0 (ecSlippage cfg)
                    spr = max 0 (ecSpread cfg)
                    c = fee + slip + spr / 2
                 in min 0.999999 (max 0 c)

              applyCost :: Double -> Double -> Double
              applyCost eq size =
                let s = min 1 (max 0 (abs size))
                 in eq * (1 - perSideCost * s)

              barHigh t1 =
                let h = highsV V.! t1
                    c = pricesV V.! t1
                 in if isNaN h || isInfinite h then c else h

              barLow t1 =
                let l = lowsV V.! t1
                    c = pricesV V.! t1
                 in if isNaN l || isInfinite l then c else l

              markToMarket :: PositionSide -> Double -> Double -> Double
              markToMarket side basePx px =
                if basePx == 0
                  then 1
                  else
                    case side of
                      SideLong -> px / basePx
                      SideShort -> 2 - px / basePx

              isBad :: Double -> Bool
              isBad x = isNaN x || isInfinite x

              clamp01 :: Double -> Double
              clamp01 x = max 0 (min 1 x)

              scale01 :: Double -> Double -> Double -> Double
              scale01 lo hi x =
                let lo' = min lo hi
                    hi' = max lo hi
                 in if hi' <= lo' + 1e-12
                      then if x >= hi' then 1 else 0
                      else clamp01 ((x - lo') / (hi' - lo'))

              metaAt :: Int -> Maybe StepMeta
              metaAt t =
                case mMetaV of
                  Nothing -> Nothing
                  Just mv -> if t >= 0 && t < V.length mv then Just (mv V.! t) else Nothing

              intervalWidth :: StepMeta -> Maybe Double
              intervalWidth m =
                case (smConformalLo m, smConformalHi m) of
                  (Just lo', Just hi') ->
                    let w = hi' - lo'
                     in if isNaN w || isInfinite w then Nothing else Just w
                  _ -> Nothing

              quantileWidth :: StepMeta -> Maybe Double
              quantileWidth m =
                case (smQuantile10 m, smQuantile90 m) of
                  (Just q10', Just q90') ->
                    let w = q90' - q10'
                     in if isNaN w || isInfinite w then Nothing else Just w
                  _ -> Nothing

              kalmanZ :: StepMeta -> Double
              kalmanZ m =
                let var = max 0 (smKalmanVar m)
                    std = sqrt var
                    mu = smKalmanMean m
                 in if std <= 0 || isNaN std || isInfinite std then 0 else abs mu / std

              confirmConformal :: StepMeta -> PositionSide -> Bool
              confirmConformal m side =
                if not (ecConfirmConformal cfg)
                  then True
                  else
                    case (smConformalLo m, smConformalHi m, side) of
                      (Just lo', _, SideLong) -> lo' > openThr
                      (_, Just hi', SideShort) -> hi' < negate openThr
                      _ -> False

              confirmQuantiles :: StepMeta -> PositionSide -> Bool
              confirmQuantiles m side =
                if not (ecConfirmQuantiles cfg)
                  then True
                  else
                    case (smQuantile10 m, smQuantile90 m, side) of
                      (Just q10', _, SideLong) -> q10' > openThr
                      (_, Just q90', SideShort) -> q90' < negate openThr
                      _ -> False

              confidenceScoreKalman :: StepMeta -> Double
              confidenceScoreKalman m =
                let zMin = max 0 (ecKalmanZMin cfg)
                    zMax = max zMin (ecKalmanZMax cfg)
                    zScore = scale01 zMin zMax (kalmanZ m)
                    hvScore =
                      case (ecMaxHighVolProb cfg, smHighVolProb m) of
                        (Just maxHv, Just hv) -> clamp01 ((maxHv - hv) / max 1e-12 maxHv)
                        _ -> 1
                    confScore =
                      case (ecMaxConformalWidth cfg, intervalWidth m) of
                        (Just maxW, Just w) -> clamp01 ((maxW - w) / max 1e-12 maxW)
                        _ -> 1
                    qScore =
                      case (ecMaxQuantileWidth cfg, quantileWidth m) of
                        (Just maxW, Just w) -> clamp01 ((maxW - w) / max 1e-12 maxW)
                        _ -> 1
                 in zScore * hvScore * confScore * qScore

              gateKalmanDir :: StepMeta -> Double -> Maybe PositionSide -> (Maybe PositionSide, Double)
              gateKalmanDir m confScore dirRaw =
                case dirRaw of
                  Nothing -> (Nothing, 0)
                  Just side ->
                    let kalZ = kalmanZ m
                        zMin = max 0 (ecKalmanZMin cfg)
                        hvOk =
                          case (ecMaxHighVolProb cfg, smHighVolProb m) of
                            (Just maxHv, Just hv) -> hv <= maxHv
                            (Just _, Nothing) -> False
                            _ -> True
                        confWidthOk =
                          case (ecMaxConformalWidth cfg, intervalWidth m) of
                            (Just maxW, Just w) -> w <= maxW
                            (Just _, Nothing) -> False
                            _ -> True
                        qWidthOk =
                          case (ecMaxQuantileWidth cfg, quantileWidth m) of
                            (Just maxW, Just w) -> w <= maxW
                            (Just _, Nothing) -> False
                            _ -> True
                        confOk = confScore >= max 0 (min 1 (ecMinPositionSize cfg))
                        size0 = if ecConfidenceSizing cfg then confScore else 1
                     in
                      if kalZ < zMin
                        then (Nothing, 0)
                        else if not hvOk
                          then (Nothing, 0)
                          else if not confWidthOk
                            then (Nothing, 0)
                            else if not qWidthOk
                              then (Nothing, 0)
                              else if not (confirmConformal m side)
                                then (Nothing, 0)
                                else if not (confirmQuantiles m side)
                                  then (Nothing, 0)
                                  else if ecConfidenceSizing cfg && (not confOk || size0 <= 0)
                                    then (Nothing, 0)
                                    else (Just side, if ecConfidenceSizing cfg then size0 else 1)

              stepFn (posSide, posSize, equity, eqAcc, posAcc, agreeAcc, changes, openTrade, tradesAcc, dead, cooldownLeft, riskState0) t =
                if dead
                  then
                    ( Nothing
                    , 0
                    , equity
                    , equity : eqAcc
                    , 0 : posAcc
                    , False : agreeAcc
                    , changes
                    , Nothing
                    , tradesAcc
                    , True
                    , 0
                    , riskState0
                    )
                  else
                    let (peakEq0, dayKey0, dayStartEq0, haltReason0) = riskState0
                        (dayKey1, dayStartEq1) =
                          case intervalSeconds of
                            Nothing -> (dayKey0, dayStartEq0)
                            Just _ ->
                              let dk = dayKeyAt t
                               in if dk /= dayKey0 then (dk, equity) else (dayKey0, dayStartEq0)
                        peakEq1 = max peakEq0 equity
                        drawdown =
                          if peakEq1 > 0
                            then max 0 (1 - equity / peakEq1)
                            else 0
                        dailyLoss =
                          if dayStartEq1 > 0
                            then max 0 (1 - equity / dayStartEq1)
                            else 0
                        riskHaltReason =
                          case haltReason0 of
                            Just _ -> Nothing
                            Nothing ->
                              case () of
                                _ | maybe False (\lim -> dailyLoss >= lim) maxDailyLossLim -> Just ExitMaxDailyLoss
                                  | maybe False (\lim -> drawdown >= lim) maxDrawdownLim -> Just ExitMaxDrawdown
                                  | otherwise -> Nothing
                        haltReason1 =
                          case haltReason0 of
                            Just r -> Just r
                            Nothing -> riskHaltReason
                        halted = haltReason1 /= Nothing

                        prev = pricesV V.! t
                        nextClose = pricesV V.! (t + 1)
                        hi = barHigh (t + 1)
                        lo = barLow (t + 1)
                        openTrade0 =
                          case posSide of
                            Nothing -> Nothing
                            Just _ -> openTrade
                        holdBars =
                          case openTrade0 of
                            Nothing -> 0
                            Just ot -> otHoldingPeriods ot
                        cooldownActive = posSide == Nothing && cooldownLeft > 0
                        cooldownNext0 = if posSide == Nothing then max 0 (cooldownLeft - 1) else 0
                        (agreeOk, desiredSideRaw, desiredSizeRaw) =
                          if t < startT
                            then (False, posSide, posSize)
                            else
                              let kp = kalPredNextV V.! t
                                  lp = lstmPredNextV V.! (t - startT)
                                  kalOpenDirRaw = direction openThr prev kp
                                  kalCloseDirRaw = direction closeThr prev kp
                                  (kalOpenDir, kalSize) =
                                    case metaAt t of
                                      Nothing -> (kalOpenDirRaw, if kalOpenDirRaw == Nothing then 0 else 1)
                                      Just m ->
                                        let confScore = confidenceScoreKalman m
                                         in gateKalmanDir m confScore kalOpenDirRaw
                                  lstmOpenDir = direction openThr prev lp
                                  lstmCloseDir = direction closeThr prev lp
                                  openAgreeDir =
                                    if kalOpenDir == lstmOpenDir
                                      then kalOpenDir
                                      else Nothing
                                  closeAgreeDir =
                                    if kalCloseDirRaw == lstmCloseDir
                                      then kalCloseDirRaw
                                      else Nothing

                                  desiredFromOpen dir =
                                    case desiredSideFromDir dir of
                                      Nothing -> (Nothing, 0)
                                      Just s -> (Just s, kalSize)

                                  (desiredSide', desiredSize') =
                                    case openAgreeDir of
                                      Just dir -> desiredFromOpen dir
                                      Nothing ->
                                        case posSide of
                                          Nothing -> (Nothing, 0)
                                          Just SideLong ->
                                            if closeAgreeDir == Just SideLong
                                              then (Just SideLong, posSize)
                                              else (Nothing, 0)
                                          Just SideShort ->
                                            if closeAgreeDir == Just SideShort
                                              then (Just SideShort, posSize)
                                              else (Nothing, 0)
                               in (openAgreeDir == Just SideLong || openAgreeDir == Just SideShort, desiredSide', desiredSize')

                        desiredSize =
                          if desiredSideRaw == Nothing
                            then 0
                            else min 1 (max 0 desiredSizeRaw)

                        desiredSide =
                          if desiredSize <= 0 then Nothing else desiredSideRaw

                        desiredSideHoldAdjusted =
                          if posSide /= Nothing && desiredSide /= posSide && holdBars < minHoldBars
                            then posSide
                            else desiredSide

                        desiredSizeHoldAdjusted =
                          if desiredSideHoldAdjusted == posSide
                            then posSize
                            else desiredSize

                        (desiredSideFinal0, desiredSizeFinal0) =
                          if cooldownActive
                            then (Nothing, 0)
                            else (desiredSideHoldAdjusted, desiredSizeHoldAdjusted)

                        (desiredSideFinal, desiredSizeFinal) =
                          if halted
                            then (Nothing, 0)
                            else (desiredSideFinal0, desiredSizeFinal0)

                        switchExitReason =
                          case (halted, haltReason1, posSide) of
                            (True, Just r, Just _) -> r
                            _ -> ExitSignal

                        closeTradeAt exitIndex why eqExit ot =
                          Trade
                            { trEntryIndex = otEntryIndex ot
                            , trExitIndex = exitIndex
                            , trEntryEquity = otEntryEquity ot
                            , trExitEquity = eqExit
                            , trReturn = eqExit / otEntryEquity ot - 1
                            , trHoldingPeriods = otHoldingPeriods ot
                            , trExitReason = Just why
                            }

                        openTradeFor side eqEntry =
                          OpenTrade
                            { otEntryIndex = t
                            , otEntryEquity = eqEntry
                            , otHoldingPeriods = 0
                            , otEntryPrice = prev
                            , otTrail = prev
                            , otSide = side
                            }

                        (posAfterSwitch, posSizeAfterSwitch, equityAfterSwitch, changes', openTrade', tradesAcc') =
                          if desiredSideFinal == posSide
                            then (posSide, posSize, equity, changes, openTrade0, tradesAcc)
                            else
                              case desiredSideFinal of
                                Nothing ->
                                  let eqExit = applyCost equity posSize
                                      tradesAcc1 =
                                        case openTrade0 of
                                          Nothing -> tradesAcc
                                          Just ot -> closeTradeAt t switchExitReason eqExit ot : tradesAcc
                                   in (Nothing, 0, eqExit, changes + 1, Nothing, tradesAcc1)
                                Just desiredSideFinal' ->
                                  case posSide of
                                    Nothing ->
                                      let eqEntry = applyCost equity desiredSizeFinal
                                       in
                                        ( Just desiredSideFinal'
                                        , desiredSizeFinal
                                        , eqEntry
                                        , changes + 1
                                        , Just (openTradeFor desiredSideFinal' eqEntry)
                                        , tradesAcc
                                        )
                                    Just _ ->
                                      let eqExit = applyCost equity posSize
                                          tradesAcc1 =
                                            case openTrade0 of
                                              Nothing -> tradesAcc
                                              Just ot -> closeTradeAt t ExitSignal eqExit ot : tradesAcc
                                          eqEntry = applyCost eqExit desiredSizeFinal
                                       in
                                        ( Just desiredSideFinal'
                                        , desiredSizeFinal
                                        , eqEntry
                                        , changes + 1
                                        , Just (openTradeFor desiredSideFinal' eqEntry)
                                        , tradesAcc1
                                        )

                        equityAtClose =
                          case posAfterSwitch of
                            Just side | posSizeAfterSwitch > 0 ->
                              let factor = markToMarket side prev nextClose
                                  eq1 = equityAfterSwitch * (1 + posSizeAfterSwitch * (factor - 1))
                               in if isBad eq1 then equityAfterSwitch else eq1
                            _ -> equityAfterSwitch

                        (posFinal, posSizeFinal, equityFinal, changesFinal, openTradeFinal, tradesFinal) =
                          case (posAfterSwitch, openTrade') of
                            (Just SideLong, Just ot0) ->
                              let otHeld = ot0 { otHoldingPeriods = otHoldingPeriods ot0 + 1 }
                                  entryPx = otEntryPrice ot0
                                  trail0 = otTrail ot0
                                  mTp =
                                    case ecTakeProfit cfg of
                                      Just tp | tp > 0 -> Just (entryPx * (1 + tp))
                                      _ -> Nothing
                                  mSl =
                                    case ecStopLoss cfg of
                                      Just sl | sl > 0 -> Just (entryPx * (1 - sl))
                                      _ -> Nothing
                                  stopPx trail =
                                    let mTs =
                                          case ecTrailingStop cfg of
                                            Just ts | ts > 0 -> Just (trail * (1 - ts))
                                            _ -> Nothing
                                     in case (mSl, mTs) of
                                          (Nothing, Nothing) -> (Nothing, Nothing)
                                          (Just slPx, Nothing) -> (Just slPx, Just ExitStopLoss)
                                          (Nothing, Just tsPx) -> (Just tsPx, Just ExitTrailingStop)
                                          (Just slPx, Just tsPx) ->
                                            if tsPx > slPx
                                              then (Just tsPx, Just ExitTrailingStop)
                                              else (Just slPx, Just ExitStopLoss)
                                  tpHit = maybe False (\tpPx -> hi >= tpPx) mTp
                                  exitOrTrail =
                                    case ecIntrabarFill cfg of
                                      StopFirst ->
                                        let (mStop, stopWhy) = stopPx trail0
                                            stopHit = maybe False (\stPx -> lo <= stPx) mStop
                                         in if stopHit
                                              then (Just (maybe nextClose id mStop, stopWhy), trail0)
                                              else if tpHit
                                                then (Just (maybe nextClose id mTp, Just ExitTakeProfit), trail0)
                                                else (Nothing, max trail0 hi)
                                      TakeProfitFirst ->
                                        if tpHit
                                          then (Just (maybe nextClose id mTp, Just ExitTakeProfit), trail0)
                                          else
                                            let trail1 = max trail0 hi
                                                (mStop, stopWhy) = stopPx trail1
                                                stopHit = maybe False (\stPx -> lo <= stPx) mStop
                                             in if stopHit
                                                  then (Just (maybe nextClose id mStop, stopWhy), trail1)
                                                  else (Nothing, trail1)
                               in case exitOrTrail of
                                    (Just (exitPx, reason), _trailUsed) ->
                                      let factor = markToMarket SideLong prev exitPx
                                          exitEq0 = equityAfterSwitch * (1 + posSizeAfterSwitch * (factor - 1))
                                          exitEq = applyCost exitEq0 posSizeAfterSwitch
                                          tr =
                                            Trade
                                              { trEntryIndex = otEntryIndex otHeld
                                              , trExitIndex = t + 1
                                              , trEntryEquity = otEntryEquity otHeld
                                              , trExitEquity = exitEq
                                              , trReturn = exitEq / otEntryEquity otHeld - 1
                                              , trHoldingPeriods = otHoldingPeriods otHeld
                                              , trExitReason = reason
                                              }
                                       in (Nothing, 0, exitEq, changes' + 1, Nothing, tr : tradesAcc')
                                    (Nothing, trail1) ->
                                      let otCont = otHeld { otTrail = trail1 }
                                       in (Just SideLong, posSizeAfterSwitch, equityAtClose, changes', Just otCont, tradesAcc')
                            (Just SideShort, Just ot0) ->
                              let otHeld = ot0 { otHoldingPeriods = otHoldingPeriods ot0 + 1 }
                                  entryPx = otEntryPrice ot0
                                  trail0 = otTrail ot0
                                  mTp =
                                    case ecTakeProfit cfg of
                                      Just tp | tp > 0 -> Just (entryPx * (1 - tp))
                                      _ -> Nothing
                                  mSl =
                                    case ecStopLoss cfg of
                                      Just sl | sl > 0 -> Just (entryPx * (1 + sl))
                                      _ -> Nothing
                                  stopPx trail =
                                    let mTs =
                                          case ecTrailingStop cfg of
                                            Just ts | ts > 0 -> Just (trail * (1 + ts))
                                            _ -> Nothing
                                     in case (mSl, mTs) of
                                          (Nothing, Nothing) -> (Nothing, Nothing)
                                          (Just slPx, Nothing) -> (Just slPx, Just ExitStopLoss)
                                          (Nothing, Just tsPx) -> (Just tsPx, Just ExitTrailingStop)
                                          (Just slPx, Just tsPx) ->
                                            if tsPx < slPx
                                              then (Just tsPx, Just ExitTrailingStop)
                                              else (Just slPx, Just ExitStopLoss)
                                  tpHit = maybe False (\tpPx -> lo <= tpPx) mTp
                                  exitOrTrail =
                                    case ecIntrabarFill cfg of
                                      StopFirst ->
                                        let (mStop, stopWhy) = stopPx trail0
                                            stopHit = maybe False (\stPx -> hi >= stPx) mStop
                                         in if stopHit
                                              then (Just (maybe nextClose id mStop, stopWhy), trail0)
                                              else if tpHit
                                                then (Just (maybe nextClose id mTp, Just ExitTakeProfit), trail0)
                                                else (Nothing, min trail0 lo)
                                      TakeProfitFirst ->
                                        if tpHit
                                          then (Just (maybe nextClose id mTp, Just ExitTakeProfit), trail0)
                                          else
                                            let trail1 = min trail0 lo
                                                (mStop, stopWhy) = stopPx trail1
                                                stopHit = maybe False (\stPx -> hi >= stPx) mStop
                                             in if stopHit
                                                  then (Just (maybe nextClose id mStop, stopWhy), trail1)
                                                  else (Nothing, trail1)
                               in case exitOrTrail of
                                    (Just (exitPx, reason), _trailUsed) ->
                                      let factor = markToMarket SideShort prev exitPx
                                          exitEq0 = equityAfterSwitch * (1 + posSizeAfterSwitch * (factor - 1))
                                          exitEq = applyCost exitEq0 posSizeAfterSwitch
                                          tr =
                                            Trade
                                              { trEntryIndex = otEntryIndex otHeld
                                              , trExitIndex = t + 1
                                              , trEntryEquity = otEntryEquity otHeld
                                              , trExitEquity = exitEq
                                              , trReturn = exitEq / otEntryEquity otHeld - 1
                                              , trHoldingPeriods = otHoldingPeriods otHeld
                                              , trExitReason = reason
                                              }
                                       in (Nothing, 0, exitEq, changes' + 1, Nothing, tr : tradesAcc')
                                    (Nothing, trail1) ->
                                      let otCont = otHeld { otTrail = trail1 }
                                       in (Just SideShort, posSizeAfterSwitch, equityAtClose, changes', Just otCont, tradesAcc')
                            _ -> (posAfterSwitch, posSizeAfterSwitch, equityAtClose, changes', openTrade', tradesAcc')

                        (posFinal2, posSizeFinal2, equityFinal2, changesFinal2, openTradeFinal2, tradesFinal2, dead2) =
                          if isBad equityFinal || equityFinal <= 0
                            then
                              let exitEq = if isBad equityFinal then 0 else max 0 equityFinal
                                  (tradesOut, changesOut) =
                                    case openTradeFinal of
                                      Nothing -> (tradesFinal, changesFinal)
                                      Just otHeld ->
                                        let tr =
                                              Trade
                                                { trEntryIndex = otEntryIndex otHeld
                                                , trExitIndex = t + 1
                                                , trEntryEquity = otEntryEquity otHeld
                                                , trExitEquity = exitEq
                                                , trReturn = exitEq / otEntryEquity otHeld - 1
                                                , trHoldingPeriods = otHoldingPeriods otHeld
                                                , trExitReason = Just ExitLiquidation
                                                }
                                         in (tr : tradesFinal, changesFinal + 1)
                               in (Nothing, 0, exitEq, changesOut, Nothing, tradesOut, True)
                            else (posFinal, posSizeFinal, equityFinal, changesFinal, openTradeFinal, tradesFinal, False)

                        exitedToFlat =
                          posFinal2 == Nothing && (posAfterSwitch /= Nothing || posSide /= Nothing)

                        cooldownNext =
                          if posFinal2 == Nothing
                            then if exitedToFlat then cooldownBars else cooldownNext0
                            else 0
                     in
                      ( posFinal2
                      , posSizeFinal2
                      , equityFinal2
                      , equityFinal2 : eqAcc
                      , (maybe 0 sideSign posAfterSwitch * posSizeAfterSwitch) : posAcc
                      , agreeOk : agreeAcc
                      , changesFinal2
                      , openTradeFinal2
                      , tradesFinal2
                      , dead2
                      , cooldownNext
                      , (max peakEq1 equityFinal2, dayKey1, dayStartEq1, haltReason1)
                      )

              (_finalPos, finalPosSize, finalEq, eqRev, posRev, agreeRev, changes, openTrade, tradesRev, _deadFinal, _cooldownFinal, _riskFinal) =
                foldl'
                  stepFn
                  ( Nothing :: Maybe PositionSide
                  , 0 :: Double
                  , 1.0
                  , [1.0]
                  , []
                  , []
                  , 0 :: Int
                  , Nothing :: Maybe OpenTrade
                  , []
                  , False
                  , 0 :: Int
                  , (1.0, dayKeyAt 0, 1.0, Nothing)
                  )
                  [0 .. stepCount - 1]

              (eqRev', tradesRev') =
                case openTrade of
                  Nothing -> (eqRev, tradesRev)
                  Just ot ->
                    let exitEq = applyCost finalEq finalPosSize
                        tr =
                          Trade
                            { trEntryIndex = otEntryIndex ot
                            , trExitIndex = stepCount
                            , trEntryEquity = otEntryEquity ot
                            , trExitEquity = exitEq
                            , trReturn = exitEq / otEntryEquity ot - 1
                            , trHoldingPeriods = otHoldingPeriods ot
                            , trExitReason = Just ExitEod
                            }
                        eqRev1 =
                          case eqRev of
                            [] -> [exitEq]
                            (_ : rest) -> exitEq : rest
                     in (eqRev1, tr : tradesRev)
              eqCurve = reverse eqRev'
           in BacktestResult
                { brEquityCurve = eqCurve
                , brPositions = reverse posRev
                , brAgreementOk = reverse agreeRev
                , brPositionChanges = changes
                , brTrades = reverse tradesRev'
                }
