module Trader.Trading
  ( Positioning(..)
  , EnsembleConfig(..)
  , IntrabarFill(..)
  , StepMeta(..)
  , Trade(..)
  , BacktestResult(..)
  , simulateEnsembleLongFlat
  , simulateEnsembleLongFlatV
  , simulateEnsembleLongFlatWithHL
  , simulateEnsembleLongFlatVWithHL
  ) where

import Data.List (foldl')
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
  , trExitReason :: !(Maybe String)
  } deriving (Eq, Show)

data OpenTrade = OpenTrade
  { otEntryIndex :: !Int
  , otEntryEquity :: !Double
  , otHoldingPeriods :: !Int
  , otEntryPrice :: !Double
  , otTrail :: !Double
  , otDir :: !Int -- 1=long, -1=short
  } deriving (Eq, Show)

data BacktestResult = BacktestResult
  { brEquityCurve :: [Double]     -- length n
  , brPositions :: [Double]       -- length n-1 (signed position size held for return t->t+1, -1..1)
  , brAgreementOk :: [Bool]       -- length n-1 (only meaningful where both preds available)
  , brPositionChanges :: !Int
  , brTrades :: [Trade]
  } deriving (Eq, Show)

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
              direction thr prev pred =
                let upEdge = prev * (1 + thr)
                    downEdge = prev * (1 - thr)
                 in if pred > upEdge
                      then Just (1 :: Int)
                      else if pred < downEdge then Just (-1) else Nothing
              desiredPosFromDir d =
                case d of
                  1 -> 1
                  (-1) ->
                    case ecPositioning cfg of
                      LongFlat -> 0
                      LongShort -> (-1)
                  _ -> 0
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

              markToMarket dir basePx px =
                if basePx == 0
                  then 1
                  else
                    case dir of
                      1 -> px / basePx
                      (-1) -> 2 - px / basePx
                      _ -> 1

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

              confirmConformal :: StepMeta -> Int -> Bool
              confirmConformal m dir =
                if not (ecConfirmConformal cfg)
                  then True
                  else
                    case (smConformalLo m, smConformalHi m, dir) of
                      (Just lo', _, 1) -> lo' > openThr
                      (_, Just hi', (-1)) -> hi' < negate openThr
                      _ -> False

              confirmQuantiles :: StepMeta -> Int -> Bool
              confirmQuantiles m dir =
                if not (ecConfirmQuantiles cfg)
                  then True
                  else
                    case (smQuantile10 m, smQuantile90 m, dir) of
                      (Just q10', _, 1) -> q10' > openThr
                      (_, Just q90', (-1)) -> q90' < negate openThr
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

              gateKalmanDir :: StepMeta -> Double -> Maybe Int -> (Maybe Int, Double)
              gateKalmanDir m confScore dirRaw =
                case dirRaw of
                  Nothing -> (Nothing, 0)
                  Just dir ->
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
                              else if not (confirmConformal m dir)
                                then (Nothing, 0)
                                else if not (confirmQuantiles m dir)
                                  then (Nothing, 0)
                                  else if ecConfidenceSizing cfg && (not confOk || size0 <= 0)
                                    then (Nothing, 0)
                                    else (Just dir, if ecConfidenceSizing cfg then size0 else 1)

              stepFn (posDir, posSize, equity, eqAcc, posAcc, agreeAcc, changes, openTrade, tradesAcc) t =
                let prev = pricesV V.! t
                    nextClose = pricesV V.! (t + 1)
                    hi = barHigh (t + 1)
                    lo = barLow (t + 1)
                    (agreeOk, desiredDirRaw, desiredSizeRaw) =
                      if t < startT
                        then (False, posDir, posSize)
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
                                let d = desiredPosFromDir dir
                                 in (d, if d == 0 then 0 else kalSize)

                              (desiredDir', desiredSize') =
                                case openAgreeDir of
                                  Just dir -> desiredFromOpen dir
                                  Nothing ->
                                    case posDir of
                                      0 -> (0, 0)
                                      1 -> if closeAgreeDir == Just 1 then (1, posSize) else (0, 0)
                                      (-1) -> if closeAgreeDir == Just (-1) then ((-1), posSize) else (0, 0)
                                      _ -> (0, 0)
                           in (openAgreeDir == Just 1 || openAgreeDir == Just (-1), desiredDir', desiredSize')

                    desiredSize =
                      if desiredDirRaw == 0
                        then 0
                        else min 1 (max 0 desiredSizeRaw)

                    (posAfterSwitch, posSizeAfterSwitch, equityAfterSwitch, changes', openTrade', tradesAcc') =
                      if desiredDirRaw /= posDir
                        then
                          let closeTradeAt exitIndex why eqExit ot =
                                Trade
                                  { trEntryIndex = otEntryIndex ot
                                  , trExitIndex = exitIndex
                                  , trEntryEquity = otEntryEquity ot
                                  , trExitEquity = eqExit
                                  , trReturn = eqExit / otEntryEquity ot - 1
                                  , trHoldingPeriods = otHoldingPeriods ot
                                  , trExitReason = Just why
                                  }
                              openTradeFor dir eqEntry =
                                OpenTrade
                                  { otEntryIndex = t
                                  , otEntryEquity = eqEntry
                                  , otHoldingPeriods = 0
                                      , otEntryPrice = prev
                                      , otTrail = prev
                                      , otDir = dir
                                      }
                           in case (posDir, desiredDirRaw, openTrade) of
                                (0, dir, Nothing) | dir /= 0 && desiredSize > 0 ->
                                  let eq1 = applyCost equity desiredSize
                                   in (dir, desiredSize, eq1, changes + 1, Just (openTradeFor dir eq1), tradesAcc)
                                (dir0, 0, Just ot) | dir0 /= 0 ->
                                  let eq1 = applyCost equity posSize
                                      tr = closeTradeAt t "SIGNAL" eq1 ot
                                   in (0, 0, eq1, changes + 1, Nothing, tr : tradesAcc)
                                (dir0, dir1, Just ot) | dir0 /= 0 && dir1 /= 0 && dir0 /= dir1 && desiredSize > 0 ->
                                  let eqExit = applyCost equity posSize
                                      tr = closeTradeAt t "SIGNAL" eqExit ot
                                      eqEntry = applyCost eqExit desiredSize
                                   in (dir1, desiredSize, eqEntry, changes + 1, Just (openTradeFor dir1 eqEntry), tr : tradesAcc)
                                (dir0, dir1, Nothing) | dir0 /= 0 && dir1 /= 0 && dir0 /= dir1 && desiredSize > 0 ->
                                  let eqExit = applyCost equity posSize
                                      eqEntry = applyCost eqExit desiredSize
                                   in (dir1, desiredSize, eqEntry, changes + 1, Just (openTradeFor dir1 eqEntry), tradesAcc)
                                _ ->
                                  let eq1 = applyCost equity posSize
                                   in (desiredDirRaw, posSize, eq1, changes + 1, openTrade, tradesAcc)
                        else (posDir, posSize, equity, changes, openTrade, tradesAcc)

                    equityAtClose =
                      if posAfterSwitch /= 0 && posSizeAfterSwitch > 0
                        then
                          let factor = markToMarket posAfterSwitch prev nextClose
                           in equityAfterSwitch * (1 + posSizeAfterSwitch * (factor - 1))
                        else equityAfterSwitch

                    (posFinal, posSizeFinal, equityFinal, changesFinal, openTradeFinal, tradesFinal) =
                      case (posAfterSwitch, openTrade') of
                        (1, Just ot0) ->
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
                                      (Just slPx, Nothing) -> (Just slPx, Just "STOP_LOSS")
                                      (Nothing, Just tsPx) -> (Just tsPx, Just "TRAILING_STOP")
                                      (Just slPx, Just tsPx) ->
                                        if tsPx > slPx
                                          then (Just tsPx, Just "TRAILING_STOP")
                                          else (Just slPx, Just "STOP_LOSS")
                              tpHit = maybe False (\tpPx -> hi >= tpPx) mTp
                              exitOrTrail =
                                case ecIntrabarFill cfg of
                                  StopFirst ->
                                    let (mStop, stopWhy) = stopPx trail0
                                        stopHit = maybe False (\stPx -> lo <= stPx) mStop
                                     in if stopHit
                                          then (Just (maybe nextClose id mStop, stopWhy), trail0)
                                          else if tpHit
                                            then (Just (maybe nextClose id mTp, Just "TAKE_PROFIT"), trail0)
                                            else (Nothing, max trail0 hi)
                                  TakeProfitFirst ->
                                    if tpHit
                                      then (Just (maybe nextClose id mTp, Just "TAKE_PROFIT"), trail0)
                                      else
                                        let trail1 = max trail0 hi
                                            (mStop, stopWhy) = stopPx trail1
                                            stopHit = maybe False (\stPx -> lo <= stPx) mStop
                                         in if stopHit
                                              then (Just (maybe nextClose id mStop, stopWhy), trail1)
                                              else (Nothing, trail1)
                           in case exitOrTrail of
                                (Just (exitPx, reason), _trailUsed) ->
                                  let factor = markToMarket 1 prev exitPx
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
                                   in (0, 0, exitEq, changes' + 1, Nothing, tr : tradesAcc')
                                (Nothing, trail1) ->
                                  let otCont = otHeld { otTrail = trail1 }
                                   in (1, posSizeAfterSwitch, equityAtClose, changes', Just otCont, tradesAcc')
                        ((-1), Just ot0) ->
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
                                      (Just slPx, Nothing) -> (Just slPx, Just "STOP_LOSS")
                                      (Nothing, Just tsPx) -> (Just tsPx, Just "TRAILING_STOP")
                                      (Just slPx, Just tsPx) ->
                                        if tsPx < slPx
                                          then (Just tsPx, Just "TRAILING_STOP")
                                          else (Just slPx, Just "STOP_LOSS")
                              tpHit = maybe False (\tpPx -> lo <= tpPx) mTp
                              exitOrTrail =
                                case ecIntrabarFill cfg of
                                  StopFirst ->
                                    let (mStop, stopWhy) = stopPx trail0
                                        stopHit = maybe False (\stPx -> hi >= stPx) mStop
                                     in if stopHit
                                          then (Just (maybe nextClose id mStop, stopWhy), trail0)
                                          else if tpHit
                                            then (Just (maybe nextClose id mTp, Just "TAKE_PROFIT"), trail0)
                                            else (Nothing, min trail0 lo)
                                  TakeProfitFirst ->
                                    if tpHit
                                      then (Just (maybe nextClose id mTp, Just "TAKE_PROFIT"), trail0)
                                      else
                                        let trail1 = min trail0 lo
                                            (mStop, stopWhy) = stopPx trail1
                                            stopHit = maybe False (\stPx -> hi >= stPx) mStop
                                         in if stopHit
                                              then (Just (maybe nextClose id mStop, stopWhy), trail1)
                                              else (Nothing, trail1)
                           in case exitOrTrail of
                                (Just (exitPx, reason), _trailUsed) ->
                                  let factor = markToMarket (-1) prev exitPx
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
                                   in (0, 0, exitEq, changes' + 1, Nothing, tr : tradesAcc')
                                (Nothing, trail1) ->
                                  let otCont = otHeld { otTrail = trail1 }
                                   in ((-1), posSizeAfterSwitch, equityAtClose, changes', Just otCont, tradesAcc')
                        _ -> (posAfterSwitch, posSizeAfterSwitch, equityAtClose, changes', openTrade', tradesAcc')
                 in ( posFinal
                    , posSizeFinal
                    , equityFinal
                    , equityFinal : eqAcc
                    , (fromIntegral posAfterSwitch * posSizeAfterSwitch) : posAcc
                    , agreeOk : agreeAcc
                    , changesFinal
                    , openTradeFinal
                    , tradesFinal
                    )

              (_finalPos, finalPosSize, finalEq, eqRev, posRev, agreeRev, changes, openTrade, tradesRev) =
                foldl'
                  stepFn
                  (0 :: Int, 0 :: Double, 1.0, [1.0], [], [], 0 :: Int, Nothing :: Maybe OpenTrade, [])
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
                            , trExitReason = Just "EOD"
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
