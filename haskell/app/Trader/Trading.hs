module Trader.Trading
  ( Positioning(..)
  , EnsembleConfig(..)
  , IntrabarFill(..)
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
  { ecTradeThreshold :: !Double
  , ecFee :: !Double
  , ecSlippage :: !Double          -- fractional per side, e.g. 0.0002
  , ecSpread :: !Double            -- fractional total spread, e.g. 0.0005 (half per side)
  , ecStopLoss :: !(Maybe Double)      -- fractional, e.g. 0.02
  , ecTakeProfit :: !(Maybe Double)    -- fractional, e.g. 0.03
  , ecTrailingStop :: !(Maybe Double)  -- fractional, e.g. 0.01
  , ecPositioning :: !Positioning
  , ecIntrabarFill :: !IntrabarFill
  } deriving (Eq, Show)

data IntrabarFill
  = StopFirst
  | TakeProfitFirst
  deriving (Eq, Show)

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
  , brPositions :: [Int]          -- length n-1 (position held for return t->t+1)
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
  -> BacktestResult
simulateEnsembleLongFlat cfg lookback prices kalPredNext lstmPredNext =
  simulateEnsembleLongFlatV
    cfg
    lookback
    (V.fromList prices)
    (V.fromList kalPredNext)
    (V.fromList lstmPredNext)

simulateEnsembleLongFlatWithHL
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> [Double]       -- closes length n
  -> [Double]       -- highs length n (aligned to closes; bar i high is for close[i-1]..close[i])
  -> [Double]       -- lows length n
  -> [Double]       -- kalman predicted next prices length n-1 (for t=0..n-2)
  -> [Double]       -- lstm predicted next prices length n-lookback (for t=lookback-1..n-2)
  -> BacktestResult
simulateEnsembleLongFlatWithHL cfg lookback closes highs lows kalPredNext lstmPredNext =
  simulateEnsembleLongFlatVWithHL
    cfg
    lookback
    (V.fromList closes)
    (V.fromList highs)
    (V.fromList lows)
    (V.fromList kalPredNext)
    (V.fromList lstmPredNext)

simulateEnsembleLongFlatV
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> V.Vector Double
  -> V.Vector Double
  -> V.Vector Double
  -> BacktestResult
simulateEnsembleLongFlatV cfg lookback pricesV kalPredNextV lstmPredNextV =
  simulateEnsembleLongFlatVWithHL cfg lookback pricesV pricesV pricesV kalPredNextV lstmPredNextV

simulateEnsembleLongFlatVWithHL
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> V.Vector Double -- closes
  -> V.Vector Double -- highs
  -> V.Vector Double -- lows
  -> V.Vector Double
  -> V.Vector Double
  -> BacktestResult
simulateEnsembleLongFlatVWithHL cfg lookback pricesV highsV lowsV kalPredNextV lstmPredNextV =
  let n = V.length pricesV
   in if n < 2
        then error "Need at least 2 prices to simulate"
        else
          let startT = max 0 (lookback - 1)
              thr = max 0 (ecTradeThreshold cfg)
              direction prev pred =
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
                  else if V.length kalPredNextV < kalNeed
                  then
                    error
                      ( "kalPredNext too short: need at least "
                          ++ show kalNeed
                          ++ ", got "
                          ++ show (V.length kalPredNextV)
                      )
                  else if V.length lstmPredNextV < lstmNeed
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

              applyCost eq = eq * (1 - perSideCost)
              applyTwoCosts eq = applyCost (applyCost eq)

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

              stepFn (pos, equity, eqAcc, posAcc, agreeAcc, changes, openTrade, tradesAcc) t =
                let prev = pricesV V.! t
                    nextClose = pricesV V.! (t + 1)
                    hi = barHigh (t + 1)
                    lo = barLow (t + 1)
                    (agreeOk, desiredPos) =
                      if t < startT
                        then (False, pos)
                        else
                          let kp = kalPredNextV V.! t
                              lp = lstmPredNextV V.! (t - startT)
                              kalDir = direction prev kp
                              lstmDir = direction prev lp
                              agreeDir =
                                if kalDir == lstmDir
                                  then kalDir
                                  else Nothing
                           in case agreeDir of
                                Just 1 -> (True, desiredPosFromDir 1)
                                Just (-1) -> (True, desiredPosFromDir (-1))
                                _ -> (False, pos)

                    (posAfterSwitch, equityAfterSwitch, changes', openTrade', tradesAcc') =
                      if desiredPos /= pos
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
                           in case (pos, desiredPos, openTrade) of
                                (0, dir, Nothing) | dir /= 0 ->
                                  let eq1 = applyCost equity
                                   in (dir, eq1, changes + 1, Just (openTradeFor dir eq1), tradesAcc)
                                (dir0, 0, Just ot) | dir0 /= 0 ->
                                  let eq1 = applyCost equity
                                      tr = closeTradeAt t "SIGNAL" eq1 ot
                                   in (0, eq1, changes + 1, Nothing, tr : tradesAcc)
                                (dir0, dir1, Just ot) | dir0 /= 0 && dir1 /= 0 && dir0 /= dir1 ->
                                  let eqExit = applyCost equity
                                      tr = closeTradeAt t "SIGNAL" eqExit ot
                                      eqEntry = applyCost eqExit
                                   in (dir1, eqEntry, changes + 1, Just (openTradeFor dir1 eqEntry), tr : tradesAcc)
                                (dir0, dir1, Nothing) | dir0 /= 0 && dir1 /= 0 && dir0 /= dir1 ->
                                  let eqExit = applyCost equity
                                      eqEntry = applyCost eqExit
                                   in (dir1, eqEntry, changes + 1, Just (openTradeFor dir1 eqEntry), tradesAcc)
                                _ ->
                                  let eq1 = applyCost equity
                                   in (desiredPos, eq1, changes + 1, openTrade, tradesAcc)
                        else (pos, equity, changes, openTrade, tradesAcc)

                    equityAtClose =
                      if posAfterSwitch /= 0
                        then equityAfterSwitch * markToMarket posAfterSwitch prev nextClose
                        else equityAfterSwitch

                    (posFinal, equityFinal, changesFinal, openTradeFinal, tradesFinal) =
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
                                  let exitEq0 = equityAfterSwitch * markToMarket 1 prev exitPx
                                      exitEq = applyCost exitEq0
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
                                   in (0, exitEq, changes' + 1, Nothing, tr : tradesAcc')
                                (Nothing, trail1) ->
                                  let otCont = otHeld { otTrail = trail1 }
                                   in (1, equityAtClose, changes', Just otCont, tradesAcc')
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
                                  let exitEq0 = equityAfterSwitch * markToMarket (-1) prev exitPx
                                      exitEq = applyCost exitEq0
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
                                   in (0, exitEq, changes' + 1, Nothing, tr : tradesAcc')
                                (Nothing, trail1) ->
                                  let otCont = otHeld { otTrail = trail1 }
                                   in ((-1), equityAtClose, changes', Just otCont, tradesAcc')
                        _ -> (posAfterSwitch, equityAtClose, changes', openTrade', tradesAcc')
                 in ( posFinal
                    , equityFinal
                    , equityFinal : eqAcc
                    , posAfterSwitch : posAcc
                    , agreeOk : agreeAcc
                    , changesFinal
                    , openTradeFinal
                    , tradesFinal
                    )

              (_finalPos, finalEq, eqRev, posRev, agreeRev, changes, openTrade, tradesRev) =
                foldl'
                  stepFn
                  (0 :: Int, 1.0, [1.0], [], [], 0 :: Int, Nothing :: Maybe OpenTrade, [])
                  [0 .. stepCount - 1]

              eqCurve0 = reverse eqRev
              (eqCurve, tradesRev') =
                case openTrade of
                  Nothing -> (eqCurve0, tradesRev)
                  Just ot ->
                    let exitEq = applyCost finalEq
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
                        eqCurve1 =
                          case eqCurve0 of
                            [] -> []
                            _ -> take (length eqCurve0 - 1) eqCurve0 ++ [exitEq]
                     in (eqCurve1, tr : tradesRev)
           in BacktestResult
                { brEquityCurve = eqCurve
                , brPositions = reverse posRev
                , brAgreementOk = reverse agreeRev
                , brPositionChanges = changes
                , brTrades = reverse tradesRev'
                }
