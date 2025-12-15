module Trader.Trading
  ( EnsembleConfig(..)
  , Trade(..)
  , BacktestResult(..)
  , simulateEnsembleLongFlat
  , simulateEnsembleLongFlatV
  ) where

import Data.List (foldl')
import qualified Data.Vector as V

data EnsembleConfig = EnsembleConfig
  { ecTradeThreshold :: !Double
  , ecFee :: !Double
  , ecStopLoss :: !(Maybe Double)      -- fractional, e.g. 0.02
  , ecTakeProfit :: !(Maybe Double)    -- fractional, e.g. 0.03
  , ecTrailingStop :: !(Maybe Double)  -- fractional, e.g. 0.01
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

simulateEnsembleLongFlatV
  :: EnsembleConfig
  -> Int            -- lookback (for LSTM alignment)
  -> V.Vector Double
  -> V.Vector Double
  -> V.Vector Double
  -> BacktestResult
simulateEnsembleLongFlatV cfg lookback pricesV kalPredNextV lstmPredNextV =
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
              stepCount = n - 1

              kalNeed = if startT < stepCount then stepCount else 0
              lstmNeed = max 0 (stepCount - startT)

              () =
                if V.length kalPredNextV < kalNeed
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

              stepFn (pos, equity, eqAcc, posAcc, agreeAcc, changes, openTrade, tradesAcc) t =
                let prev = pricesV V.! t
                    nextP = pricesV V.! (t + 1)
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
                                Just 1 -> (True, 1)
                                Just (-1) -> (True, 0)
                                _ -> (False, pos)

                    (posAfterSwitch, equityAfterSwitch, changes', openTrade', tradesAcc') =
                      if desiredPos /= pos
                        then
                          let equityFee = equity * (1 - ecFee cfg)
                           in case (pos, desiredPos, openTrade) of
                                (0, 1, Nothing) ->
                                  ( 1
                                  , equityFee
                                  , changes + 1
                                  , Just (t, equityFee, 0, prev, prev)
                                  , tradesAcc
                                  )
                                (1, 0, Just (entryT, entryEq, hold, entryPx, trailHigh)) ->
                                  let exitEq = equityFee
                                      tr = Trade
                                            { trEntryIndex = entryT
                                            , trExitIndex = t
                                            , trEntryEquity = entryEq
                                            , trExitEquity = exitEq
                                            , trReturn = exitEq / entryEq - 1
                                            , trHoldingPeriods = hold
                                            , trExitReason = Just "SIGNAL"
                                            }
                                   in ( 0
                                      , exitEq
                                      , changes + 1
                                      , Nothing
                                      , tr : tradesAcc
                                      )
                                _ ->
                                  (desiredPos, equityFee, changes + 1, openTrade, tradesAcc)
                        else (pos, equity, changes, openTrade, tradesAcc)

                    (equityNext, openTradeNext) =
                      if posAfterSwitch == 1
                        then
                          let retFactor = if prev == 0 then 1 else nextP / prev
                              eq' = equityAfterSwitch * retFactor
                              openTradeNext' =
                                case openTrade' of
                                  Just (entryT, entryEq, hold, entryPx, trailHigh) ->
                                    let trailHigh' = max trailHigh nextP
                                     in Just (entryT, entryEq, hold + 1, entryPx, trailHigh')
                                  Nothing -> Nothing
                           in (eq', openTradeNext')
                        else (equityAfterSwitch, openTrade')

                    (posFinal, equityFinal, changesFinal, openTradeFinal, tradesFinal) =
                      case (posAfterSwitch, openTradeNext) of
                        (1, Just (entryT, entryEq, hold, entryPx, trailHigh)) ->
                          let mTp =
                                case ecTakeProfit cfg of
                                  Just tp | tp > 0 -> Just (entryPx * (1 + tp))
                                  _ -> Nothing
                              mSl =
                                case ecStopLoss cfg of
                                  Just sl | sl > 0 -> Just (entryPx * (1 - sl))
                                  _ -> Nothing
                              mTs =
                                case ecTrailingStop cfg of
                                  Just ts | ts > 0 -> Just (trailHigh * (1 - ts))
                                  _ -> Nothing
                              (mStop, stopWhy) =
                                case (mSl, mTs) of
                                  (Nothing, Nothing) -> (Nothing, Nothing)
                                  (Just slPx, Nothing) -> (Just slPx, Just "STOP_LOSS")
                                  (Nothing, Just tsPx) -> (Just tsPx, Just "TRAILING_STOP")
                                  (Just slPx, Just tsPx) ->
                                    if tsPx > slPx
                                      then (Just tsPx, Just "TRAILING_STOP")
                                      else (Just slPx, Just "STOP_LOSS")
                              tpHit = maybe False (\tpPx -> nextP >= tpPx) mTp
                              stopHit = maybe False (\stPx -> nextP <= stPx) mStop
                           in if tpHit || stopHit
                                then
                                  let reason = if tpHit then Just "TAKE_PROFIT" else stopWhy
                                      exitEq = equityNext * (1 - ecFee cfg)
                                      tr = Trade
                                            { trEntryIndex = entryT
                                            , trExitIndex = t + 1
                                            , trEntryEquity = entryEq
                                            , trExitEquity = exitEq
                                            , trReturn = exitEq / entryEq - 1
                                            , trHoldingPeriods = hold
                                            , trExitReason = reason
                                            }
                                   in (0, exitEq, changes' + 1, Nothing, tr : tradesAcc')
                                else (1, equityNext, changes', openTradeNext, tradesAcc')
                        _ -> (posAfterSwitch, equityNext, changes', openTradeNext, tradesAcc')
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
                  (0 :: Int, 1.0, [1.0], [], [], 0 :: Int, Nothing :: Maybe (Int, Double, Int, Double, Double), [])
                  [0 .. stepCount - 1]

              -- Close any open trade at end (no extra fee, mark-to-market already reflected in equity curve).
              tradesRev' =
                case openTrade of
                  Nothing -> tradesRev
                  Just (entryT, entryEq, hold, _entryPx, _trailHigh) ->
                    let exitEq = finalEq
                        tr =
                          Trade
                            { trEntryIndex = entryT
                            , trExitIndex = stepCount
                            , trEntryEquity = entryEq
                            , trExitEquity = exitEq
                            , trReturn = exitEq / entryEq - 1
                            , trHoldingPeriods = hold
                            , trExitReason = Just "EOD"
                            }
                     in tr : tradesRev
           in BacktestResult
                { brEquityCurve = reverse eqRev
                , brPositions = reverse posRev
                , brAgreementOk = reverse agreeRev
                , brPositionChanges = changes
                , brTrades = reverse tradesRev'
                }
