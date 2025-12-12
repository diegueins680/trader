module Trader.Trading
  ( EnsembleConfig(..)
  , Trade(..)
  , BacktestResult(..)
  , simulateEnsembleLongFlat
  ) where

import Data.List (foldl')

data EnsembleConfig = EnsembleConfig
  { ecTradeThreshold :: !Double
  , ecAgreementThreshold :: !Double -- fraction of prev price
  , ecFee :: !Double
  } deriving (Eq, Show)

data Trade = Trade
  { trEntryIndex :: !Int
  , trExitIndex :: !Int
  , trEntryEquity :: !Double
  , trExitEquity :: !Double
  , trReturn :: !Double
  , trHoldingPeriods :: !Int
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
  let n = length prices
   in if n < 2
        then error "Need at least 2 prices to simulate"
        else
          let startT = max 0 (lookback - 1)
              stepCount = n - 1
              stepFn (t, pos, equity, eqAcc, posAcc, agreeAcc, changes, openTrade, tradesAcc) =
                if t >= stepCount
                  then (t, pos, equity, eqAcc, posAcc, agreeAcc, changes, openTrade, tradesAcc)
                  else
                    let prev = prices !! t
                        nextP = prices !! (t + 1)
                        (agreeOk, desiredPos, ensemblePred) =
                          if t < startT
                            then (False, pos, 0.0)
                            else
                              let kp = kalPredNext !! t
                                  lp = lstmPredNext !! (t - startT)
                                  ok = abs (kp - lp) <= ecAgreementThreshold cfg * prev
                                  ens = 0.5 * (kp + lp)
                                  desired = if ok && ens > prev * (1 + ecTradeThreshold cfg) then 1 else 0
                                  desired' = if ok then desired else pos
                               in (ok, desired', ens)

                        (posAfterSwitch, equityAfterSwitch, changes', openTrade', tradesAcc') =
                          if desiredPos /= pos
                            then
                              let equityFee = equity * (1 - ecFee cfg)
                               in case (pos, desiredPos, openTrade) of
                                    (0, 1, Nothing) ->
                                      ( 1
                                      , equityFee
                                      , changes + 1
                                      , Just (t, equityFee, 0)
                                      , tradesAcc
                                      )
                                    (1, 0, Just (entryT, entryEq, hold)) ->
                                      let exitEq = equityFee
                                          tr = Trade
                                                { trEntryIndex = entryT
                                                , trExitIndex = t
                                                , trEntryEquity = entryEq
                                                , trExitEquity = exitEq
                                                , trReturn = exitEq / entryEq - 1
                                                , trHoldingPeriods = hold
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
                              let eq' = equityAfterSwitch * (nextP / prev)
                                  openTradeNext' =
                                    case openTrade' of
                                      Just (entryT, entryEq, hold) -> Just (entryT, entryEq, hold + 1)
                                      Nothing -> Nothing
                               in (eq', openTradeNext')
                            else (equityAfterSwitch, openTrade')
                     in ( t + 1
                        , posAfterSwitch
                        , equityNext
                        , equityNext : eqAcc
                        , posAfterSwitch : posAcc
                        , agreeOk : agreeAcc
                        , changes'
                        , openTradeNext
                        , tradesAcc'
                        )

              -- Fold over steps, building reversed accumulators.
              (_, finalPos, finalEq, eqRev, posRev, agreeRev, changes, openTrade, tradesRev) =
                foldl'
                  (\st _ -> stepFn st)
                  (0, 0 :: Int, 1.0, [1.0], [], [], 0 :: Int, Nothing :: Maybe (Int, Double, Int), [])
                  [1..stepCount]

              -- Close any open trade at end (no extra fee, mark-to-market already reflected in equity curve).
              tradesRev' =
                case openTrade of
                  Nothing -> tradesRev
                  Just (entryT, entryEq, hold) ->
                    let exitEq = finalEq
                        tr = Trade
                              { trEntryIndex = entryT
                              , trExitIndex = stepCount
                              , trEntryEquity = entryEq
                              , trExitEquity = exitEq
                              , trReturn = exitEq / entryEq - 1
                              , trHoldingPeriods = hold
                              }
                     in tr : tradesRev
           in BacktestResult
                { brEquityCurve = reverse eqRev
                , brPositions = reverse posRev
                , brAgreementOk = reverse agreeRev
                , brPositionChanges = changes
                , brTrades = reverse tradesRev'
                }
