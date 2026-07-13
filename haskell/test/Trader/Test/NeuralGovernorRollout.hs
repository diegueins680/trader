module Trader.Test.NeuralGovernorRollout (
    neuralGovernorRolloutSuite,
) where

import Data.Maybe (isJust, isNothing)

import Trader.NeuralGovernor (
    NeuralGovernorConfig (..),
    NeuralGovernorDecision (..),
    NeuralGovernorFeatures (..),
    NeuralGovernorPendingEntry (..),
    NeuralGovernorRolloutMode (..),
    NeuralGovernorState (..),
    defaultNeuralGovernorConfig,
    initNeuralGovernorState,
    neuralGovernorCounterfactualAdvantage,
    neuralGovernorDecide,
    neuralGovernorHoldReason,
    neuralGovernorObserveTrade,
    neuralGovernorOpenBlockReason,
    neuralGovernorSizingMultiplier,
 )

neuralGovernorRolloutSuite :: [(String, IO ())]
neuralGovernorRolloutSuite =
    [ ("shadow mode reports candidates without changing orders", testShadowDoesNotEnforce)
    , ("observe mode trains without producing policy candidates", testObserveOnly)
    , ("enforce mode promotes only after counterfactual evidence", testPromotionGate)
    , ("enforced policy automatically rolls back after underperformance", testAutomaticRollback)
    ]

baseFeatures :: NeuralGovernorFeatures
baseFeatures =
    NeuralGovernorFeatures
        { ngfVolatility = Just 0.2
        , ngfConfidence = Just 0.7
        , ngfTrendProbability = Just 0.7
        , ngfMeanReversionProbability = Just 0.2
        , ngfHighVolProbability = Just 0.1
        , ngfDrawdown = 0
        , ngfLossStreak = 0
        , ngfRollingLoss = Just 0
        , ngfDirection = 1
        , ngfBasePositionSize = 1
        , ngfMarketGovernorMultiplier = 1
        , ngfMarketGovernorBlocked = False
        , ngfSymbolFeature = 0.1
        , ngfMethodFeature = 0.2
        , ngfIntervalFeature = 0.3
        }

testShadowDoesNotEnforce :: IO ()
testShadowDoesNotEnforce = do
    let cfg =
            defaultNeuralGovernorConfig
                { ngcRolloutMode = NeuralGovernorShadow
                , ngcMinTrades = 0
                , ngcOpenScoreFloor = 1
                }
        decision = neuralGovernorDecide cfg (initNeuralGovernorState cfg) baseFeatures
    expectTrue "shadow decision is ready" (ngdReady decision)
    expectTrue "shadow decision is not enforced" (not (ngdEnforced decision))
    expectTrue "shadow preserves the candidate open block" (isJust (ngdOpenBlockReason decision))
    expectTrue "shadow cannot block the actual open" (isNothing (neuralGovernorOpenBlockReason decision))
    expectNear "shadow cannot resize the actual order" 1 (neuralGovernorSizingMultiplier decision)

testObserveOnly :: IO ()
testObserveOnly = do
    let cfg =
            defaultNeuralGovernorConfig
                { ngcRolloutMode = NeuralGovernorObserve
                , ngcMinTrades = 0
                , ngcOpenScoreFloor = 1
                , ngcHoldScoreFloor = -1
                }
        decision = neuralGovernorDecide cfg (initNeuralGovernorState cfg) baseFeatures
    expectTrue "observe decision is ready" (ngdReady decision)
    expectTrue "observe has no candidate open block" (isNothing (ngdOpenBlockReason decision))
    expectTrue "observe has no candidate hold override" (isNothing (ngdHoldReason decision))
    expectTrue "observe cannot affect actual holds" (isNothing (neuralGovernorHoldReason decision))

promotionConfig :: NeuralGovernorConfig
promotionConfig =
    defaultNeuralGovernorConfig
        { ngcRolloutMode = NeuralGovernorEnforce
        , ngcMinTrades = 0
        , ngcOpenScoreFloor = -1
        , ngcHoldScoreFloor = 1
        , ngcMinMultiplier = 1.5
        , ngcMaxMultiplier = 1.5
        , ngcPromotionMinTrades = 2
        , ngcPromotionMinAdvantage = 0.009
        , ngcRollbackMinTrades = 1
        , ngcRollbackAdvantageFloor = -0.02
        }

observe :: NeuralGovernorConfig -> NeuralGovernorState -> Double -> NeuralGovernorState
observe cfg state realizedReturn =
    let decision = neuralGovernorDecide cfg state baseFeatures
        pending = NeuralGovernorPendingEntry baseFeatures decision
     in neuralGovernorObserveTrade cfg state pending realizedReturn

promotedState :: NeuralGovernorState
promotedState =
    let state0 = initNeuralGovernorState promotionConfig
        state1 = observe promotionConfig state0 0.01
     in observe promotionConfig state1 0.01

testPromotionGate :: IO ()
testPromotionGate = do
    let state0 = initNeuralGovernorState promotionConfig
        state1 = observe promotionConfig state0 0.01
        decision1 = neuralGovernorDecide promotionConfig state1 baseFeatures
        decision2 = neuralGovernorDecide promotionConfig promotedState baseFeatures
    expectTrue "one evaluation cannot promote" (not (ngdPromoted decision1) && not (ngdEnforced decision1))
    expectTrue "two positive shadow evaluations promote" (ngdPromoted decision2 && ngdEnforced decision2)
    expectTrue "promotion evidence is retained" (neuralGovernorCounterfactualAdvantage promotedState >= 0.009)
    expectNear "promoted sizing is enforced" 1.5 (neuralGovernorSizingMultiplier decision2)

testAutomaticRollback :: IO ()
testAutomaticRollback = do
    let decisionBefore = neuralGovernorDecide promotionConfig promotedState baseFeatures
        rolledBackState = observe promotionConfig promotedState (-0.1)
        decisionAfter = neuralGovernorDecide promotionConfig rolledBackState baseFeatures
    expectTrue "fixture starts enforced" (ngdEnforced decisionBefore)
    expectTrue "underperformance latches rollback" (ngsRolledBack rolledBackState && ngdRolledBack decisionAfter)
    expectTrue "rolled-back policy is no longer enforced" (not (ngdEnforced decisionAfter))
    expectNear "rollback restores neutral live sizing" 1 (neuralGovernorSizingMultiplier decisionAfter)

expectTrue :: String -> Bool -> IO ()
expectTrue label condition =
    if condition
        then pure ()
        else ioError (userError label)

expectNear :: String -> Double -> Double -> IO ()
expectNear label expected actual =
    expectTrue (label ++ ": expected " ++ show expected ++ ", got " ++ show actual) (abs (expected - actual) <= 1e-12)
