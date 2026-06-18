module Trader.Test.OnlineNeural (runOnlineNeuralTests) where

import Control.Monad (unless)
import qualified Data.Vector as V

import Trader.Method (Method (..), methodCode, parseMethod)
import Trader.Predictors.Features (FeatureInputs, featureInputsFromClose, withCoinbaseInputs, withExogenousInputs)
import Trader.Predictors.OnlineNeural (
    OnlineNeuralConfig (..),
    OnlineNeuralSignal (..),
    defaultOnlineNeuralConfig,
    onlineNeuralPredictionAt,
    onlineNeuralPredictionsRange,
 )

runOnlineNeuralTests :: IO ()
runOnlineNeuralTests = do
    testOnlineNeuralMethodParsing
    testOnlineNeuralPredictionIsFinite
    testOnlineNeuralTrainingExamplesGrow
    testOnlineNeuralPrefixInvariant

assert :: String -> Bool -> IO ()
assert message condition =
    unless condition (ioError (userError ("Assertion failed: " ++ message)))

testOnlineNeuralConfig :: OnlineNeuralConfig
testOnlineNeuralConfig =
    (defaultOnlineNeuralConfig 12 8 0.01 2 17)
        { oncTargetClamp = 0.08
        , oncGradientClip = 2.0
        }

testPrices :: V.Vector Double
testPrices =
    V.fromList
        [ 100 + fromIntegral i * 0.18 + sin (fromIntegral i / 6) * 1.25
        | i <- [0 .. 119 :: Int]
        ]

testInputsFromPrices :: V.Vector Double -> FeatureInputs
testInputsFromPrices prices =
    let n = V.length prices
        funding = V.generate n (\i -> if even i then 0.0001 else -0.00005)
        openInterest = V.generate n (\i -> 1000 + fromIntegral i * 3)
        basis = V.generate n (\i -> 0.0002 * sin (fromIntegral i / 9))
        takerFlow = V.generate n (\i -> 0.05 * cos (fromIntegral i / 7))
        coinbase = V.imap (\i p -> p * (1 + 0.0001 * sin (fromIntegral i / 4))) prices
     in withCoinbaseInputs (Just coinbase) $
            withExogenousInputs
                (Just funding)
                (Just openInterest)
                (Just basis)
                (Just takerFlow)
                (featureInputsFromClose prices)

testOnlineNeuralMethodParsing :: IO ()
testOnlineNeuralMethodParsing = do
    assert "online_nn parses" (parseMethod "online_nn" == Right MethodOnlineNeural)
    assert "online_neural alias parses" (parseMethod "online_neural" == Right MethodOnlineNeural)
    assert "methodCode for online neural is stable" (methodCode MethodOnlineNeural == "online_nn")

testOnlineNeuralPredictionIsFinite :: IO ()
testOnlineNeuralPredictionIsFinite = do
    let preds = onlineNeuralPredictionsRange testOnlineNeuralConfig Nothing (testInputsFromPrices testPrices) 40 25
    assert "online neural emits requested prediction count" (V.length preds == 25)
    assert "online neural predictions stay finite" (V.all isFiniteDouble preds)

testOnlineNeuralTrainingExamplesGrow :: IO ()
testOnlineNeuralTrainingExamplesGrow = do
    let inputs = testInputsFromPrices testPrices
        early = onlineNeuralPredictionAt testOnlineNeuralConfig Nothing inputs 40
        late = onlineNeuralPredictionAt testOnlineNeuralConfig Nothing inputs 80
    assert
        "online neural returns signals once enough history exists"
        ( case (early, late) of
            (Just e, Just l) -> isFiniteDouble (onsPrediction e) && isFiniteDouble (onsPrediction l)
            _ -> False
        )
    assert
        "online neural trains on more examples as later bars become available"
        ( case (early, late) of
            (Just e, Just l) -> onsExamples l > onsExamples e
            _ -> False
        )

testOnlineNeuralPrefixInvariant :: IO ()
testOnlineNeuralPrefixInvariant = do
    let t = 70
        fullSignal = onlineNeuralPredictionAt testOnlineNeuralConfig Nothing (testInputsFromPrices testPrices) t
        prefixPrices = V.take (t + 1) testPrices
        prefixSignal = onlineNeuralPredictionAt testOnlineNeuralConfig Nothing (testInputsFromPrices prefixPrices) t
    assert
        "online neural prediction at t is invariant to bars after t"
        ( case (fullSignal, prefixSignal) of
            (Just full, Just prefix) ->
                approxEqual (onsPrediction full) (onsPrediction prefix)
                    && approxEqual (onsReturn full) (onsReturn prefix)
                    && onsExamples full == onsExamples prefix
            _ -> False
        )

approxEqual :: Double -> Double -> Bool
approxEqual a b = abs (a - b) <= 1e-9

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)
