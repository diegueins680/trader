module Trader.Predictors.DecisionTree (
    DecisionTree (..),
    DecisionTreeModel (..),
    trainDecisionTree,
    predictDecisionTree,
) where

import Data.List (minimumBy, sort)
import Data.Ord (comparing)

import Trader.Text (dedupeStable)

data DecisionTree
    = DecisionLeaf
        { dtValue :: !Double
        , dtSigma :: !(Maybe Double)
        , dtCount :: !Int
        }
    | DecisionNode
        { dtFeature :: !Int
        , dtThreshold :: !Double
        , dtLeft :: !DecisionTree
        , dtRight :: !DecisionTree
        }
    deriving (Eq, Show)

data DecisionTreeModel = DecisionTreeModel
    { dmFeatureDim :: !Int
    , dmMaxDepth :: !Int
    , dmMinLeafSize :: !Int
    , dmRoot :: !(Maybe DecisionTree)
    , dmSigmaBase :: !(Maybe Double)
    }
    deriving (Eq, Show)

trainDecisionTree :: Int -> Int -> [([Double], Double)] -> DecisionTreeModel
trainDecisionTree maxDepth minLeafSize dataset
    | maxDepth <= 0 = emptyDecisionTreeModel
    | minLeafSize <= 0 = emptyDecisionTreeModel
    | null dataset = emptyDecisionTreeModel
    | otherwise =
        let ds = sanitizeDataset dataset
            featureDim = featureDimFromDataset ds
         in if featureDim <= 0 || null ds
                then emptyDecisionTreeModel
                else
                    let sigmaBase = sigmaFromTargets (map snd ds)
                        root = buildTree sigmaBase maxDepth minLeafSize ds
                     in DecisionTreeModel
                            { dmFeatureDim = featureDim
                            , dmMaxDepth = maxDepth
                            , dmMinLeafSize = minLeafSize
                            , dmRoot = Just root
                            , dmSigmaBase = sigmaBase
                            }

predictDecisionTree :: DecisionTreeModel -> [Double] -> (Double, Maybe Double)
predictDecisionTree model feats =
    let expected = dmFeatureDim model
        queryOk = expected > 0 && length feats == expected && all isFiniteDouble feats
     in if not queryOk
            then (0, dmSigmaBase model)
            else case dmRoot model of
                Nothing -> (0, dmSigmaBase model)
                Just root ->
                    let (mu, mSigma) = go root
                     in if isFiniteDouble mu
                            then (mu, mSigma <|> dmSigmaBase model)
                            else (0, dmSigmaBase model)
  where
    go tree =
        case tree of
            DecisionLeaf value sigma _count -> (value, sigma)
            DecisionNode feature threshold left right ->
                if feature < 0 || feature >= length feats
                    then (0, dmSigmaBase model)
                    else
                        let x = feats !! feature
                         in if x <= threshold then go left else go right

    (<|>) left right =
        case left of
            Just _ -> left
            Nothing -> right

emptyDecisionTreeModel :: DecisionTreeModel
emptyDecisionTreeModel =
    DecisionTreeModel
        { dmFeatureDim = 0
        , dmMaxDepth = 0
        , dmMinLeafSize = 0
        , dmRoot = Nothing
        , dmSigmaBase = Nothing
        }

sanitizeDataset :: [([Double], Double)] -> [([Double], Double)]
sanitizeDataset dataset =
    let finiteRows =
            [ (x, y)
            | (x, y) <- dataset
            , not (null x)
            , all isFiniteDouble x
            , isFiniteDouble y
            ]
     in case finiteRows of
            [] -> []
            ((x0, _) : _) ->
                let featureDim = length x0
                 in filter ((== featureDim) . length . fst) finiteRows

featureDimFromDataset :: [([Double], Double)] -> Int
featureDimFromDataset dataset =
    case map (length . fst) dataset of
        [] -> 0
        d : ds
            | d <= 0 -> 0
            | any (/= d) ds -> 0
            | otherwise -> d

buildTree :: Maybe Double -> Int -> Int -> [([Double], Double)] -> DecisionTree
buildTree sigmaBase depthLeft minLeafSize rows =
    let leaf = mkLeaf sigmaBase rows
        targets = map snd rows
        currentSse = sumSqErr (mean targets) targets
     in if depthLeft <= 0 || length rows < 2 * minLeafSize || currentSse <= 1e-12
            then leaf
            else case bestSplit minLeafSize rows of
                Nothing -> leaf
                Just (_, _, _leftRows, _rightRows, bestSse)
                    | bestSse >= currentSse - 1e-9 -> leaf
                Just (feature, threshold, leftRows, rightRows, _bestSse) ->
                    DecisionNode
                        { dtFeature = feature
                        , dtThreshold = threshold
                        , dtLeft = buildTree sigmaBase (depthLeft - 1) minLeafSize leftRows
                        , dtRight = buildTree sigmaBase (depthLeft - 1) minLeafSize rightRows
                        }

mkLeaf :: Maybe Double -> [([Double], Double)] -> DecisionTree
mkLeaf sigmaBase rows =
    let ys = map snd rows
        value = mean ys
        sigma = sigmaFromTargets ys <|> sigmaBase
     in DecisionLeaf
            { dtValue = if isFiniteDouble value then value else 0
            , dtSigma = sigma
            , dtCount = length rows
            }
  where
    (<|>) left right =
        case left of
            Just _ -> left
            Nothing -> right

bestSplit :: Int -> [([Double], Double)] -> Maybe (Int, Double, [([Double], Double)], [([Double], Double)], Double)
bestSplit minLeafSize rows =
    let featureDim = featureDimFromDataset rows
        candidates =
            [ bestForFeature feature
            | feature <- [0 .. featureDim - 1]
            ]
        valids =
            [ split
            | Just split <- candidates
            ]
     in case valids of
            [] -> Nothing
            _ -> Just (minimumBy (comparing (\(_, _, _, _, sse) -> sse)) valids)
  where
    bestForFeature feature =
        let xs = [x !! feature | (x, _y) <- rows]
            thresholds = candidateThresholds xs
            options =
                [ let (leftRows, rightRows) = partitionRows feature threshold rows
                      leftTargets = map snd leftRows
                      rightTargets = map snd rightRows
                      score = sumSqErr (mean leftTargets) leftTargets + sumSqErr (mean rightTargets) rightTargets
                   in if length leftRows < minLeafSize || length rightRows < minLeafSize
                        then Nothing
                        else Just (feature, threshold, leftRows, rightRows, score)
                | threshold <- thresholds
                ]
         in case [v | Just v <- options] of
                [] -> Nothing
                vs -> Just (minimumBy (comparing (\(_, _, _, _, sse) -> sse)) vs)

partitionRows :: Int -> Double -> [([Double], Double)] -> ([([Double], Double)], [([Double], Double)])
partitionRows feature threshold =
    foldr
        ( \(x, y) (leftRows, rightRows) ->
            if x !! feature <= threshold
                then ((x, y) : leftRows, rightRows)
                else (leftRows, (x, y) : rightRows)
        )
        ([], [])

candidateThresholds :: [Double] -> [Double]
candidateThresholds xs =
    let s = sort xs
        n = length s
        buckets = 16 :: Int
        atDef i ys fallback =
            case drop i ys of
                v : _ -> v
                [] -> fallback
     in if n < 3
            then []
            else
                let fallback =
                        case s of
                            v : _ -> v
                            [] -> 0
                    ix q =
                        max
                            1
                            (min (n - 2) (floor (fromIntegral q * fromIntegral n / fromIntegral buckets)))
                    rawThresholds = [atDef (ix q) s fallback | q <- [1 .. buckets - 2]]
                 in dedupeStable rawThresholds

sigmaFromTargets :: [Double] -> Maybe Double
sigmaFromTargets ys =
    case ys of
        [] -> Nothing
        [_] -> Just 1e-6
        _ ->
            let mu = mean ys
                var = sum (map (\y -> (y - mu) * (y - mu)) ys) / fromIntegral (length ys - 1)
                sigma = sqrt (max 0 var + 1e-12)
             in if isFiniteDouble sigma then Just sigma else Nothing

sumSqErr :: Double -> [Double] -> Double
sumSqErr mu ys = sum (map (\y -> let e = y - mu in e * e) ys)

mean :: [Double] -> Double
mean xs =
    case xs of
        [] -> 0
        _ -> sum xs / fromIntegral (length xs)

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)
