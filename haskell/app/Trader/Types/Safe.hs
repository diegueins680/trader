module Trader.Types.Safe (
    NonNegative,
    FinitePositive,
    Quantity,
    Leverage,
    sanitizeNonNegative,
    sanitizeFinitePositive,
    quantityFromDouble,
    leverageFromDouble,
    fromQuantity,
    fromLeverage,
) where

{- | A double that is guaranteed to be non-negative and finite.
Use 'sanitizeNonNegative' to construct from raw 'Double' input.
-}
newtype NonNegative = NonNegative {unNonNegative :: Double}
    deriving (Eq, Ord, Show)

-- | A double that is guaranteed to be strictly positive and finite.
newtype FinitePositive = FinitePositive {unFinitePositive :: Double}
    deriving (Eq, Ord, Show)

-- | A quantity of an asset. Always non-negative and finite.
newtype Quantity = Quantity {unQuantity :: NonNegative}
    deriving (Eq, Ord, Show)

-- | Leverage multiplier. Always strictly positive and finite.
newtype Leverage = Leverage {unLeverage :: FinitePositive}
    deriving (Eq, Ord, Show)

-- | Construct a quantity while preserving its finite, non-negative invariant.
quantityFromDouble :: Double -> Quantity
quantityFromDouble = Quantity . sanitizeNonNegative

-- | Construct leverage while preserving its finite, strictly-positive invariant.
leverageFromDouble :: Double -> Leverage
leverageFromDouble = Leverage . sanitizeFinitePositive

fromQuantity :: Quantity -> Double
fromQuantity = unNonNegative . unQuantity

fromLeverage :: Leverage -> Double
fromLeverage = unFinitePositive . unLeverage

sanitizeNonNegative :: Double -> NonNegative
sanitizeNonNegative x
    | isNaN x || isInfinite x || x < 0 = NonNegative 0
    | otherwise = NonNegative x

sanitizeFinitePositive :: Double -> FinitePositive
sanitizeFinitePositive x
    | isNaN x || isInfinite x || x <= 0 = FinitePositive 1e-12
    | otherwise = FinitePositive x
