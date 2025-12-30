module Trader.Optimizer.Random
  ( Rng
  , seedRng
  , nextDouble
  , nextIntRange
  , nextChoice
  , nextUniform
  , nextLogUniform
  , nextMaybe
  ) where

import Control.Monad (forM_)
import Control.Monad.ST (runST)
import Data.Bits (Bits (..), countLeadingZeros, finiteBitSize)
import Data.Vector.Unboxed (Vector)
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import Data.Word (Word32, Word64)

mtN :: Int
mtN = 624

mtM :: Int
mtM = 397

matrixA :: Word32
matrixA = 0x9908b0df

upperMask :: Word32
upperMask = 0x80000000

lowerMask :: Word32
lowerMask = 0x7fffffff

newtype Rng = Rng
  { rngState :: (Int, Vector Word32)
  }

seedRng :: Int -> Rng
seedRng seed =
  let st = initState (fromIntegral seed)
   in Rng (mtN, st)

initState :: Word32 -> Vector Word32
initState seed =
  runST $ do
    mv <- MV.new mtN
    MV.write mv 0 seed
    forM_ [1 .. mtN - 1] $ \i -> do
      prev <- MV.read mv (i - 1)
      let x = 1812433253 * (prev `xor` (prev `shiftR` 30)) + fromIntegral i
      MV.write mv i x
    V.unsafeFreeze mv

nextWord32 :: Rng -> (Word32, Rng)
nextWord32 (Rng (idx, st)) =
  let (st', idx') =
        if idx >= mtN
          then (twist st, 0)
          else (st, idx)
      y0 = st' V.! idx'
      y = temper y0
   in (y, Rng (idx' + 1, st'))

nextDouble :: Rng -> (Double, Rng)
nextDouble rng0 =
  let (a, rng1) = nextWord32 rng0
      (b, rng2) = nextWord32 rng1
      x = fromIntegral (a `shiftR` 5) * 67108864.0 + fromIntegral (b `shiftR` 6)
      d = x / 9007199254740992.0
   in (d, rng2)

nextUniform :: Double -> Double -> Rng -> (Double, Rng)
nextUniform lo hi rng =
  let (r, rng') = nextDouble rng
   in (lo + (hi - lo) * r, rng')

nextLogUniform :: Double -> Double -> Rng -> (Double, Rng)
nextLogUniform lo hi rng
  | lo <= 0 || hi <= 0 || hi < lo =
      error "log_uniform requires 0 < lo <= hi"
  | lo == hi = (lo, rng)
  | otherwise =
      let (r, rng') = nextUniform (log lo) (log hi) rng
       in (exp r, rng')

nextMaybe :: Double -> (Rng -> (a, Rng)) -> Rng -> (Maybe a, Rng)
nextMaybe pNone sampler rng
  | pNone <= 0 =
      let (v, rng') = sampler rng
       in (Just v, rng')
  | pNone >= 1 = (Nothing, rng)
  | otherwise =
      let (r, rng') = nextDouble rng
       in if r < pNone
            then (Nothing, rng')
            else
              let (v, rng'') = sampler rng'
               in (Just v, rng'')

nextIntRange :: Int -> Int -> Rng -> (Int, Rng)
nextIntRange lo hi rng
  | hi < lo = nextIntRange hi lo rng
  | otherwise =
      let spanN = fromIntegral (hi - lo + 1) :: Word64
          (r, rng') = randBelow spanN rng
       in (lo + fromIntegral r, rng')

nextChoice :: [a] -> Rng -> (Maybe a, Rng)
nextChoice xs rng =
  case xs of
    [] -> (Nothing, rng)
    _ ->
      let (idx, rng') = nextIntRange 0 (length xs - 1) rng
       in (Just (xs !! idx), rng')

randBelow :: Word64 -> Rng -> (Word64, Rng)
randBelow n rng
  | n <= 0 = error "randBelow: n must be > 0"
  | otherwise =
      let k = bitLength n
       in go k rng
  where
    go k rng0 =
      let (r, rng1) = getRandBits k rng0
       in if r < n
            then (r, rng1)
            else go k rng1

bitLength :: Word64 -> Int
bitLength 0 = 0
bitLength n = finiteBitSize n - countLeadingZeros n

getRandBits :: Int -> Rng -> (Word64, Rng)
getRandBits k rng
  | k <= 0 = (0, rng)
  | k <= 32 =
      let (w, rng') = nextWord32 rng
          r = fromIntegral (w `shiftR` (32 - k))
       in (r, rng')
  | otherwise =
      let (w, rng1) = nextWord32 rng
          (rest, rng2) = getRandBits (k - 32) rng1
          r = fromIntegral w .|. (rest `shiftL` 32)
       in (r, rng2)

temper :: Word32 -> Word32
temper y0 =
  let y1 = y0 `xor` (y0 `shiftR` 11)
      y2 = y1 `xor` ((y1 `shiftL` 7) .&. 0x9d2c5680)
      y3 = y2 `xor` ((y2 `shiftL` 15) .&. 0xefc60000)
   in y3 `xor` (y3 `shiftR` 18)

twist :: Vector Word32 -> Vector Word32
twist v =
  runST $ do
    mv <- MV.new mtN
    forM_ [0 .. mtN - 1] $ \i -> do
      let i1 = if i + 1 < mtN then i + 1 else 0
          j = if i + mtM < mtN then i + mtM else i + mtM - mtN
          x = (v V.! i .&. upperMask) + (v V.! i1 .&. lowerMask)
          xA = (x `shiftR` 1) `xor` (if x .&. 1 == 0 then 0 else matrixA)
          newVal = (v V.! j) `xor` xA
      MV.write mv i newVal
    V.unsafeFreeze mv
