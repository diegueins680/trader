module Trader.Optimizer.Json
  ( encodePretty
  ) where

import Data.Aeson (Value (..), encode)
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.ByteString.Builder (Builder)
import qualified Data.ByteString.Builder as BB
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V

indentStep :: Int
indentStep = 2

encodePretty :: Value -> BL.ByteString
encodePretty val = BB.toLazyByteString (renderValue 0 val)

renderValue :: Int -> Value -> Builder
renderValue indent val =
  case val of
    Object obj -> renderObject indent obj
    Array arr -> renderArray indent arr
    _ -> BB.lazyByteString (encode val)

renderObject :: Int -> KM.KeyMap Value -> Builder
renderObject indent obj
  | KM.null obj = BB.string7 "{}"
  | otherwise =
      let indent' = indent + indentStep
          pairs = KM.toList obj
          rendered = map (renderPair indent') pairs
          body = mconcat (intersperse (BB.string7 ",\n") rendered)
       in BB.string7 "{\n" <> body <> BB.char7 '\n' <> indentSpaces indent <> BB.char7 '}'

renderArray :: Int -> V.Vector Value -> Builder
renderArray indent arr
  | V.null arr = BB.string7 "[]"
  | otherwise =
      let indent' = indent + indentStep
          rendered = map (\v -> indentSpaces indent' <> renderValue indent' v) (V.toList arr)
          body = mconcat (intersperse (BB.string7 ",\n") rendered)
       in BB.string7 "[\n" <> body <> BB.char7 '\n' <> indentSpaces indent <> BB.char7 ']'

renderPair :: Int -> (Key.Key, Value) -> Builder
renderPair indent (k, v) =
  indentSpaces indent <> BB.lazyByteString (encode (String (Key.toText k))) <> BB.string7 ": " <> renderValue indent v

indentSpaces :: Int -> Builder
indentSpaces n = BB.string7 (replicate n ' ')

intersperse :: Builder -> [Builder] -> [Builder]
intersperse _ [] = []
intersperse _ [x] = [x]
intersperse sep (x : xs) = x : sep : intersperse sep xs
