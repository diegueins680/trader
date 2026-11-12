-- Must NOT compile: the intent constructor is not exported.
module Main (main) where

import ExposureShieldV1 (ResearchIntent (..), intentTarget)

main :: IO ()
main = print (intentTarget (ResearchIntent 9999))
