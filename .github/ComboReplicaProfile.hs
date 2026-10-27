{-# LANGUAGE OverloadedStrings #-}

import Control.Exception (bracket, bracket_)
import Control.Monad (forM, forM_, unless, void)
import qualified Data.Aeson as A
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int64)
import Data.String (fromString)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.UUID as UUID
import qualified Data.UUID.V4 as UUID
import Database.PostgreSQL.Simple
import GHC.Clock (getMonotonicTimeNSec)
import System.Environment (getEnv)
import ComboUpsertSql (comboUpsertSql, batchedComboUpsertSql)
import qualified PreviousComboParameters as Before
import qualified Batch8ComboParameters as Batch8
import qualified Batch25ComboParameters as Batch25
import Trader.Ops.ComboParameters (persistComboReplicas)
import Trader.Ops.Migrations (ensureOpsDbSchema)

type Row = (UUID.UUID,Maybe Int,Maybe T.Text,Maybe T.Text,Maybe T.Text,Maybe T.Text,Maybe Double,Maybe Double,Maybe Double,Maybe Double,Maybe Double,Maybe T.Text,Maybe T.Text,Int,Maybe Int64,Int64)
type Replica = (UUID.UUID,Row,Maybe Int,A.Object)

fixture :: Int -> Int -> Int -> [Replica]
fixture count paramsPadding metricsPadding = map (\n->make n 0) [1..count] ++ [make 1 1, make 2 1, make 1 2, make 1 3]
  where
    make :: Int -> Int -> Replica
    make n revision =
      let uuid=UUID.fromWords 0 0x4000 0x80000000 (fromIntegral n)
          params=if revision==3 then KM.empty else KM.fromList (("padding",A.String (T.replicate paramsPadding "p")):[(AK.fromString ("p"++show k),A.toJSON k)|k<-[1..168::Int]])
          metrics=A.object (["padding" A..= T.replicate metricsPadding "m"] ++ ["live" A..= A.object ["operationCount" A..= (if revision==1 then 50 else n `mod` 30),"netReturn" A..= (0.1::Double)] | revision/=3])
          encoded v=Just (TE.decodeUtf8 (BL.toStrict (A.encode v)))
          row=(uuid,Nothing,Just "SYNTHUSDT",Just "6h",Just "sharpe",Just "synthetic",Just 2,Just 0.2,Just (fromIntegral (n+revision*10000)),Just 0.01,Just 0.02,encoded (A.Object params),encoded metrics,0,(if n==1 && revision==0 then Nothing else Just (if n==1 then if revision==1 then 7 else 9 else fromIntegral n)),1234)
      in (uuid,row,Nothing,params)

fingerprint :: Connection -> IO [(Int64,Maybe String,Int64,Maybe String)]
fingerprint conn=query_ conn "SELECT (SELECT count(*) FROM combos), (SELECT md5(string_agg(row_to_json(c)::text,',' ORDER BY combo_uuid)) FROM combos c), (SELECT count(*) FROM combo_parameters), (SELECT md5(string_agg(row_to_json(p)::text,',' ORDER BY combo_uuid,name)) FROM combo_parameters p)"

main :: IO ()
main=do
 url<-getEnv "TRADER_TEST_DB_URL"
 nonce<-UUID.nextRandom
 let schema="replica_profile_"++filter (/='-') (UUID.toString nonce)
 bracket (connectPostgreSQL (BS.pack url)) close $ \conn -> bracket_
  (void (execute_ conn (fromString ("CREATE SCHEMA "++schema))))
  (void (execute_ conn (fromString ("DROP SCHEMA "++schema++" CASCADE")))) $ do
   void (execute_ conn (fromString ("SET search_path TO "++schema)))
   void (execute_ conn "SET client_min_messages=warning")
   ensureOpsDbSchema conn
   let previous rows=forM_ rows $ \(uuid,row,strategy,params)->do
        void (execute conn comboUpsertSql row)
        Before.persistComboParameters conn uuid strategy params
       measure count pp mp mode=do
        void (execute_ conn "TRUNCATE combos CASCADE")
        before<-getMonotonicTimeNSec
        withTransaction conn $ (case mode of "previous" -> previous; "batch8" -> Batch8.persistComboReplicas conn batchedComboUpsertSql; "batch25" -> Batch25.persistComboReplicas conn batchedComboUpsertSql; _ -> \_ -> fail "Unknown profile mode") (fixture count pp mp)
        after<-getMonotonicTimeNSec
        expected<-query_ conn "SELECT created_at_ms, (metrics_json->'live'->>'operationCount')::int FROM combos WHERE combo_uuid='00000000-0000-4000-8000-000000000001'" :: IO [(Maybe Int64,Maybe Int)]
        unless (expected==[(Just 7,Just 50)]) (fail "Creation or richer live evidence changed")
        result<-fingerprint conn
        pure (mode,fromIntegral (after-before)/1000000::Double,result)
   cases<-forM [(1038,3900,1000),(4544,8600,8700)] $ \(count,pp,mp)->do
    pairs<-forM [0..2::Int] $ \i->do
     let modes=if even i then ["previous"::String,"batch8","batch25"] else ["batch25","batch8","previous"]
     results<-mapM (measure count pp mp) modes
     let states=[state|(_,_,state)<-results]
     unless (all (==head states) (tail states)) (fail "Exact replica state differed")
     pure [A.object ["mode" A..= mode,"ms" A..= ms]|(mode,ms,_)<-results]
    pure (A.object ["combos" A..= count,"parametersPerCombo" A..= (169::Int),"paramsPadding" A..= pp,"metricsPadding" A..= mp,"pairs" A..= pairs,"exactStatesEqual" A..= True])
   BL.putStr (A.encode (A.object ["passed" A..= True,"scope" A..= ("Disposable PostgreSQL 16; original upsert versus the batched template with only its two redundant VALUES JSON casts removed and frozen main parameter writer; synthetic shapes, alternating order, no live traffic"::String),"cases" A..= cases])) >> putStrLn ""
