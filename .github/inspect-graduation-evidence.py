"""Read graduation coverage through the existing PostgreSQL container."""
import json
import re
import subprocess
import uuid

node = '''JSON.stringify({tenant:"binance:"+require("crypto").createHash("sha256").update(process.env.BINANCE_API_KEY.trim()+":"+process.env.BINANCE_API_SECRET.trim()).digest("hex"),started:Number(process.env.TRADER_PORTFOLIO_AUTO_GRADUATE_STARTED_AT_MS),uuids:process.env.TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS.split(",")})'''
config = json.loads(subprocess.check_output(["docker", "exec", "trader-api-1", "node", "-p", node], stderr=subprocess.DEVNULL))
assert re.fullmatch(r"binance:[0-9a-f]{64}", config["tenant"])
reviewed = ",".join("'" + str(uuid.UUID(value)) + "'" for value in config["uuids"])
day = 86_400_000
boundary = ((int(config["started"]) + day - 1) // day) * day
cutoff = boundary - 900_000
query = f"""
WITH reviewed AS (SELECT unnest(ARRAY[{reviewed}]::uuid[]) AS uuid),
rows AS (SELECT * FROM ops WHERE tenant_key = '{config['tenant']}' AND combo_uuid = ANY(ARRAY[{reviewed}]::uuid[])),
coverage AS (
 SELECT r.uuid,
 COUNT(*) FILTER (WHERE o.kind='bot.status') AS statuses,
 MIN(o.at_ms) FILTER (WHERE o.kind='bot.status') AS first_status_ms,
 MAX(o.at_ms) FILTER (WHERE o.kind='bot.status') AS latest_status_ms,
 COUNT(DISTINCT o.at_ms/{day}) FILTER (WHERE o.kind='bot.status' AND o.at_ms>={boundary}) AS observed_days,
 COUNT(*) FILTER (WHERE o.kind='bot.status' AND o.at_ms BETWEEN {cutoff} AND {boundary}
 AND o.equity>0 AND o.result_json->>'live'='true' AND o.result_json->>'tradeEnabled'='true') AS baselines,
 COUNT(*) FILTER (WHERE o.kind='bot.order' AND o.at_ms>={boundary}) AS order_attempts
 FROM reviewed r LEFT JOIN rows o ON o.combo_uuid=r.uuid GROUP BY r.uuid
)
SELECT json_build_object('boundaryMs',{boundary},'byUuid',json_agg(row_to_json(coverage))) FROM coverage;
"""
command = ["docker", "exec", "-i", "trader-postgres-1", "sh", "-c", 'exec psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At']
output = subprocess.check_output(command, input=query.encode(), stderr=subprocess.DEVNULL)
print(json.dumps(json.loads(output)))
