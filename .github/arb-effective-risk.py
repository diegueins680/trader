import datetime,hashlib,json,math,re,shlex,subprocess,sys,urllib.parse
EXPECTED=sys.argv[1]
TARGET='2f4ccff0-e4bd-123b-575d-fb490ecdff50'
if not re.fullmatch('[0-9a-f]{40}',EXPECTED):raise SystemExit('Expected commit required')
def run(args,data=None,timeout=30):
 r=subprocess.run(args,input=data,capture_output=True,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError('Probe unavailable')
 return r.stdout
def health():
 js='fetch("http://127.0.0.1:8080/health").then(r=>r.json()).then(h=>console.log(JSON.stringify({commit:h.commit,ready:h.ready,botRecoveryReady:h.botRecoveryReady})))'
 h=json.loads(run(['docker','exec','trader-api-1','node','-e',js]))
 if h['commit']!=EXPECTED:raise RuntimeError('Commit mismatch')
 return h
try:
 before=health();started=run(['docker','inspect','--format','{{.State.StartedAt}}','trader-api-1']).strip()
 start_ms=int(datetime.datetime.fromisoformat(started.replace('Z','+00:00')).timestamp()*1000)
 js='const e=process.env,c=require("crypto");console.log(JSON.stringify({db:e.TRADER_DB_URL,tenant:"binance:"+c.createHash("sha256").update(e.BINANCE_API_KEY.trim()+":"+e.BINANCE_API_SECRET.trim()).digest("hex")}))'
 config=json.loads(run(['docker','exec','trader-api-1','node','-e',js]));connection=config['db'];tenant=config['tenant']
 if not re.fullmatch('binance:[0-9a-f]{64}',tenant):raise RuntimeError('Invalid tenant')
 if connection.startswith(('postgres://','postgresql://')):
  p=urllib.parse.urlsplit(connection);host,port,db,user=p.hostname,p.port or 5432,urllib.parse.unquote(p.path[1:]),urllib.parse.unquote(p.username or '')
 else:
  p=dict(x.split('=',1) for x in shlex.split(connection));host,port,db,user=p.get('host'),int(p.get('port',5432)),p.get('dbname'),p.get('user')
 env=run(['docker','exec','trader-postgres-1','sh','-c','printf "%s\\n%s\\n" "$POSTGRES_DB" "$POSTGRES_USER"']).splitlines()
 if host not in ('postgres','trader-postgres-1') or port!=5432 or env!=[db,user]:raise RuntimeError('Database mismatch')
 sql="SELECT json_build_object('symbol',symbol,'comboUuid',combo_uuid,'running',running,'market',market,'updatedAtMs',updated_at_ms,'maxPositionSize',(args_json->>'maxPositionSize')::double precision,'minPositionSize',(args_json->>'minPositionSize')::double precision,'adoptionMaxPositionSizeCap',(args_json->>'adoptionMaxPositionSizeCap')::double precision) FROM bots WHERE tenant_key='"+tenant+"' AND symbol='ARBUSDT' AND combo_uuid='"+TARGET+"' AND market='futures' AND running AND updated_at_ms >= "+str(start_ms)+" ORDER BY updated_at_ms DESC LIMIT 1"
 body="BEGIN READ ONLY;SET LOCAL statement_timeout='10s';SET LOCAL lock_timeout='1s';\n"+sql+";\nROLLBACK;\n"
 raw=run(['docker','exec','-i','trader-postgres-1','psql','-X','-qAt','-v','ON_ERROR_STOP=1','-U',user,'-d',db],body)
 row=json.loads(raw)
 for field in ['maxPositionSize','minPositionSize','adoptionMaxPositionSizeCap']:
  if type(row[field]) not in (int,float) or not math.isfinite(row[field]):raise RuntimeError('Missing risk values')
 if run(['docker','inspect','--format','{{.State.StartedAt}}','trader-api-1']).strip()!=started:raise RuntimeError('Container changed')
 after=health();valid=0<=row['minPositionSize']<=row['maxPositionSize']<=0.01 and row['adoptionMaxPositionSizeCap']==0.01
 print('ARB_EFFECTIVE_RISK_JSON='+json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':EXPECTED,'containerStartedAt':started,'health':after,'botSnapshot':row,'capAndFloorVerified':valid,'readOnly':True,'scope':'Fresh persisted bot arguments correlated to the new ARB UUID and this container; no raw arguments, credentials or exchange orders'},separators=(',',':')))
 if not valid:raise RuntimeError('Risk values outside cap')
except Exception:
 raise SystemExit('Effective risk snapshot unavailable')
