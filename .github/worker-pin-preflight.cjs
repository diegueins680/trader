const e = process.env, f = require('fs'), c = require('crypto');
(async () => {
  const get = async path => {
    const response = await fetch('http://127.0.0.1:8080' + path, {
      headers: {authorization: 'Bearer ' + e.TRADER_API_TOKEN},
      signal: AbortSignal.timeout(120000),
    });
    if (!response.ok) throw Error();
    return response.json();
  };
  const health = await get('/health');
  if (health.commit !== 'c0982d867179e728d72eb3e96819974c000d92ec') throw Error();
  const board = await get('/optimizer/combos'), now = Date.now();
  const symbols = e.TRADER_BOT_SYMBOLS.split(','), pins = e.TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS.split(',');
  const limits = {epochs: Number(e.TRADER_API_MAX_EPOCHS || 100), hiddenSize: Number(e.TRADER_API_MAX_HIDDEN_SIZE || 32)};
  const symbolOf = row => row.params?.binanceSymbol ?? row.params?.symbol;
  const describe = row => {
    const p = row.params ?? {}, timestamp = row.backtestRefreshedAtMs ?? row.metrics?.backtestRefreshedAtMs ?? row.createdAtMs;
    const ageDays = Number.isFinite(timestamp) ? (now - timestamp) / 86400000 : null;
    const epochs = Number.isFinite(p.epochs) ? Math.max(1, p.epochs) : null;
    const hiddenSize = Number.isFinite(p.hiddenSize) ? Math.max(1, p.hiddenSize) : null;
    const computeLimitsPass = ['ta_regime_switch', 'ta_best'].includes(p.method) && epochs !== null && epochs <= limits.epochs && hiddenSize !== null && hiddenSize <= limits.hiddenSize;
    return {uuid: row.uuid, symbol: symbolOf(row), tier: row.processing?.tier, relaxed: !!row.processing?.relaxed,
      source: row.source, platform: p.platform, method: p.method, interval: p.interval, ageDays, epochs, hiddenSize, computeLimitsPass,
      finalEquity: row.finalEquity, minEdge: p.minEdge, tradeCount: row.metrics?.tradeCount ?? row.metrics?.trades};
  };
  const eligible = row => {
    const q = describe(row);
    return q.tier === 'deployable' && !q.relaxed && q.source === 'binance' && q.platform === 'binance'
      && q.ageDays !== null && q.ageDays >= 0 && q.ageDays <= 14 && q.computeLimitsPass;
  };
  const rows = ['AAVEUSDT', 'FILUSDT', 'OPUSDT'].map(symbol => {
    const candidates = board.combos.filter(q => symbolOf(q) === symbol);
    const selected = candidates.find(eligible);
    return {symbol, oldUuid: pins[symbols.indexOf(symbol)], eligibleCandidates: candidates.filter(eligible).length,
      candidate: selected ? describe(selected) : null,
      strictAlternatives: candidates.filter(q => q.processing?.tier === 'deployable' && !q.processing?.relaxed).map(describe).slice(0, 10)};
  });
  const pairs = rows.filter(q => q.candidate).map(q => q.symbol + '=' + q.candidate.uuid).sort();
  const positionCap = Number(e.TRADER_BOT_START_ADOPTION_MAX_POSITION_SIZE_CAP);
  const out = {at: new Date().toISOString(), commit: health.commit, limits, positionCap, rows, pairs,
    manifestSha256: c.createHash('sha256').update(pairs.join('\n')).digest('hex'),
    pinRows: symbols.map((symbol, i) => {const q = board.combos.find(v => v.uuid === pins[i]); return {symbol, candidate: q ? describe(q) : null};}),
    remaining: ['ARBUSDT', 'ATOMUSDT', 'ETHUSDT', 'LINKUSDT', 'OPUSDT', 'XRPUSDT'].map(symbol => ({symbol,
      strictCandidates: board.combos.filter(q => symbolOf(q) === symbol && q.processing?.tier === 'deployable' && !q.processing?.relaxed).length})),
    passed: rows.length === 3 && rows.every(q => q.candidate && q.candidate.symbol === q.symbol) && positionCap === 0.01};
  f.writeFileSync('/tmp/worker-pins-refresh-preflight-20261026.json', JSON.stringify(out), {mode: 384});
  console.log(JSON.stringify(out));
})().catch(() => console.log('pin refresh preflight unavailable'));
