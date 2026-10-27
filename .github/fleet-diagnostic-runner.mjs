import { createHash as liveCreateHash } from "node:crypto";
import { readdir as liveReadDir, writeFile as liveWriteFile } from "node:fs/promises";
try {
  const liveEnv = process.env;
  let liveOtherProbes = 0;
  for (const pid of await liveReadDir("/proc")) {
    if (!/^[0-9]+$/.test(pid) || Number(pid) === process.pid) continue;
    try { if ((await readFile("/proc/" + pid + "/cmdline", "utf8")).split("\0").includes("--input-type=module")) liveOtherProbes += 1; } catch { /* Process exited. */ }
  }
  if (liveOtherProbes !== 0) throw Error();
  const liveReportFile = "/tmp/trader-fleet-diagnostic-graduation-trace-20261027.json";
  const liveProgressFile = "/tmp/trader-fleet-diagnostic-graduation-trace-20261027-progress.json";
  const liveExpectedCommit = "3bae81e520993c9a863246c50e29572d71b09fee";
  const liveCanonicalSha256 = "24a4e82e05fcd284020c36dc51f9137ec137dedb36059557976bdc704b566f70";
  const liveExpectedManifest = "5813891079480a32452183975df8567cd4bdeb413465525b708b9293e17ce41b";
  const liveTenant = "binance:" + liveCreateHash("sha256").update(liveEnv.BINANCE_API_KEY.trim() + ":" + liveEnv.BINANCE_API_SECRET.trim()).digest("hex");
  const liveHeaders = { authorization: "Bearer " + liveEnv.TRADER_API_TOKEN, "x-tenant-key": liveTenant };
  const liveGet = async path => {
    const response = await fetch("http://127.0.0.1:8080" + path, { headers: liveHeaders, signal: AbortSignal.timeout(120000) });
    if (!response.ok) throw Error();
    return response.json();
  };
  const liveSymbols = liveEnv.TRADER_BOT_SYMBOLS.split(","), livePins = liveEnv.TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS.split(",");
  const liveRetained = new Set(liveEnv.TRADER_TOP_COMBOS_RETAIN_UUIDS.split(","));
  const liveManifest = liveCreateHash("sha256").update(liveSymbols.map((s, i) => s + "=" + livePins[i]).sort().join("\n")).digest("hex");
  const liveConfiguration = { configured: liveSymbols.length, reviewedPinCountAfterChange: 11, positionCap: Number(liveEnv.TRADER_BOT_START_ADOPTION_MAX_POSITION_SIZE_CAP), startPacing: Number(liveEnv.TRADER_BOT_AUTOSTART_MAX_STARTS_PER_CYCLE), retentionCount: liveEnv.TRADER_TOP_COMBOS_RETAIN_UUIDS.split(",").length, manifestSha256: liveManifest };
  if (liveManifest !== liveExpectedManifest || liveSymbols.length !== 18 || liveConfiguration.positionCap !== 0.01 || liveConfiguration.startPacing !== 2 || liveConfiguration.retentionCount !== 36) throw Error();
  const liveMemoryEvents = await readFile("/sys/fs/cgroup/memory.events", "utf8");
  const liveReadLimit = async name => {
    try {
      const raw = (await readFile('/sys/fs/cgroup/' + name, 'utf8')).trim();
      if (raw === 'max') return 'max';
      if (!/^[0-9]+$/.test(raw)) return null;
      const bytes = Number(raw);
      return Number.isSafeInteger(bytes) && bytes >= 0 ? bytes : null;
    } catch { return null; }
  };
  const liveCgroupLimits = { memoryMaxBytes: await liveReadLimit('memory.max'), memoryHighBytes: await liveReadLimit('memory.high'), scope: 'Configured cgroup limits observed before warmup; canonical benchmark acceptance and its unset ceiling remain unchanged' };

  const liveSnapshot = async () => {
    const health = await liveGet("/health");
    if (health.commit !== liveExpectedCommit) throw Error();
    const status = await liveGet("/bot/status?tail=1"), bots = status.multi ? status.bots : [status];
    const rows = liveSymbols.map((symbol, i) => {
      const b = bots.find(q => q.symbol === symbol), active = b?.performance?.closeTiming?.comboUuid ?? null;
      const position = b?.positions?.at(-1), origin = b?.openTrade?.comboUuid ?? null;
      const hasOpenPosition = Number.isFinite(position) && position !== 0;
      const retainedPositionOrigin = hasOpenPosition && origin === active && liveRetained.has(active);
      return { market: ["futures", "spot", "margin"].includes(b?.market) ? b.market : null, hasOpenPosition, openPositionOriginUuid: origin, retainedPositionOrigin, symbol, expectedUuid: livePins[i], activeUuid: active, matchesPin: active === livePins[i], running: b?.running === true, starting: b?.starting === true, halted: b?.halted === true, hasError: !!b?.error, live: b?.live === true, tradeEnabled: b?.settings?.tradeEnabled === true, protectionOrders: b?.settings?.protectionOrders === true };
    });
    const running = rows.filter(q => q.running);
    const verified = q => q?.running === true && (q.matchesPin || q.retainedPositionOrigin) && q.live && q.tradeEnabled && q.protectionOrders && !q.starting && !q.halted && !q.hasError;
    const baselineSymbols = ['AVAXUSDT','UNIUSDT','SUIUSDT','ETCUSDT','ADAUSDT','BCHUSDT','DOTUSDT','NEARUSDT','XRPUSDT','OPUSDT','ARBUSDT'];
    const allBaselineWorkersVerified = baselineSymbols.every(symbol => verified(rows.find(q => q.symbol === symbol)));
    const op = rows.find(q => q.symbol === 'OPUSDT');
    const opWorkerVerified = verified(op);
    const opReviewedPinRunning = opWorkerVerified && op.matchesPin;
    const arb = rows.find(q => q.symbol === 'ARBUSDT');
    const arbWorkerVerified = verified(arb);
    const arbReviewedPinRunning = arbWorkerVerified && arb.matchesPin;
    return { at: new Date().toISOString(), ready: health.ready, draining: health.draining, botRecoveryReady: health.botRecoveryReady, running: running.length, starting: rows.filter(q => q.starting).length, halted: running.filter(q => q.halted).length, errors: rows.filter(q => q.hasError).length, workerCountAtLeastBaseline: running.length >= 11, workerCountAtLeastReviewed: running.length >= 11, allBaselineWorkersVerified, opWorkerVerified, opReviewedPinRunning, arbWorkerVerified, arbReviewedPinRunning, allObservedWorkerIdentitiesVerified: running.length > 0 && running.every(q => (q.matchesPin || q.retainedPositionOrigin) && q.live && q.tradeEnabled && q.protectionOrders && !q.starting && !q.halted && !q.hasError), allObservedPinsVerified: running.length > 0 && running.every(q => q.matchesPin && q.live && q.tradeEnabled && q.protectionOrders && !q.starting && !q.halted && !q.hasError), rows, mode: status.portfolioSelector?.mode, configuredMode: status.portfolioSelector?.configuredMode, reviewDecision: status.portfolioSelector?.graduation?.review?.decision };
  };
  const liveWarmup = [];
  for (let i = 0; i < 37; i++) {
    const s = await liveSnapshot();
    liveWarmup.push({ at: s.at, ready: s.ready, draining: s.draining, botRecoveryReady: s.botRecoveryReady, running: s.running, starting: s.starting, halted: s.halted, errors: s.errors });
    await liveWriteFile(liveProgressFile, JSON.stringify({ phase: "warmup", samples: liveWarmup }), { mode: 384 });
    if (s.workerCountAtLeastReviewed && s.allBaselineWorkersVerified && s.opWorkerVerified && s.arbWorkerVerified && s.allObservedWorkerIdentitiesVerified) break;
    if (i < 36) await pause(10000);
  }
  await liveWriteFile(liveProgressFile, JSON.stringify({ phase: "benchmark", samples: liveWarmup }), { mode: 384 });
  const liveBenchmark = await runBenchmark({ baseUrl: "http://127.0.0.1:8080", rounds: 120, clients: 4, intervalMs: 1000, timeoutMs: 30000, expectedWorkers: 18, memoryFile: "/sys/fs/cgroup/memory.current" }, { token: liveEnv.TRADER_API_TOKEN, tenantKey: liveTenant });
  await liveWriteFile(liveProgressFile, JSON.stringify({ phase: "benchmark-complete", warmup: liveWarmup, benchmark: liveBenchmark }), { mode: 384 });
  const liveFinal = await liveSnapshot();
  const liveBoard = await liveGet("/optimizer/combos");
  const liveRemaining = ["AAVEUSDT", "ATOMUSDT", "DOGEUSDT", "ETHUSDT", "FILUSDT", "LINKUSDT", "SOLUSDT"].map(symbol => ({ symbol, strictCandidates: liveBoard.combos.filter(q => (q.params?.binanceSymbol ?? q.params?.symbol) === symbol && q.processing?.tier === "deployable" && !q.processing?.relaxed).length }));
  const livePinAdmission = liveSymbols.map((symbol, i) => { const q = liveBoard.combos.find(q => q.uuid === livePins[i]); return { symbol, uuid: livePins[i], present: !!q, strictCandidates: liveBoard.combos.filter(row => (row.params?.binanceSymbol ?? row.params?.symbol) === symbol && row.processing?.tier === 'deployable' && !row.processing?.relaxed).length, tier: ['deployable', 'raw', 'quarantined'].includes(q?.processing?.tier) ? q.processing.tier : 'other', relaxed: q?.processing?.relaxed === true }; });
  const liveOut = { baseline: { at: "2026-10-27T14:00:31.542Z", verifiedWorkers: 11, scope: "Observed running workers with deployable pins before graduation query tracing; canonical benchmark target remains 18" }, pinAdmission: livePinAdmission, reportFile: liveReportFile, at: new Date().toISOString(), commit: liveExpectedCommit, canonicalSha256: liveCanonicalSha256, configuration: liveConfiguration, cgroupLimits: liveCgroupLimits, warmup: liveWarmup, benchmark: liveBenchmark, final: liveFinal, remainingAdmission: liveRemaining, memoryEventsUnchanged: liveMemoryEvents === await readFile("/sys/fs/cgroup/memory.events", "utf8"), benchmarkAcceptanceChanged: false, identityScope: "API runtime pin match or retained open-position origin; position origin is preserved until flat, without independent exchange certification", memoryScope: "canonical benchmark window; startup observations and final board read are outside its memory sampling" };
  await liveWriteFile(liveReportFile, JSON.stringify(liveOut), { mode: 384 });
  console.log(JSON.stringify(liveOut));
} catch {
  console.log("LIVE_DIAGNOSTIC_UNAVAILABLE");
  process.exitCode = 1;
}
