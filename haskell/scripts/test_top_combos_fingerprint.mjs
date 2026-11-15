// Black-box regression for independent, bounded leaderboard fingerprint reads.
// TRADER_TEST_DB_URL must refer to a disposable test database with CREATE SCHEMA.
import assert from "node:assert/strict";
import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { once } from "node:events";
import { createHash } from "node:crypto";
import {
  mkdtemp,
  mkdir,
  open,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { createServer } from "node:http";
import { createServer as netServer } from "node:net";
import { resolve } from "node:path";
let stage = "setup",
  privateLog = "not created";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function stopChild(child) {
  if (!child?.pid || child.exitCode !== null || child.signalCode !== null)
    return;
  const exited = once(child, "exit");
  let timer;
  child.kill("SIGTERM");
  try {
    await Promise.race([
      exited,
      new Promise((r) => {
        timer = setTimeout(r, 5000);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
  if (child.exitCode === null && child.signalCode === null) {
    child.kill("SIGKILL");
    await exited;
  }
}
async function main() {
  const dbUrl = process.env.TRADER_TEST_DB_URL;
  assert.ok(dbUrl, "TRADER_TEST_DB_URL must name a disposable test database");
  assert.ok(process.argv[2], "Pass the built trader-hs executable");
  const program = resolve(process.argv[2]);
  const work = await mkdtemp(resolve(tmpdir(), "trader-fingerprint-"));
  privateLog = `${work}/api.log`;
  const schema =
    "trader_fingerprint_" +
    createHash("sha256").update(work).digest("hex").slice(0, 16);
  const pgEnv = {
    ...process.env,
    PGOPTIONS: `-c search_path=${schema}`,
    PGAPPNAME: schema,
  };
  const psql = process.env.TRADER_TEST_PSQL_BIN ?? "psql";
  const query = async (sql) =>
    (
      await promisify(execFile)(
        psql,
        [dbUrl, "-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", sql],
        { env: pgEnv, timeout: 5000, maxBuffer: 1024 * 1024 },
      )
    ).stdout.trim();
  let schemaCreated = false,
    passed = false,
    result,
    api,
    log,
    mock,
    keeper,
    deadline;
  try {
    await query(`CREATE SCHEMA ${schema}`);
    schemaCreated = true;
    assert.equal(
      await query("SELECT current_schema()"),
      schema,
      "Connection options must select the temporary test schema",
    );
    assert.equal(
      await query("SELECT current_setting('application_name')"),
      schema,
      "Connection options must preserve the unique test application name",
    );
    await mkdir(`${work}/board`);
    const fixture = {
      generatedAtMs: 1000,
      source: "isolated",
      combos: [
        {
          uuid: "00000000-0000-4000-8000-000000000001",
          createdAtMs: 1000,
          finalEquity: 1.2,
          annualizedReturn: 0.2,
          objective: null,
          score: 0.4,
          openThreshold: 0.005,
          closeThreshold: 0.005,
          params: {
            symbol: "BTCUSDT",
            interval: "1h",
            platform: "binance",
            method: "ta_ema",
            window: 24,
            minEdge: 0.003,
          },
          rank: 1,
          source: "isolated",
          metrics: { annualizedReturn: 0.2, finalEquity: 1.2 },
        },
      ],
    };
    let remote = Buffer.from(JSON.stringify(fixture)),
      etag = '"' + createHash("md5").update(remote).digest("hex") + '"',
      notifyGet;
    const getStarted = new Promise((r) => (notifyGet = r));
    await writeFile(`${work}/board/top-combos.json`, remote);
    mock = createServer(async (req, res) => {
      if (req.method === "HEAD") {
        res.writeHead(200, { etag });
        res.end();
        return;
      }
      if (req.method === "GET") {
        notifyGet();
        res.writeHead(200, { "content-type": "application/json", etag });
        res.end(remote);
        return;
      }
      if (req.method === "PUT") {
        const chunks = [];
        for await (const x of req) chunks.push(x);
        remote = Buffer.concat(chunks);
        etag = '"' + createHash("md5").update(remote).digest("hex") + '"';
        res.writeHead(200, { etag });
        res.end();
        return;
      }
      res.writeHead(405);
      res.end();
    });
    await new Promise((r, j) => {
      mock.once("error", j);
      mock.listen(0, "127.0.0.1", r);
    });
    const reserve = netServer();
    await new Promise((r, j) => {
      reserve.once("error", j);
      reserve.listen(0, "127.0.0.1", r);
    });
    const port = reserve.address().port;
    await new Promise((r) => reserve.close(r));
    log = await open(privateLog, "w", 0o600);
    const env = Object.fromEntries(
      ["PATH", "TMPDIR", "LANG"]
        .filter((k) => process.env[k])
        .map((k) => [k, process.env[k]]),
    );
    api = spawn(program, ["--serve", "--port", String(port)], {
      cwd: work,
      env: {
        ...env,
        TRADER_DB_URL: dbUrl,
        PGOPTIONS: `-c search_path=${schema}`,
        PGAPPNAME: schema,
        TRADER_API_BIND_HOST: "127.0.0.1",
        TRADER_API_TOKEN: "isolated-gate-token",
        TRADER_MULTI_USER: "false",
        TRADER_STATE_DIR: `${work}/state`,
        TRADER_SERVER_ROLE: "research",
        TRADER_OPTIMIZER_COMBOS_DIR: `${work}/board`,
        TRADER_BOT_AUTOSTART: "false",
        TRADER_BOT_TRADE: "false",
        TRADER_OPTIMIZER_ENABLED: "false",
        TRADER_TOP_COMBOS_BACKTEST_ENABLED: "false",
        TRADER_TOP_COMBOS_SYNC_ENABLED: "true",
        TRADER_TOP_COMBOS_SYNC_EVERY_SEC: "5",
        TRADER_TOP_COMBOS_SYNC_INITIAL_DELAY_SEC: "1",
        TRADER_TOP_COMBOS_MIN_PERSIST: "1",
        TRADER_OPTIMIZER_MAX_COMBOS: "1800",
        TRADER_GIT_COMMIT: "0".repeat(40),
        TRADER_STATE_S3_BUCKET: "gate-fixture",
        TRADER_STATE_S3_PREFIX: "fixture",
        TRADER_STATE_S3_ENDPOINT: `http://127.0.0.1:${mock.address().port}`,
        TRADER_STATE_S3_FORCE_PATH_STYLE: "true",
        AWS_ACCESS_KEY_ID: "isolated-access-key",
        AWS_SECRET_ACCESS_KEY: "isolated-secret-key",
        // Reserved documentation address keeps startup IP discovery offline.
        TRADER_BINANCE_EGRESS_IP: "203.0.113.1",
        GHCRTS: "-M2g -N2",
      },
      stdio: ["ignore", log.fd, log.fd],
    });
    await once(api, "spawn");
    deadline = setTimeout(() => api.kill("SIGTERM"), 120000);
    const trace = async () => {
      const text = await readFile(privateLog, "utf8");
      return text.split("\n").flatMap((line) => {
        try {
          const o = JSON.parse(line);
          return o.event === "top_combos_build_gate" ? [o] : [];
        } catch {
          return [];
        }
      });
    };
    const waitFor = async (check, label) => {
      const until = performance.now() + 30000;
      while (performance.now() < until) {
        if (await check()) return;
        if (api.exitCode !== null || api.signalCode !== null)
          throw Error("Test API stopped");
        await sleep(20);
      }
      throw Error(label);
    };
    await waitFor(async () => {
      try {
        return (
          await fetch(`http://127.0.0.1:${port}/health`, {
            signal: AbortSignal.timeout(1000),
          })
        ).ok;
      } catch {
        return false;
      }
    }, "API not ready");
    await Promise.race([
      getStarted,
      sleep(10000).then(() => {
        throw Error("isolated S3 GET not reached");
      }),
    ]);
    await waitFor(
      async () =>
        (await trace()).filter(
          (x) => x.origin === "sync" && x.phase === "finished",
        ).length >= 2,
      "sync did not reach steady state",
    );
    const columns = JSON.parse(
      await query(
        "SELECT json_agg(attname ORDER BY attnum) FROM pg_attribute WHERE attrelid='combos'::regclass AND attnum>0 AND NOT attisdropped",
      ),
    );
    await query(
      "ALTER TABLE combos RENAME TO combos_delay_base; CREATE FUNCTION delay_metrics(value jsonb) RETURNS jsonb LANGUAGE plpgsql STABLE AS $$ BEGIN PERFORM pg_advisory_lock(402357); PERFORM pg_advisory_unlock(402357); RETURN value; END $$; CREATE FUNCTION delay_fingerprint(value text) RETURNS text LANGUAGE plpgsql STABLE AS $$ BEGIN PERFORM pg_advisory_lock(402358); PERFORM pg_advisory_unlock(402358); RETURN value; END $$; CREATE VIEW combos AS SELECT " +
        columns
          .map((x) =>
            x === "metrics_json"
              ? "delay_metrics(metrics_json) AS metrics_json"
              : x === "params_json_md5"
                ? "delay_fingerprint(params_json_md5) AS params_json_md5"
                : '"' + x + '"',
          )
          .join(",") +
        " FROM combos_delay_base",
    );
    keeper = spawn(psql, [dbUrl, "-X", "-qAt", "-v", "ON_ERROR_STOP=1"], {
      env: pgEnv,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const locked = new Promise((resolve, reject) => {
      let output = "";
      keeper.stdout.on("data", (x) => {
        output += x.toString();
        if (output.includes("gate-locked")) resolve();
      });
      keeper.once("error", reject);
      keeper.once("exit", () =>
        reject(Error("isolated gate keeper exited early")),
      );
    });
    keeper.stdin.on("error", () => {});
    keeper.stdin.write(
      "SELECT pg_advisory_lock(402357); SELECT 'gate-locked';\n",
    );
    await Promise.race([
      locked,
      sleep(3000).then(() => {
        throw Error("isolated gate not ready");
      }),
    ]);
    remote = Buffer.from(
      JSON.stringify({
        ...JSON.parse(remote),
        source: "isolated-trigger-next-scheduled-pass",
      }),
    );
    etag = '"' + createHash("md5").update(remote).digest("hex") + '"';
    const probe = async (tag) => {
      const at = performance.now(),
        r = await fetch(`http://127.0.0.1:${port}/optimizer/combos`, {
          headers: {
            authorization: "Bearer isolated-gate-token",
            ...(tag ? { "if-none-match": tag } : {}),
          },
          signal: AbortSignal.timeout(10000),
        });
      const text = await r.text();
      let bodyHash = null;
      if (r.status === 200) {
        const v = JSON.parse(text);
        delete v.generatedAtMs;
        bodyHash = createHash("sha256").update(JSON.stringify(v)).digest("hex");
      }
      return {
        status: r.status,
        tag: r.headers.get("etag"),
        bodyHash,
        latencyMs: performance.now() - at,
      };
    };
    const warm = await probe();
    assert.equal(warm.status, 200);
    assert.ok(warm.tag);
    await waitFor(
      async () =>
        Number(
          await query(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE 'SELECT combo_uuid,%metrics_json::text%'",
          ),
        ) > 0,
      "scheduled sync did not block in full DB read",
    );
    stage = "independent reads";
    const gateEventsBefore = (await trace()).filter(
      (x) => x.origin === "response",
    ).length;
    const markKeeper = async (sql, marker) => {
      let listener, timer;
      const ready = new Promise((resolve) => {
        let output = "";
        listener = (x) => {
          output += x.toString();
          if (output.includes(marker)) resolve();
        };
        keeper.stdout.on("data", listener);
      });
      const timeout = new Promise((_, reject) => {
        timer = setTimeout(
          () => reject(Error("isolated keeper marker timeout")),
          3000,
        );
      });
      keeper.stdin.write(sql + "; SELECT '" + marker + "';\n");
      try {
        await Promise.race([ready, timeout]);
      } finally {
        clearTimeout(timer);
        keeper.stdout.off("data", listener);
      }
    };
    const responses = await Promise.race([
      Promise.all(Array.from({ length: 4 }, () => probe(warm.tag))),
      sleep(2000).then(() => {
        throw Error("conditional reads remained behind the shared connection");
      }),
    ]);
    assert.ok(responses.every((x) => x.status === 304));
    assert.ok(
      Number(
        await query(
          "SELECT count(*) FROM pg_stat_activity WHERE application_name=current_setting('application_name') AND wait_event_type='Lock' AND query LIKE 'SELECT combo_uuid,%metrics_json::text%'",
        ),
      ) > 0,
      "full replica query must remain held while 304s finish",
    );
    assert.equal(
      (await trace()).filter((x) => x.origin === "response").length,
      gateEventsBefore,
    );
    const healthAt = performance.now(),
      health = await fetch(`http://127.0.0.1:${port}/health`, {
        signal: AbortSignal.timeout(2000),
      });
    await health.arrayBuffer();
    const healthMs = performance.now() - healthAt;
    assert.equal(health.status, 200);
    stage = "connection bound";
    await markKeeper(
      "SELECT pg_advisory_unlock(402357); SELECT pg_advisory_lock(402358)",
      "fingerprint-held",
    );
    const fingerprintBackends = async () =>
      Number(
        await query(
          "SELECT count(*) FROM pg_stat_activity WHERE application_name=current_setting('application_name') AND query LIKE 'SELECT count(*)::bigint,%'",
        ),
      );
    let complete = 0;
    const pending = Promise.all(
      Array.from({ length: 4 }, () =>
        probe(warm.tag).then((x) => {
          complete++;
          return x;
        }),
      ),
    );
    pending.catch(() => {});
    await waitFor(
      async () =>
        Number(
          await query(
            "SELECT count(*) FROM pg_stat_activity WHERE application_name=current_setting('application_name') AND wait_event_type='Lock' AND query LIKE 'SELECT count(*)::bigint,%'",
          ),
        ) === 1,
      "fingerprint query did not reach isolated gate",
    );
    await sleep(300);
    assert.equal(complete, 0);
    const whileHeld = await fingerprintBackends();
    assert.equal(
      whileHeld,
      1,
      "concurrent requests opened more than one fingerprint backend",
    );
    await markKeeper(
      "SELECT pg_advisory_unlock(402358)",
      "fingerprint-released",
    );
    const afterRelease = await pending;
    assert.ok(afterRelease.every((x) => x.status === 304));
    await waitFor(
      async () => (await fingerprintBackends()) === 0,
      "fingerprint connection remained open after reads",
    );
    stage = "query failure";
    await query(
      "CREATE OR REPLACE FUNCTION delay_fingerprint(value text) RETURNS text LANGUAGE plpgsql STABLE AS $$ BEGIN RAISE EXCEPTION 'isolated fingerprint failure'; END $$",
    );
    const failedFingerprint = await probe(warm.tag);
    assert.equal(failedFingerprint.status, 200);
    assert.equal(failedFingerprint.tag, null);
    assert.equal(failedFingerprint.bodyHash, warm.bodyHash);
    await waitFor(
      async () => (await fingerprintBackends()) === 0,
      "failed query leaked its fingerprint connection",
    );
    await query(
      "CREATE OR REPLACE FUNCTION delay_fingerprint(value text) RETURNS text LANGUAGE plpgsql STABLE AS $$ BEGIN RETURN value; END $$",
    );
    const recovered = await probe(warm.tag);
    assert.equal(recovered.status, 304);
    stage = "transaction visibility";
    await markKeeper(
      "BEGIN; UPDATE combos_delay_base SET score=score+0.01",
      "uncommitted-update",
    );
    const uncommitted = await probe(warm.tag);
    assert.equal(uncommitted.status, 304);
    await markKeeper("ROLLBACK", "rolled-back-update");
    const rolledBack = await probe(warm.tag);
    assert.equal(rolledBack.status, 304);
    await query("BEGIN; UPDATE combos_delay_base SET score=score+0.01; COMMIT");
    const committed = await probe(warm.tag);
    assert.equal(committed.status, 200);
    assert.ok(committed.tag && committed.tag !== warm.tag);
    keeper.stdin.end();
    result = {
      passed: true,
      scope:
        "Isolated API, temporary PostgreSQL schema and loopback S3; no workers or venue traffic",
      programSha256: createHash("sha256")
        .update(await readFile(program))
        .digest("hex"),
      healthMs,
      conditionalResponseMs: responses.map((x) => x.latencyMs),
      all304BeforeReplicaReadReleased: true,
      enteredBuildGate: false,
      concurrentClients: 4,
      maxFingerprintBackendsWhileHeld: whileHeld,
      connectionsClosedAfterReadsAndFailure: true,
      queryFailureWithholdsEtag: true,
      queryFailurePreservesBody: true,
      nextReadRecovers: true,
      uncommittedAndRolledBackChangesKeepTag: true,
      committedChangeInvalidatesTag: true,
    };

    passed = true;
  } finally {
    clearTimeout(deadline);
    if (keeper?.pid && keeper.exitCode === null && keeper.signalCode === null)
      keeper.kill("SIGTERM");
    await stopChild(api);
    if (mock?.listening) {
      mock.closeAllConnections();
      await new Promise((r) => mock.close(r));
    }
    if (log) await log.close();
    if (schemaCreated) await query(`DROP SCHEMA ${schema} CASCADE`);
    if (passed) await rm(work, { recursive: true, force: true });
  }
  if (process.argv[3])
    await writeFile(process.argv[3], JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
}
main().catch((error) => {
  console.error(
    `Database fingerprint integration failed during ${stage}${error.code === "ERR_ASSERTION" ? `: ${error.message}` : ""}; private API log: ${privateLog}`,
  );
  process.exitCode = 1;
});
