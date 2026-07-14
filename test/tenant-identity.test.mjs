import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

const repoRoot = new URL("..", import.meta.url);

function runAwsTenantScript(script, env) {
  return spawnSync("bash", ["-c", `. ./deploy-aws-quick.sh; ${script}`], {
    cwd: repoRoot,
    encoding: "utf8",
    env: {
      ...process.env,
      TRADER_DEPLOY_ENV_FILE: "/nonexistent",
      TRADER_STATE_SYNC_TENANT_KEY: "",
      TRADER_STATE_SYNC_SOURCE_TENANT_KEY: "",
      BINANCE_API_KEY: "",
      BINANCE_API_SECRET: "",
      COINBASE_API_KEY: "",
      COINBASE_API_SECRET: "",
      COINBASE_API_PASSPHRASE: "",
      ...env,
    },
  });
}

function deriveAwsTenant(env) {
  const result = runAwsTenantScript("resolve_state_sync_tenant_key", env);
  assert.equal(result.status, 0, result.stderr);
  return result.stdout;
}

test("AWS tenant derivation matches the canonical legacy and v2 vectors", () => {
  assert.equal(
    deriveAwsTenant({ BINANCE_API_KEY: " alpha ", BINANCE_API_SECRET: " beta " }),
    "binance:8610fb69f8ec56759b2fb33fd43f9a05fefbd2d49cd35c8b5786284434537af3",
  );
  assert.equal(
    deriveAwsTenant({ BINANCE_API_KEY: "alpha:beta", BINANCE_API_SECRET: "gamma" }),
    "binance:v2:f7819a271a2175eacb13121b5ec1557b788a259f5103edf6c4c3ad05e9a28234",
  );
  assert.equal(
    deriveAwsTenant({ BINANCE_API_KEY: "alpha", BINANCE_API_SECRET: "beta:gamma" }),
    "binance:v2:b0f0389b4a3d94ff85ab3cfd9049536e3354c388284831431fee67c3c192dfc0",
  );
  assert.equal(
    deriveAwsTenant({ BINANCE_API_KEY: "cl\u00e9", BINANCE_API_SECRET: "\u79d8\u5bc6" }),
    "binance:117516d499c35af490f5de85d93a10c22e12015d1edc7c65204ecd58cb9f09f3",
  );
  assert.equal(
    deriveAwsTenant({ BINANCE_API_KEY: "\uFEFFalpha\uFEFF", BINANCE_API_SECRET: "beta" }),
    "binance:4707291792a1aa7652d15cbd0b513b35dd91f34ee7c3c92b5df042d8453a57c4",
  );
  assert.equal(
    deriveAwsTenant({ BINANCE_API_KEY: "\u00A0alpha\u00A0", BINANCE_API_SECRET: "beta" }),
    "binance:5eb0587b34a05af33bce33fe00a16f9f226e7c4974307b935faad765e6c6877d",
  );
});

test("AWS Coinbase tuple boundaries cannot alias", () => {
  const identities = [
    deriveAwsTenant({ COINBASE_API_KEY: "alpha:beta", COINBASE_API_SECRET: "gamma", COINBASE_API_PASSPHRASE: "delta" }),
    deriveAwsTenant({ COINBASE_API_KEY: "alpha", COINBASE_API_SECRET: "beta:gamma", COINBASE_API_PASSPHRASE: "delta" }),
    deriveAwsTenant({ COINBASE_API_KEY: "alpha", COINBASE_API_SECRET: "beta", COINBASE_API_PASSPHRASE: "gamma:delta" }),
  ];

  assert.equal(new Set(identities).size, identities.length);
  assert.ok(identities.every((identity) => identity.startsWith("coinbase:v2:")));
});

test("App Runner state sync uses the target tenant as the legacy export source by default", () => {
  const result = runAwsTenantScript(
    'target="$(resolve_state_sync_tenant_key)"; source="$(resolve_state_sync_source_tenant_key "$target")"; printf "%s\\n%s" "$source" "$target"',
    { BINANCE_API_KEY: "alpha", BINANCE_API_SECRET: "beta" },
  );
  assert.equal(result.status, 0, result.stderr);
  const expected = "binance:8610fb69f8ec56759b2fb33fd43f9a05fefbd2d49cd35c8b5786284434537af3";
  assert.equal(result.stdout, `${expected}\n${expected}`);
});

test("App Runner state sync rejects an implicit source when the target tenant is v2", () => {
  const result = runAwsTenantScript(
    'target="$(resolve_state_sync_tenant_key)"; resolve_state_sync_source_tenant_key "$target"',
    { BINANCE_API_KEY: "alpha:beta", BINANCE_API_SECRET: "gamma" },
  );
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /TRADER_STATE_SYNC_SOURCE_TENANT_KEY/);
  assert.equal(result.stdout, "");
});

test("App Runner state sync exports from an explicit legacy source and imports to the v2 target", () => {
  const sourceTenant = "binance:a2893f0a1ca17e7ea3a5ae4e248e911643a88d26470d369bc92130e2bd530020";
  const targetTenant = "binance:v2:f7819a271a2175eacb13121b5ec1557b788a259f5103edf6c4c3ad05e9a28234";
  const result = runAwsTenantScript(
    'target="$(resolve_state_sync_tenant_key)"; source="$(resolve_state_sync_source_tenant_key "$target")"; printf "%s\\n%s" "$source" "$target"',
    {
      BINANCE_API_KEY: "alpha:beta",
      BINANCE_API_SECRET: "gamma",
      TRADER_STATE_SYNC_SOURCE_TENANT_KEY: sourceTenant,
    },
  );
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, `${sourceTenant}\n${targetTenant}`);
});
