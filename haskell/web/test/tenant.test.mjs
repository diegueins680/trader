import assert from "node:assert/strict";
import test from "node:test";

import { buildTenantKey } from "../.tmp/web-tests/tenant.js";

test("tenant identity preserves unambiguous ASCII keys and separates ambiguous tuples", async () => {
  const legacy = await buildTenantKey("binance", "alpha", "beta");
  const separatorLeft = await buildTenantKey("binance", "alpha:beta", "gamma");
  const separatorRight = await buildTenantKey("binance", "alpha", "beta:gamma");

  assert.equal(legacy, "binance:8610fb69f8ec56759b2fb33fd43f9a05fefbd2d49cd35c8b5786284434537af3");
  assert.equal(separatorLeft, "binance:v2:f7819a271a2175eacb13121b5ec1557b788a259f5103edf6c4c3ad05e9a28234");
  assert.equal(separatorRight, "binance:v2:b0f0389b4a3d94ff85ab3cfd9049536e3354c388284831431fee67c3c192dfc0");
  assert.notEqual(separatorLeft, separatorRight);
});

test("tenant identity preserves separator-free non-ASCII credentials in the legacy namespace", async () => {
  assert.equal(
    await buildTenantKey("binance", "cl\u00e9", "\u79d8\u5bc6"),
    "binance:117516d499c35af490f5de85d93a10c22e12015d1edc7c65204ecd58cb9f09f3",
  );
  assert.equal(
    await buildTenantKey("binance", "\uFEFFalpha\uFEFF", "beta"),
    "binance:4707291792a1aa7652d15cbd0b513b35dd91f34ee7c3c92b5df042d8453a57c4",
  );
  assert.equal(
    await buildTenantKey("binance", "\u00A0alpha\u00A0", "beta"),
    "binance:5eb0587b34a05af33bce33fe00a16f9f226e7c4974307b935faad765e6c6877d",
  );
});

test("Coinbase tuple boundaries are injective", async () => {
  const tenants = await Promise.all([
    buildTenantKey("coinbase", "alpha:beta", "gamma", "delta"),
    buildTenantKey("coinbase", "alpha", "beta:gamma", "delta"),
    buildTenantKey("coinbase", "alpha", "beta", "gamma:delta"),
  ]);

  assert.equal(new Set(tenants).size, tenants.length);
  assert.ok(tenants.every((tenant) => tenant?.startsWith("coinbase:v2:")));
});
