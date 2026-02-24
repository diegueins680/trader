import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const traderConfigSource = readFileSync(new URL("../public/trader-config.js", import.meta.url), "utf8");

function runTraderConfig(hostname, existingConfig) {
  const context = {};
  context.window = { location: { hostname } };
  context.globalThis = context;
  if (existingConfig !== undefined) {
    context.__TRADER_CONFIG__ = existingConfig;
  }
  vm.runInNewContext(traderConfigSource, context, { filename: "trader-config.js" });
  return context.__TRADER_CONFIG__;
}

test("trader-config infers direct Fly API host for -web-hs naming", () => {
  const config = runTraderConfig("trader-web-hs.fly.dev");
  assert.equal(config.apiBaseUrl, "https://trader-hs.fly.dev");
  assert.equal(config.apiFallbackUrl, "/api");
});

test("trader-config infers direct Fly API host for -web suffix naming", () => {
  const config = runTraderConfig("trader-web.fly.dev");
  assert.equal(config.apiBaseUrl, "https://trader.fly.dev");
  assert.equal(config.apiFallbackUrl, "/api");
});

test("trader-config does not rewrite app names that only contain -web as a substring", () => {
  const config = runTraderConfig("price-webhook.fly.dev");
  assert.equal(config.apiBaseUrl, "/api");
  assert.equal(config.apiFallbackUrl, "");
});

test("trader-config does not rewrite ambiguous -web- names without hs backend suffix", () => {
  const config = runTraderConfig("news-web-api.fly.dev");
  assert.equal(config.apiBaseUrl, "/api");
  assert.equal(config.apiFallbackUrl, "");
});

test("trader-config strips the rightmost -web- marker for hs backend suffixes", () => {
  const config = runTraderConfig("alpha-web-api-web-hs.fly.dev");
  assert.equal(config.apiBaseUrl, "https://alpha-web-api-hs.fly.dev");
  assert.equal(config.apiFallbackUrl, "/api");
});

test("trader-config keeps existing config object intact", () => {
  const existing = { apiBaseUrl: "https://api.example.com", apiFallbackUrl: "/api", apiToken: "x" };
  const config = runTraderConfig("trader-web-hs.fly.dev", existing);
  assert.strictEqual(config, existing);
});
