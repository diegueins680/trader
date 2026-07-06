#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { writeJsonFileAtomic } from "./autoloop-lib.mjs";

const DEFAULT_STATIONS_FILE = path.join(".tmp", "radio-stations.json");
const DEFAULT_TIMEOUT_MS = 8000;
const DEFAULT_CONCURRENCY = 8;
const DEFAULT_MAX_FAILURES = 2;
const USER_AGENT = "trader-radio-maintenance/1.0";

function usage() {
  return [
    "Usage: node scripts/maintain-radio-stations.mjs [options]",
    "",
    "Checks known radio station stream URLs, purges stations that repeatedly fail,",
    "and adds live stations from configured discovery JSON sources.",
    "",
    "Options:",
    "  --stations-file <path>     JSON registry path (default: RADIO_STATIONS_FILE or .tmp/radio-stations.json)",
    "  --discovery-url <url>      JSON station discovery endpoint; may be repeated",
    "  --discovery-file <path>    Local JSON station discovery file; may be repeated",
    "  --timeout-ms <ms>          Per-request timeout (default: 8000)",
    "  --concurrency <n>          Concurrent live checks (default: 8)",
    "  --max-failures <n>         Purge after this many consecutive failures (default: 2)",
    "  --dry-run                  Print what would happen without writing the registry",
    "  --json                     Print the full result as JSON",
    "  -h, --help                 Show this help",
    "",
    "Environment:",
    "  RADIO_STATIONS_FILE",
    "  RADIO_STATION_DISCOVERY_URLS or RADIO_STATION_DISCOVERY_URL",
    "  RADIO_STATION_DISCOVERY_FILES or RADIO_STATION_DISCOVERY_FILE",
    "  RADIO_STATION_CHECK_TIMEOUT_MS",
    "  RADIO_STATION_CHECK_CONCURRENCY",
    "  RADIO_STATION_MAX_FAILURES",
  ].join("\n");
}

export function parseArgs(argv, env = process.env) {
  const args = {
    stationsFile: env.RADIO_STATIONS_FILE || DEFAULT_STATIONS_FILE,
    discoveryUrls: [
      ...splitEnvList(env.RADIO_STATION_DISCOVERY_URLS),
      ...splitEnvList(env.RADIO_STATION_DISCOVERY_URL),
    ],
    discoveryFiles: [
      ...splitEnvList(env.RADIO_STATION_DISCOVERY_FILES),
      ...splitEnvList(env.RADIO_STATION_DISCOVERY_FILE),
    ],
    timeoutMs: parseBoundedInt(env.RADIO_STATION_CHECK_TIMEOUT_MS, "RADIO_STATION_CHECK_TIMEOUT_MS", {
      defaultValue: DEFAULT_TIMEOUT_MS,
      min: 500,
      max: 60000,
    }),
    concurrency: parseBoundedInt(env.RADIO_STATION_CHECK_CONCURRENCY, "RADIO_STATION_CHECK_CONCURRENCY", {
      defaultValue: DEFAULT_CONCURRENCY,
      min: 1,
      max: 64,
    }),
    maxFailures: parseBoundedInt(env.RADIO_STATION_MAX_FAILURES, "RADIO_STATION_MAX_FAILURES", {
      defaultValue: DEFAULT_MAX_FAILURES,
      min: 1,
      max: 100,
    }),
    dryRun: false,
    json: false,
    help: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const readValue = () => {
      i += 1;
      if (i >= argv.length || argv[i].startsWith("-")) {
        throw new Error(`Missing value for ${arg}`);
      }
      return argv[i];
    };

    if (arg === "--stations-file") {
      args.stationsFile = readValue();
    } else if (arg === "--discovery-url") {
      args.discoveryUrls.push(readValue());
    } else if (arg === "--discovery-file") {
      args.discoveryFiles.push(readValue());
    } else if (arg === "--timeout-ms") {
      args.timeoutMs = parseBoundedInt(readValue(), arg, { min: 500, max: 60000 });
    } else if (arg === "--concurrency") {
      args.concurrency = parseBoundedInt(readValue(), arg, { min: 1, max: 64 });
    } else if (arg === "--max-failures") {
      args.maxFailures = parseBoundedInt(readValue(), arg, { min: 1, max: 100 });
    } else if (arg === "--dry-run") {
      args.dryRun = true;
    } else if (arg === "--json") {
      args.json = true;
    } else if (arg === "-h" || arg === "--help") {
      args.help = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  args.discoveryUrls = uniqueStrings(args.discoveryUrls);
  args.discoveryFiles = uniqueStrings(args.discoveryFiles);
  if (!args.stationsFile.trim()) {
    throw new Error("Station registry path must not be empty");
  }
  for (const url of args.discoveryUrls) {
    assertHttpUrl(url, "Discovery URL");
  }

  return args;
}

function splitEnvList(value) {
  return String(value ?? "")
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseBoundedInt(value, label, { defaultValue, min, max }) {
  if (value === undefined || value === null || value === "") {
    if (defaultValue !== undefined) return defaultValue;
    throw new Error(`${label} is required`);
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    throw new Error(`${label} must be an integer from ${min} to ${max}`);
  }
  return parsed;
}

function uniqueStrings(values) {
  return Array.from(new Set(values.map((value) => String(value).trim()).filter(Boolean)));
}

function assertHttpUrl(value, label) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${label} must be a valid URL: ${value}`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`${label} must use http or https: ${value}`);
  }
  return parsed;
}

export function normalizeStation(raw, source = "unknown") {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;

  const url = firstString(
    raw.streamUrl,
    raw.stream_url,
    raw.listenUrl,
    raw.listen_url,
    raw.urlResolved,
    raw.url_resolved,
    raw.stationUrl,
    raw.station_url,
    raw.url,
  );
  if (!url) return null;

  let parsedUrl;
  try {
    parsedUrl = assertHttpUrl(url, "Station stream URL");
  } catch {
    return null;
  }

  const cleanUrl = parsedUrl.href;
  const name = firstString(raw.name, raw.title, raw.station, raw.callsign) || parsedUrl.hostname;
  const rawId = firstString(raw.id, raw.stationId, raw.station_id, raw.stationuuid, raw.uuid);
  const id = sanitizeStationId(rawId) || stationIdFromUrl(cleanUrl);
  const tags = normalizeTags(raw.tags ?? raw.tag);
  const bitrate = parseOptionalNonNegativeNumber(raw.bitrate);

  return {
    ...raw,
    id,
    name,
    url: cleanUrl,
    homepage: firstString(raw.homepage, raw.homepageUrl, raw.homepage_url, raw.website) || undefined,
    country: firstString(raw.country, raw.countryCode, raw.countrycode) || undefined,
    language: firstString(raw.language, raw.languages) || undefined,
    codec: firstString(raw.codec) || undefined,
    bitrate: bitrate ?? undefined,
    tags: tags.length > 0 ? tags : undefined,
    source: firstString(raw.source) || source,
    consecutiveFailures: parseOptionalNonNegativeInt(raw.consecutiveFailures) ?? 0,
  };
}

function firstString(...values) {
  for (const value of values) {
    if (typeof value !== "string") continue;
    const trimmed = value.trim();
    if (trimmed) return trimmed;
  }
  return "";
}

function normalizeTags(raw) {
  if (Array.isArray(raw)) {
    return uniqueStrings(raw.map((item) => (typeof item === "string" ? item : "")));
  }
  if (typeof raw === "string") {
    return uniqueStrings(raw.split(","));
  }
  return [];
}

function parseOptionalNonNegativeNumber(raw) {
  if (raw === undefined || raw === null || raw === "") return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

function parseOptionalNonNegativeInt(raw) {
  if (raw === undefined || raw === null || raw === "") return null;
  const parsed = Number(raw);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

function sanitizeStationId(raw) {
  const value = String(raw ?? "").trim();
  if (!value) return "";
  return value
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 120);
}

function stationIdFromUrl(url) {
  return `url:${Buffer.from(canonicalUrlKey(url)).toString("base64url").slice(0, 80)}`;
}

function canonicalUrlKey(url) {
  try {
    const parsed = new URL(url);
    parsed.hash = "";
    return parsed.href.replace(/\/$/, "").toLowerCase();
  } catch {
    return String(url ?? "").trim().replace(/\/$/, "").toLowerCase();
  }
}

function stationKeys(station) {
  const keys = new Set();
  if (station?.id) keys.add(`id:${String(station.id).toLowerCase()}`);
  if (station?.url) keys.add(`url:${canonicalUrlKey(station.url)}`);
  return keys;
}

function hasAnyKey(index, station) {
  for (const key of stationKeys(station)) {
    if (index.has(key)) return true;
  }
  return false;
}

function addStationToIndex(index, station, position) {
  for (const key of stationKeys(station)) {
    index.set(key, position);
  }
}

function extractStationArray(raw) {
  if (Array.isArray(raw)) return raw;
  if (Array.isArray(raw?.stations)) return raw.stations;
  if (Array.isArray(raw?.data)) return raw.data;
  if (Array.isArray(raw?.results)) return raw.results;
  return [];
}

function normalizeRegistry(raw, source) {
  return extractStationArray(raw)
    .map((station) => normalizeStation(station, source))
    .filter(Boolean);
}

async function readJsonFileOptional(filePath, fallback) {
  try {
    const text = await fs.readFile(filePath, "utf8");
    return JSON.parse(text);
  } catch (error) {
    if (error?.code === "ENOENT") return fallback;
    throw error;
  }
}

async function fetchJson(url, { fetchImpl, timeoutMs }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  timeout.unref?.();
  try {
    const response = await fetchImpl(url, {
      headers: { accept: "application/json", "user-agent": USER_AGENT },
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

async function loadDiscoveryStations({ discoveryFiles, discoveryUrls, fetchImpl, timeoutMs }) {
  const stations = [];
  for (const file of discoveryFiles) {
    const raw = await readJsonFileOptional(file, { stations: [] });
    stations.push(...normalizeRegistry(raw, `file:${file}`));
  }
  for (const url of discoveryUrls) {
    const raw = await fetchJson(url, { fetchImpl, timeoutMs });
    stations.push(...normalizeRegistry(raw, url));
  }
  return stations;
}

export async function checkStationLive(station, options = {}) {
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  if (typeof fetchImpl !== "function") {
    throw new Error("fetch is not available; use Node 20+ or pass fetchImpl");
  }
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  timeout.unref?.();

  try {
    const response = await fetchImpl(station.url, {
      method: "GET",
      headers: {
        accept: "audio/*,application/ogg,application/octet-stream,*/*;q=0.4",
        "icy-metadata": "0",
        range: "bytes=0-1023",
        "user-agent": USER_AGENT,
      },
      redirect: "follow",
      signal: controller.signal,
    });

    if (!response || response.status < 200 || response.status >= 300) {
      return { live: false, status: response?.status ?? 0, error: `HTTP ${response?.status ?? "unknown"}` };
    }

    if (looksLikeStreamHeaders(response.headers)) {
      await cancelBody(response);
      return { live: true, status: response.status };
    }

    const hasChunk = await readFirstBodyChunk(response);
    if (hasChunk) {
      return { live: true, status: response.status };
    }
    return { live: false, status: response.status, error: "no stream data received" };
  } catch (error) {
    return { live: false, status: 0, error: error?.name === "AbortError" ? "timeout" : String(error?.message ?? error) };
  } finally {
    clearTimeout(timeout);
  }
}

function looksLikeStreamHeaders(headers) {
  if (!headers) return false;
  const contentType = headerValue(headers, "content-type").toLowerCase();
  if (
    contentType.startsWith("audio/") ||
    contentType.startsWith("video/") ||
    contentType.includes("application/ogg") ||
    contentType.includes("application/octet-stream") ||
    contentType.includes("application/vnd.apple.mpegurl") ||
    contentType.includes("application/x-mpegurl")
  ) {
    return true;
  }

  let streamHeader = false;
  if (typeof headers.forEach === "function") {
    headers.forEach((_value, key) => {
      const lower = key.toLowerCase();
      if (lower.startsWith("icy-") || lower.startsWith("x-audiocast-") || lower.startsWith("x-ogg-")) {
        streamHeader = true;
      }
    });
  }
  return streamHeader;
}

function headerValue(headers, name) {
  if (typeof headers.get === "function") return headers.get(name) || "";
  return headers[name] || headers[name.toLowerCase()] || "";
}

async function readFirstBodyChunk(response) {
  const body = response.body;
  if (!body) return false;
  if (typeof body.getReader !== "function") return true;

  const reader = body.getReader();
  try {
    const { done, value } = await reader.read();
    return !done && Boolean(value) && Number(value.byteLength ?? value.length ?? 0) > 0;
  } finally {
    await reader.cancel().catch(() => {});
  }
}

async function cancelBody(response) {
  const body = response.body;
  if (!body || typeof body.getReader !== "function") return;
  const reader = body.getReader();
  await reader.cancel().catch(() => {});
}

export async function mapWithConcurrency(items, limit, mapper) {
  const results = new Array(items.length);
  let cursor = 0;
  const workerCount = Math.min(Math.max(1, limit), items.length);
  await Promise.all(
    Array.from({ length: workerCount }, async () => {
      while (cursor < items.length) {
        const index = cursor;
        cursor += 1;
        results[index] = await mapper(items[index], index);
      }
    }),
  );
  return results;
}

export async function maintainStations(options = {}) {
  const stationsFile = options.stationsFile || DEFAULT_STATIONS_FILE;
  const discoveryFiles = options.discoveryFiles || [];
  const discoveryUrls = options.discoveryUrls || [];
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const concurrency = options.concurrency ?? DEFAULT_CONCURRENCY;
  const maxFailures = options.maxFailures ?? DEFAULT_MAX_FAILURES;
  const fetchImpl = options.fetchImpl || globalThis.fetch;
  const checkLive =
    options.checkLive ||
    ((station) =>
      checkStationLive(station, {
        fetchImpl,
        timeoutMs,
      }));
  const now = options.now instanceof Date ? options.now : new Date();
  const nowIso = now.toISOString();
  const warnings = [];

  const rawRegistry = await readJsonFileOptional(stationsFile, { version: 1, stations: [] });
  const topLevel = rawRegistry && typeof rawRegistry === "object" && !Array.isArray(rawRegistry) ? rawRegistry : {};
  const existing = normalizeRegistry(rawRegistry, "registry");
  const checked = await mapWithConcurrency(existing, concurrency, async (station) => {
    const result = await checkLive(station);
    return { station, result };
  });

  const kept = [];
  const purged = [];
  let liveCount = 0;
  let failedCount = 0;

  for (const { station, result } of checked) {
    if (result.live) {
      liveCount += 1;
      kept.push({
        ...station,
        status: "live",
        consecutiveFailures: 0,
        lastCheckedAt: nowIso,
        lastLiveAt: nowIso,
        lastCheckError: undefined,
      });
      continue;
    }

    failedCount += 1;
    const consecutiveFailures = (parseOptionalNonNegativeInt(station.consecutiveFailures) ?? 0) + 1;
    const next = {
      ...station,
      status: "down",
      consecutiveFailures,
      lastCheckedAt: nowIso,
      lastCheckError: result.error || `HTTP ${result.status ?? "unknown"}`,
    };
    if (consecutiveFailures >= maxFailures) {
      purged.push(next);
    } else {
      kept.push(next);
    }
  }

  if (discoveryFiles.length === 0 && discoveryUrls.length === 0) {
    warnings.push("no discovery source configured; only existing stations were checked");
  }

  const discovered = await loadDiscoveryStations({ discoveryFiles, discoveryUrls, fetchImpl, timeoutMs });
  const index = new Map();
  kept.forEach((station, position) => addStationToIndex(index, station, position));

  const newCandidates = [];
  const candidateIndex = new Map();
  for (const station of discovered) {
    if (hasAnyKey(index, station) || hasAnyKey(candidateIndex, station)) continue;
    addStationToIndex(candidateIndex, station, newCandidates.length);
    newCandidates.push(station);
  }

  const discoveryChecks = await mapWithConcurrency(newCandidates, concurrency, async (station) => {
    const result = await checkLive(station);
    return { station, result };
  });

  const added = [];
  const skippedDiscoveryDown = [];
  for (const { station, result } of discoveryChecks) {
    if (result.live) {
      const next = {
        ...station,
        status: "live",
        consecutiveFailures: 0,
        firstSeenAt: nowIso,
        lastCheckedAt: nowIso,
        lastLiveAt: nowIso,
      };
      addStationToIndex(index, next, kept.length + added.length);
      added.push(next);
    } else {
      skippedDiscoveryDown.push({
        ...station,
        status: "down",
        consecutiveFailures: 1,
        lastCheckedAt: nowIso,
        lastCheckError: result.error || `HTTP ${result.status ?? "unknown"}`,
      });
    }
  }

  const nextRegistry = {
    ...topLevel,
    version: Number.isInteger(topLevel.version) ? topLevel.version : 1,
    updatedAt: nowIso,
    stations: [...kept, ...added],
    lastMaintenance: {
      checked: existing.length,
      live: liveCount,
      failed: failedCount,
      purged: purged.length,
      discovered: discovered.length,
      newCandidates: newCandidates.length,
      added: added.length,
      skippedDiscoveryDown: skippedDiscoveryDown.length,
      warnings,
      ranAt: nowIso,
    },
  };

  if (!options.dryRun) {
    await writeJsonFileAtomic(stationsFile, nextRegistry);
  }

  return {
    stationsFile,
    dryRun: Boolean(options.dryRun),
    checked: existing.length,
    live: liveCount,
    failed: failedCount,
    purged,
    discovered: discovered.length,
    newCandidates: newCandidates.length,
    added,
    skippedDiscoveryDown,
    warnings,
    stationCountBefore: existing.length,
    stationCountAfter: nextRegistry.stations.length,
  };
}

function formatSummary(result) {
  return [
    "radio-stations",
    "OK",
    `file=${result.stationsFile}`,
    `checked=${result.checked}`,
    `live=${result.live}`,
    `failed=${result.failed}`,
    `purged=${result.purged.length}`,
    `discovered=${result.discovered}`,
    `added=${result.added.length}`,
    `stations=${result.stationCountAfter}`,
    result.dryRun ? "dry_run=true" : "",
  ]
    .filter(Boolean)
    .join(" ");
}

async function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    console.error("");
    console.error(usage());
    process.exitCode = 2;
    return;
  }

  if (args.help) {
    console.log(usage());
    return;
  }

  try {
    const result = await maintainStations(args);
    for (const warning of result.warnings) {
      console.error(`radio-stations warning: ${warning}`);
    }
    if (args.json) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      console.log(formatSummary(result));
    }
  } catch (error) {
    console.error(`radio-stations CRITICAL reason="${String(error?.message ?? error)}"`);
    process.exitCode = 2;
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  await main();
}
