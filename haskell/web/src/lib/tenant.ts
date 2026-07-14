import type { Platform } from "./types";

const encoder = new TextEncoder();

function toHex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function sha256Hex(input: string): Promise<string> {
  const data = encoder.encode(input);
  const hash = await crypto.subtle.digest("SHA-256", data);
  return toHex(hash);
}

function isLegacyCredentialComponentSafe(value: string): boolean {
  return !value.includes(":");
}

function encodeCredentialTupleV2(components: readonly string[]): string {
  return `tenant-key-v2|${components.map((component) => `${encoder.encode(component).byteLength}:${component}`).join("")}`;
}

function trimCredential(value: string): string {
  return value.replace(/^[\t\n\v\f\r ]+|[\t\n\v\f\r ]+$/g, "");
}

export async function buildTenantKey(platform: Platform, key: string, secret: string, passphrase?: string): Promise<string | null> {
  const k = trimCredential(key);
  const s = trimCredential(secret);
  const p = trimCredential(passphrase ?? "");
  if (!k || !s) return null;
  if (platform === "coinbase" && !p) return null;
  const components = platform === "coinbase" ? [k, s, p] : [k, s];
  const legacySafe = components.every(isLegacyCredentialComponentSafe);
  const payload = legacySafe ? components.join(":") : encodeCredentialTupleV2(components);
  const hash = await sha256Hex(payload);
  return legacySafe ? `${platform}:${hash}` : `${platform}:v2:${hash}`;
}

export async function buildTenantKeyForPlatform(
  platform: Platform,
  key: string,
  secret: string,
  passphrase?: string,
): Promise<string | null> {
  return buildTenantKey(platform, key, secret, passphrase);
}
