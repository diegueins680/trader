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

export async function buildTenantKey(platform: Platform, key: string, secret: string, passphrase?: string): Promise<string | null> {
  const k = key.trim();
  const s = secret.trim();
  const p = passphrase?.trim() ?? "";
  if (!k || !s) return null;
  if (platform === "coinbase" && !p) return null;
  const payload = platform === "coinbase" ? `${k}:${s}:${p}` : `${k}:${s}`;
  const hash = await sha256Hex(payload);
  return `${platform}:${hash}`;
}

export async function buildTenantKeyForPlatform(
  platform: Platform,
  key: string,
  secret: string,
  passphrase?: string,
): Promise<string | null> {
  return buildTenantKey(platform, key, secret, passphrase);
}
