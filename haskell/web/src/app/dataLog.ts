import { DATA_LOG_MAX_ENTRIES } from "./constants";

export type DataLogEntry = { timestamp: number; label: string; data: unknown };

export type DataLogPrefs = {
  expanded: boolean;
  indexArrays: boolean;
  autoScroll: boolean;
  logBackground: boolean;
  logErrors: boolean;
  persist: boolean;
};

export const DEFAULT_DATA_LOG_PREFS: DataLogPrefs = {
  expanded: false,
  indexArrays: true,
  autoScroll: true,
  logBackground: true,
  logErrors: true,
  persist: true,
};

export const normalizeDataLog = (raw: unknown): DataLogEntry[] => {
  if (!Array.isArray(raw)) return [];
  const out: DataLogEntry[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== "object") continue;
    const record = entry as Record<string, unknown>;
    const timestamp = typeof record.timestamp === "number" && Number.isFinite(record.timestamp) ? record.timestamp : null;
    const label = typeof record.label === "string" ? record.label : null;
    if (timestamp === null || !label) continue;
    const data = "data" in record ? record.data : null;
    out.push({ timestamp, label, data });
  }
  if (out.length <= DATA_LOG_MAX_ENTRIES) return out;
  return out.slice(out.length - DATA_LOG_MAX_ENTRIES);
};
