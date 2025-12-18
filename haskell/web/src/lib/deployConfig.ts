export type TraderUiDeployConfig = {
  apiBaseUrl: string;
  apiToken: string;
};

function readString(raw: unknown): string {
  return typeof raw === "string" ? raw : "";
}

function readConfigFromGlobal(): TraderUiDeployConfig {
  if (typeof window === "undefined") return { apiBaseUrl: "", apiToken: "" };
  const raw = window.__TRADER_CONFIG__;
  if (!raw || typeof raw !== "object") return { apiBaseUrl: "", apiToken: "" };

  return {
    apiBaseUrl: readString((raw as { apiBaseUrl?: unknown }).apiBaseUrl).trim(),
    apiToken: readString((raw as { apiToken?: unknown }).apiToken).trim(),
  };
}

export const TRADER_UI_CONFIG: TraderUiDeployConfig = readConfigFromGlobal();

