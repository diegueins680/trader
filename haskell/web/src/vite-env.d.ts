/// <reference types="vite/client" />

declare const __TRADER_API_TARGET__: string;
declare const __TRADER_UI_COMMIT__: string;

declare module "/trader-config.js";

interface Window {
  __TRADER_CONFIG__?: {
    apiBaseUrl?: string;
    apiBaseUrlInferred?: boolean;
    apiFallbackUrl?: string;
    apiToken?: string;
    timeoutsMs?: {
      requestMs?: number;
      signalMs?: number;
      backtestMs?: number;
      tradeMs?: number;
      botStartMs?: number;
      botStatusMs?: number;
    };
  };
}
