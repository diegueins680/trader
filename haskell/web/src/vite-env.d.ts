/// <reference types="vite/client" />

declare const __TRADER_API_TARGET__: string;

interface Window {
  __TRADER_CONFIG__?: {
    apiBaseUrl?: string;
    apiToken?: string;
  };
}
