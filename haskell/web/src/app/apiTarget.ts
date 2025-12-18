export const API_TARGET = (__TRADER_API_TARGET__ || "http://127.0.0.1:8080").replace(/\/+$/, "");

export const API_PORT = (() => {
  try {
    return new URL(API_TARGET).port || "8080";
  } catch {
    return "8080";
  }
})();
