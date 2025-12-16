export function fmtPct(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

export function fmtNum(x: number, digits = 6): string {
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

export function fmtMoney(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

export function fmtRatio(x: number, digits = 4): string {
  if (!Number.isFinite(x)) return "—";
  return `${x.toFixed(digits)}x`;
}
