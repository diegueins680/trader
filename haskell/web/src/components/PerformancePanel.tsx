import React, { useMemo } from "react";
import type { OpsPerformanceUiState } from "../app/appHelpers";
import { csvEscape, downloadTextFile, pnlBadgeClass } from "../app/appHelpers";
import { clamp, fmtTimeMs, formatIsoUtc, marketLabel, numFromInput, shortComboUuid, shortCommitHash } from "../app/utils";
import { fmtPct } from "../lib/format";
import type { Market } from "../lib/types";

export type PerformancePanelProps = {
  apiOk: "unknown" | "ok" | "down" | "auth";
  opsPerformanceUi: OpsPerformanceUiState;
  opsPerformanceCommitLimit: number;
  setOpsPerformanceCommitLimit: (value: number) => void;
  opsPerformanceComboLimit: number;
  setOpsPerformanceComboLimit: (value: number) => void;
  opsPerformanceComboScope: "latest" | "all";
  setOpsPerformanceComboScope: (value: "latest" | "all") => void;
  opsPerformanceComboOrder: "delta" | "recent" | "return";
  setOpsPerformanceComboOrder: (value: "delta" | "recent" | "return") => void;
  fetchOpsPerformance: () => Promise<void>;
};

export function PerformancePanel({
  apiOk,
  opsPerformanceUi,
  opsPerformanceCommitLimit,
  setOpsPerformanceCommitLimit,
  opsPerformanceComboLimit,
  setOpsPerformanceComboLimit,
  opsPerformanceComboScope,
  setOpsPerformanceComboScope,
  opsPerformanceComboOrder,
  setOpsPerformanceComboOrder,
  fetchOpsPerformance,
}: PerformancePanelProps) {
  const commitsCsv = useMemo(() => {
    if (opsPerformanceUi.commits.length === 0) return "";
    const header = [
      "commitHash",
      "committedAtMs",
      "committedAtIso",
      "medianReturn",
      "deltaMedianReturn",
      "medianDrawdown",
      "deltaMedianDrawdown",
      "worstDrawdown",
      "deltaWorstDrawdown",
      "rollups",
      "combos",
      "symbols",
    ];
    const rows = opsPerformanceUi.commits.map((row) => {
      const committedAtMs = row.committedAtMs;
      const committedAtIso = formatIsoUtc(committedAtMs);
      return [
        row.commitHash ?? "",
        committedAtMs ?? "",
        committedAtIso,
        row.medianReturn ?? "",
        row.deltaMedianReturn ?? "",
        row.medianDrawdown ?? "",
        row.deltaMedianDrawdown ?? "",
        row.worstDrawdown ?? "",
        row.deltaWorstDrawdown ?? "",
        row.rollups ?? "",
        row.combos ?? "",
        row.symbols ?? "",
      ];
    });
    return [header.map(csvEscape).join(","), ...rows.map((row) => row.map(csvEscape).join(","))].join("\n");
  }, [opsPerformanceUi.commits]);

  const combosCsv = useMemo(() => {
    if (opsPerformanceUi.combos.length === 0) return "";
    const header = [
      "symbol",
      "market",
      "interval",
      "comboUuid",
      "return",
      "deltaReturn",
      "maxDrawdown",
      "deltaDrawdown",
      "commitHash",
      "committedAtMs",
      "committedAtIso",
    ];
    const rows = opsPerformanceUi.combos.map((row) => {
      const committedAtMs = row.committedAtMs;
      const committedAtIso = formatIsoUtc(committedAtMs);
      return [
        row.symbol ?? "",
        row.market ?? "",
        row.interval ?? "",
        row.comboUuid ?? "",
        row.return ?? "",
        row.deltaReturn ?? "",
        row.maxDrawdown ?? "",
        row.deltaDrawdown ?? "",
        row.commitHash ?? "",
        committedAtMs ?? "",
        committedAtIso,
      ];
    });
    return [header.map(csvEscape).join(","), ...rows.map((row) => row.map(csvEscape).join(","))].join("\n");
  }, [opsPerformanceUi.combos]);

  return (
    <>
<div className="row" style={{ marginBottom: 10, gridTemplateColumns: "repeat(4, minmax(0, 1fr)) auto", alignItems: "end" }}>
  <div className="field">
    <label className="label" htmlFor="opsPerfCommitLimit">
      Commit rows
    </label>
    <input
      id="opsPerfCommitLimit"
      className="input"
      type="number"
      min={1}
      max={200}
      value={opsPerformanceCommitLimit}
      onChange={(e) =>
        setOpsPerformanceCommitLimit(
          clamp(Math.trunc(numFromInput(e.target.value, opsPerformanceCommitLimit)), 1, 200),
        )
      }
    />
  </div>
  <div className="field">
    <label className="label" htmlFor="opsPerfComboLimit">
      Combo rows
    </label>
    <input
      id="opsPerfComboLimit"
      className="input"
      type="number"
      min={1}
      max={200}
      value={opsPerformanceComboLimit}
      onChange={(e) =>
        setOpsPerformanceComboLimit(
          clamp(Math.trunc(numFromInput(e.target.value, opsPerformanceComboLimit)), 1, 200),
        )
      }
    />
  </div>
  <div className="field">
    <label className="label" htmlFor="opsPerfComboScope">
      Combo scope
    </label>
    <select
      id="opsPerfComboScope"
      className="select"
      value={opsPerformanceComboScope}
      onChange={(e) => setOpsPerformanceComboScope(e.target.value === "all" ? "all" : "latest")}
    >
      <option value="latest">Latest commit</option>
      <option value="all">All history</option>
    </select>
  </div>
  <div className="field">
    <label className="label" htmlFor="opsPerfComboOrder">
      Combo order
    </label>
    <select
      id="opsPerfComboOrder"
      className="select"
      value={opsPerformanceComboOrder}
      onChange={(e) => {
        const value = e.target.value;
        if (value === "recent" || value === "return" || value === "delta") setOpsPerformanceComboOrder(value);
      }}
    >
      <option value="delta">Worst delta return</option>
      <option value="recent">Most recent</option>
      <option value="return">Highest return</option>
    </select>
  </div>
  <button
    className="btn"
    type="button"
    disabled={opsPerformanceUi.loading || apiOk !== "ok"}
    onClick={() => void fetchOpsPerformance()}
  >
    {opsPerformanceUi.loading ? "Loading..." : "Refresh"}
  </button>
</div>

{!opsPerformanceUi.enabled ? (
  <div className="hint">{opsPerformanceUi.hint ?? "Enable TRADER_DB_URL to load performance rollups."}</div>
) : opsPerformanceUi.hint ? (
  <div className="hint">{opsPerformanceUi.hint}</div>
) : null}

{opsPerformanceUi.error ? (
  <div className="hint" style={{ color: "rgba(239, 68, 68, 0.9)", marginBottom: 6 }}>
    {opsPerformanceUi.error}
  </div>
) : null}

<div className="pillRow" style={{ marginBottom: 10 }}>
  <span className="badge">Commits {opsPerformanceUi.commits.length}</span>
  <span className="badge">Combos {opsPerformanceUi.combos.length}</span>
  {opsPerformanceUi.lastFetchedAtMs ? <span className="badge">Synced {fmtTimeMs(opsPerformanceUi.lastFetchedAtMs)}</span> : null}
</div>
<div className="actions" style={{ marginBottom: 12 }}>
  <button
    className="btnSmall"
    type="button"
    disabled={!commitsCsv}
    onClick={() => {
      const stamp = Date.now();
      downloadTextFile(`performance-commits-${stamp}.csv`, commitsCsv, "text/csv");
    }}
  >
    Download commits CSV
  </button>
  <button
    className="btnSmall"
    type="button"
    disabled={!combosCsv}
    onClick={() => {
      const stamp = Date.now();
      downloadTextFile(`performance-combos-${stamp}.csv`, combosCsv, "text/csv");
    }}
  >
    Download combos CSV
  </button>
</div>

<div style={{ marginBottom: 12 }}>
  <div className="hint" style={{ marginBottom: 6 }}>
    Commit deltas
  </div>
  {opsPerformanceUi.commits.length === 0 ? (
    <div className="hint">No commit rollups yet.</div>
  ) : (
    <div className="tableWrap" role="region" aria-label="Commit performance rollups">
      <table className="table">
        <thead>
          <tr>
            <th>Commit</th>
            <th>Committed</th>
            <th>Median return</th>
            <th>Δ return</th>
            <th>Median DD</th>
            <th>Δ DD</th>
            <th>Worst DD</th>
            <th>Δ worst</th>
            <th>Samples</th>
          </tr>
        </thead>
        <tbody>
          {opsPerformanceUi.commits.map((row) => {
            const commitLabel = shortCommitHash(row.commitHash);
            const commitTitle = row.commitHash ?? "—";
            const committedLabel =
              typeof row.committedAtMs === "number" && Number.isFinite(row.committedAtMs)
                ? fmtTimeMs(row.committedAtMs)
                : "—";
            const medianReturn = typeof row.medianReturn === "number" && Number.isFinite(row.medianReturn)
              ? fmtPct(row.medianReturn, 2)
              : "—";
            const deltaReturn = typeof row.deltaMedianReturn === "number" && Number.isFinite(row.deltaMedianReturn)
              ? fmtPct(row.deltaMedianReturn, 2)
              : "—";
            const medianDd = typeof row.medianDrawdown === "number" && Number.isFinite(row.medianDrawdown)
              ? fmtPct(row.medianDrawdown, 2)
              : "—";
            const deltaDd = typeof row.deltaMedianDrawdown === "number" && Number.isFinite(row.deltaMedianDrawdown)
              ? fmtPct(row.deltaMedianDrawdown, 2)
              : "—";
            const worstDd = typeof row.worstDrawdown === "number" && Number.isFinite(row.worstDrawdown)
              ? fmtPct(row.worstDrawdown, 2)
              : "—";
            const deltaWorst = typeof row.deltaWorstDrawdown === "number" && Number.isFinite(row.deltaWorstDrawdown)
              ? fmtPct(row.deltaWorstDrawdown, 2)
              : "—";
            const samples = `${row.rollups ?? 0} rollups • ${row.combos ?? 0} combos • ${row.symbols ?? 0} symbols`;
            return (
              <tr key={`commit-${row.gitCommitId}`}>
                <td className="tdMono" title={commitTitle}>
                  {commitLabel}
                </td>
                <td>{committedLabel}</td>
                <td>
                  <span className={pnlBadgeClass(row.medianReturn)}>{medianReturn}</span>
                </td>
                <td>
                  <span className={pnlBadgeClass(row.deltaMedianReturn)}>{deltaReturn}</span>
                </td>
                <td>
                  <span className={pnlBadgeClass(row.medianDrawdown)}>{medianDd}</span>
                </td>
                <td>
                  <span className={pnlBadgeClass(row.deltaMedianDrawdown)}>{deltaDd}</span>
                </td>
                <td>
                  <span className={pnlBadgeClass(row.worstDrawdown)}>{worstDd}</span>
                </td>
                <td>
                  <span className={pnlBadgeClass(row.deltaWorstDrawdown)}>{deltaWorst}</span>
                </td>
                <td>{samples}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  )}
</div>

<div>
  <div className="hint" style={{ marginBottom: 6 }}>
    Combo deltas
  </div>
  {!opsPerformanceUi.combosReady ? (
    <div className="hint">Combo deltas are unavailable until rollups are built.</div>
  ) : opsPerformanceUi.combos.length === 0 ? (
    <div className="hint">No combo deltas available for the selected scope.</div>
  ) : (
    <div className="tableWrap" role="region" aria-label="Combo performance deltas">
      <table className="table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Market</th>
            <th>Interval</th>
            <th>Combo</th>
            <th>Return</th>
            <th>Δ return</th>
            <th>Max DD</th>
            <th>Δ DD</th>
            <th>Commit</th>
          </tr>
        </thead>
        <tbody>
          {opsPerformanceUi.combos.map((row, idx) => {
            const symbol = row.symbol ?? "—";
            const market = row.market ? marketLabel(row.market as Market) : "—";
            const interval = row.interval ?? "—";
            const combo = shortComboUuid(row.comboUuid);
            const commitLabel = shortCommitHash(row.commitHash, 7);
            const commitTitle = row.commitHash ?? "—";
            const ret = typeof row.return === "number" && Number.isFinite(row.return) ? fmtPct(row.return, 2) : "—";
            const deltaRet = typeof row.deltaReturn === "number" && Number.isFinite(row.deltaReturn)
              ? fmtPct(row.deltaReturn, 2)
              : "—";
            const dd = typeof row.maxDrawdown === "number" && Number.isFinite(row.maxDrawdown)
              ? fmtPct(row.maxDrawdown, 2)
              : "—";
            const deltaDd = typeof row.deltaDrawdown === "number" && Number.isFinite(row.deltaDrawdown)
              ? fmtPct(row.deltaDrawdown, 2)
              : "—";
            return (
              <tr key={`combo-${row.comboUuid ?? idx}-${row.gitCommitId}`}>
                <td className="tdMono">{symbol}</td>
                <td>{market}</td>
                <td>{interval}</td>
                <td className="tdMono" title={row.comboUuid ?? "—"}>
                  {combo}
                </td>
                <td>
                  <span className={pnlBadgeClass(row.return)}>{ret}</span>
                </td>
                <td>
                  <span className={pnlBadgeClass(row.deltaReturn)}>{deltaRet}</span>
                </td>
                <td>
                  <span className={pnlBadgeClass(row.maxDrawdown)}>{dd}</span>
                </td>
                <td>
                  <span className={pnlBadgeClass(row.deltaDrawdown)}>{deltaDd}</span>
                </td>
                <td className="tdMono" title={commitTitle}>
                  {commitLabel}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  )}
</div>
    </>
  );
}
