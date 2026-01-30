import React, { useEffect, useState } from "react";
import type { DataLogEntry } from "../app/dataLog";
import { DATA_LOG_COLLAPSED_MAX_LINES } from "../app/constants";
import { indexTopLevelPrimitiveArrays } from "../app/utils";

export type DataLogPanelProps = {
  dataLog: DataLogEntry[];
  dataLogShown: DataLogEntry[];
  dataLogShownDeferred: DataLogEntry[];
  dataLogFiltered: DataLogEntry[];
  dataLogFilterText: string;
  setDataLogFilterText: (value: string) => void;
  dataLogExpanded: boolean;
  setDataLogExpanded: (value: boolean) => void;
  dataLogLogErrors: boolean;
  setDataLogLogErrors: (value: boolean) => void;
  dataLogLogBackground: boolean;
  setDataLogLogBackground: (value: boolean) => void;
  dataLogIndexArrays: boolean;
  setDataLogIndexArrays: (value: boolean) => void;
  dataLogAutoScroll: boolean;
  setDataLogAutoScroll: (value: boolean) => void;
  dataLogPersist: boolean;
  setDataLogPersist: (value: boolean) => void;
  setDataLog: (value: DataLogEntry[]) => void;
  copyText: (value: string) => Promise<void> | void;
  showToast: (value: string) => void;
  scrollDataLogToBottom: () => void;
  dataLogRef: React.RefObject<HTMLDivElement>;
  handleDataLogScroll: () => void;
};

export function DataLogPanel({
  dataLog,
  dataLogShown,
  dataLogShownDeferred,
  dataLogFiltered,
  dataLogFilterText,
  setDataLogFilterText,
  dataLogExpanded,
  setDataLogExpanded,
  dataLogLogErrors,
  setDataLogLogErrors,
  dataLogLogBackground,
  setDataLogLogBackground,
  dataLogIndexArrays,
  setDataLogIndexArrays,
  dataLogAutoScroll,
  setDataLogAutoScroll,
  dataLogPersist,
  setDataLogPersist,
  setDataLog,
  copyText,
  showToast,
  scrollDataLogToBottom,
  dataLogRef,
  handleDataLogScroll,
}: DataLogPanelProps) {
  const [confirmClear, setConfirmClear] = useState(false);
  const [clearedCache, setClearedCache] = useState<DataLogEntry[] | null>(null);

  useEffect(() => {
    if (!confirmClear) return;
    const t = window.setTimeout(() => setConfirmClear(false), 4000);
    return () => window.clearTimeout(t);
  }, [confirmClear]);

  useEffect(() => {
    if (!clearedCache) return;
    if (dataLog.length > 0) {
      setClearedCache(null);
    }
  }, [clearedCache, dataLog.length]);

  return (
    <>
        <div className="actions dataLogActions">
          <button
            className="btn"
            onClick={() => {
              if (!confirmClear) {
                setConfirmClear(true);
                return;
              }
              if (dataLog.length === 0) return;
              setClearedCache(dataLog);
              setDataLog([]);
              setConfirmClear(false);
              showToast("Data log cleared");
            }}
            disabled={dataLog.length === 0}
          >
            {confirmClear ? "Confirm clear" : "Clear log"}
          </button>
          {confirmClear ? (
            <button className="btnSmall" type="button" onClick={() => setConfirmClear(false)}>
              Cancel
            </button>
          ) : null}
	            <button
	              className="btn"
                disabled={dataLogShown.length === 0}
	              onClick={() => {
	                const logText = dataLogShown
	                  .map((entry) => `[${new Date(entry.timestamp).toISOString()}] ${entry.label}:\n${JSON.stringify(entry.data, null, 2)}`)
	                  .join("\n\n");
	                copyText(logText);
	                showToast("Copied log to clipboard");
	              }}
	            >
	              {dataLogFilterText.trim() ? "Copy shown" : "Copy all"}
	            </button>
              <input
                className="input dataLogFilter"
                value={dataLogFilterText}
                onChange={(e) => setDataLogFilterText(e.target.value)}
                placeholder="Filter log…"
                spellCheck={false}
                aria-label="Filter data log"
              />
              {dataLogFilterText.trim() ? (
                <button className="btnSmall" type="button" onClick={() => setDataLogFilterText("")}>
                  Clear filter
                </button>
              ) : null}
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogExpanded} onChange={(e) => setDataLogExpanded(e.target.checked)} />
                Expand
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogLogErrors} onChange={(e) => setDataLogLogErrors(e.target.checked)} />
                Log errors
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input
                  type="checkbox"
                  checked={dataLogLogBackground}
                  onChange={(e) => setDataLogLogBackground(e.target.checked)}
                />
                Log auto-refresh
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogIndexArrays} onChange={(e) => setDataLogIndexArrays(e.target.checked)} />
                Index arrays
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogAutoScroll} onChange={(e) => setDataLogAutoScroll(e.target.checked)} />
                Auto-scroll
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogPersist} onChange={(e) => setDataLogPersist(e.target.checked)} />
                Remember
              </label>
              <button className="btnSmall" type="button" onClick={scrollDataLogToBottom} disabled={dataLog.length === 0}>
                Jump to latest
              </button>
              {dataLogFilterText.trim() ? (
                <span className="hint">
                  Showing {dataLogFiltered.length} of {dataLog.length}
                </span>
              ) : null}
        </div>
        {clearedCache ? (
          <div className="undoBar">
            <span>Log cleared.</span>
            <button
              className="btnSmall"
              type="button"
              onClick={() => {
                setDataLog(clearedCache);
                setClearedCache(null);
                showToast("Log restored");
              }}
            >
              Undo
            </button>
            <button className="btnSmall" type="button" onClick={() => setClearedCache(null)}>
              Dismiss
            </button>
          </div>
        ) : null}
        <div ref={dataLogRef} className="dataLogBox" onScroll={handleDataLogScroll}>
            {dataLogShownDeferred.length === 0 ? (
              <div className="dataLogEmpty">
                {dataLog.length === 0
                  ? "No data logged yet. Run a signal, backtest, trade, or bot action to see incoming data."
                  : "No entries match the current filter."}
              </div>
            ) : (
              dataLogShownDeferred.map((entry, idx) => (
                <div key={idx} className="dataLogEntry">
                  <div className="dataLogEntryHeader">
                    [{new Date(entry.timestamp).toLocaleTimeString()}] <span className="dataLogEntryLabel">{entry.label}</span>
                  </div>
                  <div className="dataLogEntryBody">
                      {(() => {
                        const data = dataLogIndexArrays ? indexTopLevelPrimitiveArrays(entry.data) : entry.data;
                        const json = JSON.stringify(data, null, 2);
                        if (dataLogExpanded) return json;
                        const lines = json.split("\n");
                        const head = lines.slice(0, DATA_LOG_COLLAPSED_MAX_LINES).join("\n");
                        return lines.length > DATA_LOG_COLLAPSED_MAX_LINES ? `${head}\n... (truncated)` : head;
                      })()}
                  </div>
                </div>
              ))
            )}
          </div>
    </>
  );
}
