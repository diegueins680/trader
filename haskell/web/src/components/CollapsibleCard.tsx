import React from "react";

type CollapsibleCardProps = {
  panelId: string;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  open: boolean;
  onToggle?: (event: React.SyntheticEvent<HTMLDetailsElement>) => void;
  className?: string;
  summaryId?: string;
  style?: React.CSSProperties;
  containerRef?: React.Ref<HTMLDetailsElement>;
  maximized?: boolean;
  onToggleMaximize?: () => void;
};

export const CollapsibleCard = ({
  panelId,
  title,
  subtitle,
  children,
  open,
  onToggle,
  className,
  summaryId,
  style,
  containerRef,
  maximized = false,
  onToggleMaximize,
}: CollapsibleCardProps) => (
  <details
    className={`card cardCollapsible${maximized ? " cardMaximized" : ""}${className ? ` ${className}` : ""}`}
    open={open}
    onToggle={onToggle}
    data-panel={panelId}
    ref={containerRef}
    style={style}
  >
    <summary className="cardHeader cardSummary" id={summaryId}>
      <div className="cardHeaderText">
        <h2 className="cardTitle">{title}</h2>
        {subtitle ? <p className="cardSubtitle">{subtitle}</p> : null}
      </div>
      <div className="cardControls">
        {onToggleMaximize ? (
          <button
            className="cardControl"
            type="button"
            aria-pressed={maximized}
            aria-label={maximized ? "Restore panel size" : "Maximize panel"}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onToggleMaximize();
            }}
          >
            {maximized ? "Restore" : "Maximize"}
          </button>
        ) : null}
        <span className="cardToggle" aria-hidden="true">
          <span className="cardToggleLabel" data-open="Collapse" data-closed="Expand" />
          <span className="cardToggleIcon" />
        </span>
      </div>
    </summary>
    <div className="cardBody">{children}</div>
  </details>
);
