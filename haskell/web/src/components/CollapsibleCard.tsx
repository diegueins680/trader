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
    </summary>
    <div className="cardBody">{children}</div>
  </details>
);
