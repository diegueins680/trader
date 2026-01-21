import React from "react";

type InfoPopoverProps = {
  label: string;
  children: React.ReactNode;
  align?: "left" | "right";
};

type InfoListProps = {
  items: string[];
};

export const InfoPopover = ({ label, children, align = "right" }: InfoPopoverProps) => (
  <details className={`infoPopover${align === "left" ? " infoPopoverLeft" : ""}`}>
    <summary className="infoButton" aria-label={label} title={label}>
      i
    </summary>
    <div className="infoContent" role="note">
      {children}
    </div>
  </details>
);

export const InfoList = ({ items }: InfoListProps) => (
  <ul className="infoList">
    {items.map((item) => (
      <li key={item}>{item}</li>
    ))}
  </ul>
);
