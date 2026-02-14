import React, { useEffect, useRef } from "react";

type InfoPopoverProps = {
  label: string;
  children: React.ReactNode;
  align?: "left" | "right";
};

type InfoListProps = {
  items: string[];
};

export const InfoPopover = ({ label, children, align = "right" }: InfoPopoverProps) => {
  const detailsRef = useRef<HTMLDetailsElement | null>(null);

  useEffect(() => {
    if (typeof document === "undefined") return;

    const handlePointerDown = (event: PointerEvent) => {
      const el = detailsRef.current;
      if (!el || !el.open) return;
      const target = event.target;
      if (target instanceof Node && !el.contains(target)) {
        el.open = false;
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      const el = detailsRef.current;
      if (el && el.open) {
        el.open = false;
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  return (
    <details ref={detailsRef} className={`infoPopover${align === "left" ? " infoPopoverLeft" : ""}`}>
      <summary className="infoButton" aria-label={label} title={label}>
        i
      </summary>
      <div className="infoContent" role="note">
        {children}
      </div>
    </details>
  );
};

export const InfoList = ({ items }: InfoListProps) => (
  <ul className="infoList">
    {items.map((item) => (
      <li key={item}>{item}</li>
    ))}
  </ul>
);
