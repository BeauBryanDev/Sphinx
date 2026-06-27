// ============================================================
// <Card /> — ornate Egyptian-styled panel wrapper
// ============================================================

import { type HTMLAttributes, type ReactNode } from "react";
import { cn } from "@/utils/cn";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  icon?: ReactNode;
  footer?: ReactNode;
  padded?: boolean;
}

export const Card = ({
  title,
  subtitle,
  icon,
  footer,
  padded = true,
  className,
  children,
  ...rest
}: CardProps) => {
  return (
    <div
      className={cn(
        "relative rounded-lg border border-amber-700/60 bg-gradient-to-b from-amber-950/80 to-stone-900/80 shadow-[0_8px_30px_rgba(0,0,0,0.45)]",
        className,
      )}
      {...rest}
    >
      {/* Decorative corner glyphs */}
      <span className="pointer-events-none absolute left-2 top-2 text-amber-600/60 text-xs">𓋹</span>
      <span className="pointer-events-none absolute right-2 top-2 text-amber-600/60 text-xs">𓂀</span>

      {(title || icon) && (
        <header className="flex flex-col items-center gap-1 border-b border-amber-700/40 px-6 pt-5 pb-3 text-center">
          {icon && <span className="text-amber-400">{icon}</span>}
          {title && (
            <h3 className="text-lg font-bold tracking-[0.25em] text-amber-300 uppercase">
              {title}
            </h3>
          )}
          {subtitle && (
            <p className="text-sm text-amber-200/80 italic">{subtitle}</p>
          )}
          <div className="mt-1 flex items-center gap-2 text-amber-500/70 text-xs">
            <span>—</span>
            <span>☥</span>
            <span>—</span>
          </div>
        </header>
      )}

      <div className={cn(padded && "p-6")}>{children}</div>

      {footer && (
        <footer className="border-t border-amber-700/40 px-6 py-4">
          {footer}
        </footer>
      )}
    </div>
  );
};

export default Card;
