// ============================================================
// <Button /> — reusable themed button with Egyptian styling
// ============================================================

import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cn } from "@/utils/cn";

type Variant = "primary" | "secondary" | "ghost";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  isLoading?: boolean;
}

const VARIANT_STYLES: Record<Variant, string> = {
  primary:
    "bg-gradient-to-b from-amber-400 to-amber-600 text-amber-950 border border-amber-300 shadow-[0_4px_14px_rgba(180,120,30,0.35)] hover:from-amber-300 hover:to-amber-500",
  secondary:
    "bg-amber-900/40 text-amber-200 border border-amber-700/60 hover:bg-amber-800/60",
  ghost:
    "bg-transparent text-amber-200 border border-transparent hover:bg-amber-900/40",
};

const SIZE_STYLES: Record<Size, string> = {
  sm: "px-3 py-1.5 text-sm",
  md: "px-5 py-2.5 text-base",
  lg: "px-7 py-3.5 text-lg",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = "primary",
      size = "md",
      leftIcon,
      rightIcon,
      isLoading,
      disabled,
      children,
      ...rest
    },
    ref,
  ) => {
    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-md font-semibold tracking-wide transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-amber-400/70 disabled:opacity-60 disabled:cursor-not-allowed",
          VARIANT_STYLES[variant],
          SIZE_STYLES[size],
          className,
        )}
        {...rest}
      >
        {isLoading ? (
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-amber-950/30 border-t-amber-950" />
        ) : (
          leftIcon
        )}
        {children}
        {!isLoading && rightIcon}
      </button>
    );
  },
);

Button.displayName = "Button";

export default Button;
