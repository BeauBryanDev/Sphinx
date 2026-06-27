// ============================================================
// <Input /> — themed text input / textarea
// ============================================================

import { forwardRef, type InputHTMLAttributes, type ReactNode } from "react";
import { cn } from "@/utils/cn";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  leftIcon?: ReactNode;
  rightSlot?: ReactNode;
  invalid?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, leftIcon, rightSlot, invalid, disabled, ...rest }, ref) => {
    return (
      <div
        className={cn(
          "flex items-center gap-2 rounded-md border bg-amber-950/60 px-3 py-2 transition-colors",
          invalid
            ? "border-red-500/70 focus-within:ring-2 focus-within:ring-red-400/60"
            : "border-amber-700/60 focus-within:border-amber-400 focus-within:ring-2 focus-within:ring-amber-400/40",
          disabled && "opacity-60",
          className,
        )}
      >
        {leftIcon && <span className="text-amber-400">{leftIcon}</span>}
        <input
          ref={ref}
          disabled={disabled}
          className="flex-1 bg-transparent text-amber-100 placeholder:text-amber-400/50 focus:outline-none disabled:cursor-not-allowed"
          {...rest}
        />
        {rightSlot}
      </div>
    );
  },
);

Input.displayName = "Input";

export default Input;
