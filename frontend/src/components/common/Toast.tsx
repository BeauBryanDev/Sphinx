// ============================================================
// <Toast /> — small ephemeral notifications
// ============================================================

import { useApp } from "@/contexts/AppContext";
import { cn } from "@/utils/cn";

const VARIANT_STYLES = {
  info: "border-amber-500/60 bg-amber-900/80 text-amber-100",
  success: "border-emerald-500/60 bg-emerald-900/80 text-emerald-100",
  error: "border-red-500/60 bg-red-900/80 text-red-100",
} as const;

export const ToastContainer = () => {
  const { toasts, dismissToast } = useApp();

  return (
    <div className="pointer-events-none fixed right-4 top-4 z-50 flex w-80 flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={cn(
            "pointer-events-auto flex items-start justify-between gap-3 rounded-md border px-4 py-3 text-sm shadow-lg",
            VARIANT_STYLES[t.variant],
          )}
        >
          <span>{t.message}</span>
          <button
            onClick={() => dismissToast(t.id)}
            className="text-white/70 hover:text-white"
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
};

export default ToastContainer;
