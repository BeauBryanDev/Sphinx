// ============================================================
// <MobileNav /> — fixed bottom tab bar (mobile / tablet only)
// Mirrors the sidebar nav; hidden on lg+ where the Sidebar shows.
// ============================================================

import { NAV_ITEMS } from "@/constants";
import { useApp } from "@/contexts/AppContext";
import { cn } from "@/utils/cn";

export const MobileNav = () => {
  const { activePage, setActivePage } = useApp();

  return (
    <nav
      className="fixed inset-x-0 bottom-0 z-30 flex items-stretch justify-around border-t border-amber-700/50 bg-stone-950/95 px-1 py-1.5 backdrop-blur lg:hidden"
      aria-label="Primary"
    >
      {NAV_ITEMS.map((item) => {
        const isActive = item.key === activePage;
        return (
          <button
            key={item.key}
            onClick={() => setActivePage(item.key)}
            aria-current={isActive ? "page" : undefined}
            className={cn(
              "flex min-w-0 flex-1 flex-col items-center gap-0.5 rounded-md px-1 py-1.5 transition-colors",
              isActive
                ? "text-amber-200"
                : "text-amber-400/70 hover:text-amber-200",
            )}
          >
            <span
              className={cn(
                "flex h-8 w-8 items-center justify-center rounded-md text-base transition-colors",
                isActive
                  ? "bg-amber-500 text-amber-950"
                  : "bg-amber-900/40 text-amber-300",
              )}
            >
              {item.icon}
            </span>
            <span className="w-full truncate text-center text-[10px] font-medium tracking-wide">
              {item.label}
            </span>
          </button>
        );
      })}
    </nav>
  );
};

export default MobileNav;
