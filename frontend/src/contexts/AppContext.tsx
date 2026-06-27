// ============================================================
// AppContext — global UI state (active page, toast, sidebar)
// ============================================================

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { PageKey } from "@/types";

interface Toast {
  id: number;
  message: string;
  variant: "info" | "success" | "error";
}

interface AppContextValue {
  activePage: PageKey;
  setActivePage: (page: PageKey) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  closeSidebar: () => void;
  toasts: Toast[];
  pushToast: (message: string, variant?: Toast["variant"]) => void;
  dismissToast: (id: number) => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [activePage, setActivePage] = useState<PageKey>("home");
  // Default closed: mobile-first. On lg+ the sidebar is forced visible via CSS,
  // so this state only drives the mobile drawer.
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev);
  }, []);

  const closeSidebar = useCallback(() => {
    setSidebarOpen(false);
  }, []);

  const pushToast = useCallback(
    (message: string, variant: Toast["variant"] = "info") => {
      const id = Date.now();
      setToasts((prev) => [...prev, { id, message, variant }]);
      window.setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 3500);
    },
    [],
  );

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const value = useMemo<AppContextValue>(
    () => ({
      activePage,
      setActivePage,
      sidebarOpen,
      toggleSidebar,
      closeSidebar,
      toasts,
      pushToast,
      dismissToast,
    }),
    [
      activePage,
      sidebarOpen,
      toggleSidebar,
      closeSidebar,
      toasts,
      pushToast,
      dismissToast,
    ],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useApp = (): AppContextValue => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside <AppProvider>");
  return ctx;
};
