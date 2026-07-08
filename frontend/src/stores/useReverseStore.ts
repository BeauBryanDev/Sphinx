// ============================================================
// useReverseStore — Zustand store for the reverse-translation page
//
// Global store: the draft text and the composed Middle Egyptian
// SURVIVE navigating to other pages and back.
// ============================================================

import { create } from "zustand";
import { reverseTranslate } from "@/api/reverseApi";
import type { ReverseRegister, ReverseTranslationOut } from "@/types";

export const MAX_REVERSE_CHARS = 300;

export interface ReverseStore {
  text: string;
  register: ReverseRegister;
  result: ReverseTranslationOut | null;
  isLoading: boolean;
  error: string | null;
  setText: (t: string) => void;
  setRegister: (r: ReverseRegister) => void;
  translate: () => Promise<void>;
  reset: () => void;
}

export const useReverseStore = create<ReverseStore>((set, get) => ({
  text: "",
  register: "unknown",
  result: null,
  isLoading: false,
  error: null,

  setText: (t) => set({ text: t.slice(0, MAX_REVERSE_CHARS) }),
  setRegister: (r) => set({ register: r }),

  translate: async () => {
    const { text, register, isLoading } = get();
    if (!text.trim() || isLoading) return;
    set({ isLoading: true, error: null });
    try {
      set({ result: await reverseTranslate(text.trim(), register) });
    } catch (err) {
      set({
        result: null,
        error:
          err instanceof Error ? err.message : "Reverse translation failed.",
      });
    } finally {
      set({ isLoading: false });
    }
  },

  reset: () =>
    set({ text: "", register: "unknown", result: null, error: null }),
}));
