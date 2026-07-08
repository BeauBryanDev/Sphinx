// ============================================================
// Reverse API — English -> Middle Egyptian (POST /reverse/)
//
// Pure LLM feature on the backend (GPT-4o composes the Egyptian).
// Bare backend response — do NOT use the ApiResponse envelope helper.
// ============================================================

import http from "@/services/http";
import type { ReverseRegister, ReverseTranslationOut } from "@/types";

const REVERSE_TIMEOUT_MS = 120_000;

export const reverseTranslate = async (
  text: string,
  register: ReverseRegister = "unknown",
): Promise<ReverseTranslationOut> => {
  const { data } = await http.post<ReverseTranslationOut>(
    "/reverse/",
    { text, register },
    { timeout: REVERSE_TIMEOUT_MS },
  );
  return data;
};
