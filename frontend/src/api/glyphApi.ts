// ============================================================
// Glyph API — endpoint wrappers for the hieroglyph decoder
// ============================================================

import { request } from "@/services/http";
import type { GlyphDecodingResult } from "@/types";

/** Upload an image of hieroglyphs for transliteration. */
export const decodeGlyphs = (file: File): Promise<GlyphDecodingResult> => {
  const form = new FormData();
  form.append("image", file);

  return request<GlyphDecodingResult>({
    method: "POST",
    url: "/v1/glyphs/decode",
    data: form,
    headers: { "Content-Type": "multipart/form-data" },
  });
};

/** Fetch previous decodings for the current user. */
export const fetchDecodingHistory = (): Promise<GlyphDecodingResult[]> => {
  return request<GlyphDecodingResult[]>({
    method: "GET",
    url: "/v1/glyphs/history",
  });
};
