// ============================================================
// useGlyphDecoder — hieroglyph upload/decode flow hook
//
// Now a thin wrapper over the global Zustand store, so the
// uploaded image, background detection and decode result survive
// page navigation. Same interface as the old useState
// implementation — components are unchanged.
//
// Two-stage latency-hiding flow (implemented in the store):
//   1. picking a reading direction FIRES DETECTION in the
//      background while the user fills the context form;
//   2. "Decode Glyphs" awaits it, then calls /transliterate/
//      (only this LLM stage shows the scanning animation).
// ============================================================

import { useGlyphStore, type GlyphStore } from "@/stores";

export type UseGlyphDecoderReturn = GlyphStore;

export const useGlyphDecoder = (): UseGlyphDecoderReturn => useGlyphStore();

export default useGlyphDecoder;
