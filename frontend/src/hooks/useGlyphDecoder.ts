// ============================================================
// useGlyphDecoder — stateful hook for the hieroglyph upload flow
// ============================================================

import { useCallback, useState } from "react";
import { decodeGlyphs } from "@/api/glyphApi";
import type { GlyphDecodingResult, UploadedImage } from "@/types";

export interface UseGlyphDecoderReturn {
  image: UploadedImage | null;
  result: GlyphDecodingResult | null;
  isLoading: boolean;
  error: string | null;
  setImage: (file: File) => void;
  clearImage: () => void;
  decode: () => Promise<void>;
}

const readAsDataUrl = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });

export const useGlyphDecoder = (): UseGlyphDecoderReturn => {
  const [image, setImageState] = useState<UploadedImage | null>(null);
  const [result, setResult] = useState<GlyphDecodingResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setImage = useCallback(async (file: File) => {
    const preview = await readAsDataUrl(file);
    setImageState({ file, preview });
    setResult(null);
    setError(null);
  }, []);

  const clearImage = useCallback(() => {
    setImageState(null);
    setResult(null);
    setError(null);
  }, []);

  const decode = useCallback(async () => {
    if (!image) return;
    setIsLoading(true);
    setError(null);
    try {
      const res = await decodeGlyphs(image.file);
      setResult(res);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Decoding failed.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [image]);

  return { image, result, isLoading, error, setImage, clearImage, decode };
};

export default useGlyphDecoder;
