// ============================================================
// <GlyphDecoder /> — image upload + transliteration result
// ============================================================

import { useCallback, useRef, type DragEvent, type ChangeEvent } from "react";
import { Card, Button } from "@/components/common";
import { useGlyphDecoder } from "@/hooks/useGlyphDecoder";
import { formatConfidence } from "@/utils/formatters";

export const GlyphDecoder = () => {
  const { image, result, isLoading, error, setImage, clearImage, decode } =
    useGlyphDecoder();
  const inputRef = useRef<HTMLInputElement>(null);
  const dragActiveRef = useRef(false);

  const onFile = useCallback(
    (file: File | undefined | null) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) return;
      setImage(file);
    },
    [setImage],
  );

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragActiveRef.current = false;
    onFile(e.dataTransfer.files?.[0]);
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragActiveRef.current = true;
  };

  const onDragLeave = () => {
    dragActiveRef.current = false;
  };

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    onFile(e.target.files?.[0]);
  };

  return (
    <Card
      title="Glyph Decoder"
      subtitle="Upload an image to transliterate hieroglyphs"
      icon="𓂀"
      className="flex h-full flex-col"
    >
      {/* Dropzone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        role="button"
        tabIndex={0}
        className="flex cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed border-amber-700/60 bg-amber-950/40 px-6 py-10 text-center transition-colors hover:border-amber-500/70 hover:bg-amber-900/40"
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={onChange}
        />

        {image ? (
          <div className="flex w-full flex-col items-center gap-3">
            <img
              src={image.preview}
              alt="Uploaded hieroglyph preview"
              className="max-h-48 rounded border border-amber-700/50 object-contain shadow"
            />
            <p className="text-xs text-amber-300/80">{image.file.name}</p>
            <button
              onClick={(e) => {
                e.stopPropagation();
                clearImage();
              }}
              className="text-xs text-amber-400 underline-offset-2 hover:underline"
            >
              Clear image
            </button>
          </div>
        ) : (
          <>
            <div className="mb-2 text-4xl text-amber-500/80">🖼</div>
            <p className="text-sm font-semibold tracking-wide text-amber-200">
              DRAG &amp; DROP AN IMAGE HERE
            </p>
            <p className="mt-1 text-xs text-amber-400/70">
              or click to browse
            </p>
          </>
        )}
      </div>

      {/* Result block */}
      <div className="mt-5 rounded-md border border-amber-700/50 bg-stone-900/60 p-5 text-center">
        <p className="mb-3 text-xs uppercase tracking-[0.3em] text-amber-400/80">
          — Transliteration Result —
        </p>
        {result ? (
          <>
            <div className="mb-3 flex flex-wrap items-center justify-center gap-3 text-3xl text-amber-300">
              {result.detectedGlyphs.map((g, i) => (
                <span key={i} title={g}>
                  {g}
                </span>
              ))}
            </div>
            <p className="font-serif text-lg italic text-amber-200">
              {result.transliteration}
            </p>
            <p className="mt-2 text-sm text-amber-100/90">
              “{result.translation}”
            </p>
            <p className="mt-3 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
              Confidence {formatConfidence(result.confidence)}
            </p>
          </>
        ) : (
          <>
            <div className="mb-3 flex flex-wrap items-center justify-center gap-3 text-3xl text-amber-500/60">
              <span>𓇋</span>
              <span>𓂀</span>
              <span>𓃭</span>
              <span>𓇋</span>
              <span>𓊪</span>
              <span>𓋴</span>
            </div>
            <p className="font-serif text-lg italic text-amber-500/80">
              ntr nfr iri m htp
            </p>
            <p className="mt-2 text-sm text-amber-400/70">
              “Good god, in peace”
            </p>
          </>
        )}
      </div>

      {error && (
        <p className="mt-3 rounded border border-red-500/50 bg-red-900/40 px-3 py-2 text-xs text-red-200">
          {error}
        </p>
      )}

      <div className="mt-5">
        <Button
          className="w-full"
          size="lg"
          onClick={decode}
          isLoading={isLoading}
          disabled={!image}
          leftIcon="𓂀"
        >
          Decode Glyphs
        </Button>
      </div>
    </Card>
  );
};

export default GlyphDecoder;
