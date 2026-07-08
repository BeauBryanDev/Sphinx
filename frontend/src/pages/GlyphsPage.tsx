// ============================================================
// <GlyphsPage /> — dedicated full-page glyph decoder
//
// The decoder state lives HERE (shared controller) so the
// right-hand <GlyphWall /> can render the decoded signs as real
// Unicode hieroglyphs in the same spatial order.
// ============================================================

import {
  GlyphDecoder,
  GlyphWall,
  SignConfidenceTable,
} from "@/components/glyphs";
import { useGlyphDecoder } from "@/hooks/useGlyphDecoder";

export const GlyphsPage = () => {
  const decoder = useGlyphDecoder();

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <GlyphDecoder controller={decoder} />
      </div>
      <div className="space-y-6">
        <GlyphWall result={decoder.result} />
        <SignConfidenceTable result={decoder.result} />
      </div>
    </div>
  );
};

export default GlyphsPage;
