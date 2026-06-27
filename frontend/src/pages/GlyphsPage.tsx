// ============================================================
// <GlyphsPage /> — dedicated full-page glyph decoder
// ============================================================

import { GlyphDecoder } from "@/components/glyphs";
import { Card } from "@/components/common";

const GLYPH_KEY = [
  { glyph: "𓁹", translit: "ḥr", meaning: "Horus / face" },
  { glyph: "𓂀", translit: "wḏꜣt", meaning: "Eye of Horus" },
  { glyph: "𓋹", translit: "ꜥnḫ", meaning: "Life (ankh)" },
  { glyph: "𓊪", translit: "pr", meaning: "House" },
  { glyph: "𓇋", translit: "ꜣ", meaning: "Reed / vocalised a" },
  { glyph: "𓃭", translit: "r", meaning: "Lion / r sound" },
  { glyph: "𓏏", translit: "t", meaning: "Bread / t sound" },
];

export const GlyphsPage = () => {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <GlyphDecoder />
      </div>
      <Card title="Glyph Key" subtitle="Common hieroglyph transliterations" icon="𓊹">
        <ul className="divide-y divide-amber-700/40 text-sm">
          {GLYPH_KEY.map((g) => (
            <li
              key={g.glyph}
              className="flex items-center justify-between gap-3 py-2"
            >
              <span className="text-2xl text-amber-300">{g.glyph}</span>
              <span className="flex-1 text-right font-mono text-xs text-amber-200">
                {g.translit}
              </span>
              <span className="w-28 text-right text-xs text-amber-400/80">
                {g.meaning}
              </span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
};

export default GlyphsPage;
