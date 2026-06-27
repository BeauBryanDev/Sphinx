// ============================================================
// <TransliterationPage /> — text transliteration workspace
// ============================================================

import { useState } from "react";
import { Card, Button, Input } from "@/components/common";

const DEMO_MAP: Record<string, string> = {
  "𓁹": "ḥr",
  "𓂀": "wḏꜣt",
  "𓋹": "ꜥnḫ",
  "𓊪": "pr",
  "𓇋": "ꜣ",
  "𓃭": "r",
  "𓏏": "t",
  "𓅓": "ms",
  "𓐍": "ḫ",
  "𓈖": "n",
};

export const TransliterationPage = () => {
  const [input, setInput] = useState("𓁹 𓂀 𓋹 𓊪 𓇋 𓈖 𓏏");
  const [output, setOutput] = useState("");

  const onTransliterate = () => {
    const result = input
      .split("")
      .map((ch) => DEMO_MAP[ch] ?? (ch.trim() ? ch : ""))
      .filter(Boolean)
      .join(" ");
    setOutput(result);
  };

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <Card title="Input" subtitle="Paste hieroglyph text below" icon="𓍯">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={6}
          className="w-full rounded-md border border-amber-700/60 bg-amber-950/60 p-3 text-2xl text-amber-200 focus:border-amber-400 focus:outline-none"
          placeholder="Paste hieroglyph characters…"
        />
        <div className="mt-4 flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Or type a short phrase here"
          />
          <Button onClick={onTransliterate} leftIcon="𓂀">
            Transliterate
          </Button>
        </div>
      </Card>

      <Card title="Output" subtitle="Phonetic transliteration" icon="𓊹">
        <div className="min-h-[160px] rounded-md border border-amber-700/60 bg-stone-900/60 p-4 font-mono text-lg text-amber-200">
          {output || (
            <span className="text-amber-500/70">
              Transliteration will appear here…
            </span>
          )}
        </div>
        <p className="mt-3 text-xs text-amber-400/70">
          Tip: replace <code className="text-amber-300">DEMO_MAP</code> with a
          call to <code className="text-amber-300">/v1/glyphs/transliterate</code> in{" "}
          <code className="text-amber-300">src/api/glyphApi.ts</code>.
        </p>
      </Card>
    </div>
  );
};

export default TransliterationPage;
