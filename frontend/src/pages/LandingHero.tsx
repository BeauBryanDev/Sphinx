// ============================================================
// <LandingHero /> — public landing page (default "home")
// Hero banner + mission board. No chat / decoder panels here;
// those live on their own authenticated pages.
// ============================================================

import { Card } from "@/components/common";
import sphinxBackground from "@/assets/sphinx_background.jpg";
import pharaohMask from "@/assets/pharaoh_mask.webp";
import horusEye from "@/assets/horus_eye.webp";


// Mission pipeline — sourced from the project README (System Overview).
const PIPELINE_STAGES: { step: string; title: string; description: string }[] = [
  {
    step: "I",
    title: "Image Enhancement",
    description: "CLAHE, denoising and unsharp masking lift weathered carvings out of the stone.",
  },
  {
    step: "II",
    title: "Glyph Detection",
    description: "YOLOv11-Large reads 150 Gardiner classes natively, cartouche interiors included.",
  },
  {
    step: "III",
    title: "Layout & Reading Order",
    description: "Geometry decides rows vs. columns; quadrat clustering recovers the reading sequence.",
  },
  {
    step: "IV",
    title: "Semantic Correction",
    description: "A Viterbi pass over a lexicon trie and a bigram language model repairs misreads.",
  },
  {
    step: "V",
    title: "Cartouche Reading",
    description: "Needleman-Wunsch alignment matches royal names against an attested database.",
  },
  {
    step: "VI",
    title: "Transliteration",
    description: "The corrected sign sequence becomes structured, readable phonetic output.",
  },
];

const HorusDivider = () => (
  <div className="flex items-center justify-center gap-4">
    <span className="h-px w-16 bg-gradient-to-r from-transparent to-amber-600/60" />
    <img
      src={horusEye}
      alt=""
      aria-hidden="true"
      className="h-12 w-auto opacity-90 select-none"
      draggable={false}
    />
    <span className="h-px w-16 bg-gradient-to-l from-transparent to-amber-600/60" />
  </div>
);

export const LandingHero = () => {
  
  return (
    <div className="space-y-8">
      {/* Hero banner */}
      <section className="relative overflow-hidden rounded-2xl border-2 border-amber-500/60 bg-stone-950 shadow-[0_10px_40px_rgba(0,0,0,0.45)]">
        <img
          src={sphinxBackground}
          alt=""
          aria-hidden="true"
          className="absolute inset-0 h-full w-full object-cover object-center select-none"
          draggable={false}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-stone-950/70 via-stone-950/45 to-stone-950/85" />

        <div className="relative z-10 flex flex-col items-center px-6 py-14 text-center md:py-20">
          <img
            src={pharaohMask}
            alt="SphinxEyes pharaoh emblem"
            className="h-32 w-auto drop-shadow-[0_4px_18px_rgba(180,120,30,0.55)] select-none md:h-44"
            draggable={false}
          />
          <h1 className="mt-4 text-4xl font-black tracking-[0.15em] text-amber-300 drop-shadow md:text-6xl">
            SphinxEyes
          </h1>
          <p className="mt-3 text-sm uppercase tracking-[0.35em] text-amber-200/80 md:text-base">
            Advanced Computer Vision · AI Expert Chatbot
          </p>
          <p className="mt-3 font-serif text-lg text-amber-100/90 md:text-xl">
            Unlock the Secrets of Ancient Egypt
          </p>
          <div className="mx-auto mt-4 flex items-center justify-center gap-2 text-amber-500/80">
            <span>—</span>
            <span>☥</span>
            <span>—</span>
          </div>
        </div>
      </section>

      {/* Mission board */}
      <Card
        title="Our Mission"
        subtitle="From weathered stone to readable word"
        icon={
          <img
            src={horusEye}
            alt=""
            aria-hidden="true"
            className="h-14 w-auto opacity-90 select-none"
            draggable={false}
          />
        }
      >
        <p className="mx-auto max-w-3xl text-center text-base leading-relaxed text-amber-100/90">
          SphinxEyes reads Middle Egyptian hieroglyphs straight from photographs.
          Dense, weathered, and arranged in rows or columns, the signs of antiquity
          resist easy reading — so the pipeline carries each inscription end to end,
          from raw image bytes to a structured, corrected transliteration.
        </p>

        <div className="my-8">
          <HorusDivider />
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {PIPELINE_STAGES.map((stage) => (
            <div
              key={stage.step}
              className="rounded-lg border border-amber-800/50 bg-stone-950/40 p-4 transition-colors hover:border-amber-600/70 hover:bg-stone-950/60"
            >
              <div className="flex items-center gap-3">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-amber-500/60 font-serif text-sm font-bold text-amber-300">
                  {stage.step}
                </span>
                <h4 className="text-sm font-bold uppercase tracking-[0.18em] text-amber-300">
                  {stage.title}
                </h4>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-amber-200/80">
                {stage.description}
              </p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default LandingHero;
