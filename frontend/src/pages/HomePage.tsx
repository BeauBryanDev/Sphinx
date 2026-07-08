// ============================================================
// <HomePage /> — main landing / dashboard
// ============================================================

import { ChatPanel } from "@/components/chat";
import { GlyphDecoder } from "@/components/glyphs";
import { ActivityChart } from "@/components/dashboard";
import { Card } from "@/components/common";
import { FEATURES } from "@/constants";
import { useAnalytics } from "@/hooks/useAnalytics";
import sphinxBackground from "@/assets/sphinx_background.jpg";
import pharaohMask from "@/assets/pharaoh_mask.webp";

export const HomePage = () => {
  const { metrics } = useAnalytics("7d");

  return (
    <div className="space-y-6">
      {/* Hero banner */}
      <section className="relative overflow-hidden rounded-2xl border-2 border-amber-500/60 bg-stone-950 shadow-[0_10px_40px_rgba(0,0,0,0.45)]">
        {/* Sphinx background */}
        <img
          src={sphinxBackground}
          alt=""
          aria-hidden="true"
          className="absolute inset-0 h-full w-full object-cover object-center select-none"
          draggable={false}
        />
        {/* Legibility overlay */}
        <div className="absolute inset-0 bg-gradient-to-b from-stone-950/70 via-stone-950/45 to-stone-950/85" />

        <div className="relative z-10 flex flex-col items-center px-6 py-12 text-center md:py-16">
          <img
            src={pharaohMask}
            alt="SphinxEyes pharaoh emblem"
            className="h-32 w-auto drop-shadow-[0_4px_18px_rgba(180,120,30,0.55)] select-none md:h-40"
            draggable={false}
          />
          <h1 className="mt-4 text-4xl font-black tracking-[0.15em] text-amber-300 drop-shadow md:text-5xl">
            SphinxEyes
          </h1>
          <p className="mt-2 text-sm uppercase tracking-[0.35em] text-amber-200/80">
            Advanced Computer Vision · AI Expert Chatbot
          </p>
          <p className="mt-3 text-base text-amber-100/90">
            Unlock the Secrets of Ancient Egypt
          </p>
          <div className="mx-auto mt-3 flex items-center justify-center gap-2 text-amber-500/80">
            <span>—</span>
            <span>☥</span>
            <span>—</span>
          </div>
        </div>
      </section>

      {/* Metric tiles */}
      {metrics.length > 0 && (
        <section className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {metrics.map((m) => (
            <Card key={m.label} className="text-center">
              <p className="text-xs uppercase tracking-[0.3em] text-amber-400/80">
                {m.label}
              </p>
              <p className="mt-2 text-3xl font-bold text-amber-200">
                {m.value.toLocaleString()}
                {m.unit && <span className="ml-1 text-sm text-amber-400">{m.unit}</span>}
              </p>
              {m.trend && (
                <p className="mt-1 text-xs text-amber-500/80">
                  trend · {m.trend}
                </p>
              )}
            </Card>
          ))}
        </section>
      )}

      {/* Main two-column workspace */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <ChatPanel />
        <GlyphDecoder />
      </section>

      {/* Feature highlights + activity chart */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-1">
          {FEATURES.map((f) => (
            <Card key={f.title}>
              <div className="flex items-start gap-3">
                <span className="text-2xl text-amber-400">{f.icon}</span>
                <div>
                  <h4 className="text-sm font-bold uppercase tracking-[0.2em] text-amber-300">
                    {f.title}
                  </h4>
                  <p className="mt-1 text-sm text-amber-200/80">
                    {f.description}
                  </p>
                </div>
              </div>
            </Card>
          ))}
        </div>
        <div className="lg:col-span-2">
          <ActivityChart />
        </div>
      </section>
    </div>
  );
};

export default HomePage;
