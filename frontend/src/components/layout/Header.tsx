// ============================================================
// <Header /> — top bar with title + auth
// ============================================================

import { APP_DESCRIPTION, APP_NAME } from "@/constants";
import { useApp } from "@/contexts/AppContext";
import { Button } from "@/components/common/Button";
import horusEye from "@/assets/horus_eye.webp";
import ankh from "@/assets/ankh.svg";
import amphora from "@/assets/amphora.svg";
import pyramid from "@/assets/pyramid.svg";

// Decorative icons flanking the Horus eye — rendered left-to-right on the
// left side and mirrored (reversed order + horizontal flip) on the right
// for perfect symmetry around the emblem.
const FLANK_ICONS = [
  { src: ankh, alt: "Ankh" },
  { src: amphora, alt: "Amphora" },
  { src: pyramid, alt: "Pyramid" },
];

export const Header = () => {
  const { toggleSidebar } = useApp();

  return (
    <header className="relative flex items-center justify-between border-b border-amber-700/40 bg-gradient-to-r from-stone-900/80 via-amber-950/60 to-stone-900/80 px-4 py-4 sm:px-6">
      {/* Decorative hieroglyph strip */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-6 overflow-hidden text-[11px] tracking-[0.4em] text-amber-600/40">
        <div className="whitespace-nowrap px-4 pt-1">
          {"𓀀 𓁿 𓂀 𓃭 𓄿 𓅓 𓆣 𓇋 𓈖 𓉐 𓊪 𓋹 𓌻 𓍯 𓎼 𓏏 ".repeat(6)}
        </div>
      </div>

      {/* Centered winged Horus eye emblem, symmetrically flanked */}
      <div className="pointer-events-none absolute left-1/2 top-1/2 z-10 flex -translate-x-1/2 -translate-y-1/2 items-center gap-3 md:gap-4">
        <div className="hidden items-center gap-2 md:flex md:gap-3">
          {FLANK_ICONS.map((icon) => (
            <img
              key={icon.alt}
              src={icon.src}
              alt={icon.alt}
              className="h-7 w-auto opacity-80 drop-shadow-[0_2px_8px_rgba(180,120,30,0.4)] select-none lg:h-8"
              draggable={false}
            />
          ))}
        </div>
        <img
          src={horusEye}
          alt="Winged Horus eye"
          className="h-16 w-auto drop-shadow-[0_3px_12px_rgba(180,120,30,0.5)] select-none md:h-20"
          draggable={false}
        />
        <div className="hidden items-center gap-2 md:flex md:gap-3">
          {[...FLANK_ICONS].reverse().map((icon) => (
            <img
              key={icon.alt}
              src={icon.src}
              alt={icon.alt}
              className="h-7 w-auto -scale-x-100 opacity-80 drop-shadow-[0_2px_8px_rgba(180,120,30,0.4)] select-none lg:h-8"
              draggable={false}
            />
          ))}
        </div>
      </div>

      <div className="relative z-20 flex items-center gap-3 pt-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
          className="lg:hidden"
        >
          ☰
        </Button>
        <div className="hidden md:block">
          <h2 className="text-xl font-bold tracking-[0.3em] text-amber-300">
            {APP_NAME}
          </h2>
          <p className="text-xs text-amber-200/70">{APP_DESCRIPTION}</p>
        </div>
      </div>

      <div className="relative z-20 flex items-center gap-3 pt-3">
        <Button variant="secondary" size="sm" rightIcon="▾">
          <span className="inline-flex items-center gap-2">
            <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-amber-700/60 text-xs">
              👤
            </span>
            <span className="hidden sm:inline">Log In</span>
          </span>
        </Button>
      </div>
    </header>
  );
};

export default Header;
