// ============================================================
// <Footer /> — bottom attribution bar
// ============================================================

import { FOOTER_LINES } from "@/constants";

export const Footer = () => {
  return (
    <footer className="border-t border-amber-700/40 bg-stone-950/60 px-6 py-4 text-center text-xs tracking-[0.25em] text-amber-400/80">
      <div className="flex items-center justify-center gap-3">
        <span>𓂀</span>
        <span>{FOOTER_LINES[0]}</span>
        <span>•</span>
        <span>{FOOTER_LINES[1]}</span>
        <span>𓋹</span>
      </div>
    </footer>
  );
};

export default Footer;
