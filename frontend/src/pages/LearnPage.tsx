// ============================================================
// <LearnPage /> — educational content / about
// ============================================================

import { Card } from "@/components/common";

const TOPICS = [
  {
    title: "The Pyramids of Giza",
    body: "Built during the Old Kingdom, the pyramids are architectural marvels aligned with extraordinary precision to the cardinal directions.",
  },
  {
    title: "Hieroglyphs & Writing",
    body: "Hieroglyphs combined logographic, syllabic, and alphabetic elements. The Rosetta Stone was the key to unlocking their meaning.",
  },
  {
    title: "Gods & Mythology",
    body: "Ra, Osiris, Isis, Horus, Anubis — the Egyptian pantheon shaped religion, kingship, and the journey to the afterlife.",
  },
  {
    title: "Daily Life in Egypt",
    body: "Farmers, scribes, artisans, and nobles lived along the Nile. Bread, beer, and festivals were central to everyday life.",
  },
];

export const LearnPage = () => {
  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
      {TOPICS.map((t) => (
        <Card key={t.title} title={t.title} icon="𓉴">
          <p className="text-sm leading-relaxed text-amber-200/90">
            {t.body}
          </p>
        </Card>
      ))}
    </div>
  );
};

export default LearnPage;
