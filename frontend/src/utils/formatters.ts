// ============================================================
// Formatting utilities
// ============================================================

/** Format a Unix timestamp (ms) into a short human-readable time. */
export const formatTime = (ts: number): string => {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
};

/** Format a Unix timestamp (ms) into a date string. */
export const formatDate = (ts: number): string => {
  return new Date(ts).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
};

/** Format a confidence score (0-1) as a percentage. */
export const formatConfidence = (score: number): string => {
  return `${Math.round(score * 100)}%`;
};

/** Generate a short random id — sufficient for client-side keys. */
export const generateId = (): string => {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
};

/** Truncate text to a maximum length with an ellipsis. */
export const truncate = (text: string, max = 80): string => {
  if (text.length <= max) return text;
  return text.slice(0, max - 1).trimEnd() + "…";
};
