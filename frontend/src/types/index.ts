// ============================================================
// Domain Types — Central Type Definitions for SphinxEyes
// ============================================================

// ---------- Chat / AI ----------
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
}

// ---------- Glyph Decoder ----------
export interface GlyphDecodingResult {
  id: string;
  originalImage: string;
  detectedGlyphs: string[];
  transliteration: string;
  translation: string;
  confidence: number;
  processedAt: number;
}

export interface UploadedImage {
  file: File;
  preview: string;
}

// ---------- Analytics / Dashboard ----------
export interface AnalyticsMetric {
  label: string;
  value: number;
  unit?: string;
  trend?: "up" | "down" | "neutral";
}

export interface TimeSeriesPoint {
  name: string; // x-axis label (e.g., day, month)
  queries: number;
  decodings: number;
}

// ---------- Navigation ----------
export type PageKey =
  | "home"
  | "chat"
  | "glyphs"
  | "transliteration"
  | "learn"
  | "profile";

export interface NavItem {
  key: PageKey;
  label: string;
  icon: string; // emoji or svg id
}

// ---------- User ----------
export interface UserProfile {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  plan: "free" | "scholar" | "pharaoh";
  joinedAt: number;
}

// ---------- Generic API envelope ----------
export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}
