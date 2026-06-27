// ============================================================
// Chat API — endpoint wrappers for SphinxChat
// ============================================================

import { request } from "@/services/http";
import type { ChatMessage } from "@/types";

export interface SendMessagePayload {
  prompt: string;
  history?: Pick<ChatMessage, "role" | "content">[];
}

export interface SendMessageResponse {
  reply: string;
  messageId: string;
}

/** Send a prompt to the AI chat endpoint. */
export const sendChatMessage = (
  payload: SendMessagePayload,
): Promise<SendMessageResponse> => {
  return request<SendMessageResponse>({
    method: "POST",
    url: "/v1/chat/messages",
    data: payload,
  });
};

/** Fetch conversation history. */
export const fetchChatHistory = (
  sessionId: string,
): Promise<ChatMessage[]> => {
  return request<ChatMessage[]>({
    method: "GET",
    url: `/v1/chat/sessions/${sessionId}/messages`,
  });
};
