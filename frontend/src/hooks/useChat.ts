// ============================================================
// useChat — stateful hook for a chat conversation
// ============================================================

import { useCallback, useState } from "react";
import { sendChatMessage } from "@/api/chatApi";
import { generateId } from "@/utils/formatters";
import type { ChatMessage } from "@/types";

const INITIAL_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "Greetings, seeker of knowledge. I am Sphinx, guardian of ancient wisdom. Ask me anything about Ancient Egypt — its history, gods, pharaohs, and mysteries.",
  timestamp: Date.now(),
};

export interface UseChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  send: (prompt: string) => Promise<void>;
  reset: () => void;
}

export const useChat = (): UseChatReturn => {
  const [messages, setMessages] = useState<ChatMessage[]>([INITIAL_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = useCallback(async (prompt: string) => {
    const trimmed = prompt.trim();
    if (!trimmed) return;

    const userMsg: ChatMessage = {
      id: generateId(),
      role: "user",
      content: trimmed,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    setError(null);

    try {
      const { reply, messageId } = await sendChatMessage({
        prompt: trimmed,
        history: messages.map((m) => ({ role: m.role, content: m.content })),
      });

      const assistantMsg: ChatMessage = {
        id: messageId || generateId(),
        role: "assistant",
        content: reply,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to reach the Sphinx.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [messages]);

  const reset = useCallback(() => {
    setMessages([INITIAL_MESSAGE]);
    setError(null);
  }, []);

  return { messages, isLoading, error, send, reset };
};

export default useChat;
