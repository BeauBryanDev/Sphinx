// ============================================================
// useChat — chat conversation hook
//
// Now a thin wrapper over the global Zustand store, so the
// conversation survives page navigation. Same interface as the
// old useState implementation — components are unchanged.
// ============================================================

import { useChatStore, type ChatStore } from "@/stores";

export type UseChatReturn = ChatStore;

export const useChat = (): UseChatReturn => useChatStore();

export default useChat;
