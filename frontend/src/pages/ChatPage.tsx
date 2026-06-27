// <ChatPage /> — dedicated full-page chat experience

import { ChatPanel } from "@/components/chat";
import { Card } from "@/components/common";
import { CHAT_SUGGESTIONS } from "@/constants";
import { useChat } from "@/hooks/useChat";

export const ChatPage = () => {
  const { send } = useChat();

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <ChatPanel />
      </div>
      <Card title="Topics" subtitle="Suggested avenues of inquiry" icon="📜">
        <ul className="space-y-2 text-sm">
          {CHAT_SUGGESTIONS.map((q) => (
            <li key={q}>
              <button
                onClick={() => send(q)}
                className="w-full rounded border border-amber-700/50 bg-amber-950/40 px-3 py-2 text-left text-amber-200 transition-colors hover:bg-amber-900/60"
              >
                {q}
              </button>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
};

export default ChatPage;
