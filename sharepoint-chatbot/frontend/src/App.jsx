
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Plus, MessageSquare, Send, FileText, PanelLeftClose,
  PanelLeft, Sparkles, Bot, User, LoaderCircle,
  AlertCircle, ExternalLink, Trash2
} from "lucide-react";
import "./App.css";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8003";

const suggestions = [
  "Summarize the key points in the documents.",
  "What are the main topics covered?",
  "Find important technical details.",
  "Compare the information across documents."
];

function makeChat() {
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    messages: [],
    createdAt: Date.now()
  };
}

export default function App() {
  const [chats, setChats] = useState(() => {
    try {
      const saved = sessionStorage.getItem("sp-chat-history");
      return saved ? JSON.parse(saved) : [makeChat()];
    } catch {
      return [makeChat()];
    }
  });

  const [activeId, setActiveId] = useState(() => chats[0]?.id);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [apiOnline, setApiOnline] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  const activeChat = chats.find((chat) => chat.id === activeId) || chats[0];
  const messages = activeChat?.messages || [];

  useEffect(() => {
    sessionStorage.setItem("sp-chat-history", JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    fetch(`${API}/health`)
      .then((res) => setApiOnline(res.ok))
      .catch(() => setApiOnline(false));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function updateChat(id, updater) {
    setChats((previous) =>
      previous.map((chat) => chat.id === id ? updater(chat) : chat)
    );
  }

  function newChat() {
    if (loading) return;
    const chat = makeChat();
    setChats((previous) => [chat, ...previous]);
    setActiveId(chat.id);
    setInput("");
    setError("");
    inputRef.current?.focus();
  }

  function deleteChat(id) {
    if (loading) return;
    const remaining = chats.filter((chat) => chat.id !== id);
    const next = remaining.length ? remaining : [makeChat()];
    setChats(next);
    if (activeId === id) setActiveId(next[0].id);
  }

  async function sendMessage(text = input) {
    const question = text.trim();
    if (!question || loading || !activeChat) return;

    const chatId = activeChat.id;
    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question
    };

    updateChat(chatId, (chat) => ({
      ...chat,
      title: chat.messages.length ? chat.title : question.slice(0, 42),
      messages: [...chat.messages, userMessage]
    }));

    setInput("");
    setError("");
    setLoading(true);

    try {
      const response = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "The request could not be completed.");
      }

      updateChat(chatId, (chat) => ({
        ...chat,
        messages: [
          ...chat.messages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: data.answer || "No answer was returned.",
            sources: Array.isArray(data.sources) ? data.sources : []
          }
        ]
      }));
    } catch (err) {
      setError(
        `${err.message}. Check that the FastAPI backend is running at ${API}.`
      );
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    sendMessage();
  }

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark"><FileText size={21} /></div>
            <div>
              <strong>DocuMind</strong>
              <span>Document AI Assistant</span>
            </div>
          </div>

          <button className="new-chat-button" onClick={newChat}>
            <Plus size={18} /> New chat
          </button>

          <div className="history-heading">YOUR CONVERSATIONS</div>
          <div className="history-list">
            {chats.map((chat) => (
              <div
                key={chat.id}
                className={`history-item ${chat.id === activeId ? "selected" : ""}`}
              >
                <button
                  className="history-select"
                  onClick={() => {
                    if (!loading) {
                      setActiveId(chat.id);
                      setError("");
                    }
                  }}
                  title={chat.title}
                >
                  <MessageSquare size={16} />
                  <span>{chat.title}</span>
                </button>
                <button
                  className="icon-button delete-chat"
                  title="Delete conversation"
                  onClick={() => deleteChat(chat.id)}
                  disabled={loading}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>

          <div className="sidebar-footer">
            <div className={`status-dot ${apiOnline ? "online" : ""}`} />
            <span>{apiOnline ? "Backend connected" : "Backend unavailable"}</span>
          </div>
        </aside>
      )}

      <main className="main-panel">
        <header className="topbar">
          <button
            className="icon-button"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            title="Toggle sidebar"
          >
            {sidebarOpen ? <PanelLeftClose size={19} /> : <PanelLeft size={19} />}
          </button>
          <div className="topbar-title">
            <strong>Document Chat</strong>
            <span>Ask questions from your indexed documents</span>
          </div>
          <div className="secure-label"><Sparkles size={15} /> AI Assistant</div>
        </header>

        <section className="conversation">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon"><Sparkles size={25} /></div>
              <h1>Ask your documents anything.</h1>
              <p>
                Get answers grounded in your indexed SharePoint PDFs and
                PowerPoint files, with source references.
              </p>

              <div className="suggestion-grid">
                {suggestions.map((suggestion) => (
                  <button
                    className="suggestion-card"
                    key={suggestion}
                    disabled={loading}
                    onClick={() => sendMessage(suggestion)}
                  >
                    <span>{suggestion}</span>
                    <Send size={15} />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article
                  className={`message-row ${message.role}`}
                  key={message.id}
                >
                  <div className={`avatar ${message.role}`}>
                    {message.role === "assistant"
                      ? <Bot size={18} />
                      : <User size={18} />}
                  </div>

                  <div className="message-body">
                    <div className="message-author">
                      {message.role === "assistant" ? "Document AI" : "You"}
                    </div>

                    {message.role === "assistant" ? (
                      <div className="markdown-content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div className="user-message">{message.content}</div>
                    )}

                    {message.role === "assistant" &&
                      message.sources?.length > 0 && (
                        <div className="sources">
                          <div className="sources-title">
                            <FileText size={15} /> Sources used
                          </div>
                          {message.sources.map((source, index) => (
                            <div
                              className="source-card"
                              key={`${source.source_file}-${source.location}-${index}`}
                            >
                              <div className="source-icon"><FileText size={17} /></div>
                              <div className="source-details">
                                <strong>{source.source_file || `Source ${index + 1}`}</strong>
                                <span>
                                  {source.location || "Document reference"}
                                </span>
                              </div>
                              {source.source_url && (
                                <a
                                  href={source.source_url}
                                  target="_blank"
                                  rel="noreferrer"
                                  title="Open source document"
                                >
                                  <ExternalLink size={15} />
                                </a>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                  </div>
                </article>
              ))}

              {loading && (
                <div className="message-row assistant">
                  <div className="avatar assistant"><Bot size={18} /></div>
                  <div className="message-body">
                    <div className="message-author">Document AI</div>
                    <div className="loading-indicator">
                      <LoaderCircle className="spinner" size={17} />
                      Searching documents and preparing an answer…
                    </div>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}

          {error && (
            <div className="error-banner">
              <AlertCircle size={17} />
              <span>{error}</span>
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
        </section>

        <footer className="composer-area">
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  handleSubmit(event);
                }
              }}
              placeholder="Ask a question about your documents…"
              rows={1}
              maxLength={4000}
              disabled={loading}
            />
            <button
              className="send-button"
              type="submit"
              disabled={!input.trim() || loading}
              title="Send message"
            >
              {loading
                ? <LoaderCircle className="spinner" size={18} />
                : <Send size={18} />}
            </button>
          </form>
          <p className="composer-note">
            Answers are based on indexed document content. Verify critical
            information against the original source.
          </p>
        </footer>
      </main>
    </div>
  );
}
