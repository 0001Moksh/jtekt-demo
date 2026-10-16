import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Plus, MessageSquare, Send, FileText, PanelLeftClose, PanelLeft,
  PanelRightClose, PanelRight, Sparkles, Bot, User, AlertCircle,
  ExternalLink, Trash2, Sun, Moon, Copy, Check, RefreshCw, ThumbsUp,
  ThumbsDown, Mic, Volume2, Square, Search, Pin, StickyNote, Download,
  Library, WandSparkles, X, ListChecks, CircleHelp, Clock, GraduationCap,
  BookOpen, Pencil
} from "lucide-react";
import "./App.css";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8003";
const CHATS_KEY = "documind-chats";
const NOTES_KEY = "documind-notes";
const THEME_KEY = "documind-theme";

const suggestions = [
  "Summarize the key points in the documents.",
  "What are the main topics covered?",
  "Find important technical details.",
  "Compare the information across documents."
];

const studioTools = [
  { icon: BookOpen, label: "Briefing doc", prompt: "Create a briefing document: key themes, important facts and conclusions from the documents." },
  { icon: GraduationCap, label: "Study guide", prompt: "Create a study guide with key terms, short-answer questions and an answer key." },
  { icon: CircleHelp, label: "FAQ", prompt: "Write a FAQ with the 8 most useful questions and answers from the documents." },
  { icon: Clock, label: "Timeline", prompt: "Build a timeline of events, milestones or steps mentioned in the documents." },
  { icon: ListChecks, label: "Action items", prompt: "List all action items, requirements and decisions as a checklist." },
  { icon: WandSparkles, label: "Explain simply", prompt: "Explain the main idea of the documents in simple language for a beginner." }
];

const defaultFollowUps = ["Explain in simpler terms", "Give an example", "Summarize in 5 bullets"];

const load = (key, fallback) => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
};

const makeChat = () => ({
  id: crypto.randomUUID(),
  title: "New chat",
  messages: [],
  createdAt: Date.now()
});

function Typed({ text, fresh, onTick }) {
  const [count, setCount] = useState(fresh ? 0 : text.length);
  useEffect(() => {
    if (!fresh) return;
    const step = Math.max(2, Math.ceil(text.length / 160));
    const timer = setInterval(() => {
      setCount((value) => {
        if (value >= text.length) {
          clearInterval(timer);
          return value;
        }
        return Math.min(text.length, value + step);
      });
      onTick?.();
    }, 18);
    return () => clearInterval(timer);
  }, [fresh, text]);
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{text.slice(0, count)}</ReactMarkdown>;
}

export default function App() {
  const [chats, setChats] = useState(() => {
    const saved = load(CHATS_KEY, null);
    if (!saved?.length) return [makeChat()];
    return saved.map((c) => ({ ...c, messages: c.messages.map((m) => ({ ...m, fresh: false })) }));
  });
  const [activeId, setActiveId] = useState(() => chats[0]?.id);
  const [notes, setNotes] = useState(() => load(NOTES_KEY, []));
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_KEY) || "light");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [panelOpen, setPanelOpen] = useState(true);
  const [tab, setTab] = useState("sources");
  const [apiOnline, setApiOnline] = useState(false);
  const [search, setSearch] = useState("");
  const [toast, setToast] = useState("");
  const [copiedId, setCopiedId] = useState(null);
  const [speakingId, setSpeakingId] = useState(null);
  const [listening, setListening] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [draftTitle, setDraftTitle] = useState("");
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const recognitionRef = useRef(null);
  const toastTimer = useRef(null);

  const activeChat = chats.find((c) => c.id === activeId) || chats[0];
  const messages = activeChat?.messages || [];
  const lastAssistantId = [...messages].reverse().find((m) => m.role === "assistant")?.id;

  const sources = useMemo(() => {
    const map = new Map();
    messages.forEach((m) =>
      (m.sources || []).forEach((s) => {
        const key = s.source_file || s.source_url || "Source";
        const entry = map.get(key) || { ...s, key, count: 0, locations: new Set() };
        entry.count += 1;
        if (s.location) entry.locations.add(s.location);
        map.set(key, entry);
      })
    );
    return [...map.values()];
  }, [messages]);

  const filteredChats = chats.filter((c) => c.title.toLowerCase().includes(search.toLowerCase()));

  useEffect(() => { localStorage.setItem(CHATS_KEY, JSON.stringify(chats)); }, [chats]);
  useEffect(() => { localStorage.setItem(NOTES_KEY, JSON.stringify(notes)); }, [notes]);
  useEffect(() => { localStorage.setItem(THEME_KEY, theme); }, [theme]);

  useEffect(() => {
    fetch(`${API}/health`)
      .then((res) => setApiOnline(res.ok))
      .catch(() => setApiOnline(false));
  }, []);

  const scrollBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  };
  useEffect(() => {
    scrollBottom();
  }, [messages.length, loading]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  useEffect(() => {
    function onKey(event) {
      const mod = event.ctrlKey || event.metaKey;
      if (!mod) return;
      if (event.key === "k") { event.preventDefault(); newChat(); }
      if (event.key === "b") { event.preventDefault(); setSidebarOpen((v) => !v); }
      if (event.key === "/") { event.preventDefault(); inputRef.current?.focus(); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  function showToast(message) {
    setToast(message);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 1900);
  }

  function updateChat(id, updater) {
    setChats((prev) => prev.map((c) => (c.id === id ? updater(c) : c)));
  }

  function newChat() {
    if (loading) return;
    const chat = makeChat();
    setChats((prev) => [chat, ...prev]);
    setActiveId(chat.id);
    setInput("");
    setError("");
    inputRef.current?.focus();
  }

  function deleteChat(id) {
    if (loading) return;
    const remaining = chats.filter((c) => c.id !== id);
    const next = remaining.length ? remaining : [makeChat()];
    setChats(next);
    if (activeId === id) setActiveId(next[0].id);
    showToast("Conversation deleted");
  }

  function commitRename(id) {
    const title = draftTitle.trim();
    if (title) updateChat(id, (c) => ({ ...c, title }));
    setEditingId(null);
  }

  async function ask(question, { addUser = true } = {}) {
    if (!question || loading || !activeChat) return;
    const chatId = activeChat.id;

    if (addUser) {
      const userMessage = { id: crypto.randomUUID(), role: "user", content: question };
      updateChat(chatId, (chat) => ({
        ...chat,
        title: chat.messages.length ? chat.title : question.slice(0, 42),
        messages: [...chat.messages, userMessage]
      }));
    }

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
      if (!response.ok) throw new Error(data.detail || "The request could not be completed.");

      const reply = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer || "No answer was returned.",
        sources: Array.isArray(data.sources) ? data.sources : [],
        followUps: Array.isArray(data.follow_ups) && data.follow_ups.length ? data.follow_ups : defaultFollowUps,
        fresh: true
      };
      updateChat(chatId, (chat) => ({ ...chat, messages: [...chat.messages, reply] }));
    } catch (err) {
      setError(`${err.message}. Check that the FastAPI backend is running at ${API}.`);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function regenerate() {
    if (loading || !activeChat) return;
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    updateChat(activeChat.id, (chat) => {
      const idx = chat.messages.map((m) => m.role).lastIndexOf("assistant");
      return { ...chat, messages: chat.messages.filter((_, i) => i !== idx) };
    });
    ask(lastUser.content, { addUser: false });
  }

  async function copyText(text, id) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
      showToast("Copied to clipboard");
    } catch {
      showToast("Copy failed. Select the text and copy manually.");
    }
  }

  function setFeedback(id, value) {
    updateChat(activeChat.id, (chat) => ({
      ...chat,
      messages: chat.messages.map((m) => (m.id === id ? { ...m, feedback: m.feedback === value ? null : value } : m))
    }));
  }

  function pinNote(message) {
    setNotes((prev) => [
      { id: crypto.randomUUID(), text: message.content, chat: activeChat.title, at: Date.now() },
      ...prev
    ]);
    setPanelOpen(true);
    setTab("notes");
    showToast("Saved to notes");
  }

  function speak(message) {
    if (!("speechSynthesis" in window)) return showToast("Read aloud is not supported in this browser");
    if (speakingId === message.id) {
      window.speechSynthesis.cancel();
      setSpeakingId(null);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message.content.replace(/[#*_`>|-]/g, " "));
    utterance.onend = () => setSpeakingId(null);
    utterance.onerror = () => setSpeakingId(null);
    setSpeakingId(message.id);
    window.speechSynthesis.speak(utterance);
  }

  function toggleMic() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) return showToast("Voice input is not supported in this browser");
    if (listening) {
      recognitionRef.current?.stop();
      return;
    }
    const rec = new Recognition();
    rec.lang = "en-IN";
    rec.interimResults = true;
    rec.onresult = (event) => {
      setInput(Array.from(event.results).map((r) => r[0].transcript).join(" "));
    };
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    recognitionRef.current = rec;
    setListening(true);
    rec.start();
  }

  function download(filename, content) {
    const url = URL.createObjectURL(new Blob([content], { type: "text/markdown" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  function exportChat() {
    if (!messages.length) return showToast("Nothing to export yet");
    const body = messages
      .map((m) => `### ${m.role === "user" ? "You" : "Document AI"}\n\n${m.content}`)
      .join("\n\n---\n\n");
    download(`${activeChat.title || "chat"}.md`, `# ${activeChat.title}\n\n${body}\n`);
    showToast("Conversation exported");
  }

  function exportNotes() {
    if (!notes.length) return showToast("No notes to export");
    download("documind-notes.md", notes.map((n) => `## ${n.chat}\n\n${n.text}`).join("\n\n---\n\n"));
  }

  return (
    <div className="app-shell" data-theme={theme}>
      <aside className={`sidebar ${sidebarOpen ? "" : "closed"}`}>
        <div className="sidebar-inner">
          <div className="brand">
            <div className="brand-mark neu"><FileText size={21} /></div>
            <div>
              <strong>DocuMind</strong>
              <span>Document AI Assistant</span>
            </div>
          </div>

          <button className="new-chat-button neu-btn" onClick={newChat}>
            <Plus size={18} /> New chat
          </button>

          <label className="search-box neu-in">
            <Search size={15} />
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search conversations" />
          </label>

          <div className="history-heading">Your conversations</div>
          <div className="history-list">
            {filteredChats.length === 0 && <p className="empty-hint">No conversations match your search.</p>}
            {filteredChats.map((chat) => (
              <div key={chat.id} className={`history-item ${chat.id === activeId ? "selected" : ""}`}>
                {editingId === chat.id ? (
                  <input
                    className="rename-input"
                    autoFocus
                    value={draftTitle}
                    onChange={(e) => setDraftTitle(e.target.value)}
                    onBlur={() => commitRename(chat.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitRename(chat.id);
                      if (e.key === "Escape") setEditingId(null);
                    }}
                  />
                ) : (
                  <button
                    className="history-select"
                    title={chat.title}
                    onClick={() => { if (!loading) { setActiveId(chat.id); setError(""); } }}
                    onDoubleClick={() => { setEditingId(chat.id); setDraftTitle(chat.title); }}
                  >
                    <MessageSquare size={16} />
                    <span>{chat.title}</span>
                  </button>
                )}
                <button
                  className="icon-button row-action"
                  title="Rename"
                  onClick={() => { setEditingId(chat.id); setDraftTitle(chat.title); }}
                >
                  <Pencil size={13} />
                </button>
                <button
                  className="icon-button row-action"
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
            <kbd title="New chat">Ctrl K</kbd>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <button className="icon-button neu-btn" onClick={() => setSidebarOpen(!sidebarOpen)} title="Toggle sidebar (Ctrl B)">
            {sidebarOpen ? <PanelLeftClose size={19} /> : <PanelLeft size={19} />}
          </button>
          <div className="topbar-title">
            <strong>{activeChat?.title === "New chat" ? "Document chat" : activeChat?.title}</strong>
            <span>Ask questions from your indexed documents</span>
          </div>
          <button className="icon-button neu-btn" onClick={exportChat} title="Export conversation">
            <Download size={18} />
          </button>
          <button
            className="icon-button neu-btn theme-toggle"
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
            title="Switch theme"
          >
            {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
          </button>
          <button className="icon-button neu-btn" onClick={() => setPanelOpen(!panelOpen)} title="Toggle side panel">
            {panelOpen ? <PanelRightClose size={19} /> : <PanelRight size={19} />}
          </button>
        </header>

        <section className="conversation">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon neu"><Sparkles size={26} /></div>
              <h1>Ask your documents anything.</h1>
              <p>
                Get answers grounded in your indexed SharePoint PDFs and PowerPoint files,
                with source references you can open, save and export.
              </p>
              <div className="suggestion-grid">
                {suggestions.map((suggestion, i) => (
                  <button
                    className="suggestion-card neu-btn"
                    key={suggestion}
                    style={{ animationDelay: `${120 + i * 90}ms` }}
                    disabled={loading}
                    onClick={() => ask(suggestion)}
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
                <article className={`message-row ${message.role}`} key={message.id}>
                  <div className={`avatar neu ${message.role}`}>
                    {message.role === "assistant" ? <Bot size={18} /> : <User size={18} />}
                  </div>

                  <div className="message-body">
                    <div className="message-author">{message.role === "assistant" ? "Document AI" : "You"}</div>

                    {message.role === "assistant" ? (
                      <div className="markdown-content">
                        <Typed text={message.content} fresh={message.fresh} onTick={scrollBottom} />
                      </div>
                    ) : (
                      <div className="user-message">{message.content}</div>
                    )}

                    {message.role === "assistant" && (
                      <div className="message-actions">
                        <button className="icon-button" title="Copy" onClick={() => copyText(message.content, message.id)}>
                          {copiedId === message.id ? <Check size={15} /> : <Copy size={15} />}
                        </button>
                        <button className="icon-button" title="Save to notes" onClick={() => pinNote(message)}>
                          <Pin size={15} />
                        </button>
                        <button
                          className={`icon-button ${speakingId === message.id ? "active" : ""}`}
                          title={speakingId === message.id ? "Stop reading" : "Read aloud"}
                          onClick={() => speak(message)}
                        >
                          {speakingId === message.id ? <Square size={14} /> : <Volume2 size={15} />}
                        </button>
                        <button
                          className={`icon-button ${message.feedback === "up" ? "active" : ""}`}
                          title="Good answer"
                          onClick={() => setFeedback(message.id, "up")}
                        >
                          <ThumbsUp size={15} />
                        </button>
                        <button
                          className={`icon-button ${message.feedback === "down" ? "active" : ""}`}
                          title="Poor answer"
                          onClick={() => setFeedback(message.id, "down")}
                        >
                          <ThumbsDown size={15} />
                        </button>
                        {message.id === lastAssistantId && (
                          <button className="icon-button" title="Regenerate" onClick={regenerate} disabled={loading}>
                            <RefreshCw size={15} />
                          </button>
                        )}
                      </div>
                    )}

                    {message.role === "assistant" && message.sources?.length > 0 && (
                      <div className="sources" style={{ marginTop: '8px' }}>
                        <div className="sources-title" style={{ fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px', opacity: 0.8 }}>
                          <FileText size={14} /> Sources used
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                          {message.sources.map((source, index) => (
                            <div
                              key={`${source.source_file}-${source.location}-${index}`}
                              className="source-card neu-in"
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '6px',
                                padding: '4px 8px',
                                borderRadius: '6px',
                                fontSize: '12px',
                                maxWidth: '180px',
                                background: 'rgba(0,0,0,0.03)'
                              }}
                            >
                              <FileText size={13} style={{ flexShrink: 0 }} />
                              <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                <span style={{ fontWeight: 500, display: 'inline' }}>{source.source_file || `Source ${index + 1}`}</span>
                              </div>
                              {source.source_url && (
                                <a href={source.source_url} target="_blank" rel="noreferrer" title="Open" style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
                                  <ExternalLink size={12} />
                                </a>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {message.id === lastAssistantId && !loading && (
                      <div className="followups">
                        {(message.followUps || defaultFollowUps).map((q) => (
                          <button key={q} className="chip neu-btn" onClick={() => ask(q)}>{q}</button>
                        ))}
                      </div>
                    )}
                  </div>
                </article>
              ))}

              {loading && (
                <div className="message-row assistant">
                  <div className="avatar neu assistant pulse"><Bot size={18} /></div>
                  <div className="message-body">
                    <div className="message-author">Document AI</div>
                    <div className="loading-indicator">
                      <span className="dots"><i /><i /><i /></span>
                      Searching documents and preparing an answer…
                    </div>
                    <div className="skeleton"><b /><b /><b /></div>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}

          {error && (
            <div className="error-banner neu-in">
              <AlertCircle size={17} />
              <span>{error}</span>
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
        </section>

        <footer className="composer-area">
          <form
            className="composer neu-in"
            onSubmit={(e) => { e.preventDefault(); ask(input.trim()); }}
          >
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(input.trim());
                }
              }}
              placeholder={listening ? "Listening…" : "Ask a question about your documents…"}
              rows={1}
              maxLength={4000}
              disabled={loading}
            />
            {input.length > 3200 && <span className="counter">{input.length}/4000</span>}
            <button
              className={`icon-button neu-btn mic ${listening ? "listening" : ""}`}
              type="button"
              onClick={toggleMic}
              title="Voice input"
            >
              <Mic size={18} />
            </button>
            <button className="send-button" type="submit" disabled={!input.trim() || loading} title="Send message">
              <Send size={18} />
            </button>
          </form>
          <p className="composer-note">
            Answers are based on indexed document content. Verify critical information against the original source.
          </p>
        </footer>
      </main>

      <aside className={`side-panel ${panelOpen ? "" : "closed"}`}>
        <div className="panel-inner">
          <div className="tabs neu-in">
            {[
              ["sources", Library, "Sources"],
              ["notes", StickyNote, "Notes"],
              ["studio", WandSparkles, "Studio"]
            ].map(([id, Icon, label]) => (
              <button key={id} className={tab === id ? "tab active neu-btn" : "tab"} onClick={() => setTab(id)}>
                <Icon size={15} /> {label}
              </button>
            ))}
          </div>

          <div className="panel-body" key={tab}>
            {tab === "sources" && (
              <>
                <p className="panel-lead">Documents cited in this conversation.</p>
                {sources.length === 0 && <p className="empty-hint">Sources will appear here once you get an answer.</p>}
                {sources.map((s) => (
                  <div className="source-card neu" key={s.key}>
                    <div className="source-icon neu-in"><FileText size={16} /></div>
                    <div className="source-details">
                      <strong>{s.key}</strong>
                      <span>Cited {s.count}× {s.locations.size ? `· ${[...s.locations].slice(0, 2).join(", ")}` : ""}</span>
                    </div>
                    {s.source_url && (
                      <a href={s.source_url} target="_blank" rel="noreferrer" title="Open source document">
                        <ExternalLink size={15} />
                      </a>
                    )}
                  </div>
                ))}
              </>
            )}

            {tab === "notes" && (
              <>
                <div className="panel-row">
                  <p className="panel-lead">Answers you saved.</p>
                  <button className="icon-button" onClick={exportNotes} title="Export notes"><Download size={15} /></button>
                </div>
                {notes.length === 0 && <p className="empty-hint">Use the pin under any answer to save it here.</p>}
                {notes.map((n) => (
                  <div className="note-card neu" key={n.id}>
                    <div className="note-text"><ReactMarkdown remarkPlugins={[remarkGfm]}>{n.text}</ReactMarkdown></div>
                    <div className="note-foot">
                      <span>{n.chat}</span>
                      <button className="icon-button" title="Copy" onClick={() => copyText(n.text, n.id)}><Copy size={14} /></button>
                      <button className="icon-button" title="Delete note" onClick={() => setNotes((p) => p.filter((x) => x.id !== n.id))}>
                        <X size={14} />
                      </button>
                    </div>
                  </div>
                ))}
              </>
            )}

            {tab === "studio" && (
              <>
                <p className="panel-lead">Turn your documents into ready-made outputs.</p>
                <div className="studio-grid">
                  {studioTools.map(({ icon: Icon, label, prompt }) => (
                    <button key={label} className="studio-card neu-btn" disabled={loading} onClick={() => ask(prompt)}>
                      <Icon size={19} />
                      <span>{label}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </aside>

      <div className={`toast neu ${toast ? "show" : ""}`} role="status">{toast}</div>
    </div>
  );
}