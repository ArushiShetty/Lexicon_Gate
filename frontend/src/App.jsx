import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api, ms, pct } from "./api";
import AuthScreen, { Logo } from "./Auth";
import {
  EmptyState,
  StatusBanner,
} from "./ui";
import WritePage, { StoryReader } from "./Write";
import CollaboratorsPage from "./Collaborators";
import { parseStoryHash } from "./markdown";
import { MarkdownView } from "./MarkdownView";
import {
  createRecognition,
  speakText,
  speakViaServer,
  speechRecognitionSupported,
  speechSynthesisSupported,
  stopSpeaking,
  unlockAudio,
  warmSpeechVoices,
} from "./voice";

const NAV = [
  ["ask", "Ask"],
  ["write", "Write"],
  ["find", "Find papers"],
  ["collaborators", "Collaborators"],
  ["library", "Library"],
  ["compare", "Compare"],
  ["events", "Events"],
];


const STATUS_COPY = {
  ANSWERED: { label: "Answered", tone: "good" },
  INSUFFICIENT_EVIDENCE: { label: "Not enough evidence", tone: "warn" },
  CONFLICTING_EVIDENCE: { label: "Sources disagree", tone: "warn" },
  CLARIFICATION_NEEDED: { label: "Needs a clearer question", tone: "warn" },
  NO_RETRIEVAL_NEEDED: { label: "No documents needed", tone: "" },
};

export default function App() {
  const [session, setSession] = useState(undefined);
  const [page, setPage] = useState("ask");
  const [health, setHealth] = useState(null);
  const [docs, setDocs] = useState(() => {
    try {
      const saved = localStorage.getItem("lexicon-cached-docs");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const updateDocs = useCallback((newDocs) => {
    setDocs(newDocs || []);
    try {
      if (Array.isArray(newDocs)) {
        localStorage.setItem("lexicon-cached-docs", JSON.stringify(newDocs));
      }
    } catch {
      /* ignore */
    }
  }, []);
  const [lastResult, setLastResult] = useState(null);
  const [selectedEvidence, setSelectedEvidence] = useState(null);
  const [history, setHistory] = useState([]);
  const [askScope, setAskScope] = useState(null);
  const [askDraft, setAskDraft] = useState("");
  const [chatEpoch, setChatEpoch] = useState(0);
  const [askChatId, setAskChatId] = useState(null);
  const [askRestoreTurns, setAskRestoreTurns] = useState(null);
  const [sidebarRecents, setSidebarRecents] = useState([]);
  const [activeRecentId, setActiveRecentId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    try {
      return localStorage.getItem("lexicon-sidebar-open") !== "0";
    } catch {
      return true;
    }
  });
  const [chatsOpen, setChatsOpen] = useState(() => {
    try {
      return localStorage.getItem("lexicon-chats-open") === "1";
    } catch {
      return false;
    }
  });
  const [findRestore, setFindRestore] = useState(null);
  const [comparePrefill, setComparePrefill] = useState(null);
  const [writeStoryId, setWriteStoryId] = useState(null);
  const [publicSlug, setPublicSlug] = useState(() => parseStoryHash());
  const navRef = useRef(null);

  useEffect(() => {
    api.me().then(setSession).catch(() => setSession(null));
  }, []);

  useEffect(() => {
    function onHash() {
      setPublicSlug(parseStoryHash());
    }
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (session && publicSlug) {
      setPage("write");
    }
  }, [session, publicSlug]);

  useEffect(() => {
    function onLost() {
      setSession(null);
    }
    window.addEventListener("selfrag-auth-lost", onLost);
    return () => window.removeEventListener("selfrag-auth-lost", onLost);
  }, []);

  const loadSidebarRecents = useCallback(async () => {
    if (!session) return;
    try {
      const payload = await api.workspaceRecents(24);
      setSidebarRecents(payload.items || []);
    } catch {
      /* keep prior list */
    }
  }, [session]);

  const refresh = useCallback(async () => {
    try {
      const [h, d, recent] = await Promise.all([
        api.health().catch(() => null),
        api.documents().catch(() => null),
        api.recent().catch(() => []),
      ]);
      if (h) setHealth(h);
      if (d?.documents) updateDocs(d.documents);
      setHistory(Array.isArray(recent) ? recent : []);
      if (session) await loadSidebarRecents();
    } catch (err) {
      console.warn(err);
    }
  }, [session, loadSidebarRecents, updateDocs]);

  useEffect(() => {
    // Immediately fetch library documents so they populate right away
    api.documents().then((d) => {
      if (d?.documents) updateDocs(d.documents);
    }).catch(console.warn);

    refresh();

    // Send a non-blocking warming ping and keepalive heartbeat every 8 minutes
    // so Render does not spin down into cold sleep while the user has the app open
    api.health().catch(() => {});
    const pingTimer = setInterval(() => {
      api.health().catch(() => {});
    }, 8 * 60 * 1000);
    return () => clearInterval(pingTimer);
  }, [refresh, updateDocs]);

  function toggleSidebar() {
    setSidebarOpen((open) => {
      const next = !open;
      try {
        localStorage.setItem("lexicon-sidebar-open", next ? "1" : "0");
      } catch {
        /* ignore */
      }
      return next;
    });
  }

  function toggleChats() {
    setChatsOpen((open) => {
      const next = !open;
      try {
        localStorage.setItem("lexicon-chats-open", next ? "1" : "0");
      } catch {
        /* ignore */
      }
      return next;
    });
  }

  function startNewChat() {
    setLastResult(null);
    setSelectedEvidence(null);
    setAskScope(null);
    setAskDraft("");
    setAskChatId(null);
    setAskRestoreTurns(null);
    setActiveRecentId(null);
    setFindRestore(null);
    setChatEpoch((n) => n + 1);
    setPage("ask");
  }

  async function openSidebarRecent(item) {
    setActiveRecentId(`${item.kind}:${item.id}`);
    if (item.kind === "ask") {
      setAskDraft("");
      setSelectedEvidence(null);
      setAskScope(null);
      setLastResult(null);
      setAskChatId(null);
      setAskRestoreTurns(null);
      setPage("ask");
      try {
        const chat = await api.askChatItem(item.id);
        const turns = (chat.turns || []).map((turn, index) => ({
          id: turn?.result?.query_id || `${chat.chat_id}-${index}`,
          query: turn.query || "",
          result: turn.result || null,
          pending: false,
          error: turn.error || "",
        }));
        setAskChatId(chat.chat_id);
        setAskRestoreTurns(turns);
        const lastWithResult = [...turns].reverse().find((t) => t.result);
        if (lastWithResult?.result) setLastResult(lastWithResult.result);
      } catch {
        try {
          const saved = await api.historyItem(item.id);
          setAskChatId(null);
          setAskRestoreTurns([
            {
              id: saved.query_id || item.id,
              query: saved.query || item.query || "",
              result: saved,
              pending: false,
              error: "",
            },
          ]);
          setLastResult(saved);
        } catch {
          setAskDraft(item.query || "");
        }
      }
      setChatEpoch((n) => n + 1);
      return;
    }
    try {
      const saved = await api.paperSearchItem(item.id);
      setFindRestore({
        id: saved.search_id || item.id,
        query: saved.query,
        provider: saved.provider,
        papers: saved.papers || [],
      });
      setPage("find");
    } catch {
      setFindRestore(null);
    }
  }

  async function deleteSidebarRecent(item, event) {
    event?.stopPropagation?.();
    event?.preventDefault?.();
    if (!item?.id) return;
    const key = `${item.kind}:${item.id}`;
    const previous = sidebarRecents;
    // Remove immediately so the × feels instant.
    setSidebarRecents((prev) => prev.filter((row) => `${row.kind}:${row.id}` !== key));
    if (activeRecentId === key) {
      setActiveRecentId(null);
      if (item.kind === "ask") {
        setLastResult(null);
        setSelectedEvidence(null);
        setAskScope(null);
        setAskDraft("");
        setAskChatId(null);
        setAskRestoreTurns(null);
        setChatEpoch((n) => n + 1);
      } else {
        setFindRestore(null);
      }
    }
    setHistory((prev) => (item.kind === "ask" ? prev.filter((h) => h.query_id !== item.id) : prev));
    try {
      if (item.kind === "ask") {
        await api.deleteRecentAsk(item.id);
      } else {
        await api.deleteRecentPapers(item.id);
      }
      await loadSidebarRecents();
    } catch (err) {
      console.warn(err);
      setSidebarRecents(previous);
      await loadSidebarRecents();
    }
  }

  async function logout() {
    try {
      await api.logout();
    } catch {
      /* still leave the session locally */
    }
    setSession(null);
    setDocs([]);
    setLastResult(null);
    setSelectedEvidence(null);
    setAskScope(null);
    setAskDraft("");
    setAskChatId(null);
    setAskRestoreTurns(null);
    setHistory([]);
    setSidebarRecents([]);
    setActiveRecentId(null);
    setFindRestore(null);
    setPage("ask");
  }

  const askChatRecents = useMemo(
    () => sidebarRecents.filter((item) => item.kind === "ask"),
    [sidebarRecents],
  );

  const navIndex = Math.max(0, NAV.findIndex(([id]) => id === page));

  useLayoutEffect(() => {
    const nav = navRef.current;
    if (!nav) return;
    function place() {
      const btn =
        nav.querySelector(":scope > .nav-chats > button.active") ||
        nav.querySelector(":scope > button.active");
      if (!btn) return;
      nav.style.setProperty("--thumb-x", `${btn.offsetLeft}px`);
      nav.style.setProperty("--thumb-w", `${btn.offsetWidth}px`);
      nav.style.setProperty("--thumb-y", `${btn.offsetTop}px`);
      nav.style.setProperty("--thumb-h", `${btn.offsetHeight}px`);
    }
    place();
    const raf = requestAnimationFrame(place);
    window.addEventListener("resize", place);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", place);
    };
  }, [page, session, askChatRecents.length, chatsOpen, sidebarOpen]);

  if (session === undefined) {
    return (
      <div className="auth-screen session-splash">
        <div className="auth-orbs" aria-hidden="true">
          <span className="shape shape-sphere shape-a is-near" />
          <span className="shape shape-torus shape-c is-mid" />
          <span className="shape shape-sphere shape-b is-far" />
          <span className="shape shape-torus shape-d is-far" />
          <span className="shape shape-sphere shape-e is-near" />
        </div>
        <Logo size="hero" />
        <p className="muted">Opening the gate…</p>
      </div>
    );
  }

  if (!session) {
    if (publicSlug) {
      return (
        <div className="app story-public-shell">
          <div className="app-atmosphere" aria-hidden="true">
            <span className="shape shape-sphere shape-a is-near" />
            <span className="shape shape-torus shape-c is-mid" />
            <span className="shape shape-sphere shape-b is-far" />
            <span className="app-grain" />
          </div>
          <main className="main story-public-main">
            <StoryReader
              slug={publicSlug}
              showSignIn
              onClose={() => {
                window.location.hash = "";
                setPublicSlug(null);
              }}
            />
          </main>
        </div>
      );
    }
    return <AuthScreen onSignedIn={setSession} />;
  }

  return (
    <div className={`app${sidebarOpen ? "" : " sidebar-collapsed"}`}>
      <div className="app-atmosphere" aria-hidden="true">
        <span className="shape shape-sphere shape-a is-near" />
        <span className="shape shape-sphere shape-b is-far" />
        <span className="shape shape-torus shape-c is-mid" />
        <span className="shape shape-torus shape-d is-far" />
        <span className="shape shape-sphere shape-e is-near" />
        <span className="app-grain" />
      </div>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside
        className="sidebar"
        aria-hidden={!sidebarOpen}
        {...(!sidebarOpen ? { inert: "" } : {})}
      >
        <div className="sidebar-top">
          <div className="brand">
            <Logo size="nav" />
            <p>Find sources. Ask with evidence. Trust the answer.</p>
          </div>
          <button
            type="button"
            className="sidebar-toggle"
            onClick={toggleSidebar}
            aria-label="Close sidebar"
            title="Close sidebar"
          >
            «
          </button>
        </div>
        <button type="button" className="new-chat-btn" onClick={startNewChat}>
          New Chat
        </button>
        <nav ref={navRef} data-active={page} data-index={navIndex} aria-label="Primary">
          <span className="nav-thumb" aria-hidden="true" />
          <div className={`nav-chats${chatsOpen ? " is-open" : ""}`}>
            <button
              type="button"
              className={chatsOpen ? "active" : ""}
              onClick={toggleChats}
              aria-expanded={chatsOpen}
              aria-controls="sidebar-chats-list"
            >
              <span>Chats</span>
            </button>
            {chatsOpen && (
              <div id="sidebar-chats-list" className="nav-chats-panel">
                {!askChatRecents.length ? (
                  <p className="sidebar-recents-empty">Your conversations will show up here.</p>
                ) : (
                  <ul className="sidebar-recents-list">
                    {askChatRecents.map((item) => {
                      const key = `${item.kind}:${item.id}`;
                      const topic = String(item.query || "Untitled chat").trim() || "Untitled chat";
                      return (
                        <li key={key} className="sidebar-recent-row">
                          <button
                            type="button"
                            className={`sidebar-recent-item${activeRecentId === key ? " active" : ""}`}
                            onClick={() => openSidebarRecent(item)}
                            title={topic}
                          >
                            <span className="sidebar-recent-title">{topic}</span>
                          </button>
                          <button
                            type="button"
                            className="sidebar-recent-delete"
                            aria-label={`Delete chat: ${topic}`}
                            title="Remove from history"
                            onMouseDown={(event) => {
                              event.preventDefault();
                              event.stopPropagation();
                            }}
                            onClick={(event) => {
                              event.preventDefault();
                              event.stopPropagation();
                              deleteSidebarRecent(item, event);
                            }}
                          >
                            ×
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            )}
          </div>
          {NAV.map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={!chatsOpen && page === id ? "active" : ""}
              onClick={() => {
                setChatsOpen(false);
                try {
                  localStorage.setItem("lexicon-chats-open", "0");
                } catch {
                  /* ignore */
                }
                setPage(id);
              }}
            >
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className="topbar-user">{session.name || session.email}</span>
          <button type="button" className="ghost" onClick={logout}>
            Log out
          </button>
        </div>
      </aside>
      <div className="workspace">
        <header className="workspace-bar">
          {!sidebarOpen && (
            <button
              type="button"
              className="sidebar-toggle sidebar-toggle-open"
              onClick={toggleSidebar}
              aria-label="Open sidebar"
              title="Open sidebar"
            >
              »
            </button>
          )}
          <span>
            {health
              ? `${docs.length} source${docs.length === 1 ? "" : "s"} · ${health.knowledge_base?.chunks ?? 0} passages`
              : "Connecting to API…"}
          </span>
        </header>
        <main className="main" id="main" data-page={page} key={page}>
          {page === "collaborators" && (
            <CollaboratorsPage
              session={session}
              onNavigate={setPage}
              onOpenPaper={(id) => {
                setWriteStoryId(id);
                setPage("write");
              }}
            />
          )}
          {page === "ask" && (
            <Ask
              key={chatEpoch}
              docs={docs}
              query={askDraft}
              setQuery={setAskDraft}
              result={lastResult}
              setResult={setLastResult}
              selected={selectedEvidence}
              setSelected={setSelectedEvidence}
              onQueried={refresh}
              onRecentsChange={loadSidebarRecents}
              askScope={askScope}
              onClearScope={() => setAskScope(null)}
              chatId={askChatId}
              onChatId={(id) => {
                setAskChatId(id);
                if (id) setActiveRecentId(`ask:${id}`);
              }}
              initialTurns={askRestoreTurns}
            />
          )}
          {page === "write" && (
            <WritePage
              session={session}
              openStoryId={writeStoryId}
              onOpenStoryConsumed={() => setWriteStoryId(null)}
            />
          )}
          {page === "find" && (
            <FindPapers
              onImported={refresh}
              onRecentsChange={loadSidebarRecents}
              restore={findRestore}
            />
          )}
          {page === "library" && (
            <Library
              docs={docs}
              onChange={refresh}
            />
          )}
          {page === "compare" && (
            <ComparePapers
              docs={docs}
              prefill={comparePrefill}
              onPrefillConsumed={() => setComparePrefill(null)}
            />
          )}
          {page === "events" && <Events />}
        </main>
      </div>
    </div>
  );
}

function formatEventDate(value) {
  if (!value) return "—";
  try {
    return new Date(`${value}T00:00:00`).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return value;
  }
}

const EVENT_TYPE_LABELS = {
  conference: "Conference",
  ieee: "IEEE",
  research_cfp: "Research CFP",
  workshop: "Workshop",
  hackathon: "Hackathon",
  meetup: "Meetup",
};

const EVENT_TYPE_ICONS = {
  conference: "▣",
  ieee: "⚡",
  research_cfp: "¶",
  workshop: "◇",
  hackathon: "⌘",
  meetup: "◎",
};

async function copyText(value) {
  const text = String(value || "");
  if (!text) return false;
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}

function Events() {
  const [q, setQ] = useState("");
  const [region, setRegion] = useState("all");
  const [city, setCity] = useState("");
  const [topic, setTopic] = useState("");
  const [eventType, setEventType] = useState("");
  const [status, setStatus] = useState("active");
  const [items, setItems] = useState([]);
  const [cities, setCities] = useState([]);
  const [topics, setTopics] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [brokenIcons, setBrokenIcons] = useState({});

  const load = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const list = await api.events({ q, region, city, topic, event_type: eventType, status });
      setItems(list.events || []);
      setCities(list.cities || []);
      setTopics(list.topics || []);
      setBrokenIcons({});
    } catch (err) {
      setError(err.message || "Could not load events.");
    } finally {
      setBusy(false);
    }
  }, [q, region, city, topic, eventType, status]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <>
      <div className="page-title">
        <div>
          <h2>Events</h2>
          <p>Current and upcoming conferences, IEEE, research CFPs, and hackathons — apply on the official site.</p>
        </div>
      </div>
      <div className="spotlight spotlight-forest">
        <p className="spotlight-kicker">Live & upcoming</p>
        <p>Browse events happening now or coming up next. Filter by type and city, then open Apply now.</p>
      </div>
      <form
        className="card event-filters"
        onSubmit={(event) => {
          event.preventDefault();
          load();
        }}
      >
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search events, cities, topics…"
          aria-label="Search events"
        />
        <div className="chips" style={{ marginTop: 12 }}>
          {[
            ["all", "All regions"],
            ["india", "India"],
            ["worldwide", "Worldwide"],
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={`chip${region === id ? " active" : ""}`}
              onClick={() => setRegion(id)}
            >
              {label}
            </button>
          ))}
          {[
            ["active", "Happening & upcoming"],
            ["live", "Happening now"],
            ["upcoming", "Upcoming"],
            ["open", "Registration open"],
            ["closed", "Registration closed"],
            ["past", "Past"],
          ].map(([id, label]) => (
            <button
              key={`status-${id}`}
              type="button"
              className={`chip${status === id ? " active" : ""}`}
              onClick={() => setStatus(id)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="chips" style={{ marginTop: 10 }}>
          <button
            type="button"
            className={`chip${!eventType ? " active" : ""}`}
            onClick={() => setEventType("")}
          >
            All types
          </button>
          {Object.entries(EVENT_TYPE_LABELS).map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={`chip${eventType === id ? " active" : ""}`}
              onClick={() => setEventType(id)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <label className="event-select">
            City
            <select value={city} onChange={(e) => setCity(e.target.value)}>
              <option value="">All cities</option>
              {cities.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label className="event-select">
            Topic
            <select value={topic} onChange={(e) => setTopic(e.target.value)}>
              <option value="">All topics</option>
              {topics.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <button className={`primary${busy ? " is-busy" : ""}`} type="submit" disabled={busy}>
            {busy ? "Loading…" : "Apply filters"}
          </button>
        </div>
      </form>
      {error && <StatusBanner tone="error">{error}</StatusBanner>}
      <div className="event-grid">
        {items.map((item) => {
          const typeLabel = EVENT_TYPE_LABELS[item.event_type] || "Event";
          const typeIcon = EVENT_TYPE_ICONS[item.event_type] || "▣";
          const imageUrl = (item.image_url || "").trim();
          const showImg = Boolean(imageUrl) && !brokenIcons[item.event_id];
          const registrationOpen = item.registration_open !== false;
          const timing = item.timing || "upcoming";
          const timingLabel =
            timing === "live" ? "Happening now" : timing === "past" ? "Past" : "Upcoming";
          const timingTone = timing === "live" ? "good" : timing === "past" ? "neutral" : "neutral";
          return (
            <article key={item.event_id} className="card event-card">
              <div className={`event-media type-${item.event_type || "conference"}`}>
                {showImg ? (
                  <img
                    className="event-logo"
                    src={imageUrl}
                    alt=""
                    loading="lazy"
                    onError={() =>
                      setBrokenIcons((prev) => ({ ...prev, [item.event_id]: true }))
                    }
                  />
                ) : (
                  <span className="event-icon" aria-hidden="true">
                    {typeIcon}
                  </span>
                )}
                <span className="event-type-badge">{typeLabel}</span>
                {timing === "live" && <span className="event-live-badge">Live</span>}
              </div>
              <div className="event-card-body">
                <div className="event-card-top">
                  <h3>{item.name}</h3>
                  <span className={`flag ${item.region === "india" ? "good" : "neutral"}`}>
                    {item.region === "india" ? "India" : "Worldwide"}
                  </span>
                </div>
                <p className="event-location">
                  {item.city}, {item.country}
                  {item.venue ? ` · ${item.venue}` : ""}
                </p>
                <p className="status" style={{ marginTop: 6 }}>
                  <span className={`flag ${timingTone}`} style={{ marginRight: 8 }}>
                    {timingLabel}
                  </span>
                  {formatEventDate(item.start_date)} – {formatEventDate(item.end_date)}
                </p>
                <p className="status" style={{ marginTop: 4 }}>
                  {registrationOpen ? (
                    <>
                      {item.cfp_deadline ? `CFP ${formatEventDate(item.cfp_deadline)}` : ""}
                      {item.cfp_deadline && item.registration_deadline ? " · " : ""}
                      {item.registration_deadline
                        ? `Apply by ${formatEventDate(item.registration_deadline)}`
                        : !item.cfp_deadline
                          ? "Registration open"
                          : ""}
                    </>
                  ) : (
                    <span className="event-closed">Registration closed</span>
                  )}
                </p>
                {item.summary && <p className="event-summary">{item.summary}</p>}
                <div className="chips" style={{ marginTop: 10 }}>
                  {item.topics.map((tag) => (
                    <span key={tag} className="chip" style={{ cursor: "default" }}>
                      {tag}
                    </span>
                  ))}
                </div>
                <div className="row" style={{ marginTop: 14 }}>
                  {registrationOpen && (item.apply_url || item.website) ? (
                    <a
                      className="primary"
                      href={item.apply_url || item.website}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Apply now
                    </a>
                  ) : (
                    <span className="event-closed-pill">Registration closed</span>
                  )}
                  {item.website && item.website !== item.apply_url && (
                    <a className="ghost" href={item.website} target="_blank" rel="noreferrer">
                      Open site
                    </a>
                  )}
                </div>
              </div>
            </article>
          );
        })}
      </div>
      {!busy && items.length === 0 && (
        <div className="card">
          <h3>No events match</h3>
          <p className="status">Try Happening & upcoming, or clear the city and topic filters.</p>
        </div>
      )}
    </>
  );
}

function Ask({
  docs,
  query,
  setQuery,
  result,
  setResult,
  selected,
  setSelected,
  onQueried,
  onRecentsChange,
  askScope,
  onClearScope,
  chatId,
  onChatId,
  initialTurns,
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [askMode, setAskMode] = useState("research"); // "research" | "general"
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [turns, setTurns] = useState(() =>
    Array.isArray(initialTurns) && initialTurns.length
      ? initialTurns
      : result
        ? [
            {
              id: result.query_id || "restored",
              query: result.query || "",
              result,
              pending: false,
              error: "",
            },
          ]
        : [],
  );
  const [activeChatId, setActiveChatId] = useState(chatId || null);
  const chatIdRef = useRef(chatId || null);
  const turnsRef = useRef(
    Array.isArray(initialTurns) && initialTurns.length
      ? initialTurns
      : result
        ? [
            {
              id: result.query_id || "restored",
              query: result.query || "",
              result,
              pending: false,
              error: "",
            },
          ]
        : [],
  );
  const submittingRef = useRef(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceHint, setVoiceHint] = useState("");
  const threadEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const baseQueryRef = useRef("");
  const voiceSupported = speechRecognitionSupported();
  const emptyLibrary = docs.length === 0;
  const scopedDocs = useMemo(() => {
    if (!askScope?.document_ids?.length) return docs;
    return docs.filter((d) => askScope.document_ids.includes(d.document_id));
  }, [askScope, docs]);

  const pickDocs = useMemo(() => {
    const list = scopedDocs.length ? scopedDocs : docs;
    const seen = new Set();
    const unique = [];
    for (const d of list) {
      if (!d || !d.document_id) continue;
      if (!seen.has(d.document_id)) {
        seen.add(d.document_id);
        unique.push(d);
      }
    }
    return unique;
  }, [scopedDocs, docs]);

  useEffect(() => {
    if (askScope?.document_ids?.length) {
      setSelectedDocIds(askScope.document_ids.filter((id) => docs.some((d) => d.document_id === id)));
    }
  }, [askScope, docs]);

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, busy]);

  useEffect(
    () => () => {
      recognitionRef.current?.abort();
      recognitionRef.current = null;
      stopSpeaking();
    },
    [],
  );

  useEffect(() => {
    warmSpeechVoices();
  }, []);

  async function persistChat(nextTurns, existingChatId = chatIdRef.current || activeChatId) {
    const durable = (nextTurns || []).filter((t) => t.query && (t.result || t.error));
    if (!durable.length) return existingChatId;
    try {
      const saved = await api.saveAskChat({
        chat_id: existingChatId || undefined,
        title: durable[0]?.query || "",
        turns: durable.map((t) => ({
          query: t.query,
          result: t.result || null,
          error: t.error || "",
        })),
      });
      chatIdRef.current = saved.chat_id;
      setActiveChatId(saved.chat_id);
      onChatId?.(saved.chat_id);
      await onRecentsChange?.();
      return saved.chat_id;
    } catch (err) {
      console.warn("Failed to save chat to Recents:", err);
      setError((prev) => prev || err?.message || "Could not save this chat to history.");
      return existingChatId;
    }
  }

  function toggleDoc(id) {
    setSelectedDocIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  function stopListening() {
    const rec = recognitionRef.current;
    recognitionRef.current = null;
    if (rec) {
      try {
        if (typeof rec.abort === "function") rec.abort();
        else if (typeof rec.stop === "function") rec.stop();
      } catch {
        /* ignore */
      }
    }
    setListening(false);
    setVoiceHint("");
  }

  function startListening() {
    if (!voiceSupported || emptyLibrary || busy || submittingRef.current) return;
    stopSpeaking();
    stopListening();
    baseQueryRef.current = (query || "").trim();
    const session = createRecognition({
      onResult: ({ display }) => {
        if (submittingRef.current) return;
        const base = baseQueryRef.current;
        const next = base ? `${base} ${display}`.trim() : display;
        setQuery(next);
        setVoiceHint(display ? "Listening…" : "Speak your question…");
      },
      onError: (err) => {
        if (submittingRef.current) return;
        setError(err.message);
        stopListening();
      },
      onEnd: () => {
        recognitionRef.current = null;
        setListening(false);
        setVoiceHint("");
      },
    });
    if (!session) {
      setError("Voice input is not supported in this browser. Try Chrome or Edge.");
      return;
    }
    recognitionRef.current = session;
    setListening(true);
    setVoiceHint("Speak your question…");
    setError("");
    try {
      session.start();
    } catch (err) {
      setError(err?.message || "Could not start the microphone.");
      stopListening();
    }
  }

  function toggleMic() {
    if (listening) stopListening();
    else startListening();
  }

  async function ask(next = query) {
    const text = (typeof next === "string" ? next : query || "").trim();
    if (!text || (emptyLibrary && askMode !== "general") || busy || submittingRef.current) return;
    submittingRef.current = true;
    stopListening();
    stopSpeaking();
    baseQueryRef.current = "";
    setQuery("");
    setBusy(true);
    setError("");
    setSelected(null);
    const turnId =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `turn-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const pendingTurn = { id: turnId, query: text, result: null, pending: true, error: "" };
    const withPending = [...turnsRef.current, pendingTurn];
    turnsRef.current = withPending;
    setTurns(withPending);
    try {
      const scopedIds = selectedDocIds.length
        ? selectedDocIds
        : askScope?.document_ids?.length
          ? askScope.document_ids
          : undefined;
      const payload = await api.query({
        query: text,
        include_trace: true,
        document_ids: scopedIds,
        chat_id: chatIdRef.current || activeChatId || undefined,
        mode: askMode,
      });
      const serverChatId = payload?.config_snapshot?.chat_id;
      if (serverChatId) {
        chatIdRef.current = serverChatId;
        setActiveChatId(serverChatId);
        onChatId?.(serverChatId);
      }
      const nextTurns = turnsRef.current.map((t) =>
        t.id === turnId
          ? {
              ...t,
              id: payload.query_id || turnId,
              result: payload,
              pending: false,
              error: "",
            }
          : t,
      );
      turnsRef.current = nextTurns;
      setTurns(nextTurns);
      setResult(payload);
      setQuery("");
      await persistChat(nextTurns, chatIdRef.current || serverChatId || activeChatId);
      await onQueried();
      await onRecentsChange?.();
    } catch (err) {
      const message = err?.message || "Request failed.";
      setError(message);
      const nextTurns = turnsRef.current.map((t) =>
        t.id === turnId ? { ...t, pending: false, error: message } : t,
      );
      turnsRef.current = nextTurns;
      setTurns(nextTurns);
      setQuery("");
      await persistChat(nextTurns);
    } finally {
      submittingRef.current = false;
      setBusy(false);
      setQuery("");
    }
  }

  const scopeLabel = selectedDocIds.length
    ? `${selectedDocIds.length} source${selectedDocIds.length === 1 ? "" : "s"}`
    : askScope?.label || "";
  const showThread = turns.length > 0;

  return (
    <div className={`ask-chat${showThread ? " has-thread" : " is-empty"}`}>
      <div className="ask-thread" role="log" aria-live="polite" aria-relevant="additions">
        {!showThread && (
          <div className="ask-empty">
            <Logo size="hero" />
            <h2 className="ask-empty-title">What do you want to know?</h2>
            <p className="ask-empty-copy">
              {emptyLibrary
                ? "Add sources in Find papers or Library, then ask here."
                : "Ask in plain language. Answers stay grounded in your library."}
            </p>
          </div>
        )}

        {showThread && (
          <>
            {turns.map((turn) => {
              const badge = turn.result ? verificationBadge(turn.result) : null;
              return (
                <Fragment key={turn.id}>
                  <div className="chat-turn chat-turn-user">
                    <div className="chat-bubble chat-bubble-user">{turn.query}</div>
                  </div>

                  {turn.pending && (
                    <div className="chat-turn chat-turn-assistant">
                      <div className="chat-bubble chat-bubble-assistant is-thinking">
                        <span className="ask-thinking-dot" />
                        <span className="ask-thinking-dot" />
                        <span className="ask-thinking-dot" />
                        Checking sources…
                      </div>
                    </div>
                  )}

                  {turn.error && !turn.pending && !turn.result && (
                    <div className="chat-turn chat-turn-assistant">
                      <div className="chat-bubble chat-bubble-assistant">
                        <p className="error">{turn.error}</p>
                      </div>
                    </div>
                  )}

                  {turn.result && (
                    <div className="chat-turn chat-turn-assistant">
                      <div className="chat-bubble chat-bubble-assistant">
                        {badge && (
                          <div className="chat-answer-meta">
                            <span className={`flag ${badge.tone}`}>{badge.label}</span>
                          </div>
                        )}
                        <AnswerView
                          result={turn.result}
                          docs={docs}
                          selected={selected}
                          onSelect={setSelected}
                        />
                      </div>
                    </div>
                  )}
                </Fragment>
              );
            })}
            <div ref={threadEndRef} />
          </>
        )}
      </div>

      <div className="ask-composer">
        {scopeLabel && (
          <div className="theme-chip-row ask-composer-scope">
            <span className="theme-chip">
              {scopeLabel}
              <button
                type="button"
                className="theme-chip-clear"
                onClick={() => {
                  setSelectedDocIds([]);
                  onClearScope?.();
                }}
                aria-label="Clear source filter"
              >
                ×
              </button>
            </span>
          </div>
        )}

        {!emptyLibrary && (
          <details
            className="ask-sources"
            open={sourcesOpen}
            onToggle={(e) => setSourcesOpen(e.currentTarget.open)}
          >
            <summary>
              Sources
              <span className="ask-sources-hint">
                {selectedDocIds.length
                  ? `${selectedDocIds.length} selected`
                  : askScope?.document_ids?.length
                    ? "Theme scoped"
                    : "Entire library"}
              </span>
            </summary>
            <div className="chips write-source-chips ask-source-chips">
              {pickDocs.map((doc) => {
                const id = doc.document_id;
                const active = selectedDocIds.includes(id);
                const displayTitle = doc.title && doc.title !== doc.name ? doc.title : doc.name;
                const tooltip = doc.title && doc.title !== doc.name ? `${doc.title} (file: ${doc.name})` : doc.name;
                return (
                  <button
                    key={id}
                    type="button"
                    className={`chip${active ? " active" : ""}`}
                    onClick={() => toggleDoc(id)}
                    title={tooltip}
                  >
                    {displayTitle}
                  </button>
                );
              })}
            </div>
          </details>
        )}

        <div className="ask-mode-bar">
          <div className="ask-mode-toggle">
            <button
              type="button"
              className={`chip${askMode === "research" ? " active" : ""}`}
              onClick={() => setAskMode("research")}
              title="Grounded in your indexed research papers with Evidence Gate verification"
            >
              Research Sources Mode
            </button>
            <button
              type="button"
              className={`chip${askMode === "general" ? " active" : ""}`}
              onClick={() => setAskMode("general")}
              title="General academic synthesis (labeled, without citing indexed library)"
            >
              General Knowledge Mode
            </button>
          </div>
          {askMode === "general" && (
            <span className="muted" style={{ fontSize: "11px", color: "var(--apricot-soft)" }}>
              General mode answers broad questions without citing papers
            </span>
          )}
        </div>

        <form
          className="ask-composer-box"
          onSubmit={(e) => {
            e.preventDefault();
            ask();
          }}
        >
          <textarea
            value={query}
            rows={1}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              emptyLibrary && askMode !== "general"
                ? "Add sources first (or switch to General Knowledge mode)…"
                : busy
                  ? "Processing query…"
                  : listening
                    ? "Listening…"
                    : askMode === "general"
                      ? "Ask any question in General Knowledge mode…"
                      : "Message Lexicon Gate… or tap the mic"
            }
            disabled={emptyLibrary && askMode !== "general"}
            readOnly={busy}
            autoComplete="off"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (!busy) ask();
              }
            }}
          />
          {voiceSupported && (
            <button
              type="button"
              className={`ghost ask-mic${listening ? " is-live" : ""}`}
              onClick={toggleMic}
              disabled={busy || (emptyLibrary && askMode !== "general")}
              aria-pressed={listening}
              aria-label={listening ? "Stop listening" : "Speak your question"}
              title={listening ? "Stop listening" : "Speak your question"}
            >
              {listening ? "Stop" : "Mic"}
            </button>
          )}
          <button
            type="submit"
            className={`primary ask-send${busy ? " is-busy" : ""}`}
            disabled={busy || (emptyLibrary && askMode !== "general") || !query.trim()}
            aria-label={busy ? "Checking sources" : "Send"}
          >
            {busy ? "…" : "Send"}
          </button>
        </form>
        {listening && voiceHint ? <p className="ask-voice-hint">{voiceHint}</p> : null}
        {error && <p className="ask-composer-error error">{error}</p>}
        {!voiceSupported && !emptyLibrary && (
          <p className="ask-voice-hint muted">
            Voice input needs Chrome or Edge. You can still type questions.
          </p>
        )}
      </div>
    </div>
  );
}

function paperHref(paper) {
  if (paper.doi) {
    const doi = String(paper.doi).replace(/^https?:\/\/doi\.org\//i, "");
    return `https://doi.org/${doi}`;
  }
  return paper.url || paper.pdf_url || "";
}

function FindPapers({ onImported, onRecentsChange, restore }) {
  const FILTERS = [
    { id: "all", label: "All" },
    { id: "academic", label: "Academic Papers" },
    { id: "ieee", label: "IEEE Xplore" },
    { id: "open_access", label: "Open Access" },
    { id: "research_web", label: "Research Websites" },
  ];
  const [query, setQuery] = useState(restore?.query || "");
  const [filter, setFilter] = useState(restore?.filter || "all");
  const [busy, setBusy] = useState(false);
  const [importing, setImporting] = useState(null);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [results, setResults] = useState(
    restore
      ? { query: restore.query, provider: restore.provider, papers: restore.papers || [], notes: restore.notes || [] }
      : null
  );

  useEffect(() => {
    if (!restore?.id) return;
    setQuery(restore.query || "");
    setFilter(restore.filter || "all");
    setResults({
      query: restore.query,
      provider: restore.provider,
      papers: restore.papers || [],
      notes: restore.notes || [],
      filter: restore.filter,
    });
    setError("");
    setMessage("");
  }, [restore?.id]);

  async function search(event) {
    event?.preventDefault();
    const text = query.trim();
    if (!text || busy) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const payload = await api.searchPapers(text, { filter });
      setResults(payload);
      await api.savePaperSearch({
        query: text,
        provider: payload.provider || "unknown",
        filter,
        papers: (payload.papers || []).slice(0, 20).map((paper) => ({
          paper_id: paper.paper_id,
          title: paper.title,
          authors: paper.authors || [],
          year: paper.year,
          source: paper.source,
          url: paper.url,
          pdf_url: paper.pdf_url,
          doi: paper.doi,
          venue: paper.venue,
          abstract: paper.abstract,
          summary: paper.summary,
          open_access: paper.open_access,
          citation_count: paper.citation_count,
          result_kind: paper.result_kind,
          full_text_available: paper.full_text_available,
        })),
      });
      await onRecentsChange?.();
      if (payload.notes?.length) setMessage(payload.notes.join(" "));
      if (!payload.papers?.length) {
        setMessage((prev) => prev || "No results. Try another filter or a more specific query.");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function addPaper(paper) {
    setImporting(paper.paper_id);
    setError("");
    setMessage("");
    const payload = {
      paper_id: paper.paper_id,
      source: paper.source,
      title: paper.title,
      authors: paper.authors,
      year: paper.year,
      venue: paper.venue,
      abstract: paper.abstract,
      summary: paper.summary,
      doi: paper.doi,
      pdf_url: paper.pdf_url,
      url: paper.url,
      result_kind: paper.result_kind,
    };
    try {
      const result = await api.importPaper(payload);
      if (result.ingested === "pdf") {
        setMessage(`Added “${paper.title}”. Full PDF indexed. Ask over it from Ask.`);
      } else if (result.ingested === "web") {
        setMessage(`Added “${paper.title}”. Page text indexed (no open PDF).`);
      } else {
        setMessage(
          `Added “${paper.title}”. Could not fetch a PDF (publisher blocked or paywalled). Abstract saved. Use Open paper for the official copy.`
        );
      }
      if (result.warnings?.length) setError(result.warnings.join(" · "));
      await onImported();
    } catch (err) {
      setError(err.message);
    } finally {
      setImporting(null);
    }
  }

  function availabilityLabel(paper) {
    if (paper.full_text_available || paper.pdf_url) return { text: "Full text", tone: "good" };
    if (paper.result_kind === "web") return { text: "Web page", tone: "" };
    return { text: "Abstract only", tone: "warn" };
  }

  function canAdd(paper) {
    return Boolean(
      paper.pdf_url ||
        paper.full_text_available ||
        paper.open_access ||
        paper.abstract ||
        paper.summary ||
        paper.url
    );
  }

  return (
    <>
      <div className="page-title">
        <div>
          <h2>Find papers</h2>
          <p>
            Search academic indexes and research websites. Prefer open-access PDFs when adding to
            Library, then Ask.
          </p>
        </div>
      </div>
      <div className="spotlight spotlight-orange">
        <p className="spotlight-kicker">Unified search</p>
        <p>Filter by All, Academic Papers, Research Websites, or Open Access. Full PDFs are indexed when legally available.</p>
      </div>
      <form className="card" style={{ marginBottom: 16 }} onSubmit={search}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Paper title, topic, author, or research question…"
        />
        <div className="chips find-filter-chips" style={{ marginTop: 10 }}>
          {FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              className={`chip${filter === f.id ? " active" : ""}`}
              onClick={() => setFilter(f.id)}
            >
              {f.label}
            </button>
          ))}
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <button className={`primary${busy ? " is-busy" : ""}`} type="submit" disabled={busy || !query.trim()}>
            {busy ? "Searching…" : "Search"}
          </button>
          {results?.provider && (
            <span className="status">
              Results from {(results.providers_used || String(results.provider).split("+")).filter(Boolean).map(p => ({
                ieee: "IEEE Xplore",
                europe_pmc: "Europe PMC",
                semantic_scholar: "Semantic Scholar",
                openalex: "OpenAlex",
                arxiv: "arXiv",
                crossref: "CrossRef",
                tavily: "Web",
              }[p] || p.replaceAll("_", " "))).join(", ")}
            </span>
          )}
        </div>
        {query.trim() && (
          <div className="find-external-sources" style={{ marginTop: 12, display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
            <span className="status" style={{ margin: 0, fontSize: "0.82rem" }}>Quick launch query in:</span>
            <a
              className="chip ghost"
              href={`https://scholar.google.com/scholar?q=${encodeURIComponent(query.trim())}`}
              target="_blank"
              rel="noreferrer"
              style={{ textDecoration: "none", display: "inline-flex", alignItems: "center", gap: 4 }}
            >
              🎓 Google Scholar
            </a>
            <a
              className="chip ghost"
              href={`https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=${encodeURIComponent(query.trim())}`}
              target="_blank"
              rel="noreferrer"
              style={{ textDecoration: "none", display: "inline-flex", alignItems: "center", gap: 4 }}
            >
              ⚡ IEEE Xplore
            </a>
            <a
              className="chip ghost"
              href={`https://arxiv.org/search/?query=${encodeURIComponent(query.trim())}&searchtype=all`}
              target="_blank"
              rel="noreferrer"
              style={{ textDecoration: "none", display: "inline-flex", alignItems: "center", gap: 4 }}
            >
              📚 arXiv
            </a>
            <a
              className="chip ghost"
              href={`https://www.semanticscholar.org/search?q=${encodeURIComponent(query.trim())}`}
              target="_blank"
              rel="noreferrer"
              style={{ textDecoration: "none", display: "inline-flex", alignItems: "center", gap: 4 }}
            >
              🌐 Semantic Scholar
            </a>
          </div>
        )}
      </form>
      {error && <StatusBanner tone="error">{error}</StatusBanner>}
      {message && <StatusBanner tone="success">{message}</StatusBanner>}
      {(results?.papers || []).map((paper) => {
        const avail = availabilityLabel(paper);
        const blurb = paper.summary || paper.abstract || "";
        const sourceLabel = {
          ieee: "IEEE Xplore",
          europe_pmc: "Europe PMC",
          semantic_scholar: "Semantic Scholar",
          openalex: "OpenAlex",
          arxiv: "arXiv",
          crossref: "CrossRef",
          tavily: "Web",
        }[paper.source] || (paper.source || "").replaceAll("_", " ");
        return (
          <div key={`${paper.source}-${paper.paper_id}`} className="card paper-row">
            <div>
              <strong>{paper.title}</strong>
              <div className="status" style={{ margin: "4px 0 0" }}>
                <span className="paper-source-badge">{sourceLabel}</span>
                {(paper.authors || []).slice(0, 8).join(", ") || (paper.result_kind === "web" ? "Web result" : "Unknown author")}
                {paper.year ? ` · ${paper.year}` : ""}
                {paper.venue ? ` · ${paper.venue}` : ""}
                {paper.citation_count ? ` · cited ${paper.citation_count}` : ""}
              </div>
              {paper.url && (
                <div className="status" style={{ margin: "2px 0 0", wordBreak: "break-all" }}>
                  {paper.url}
                </div>
              )}
              <div className="paper-flags" style={{ marginTop: 8 }}>
                {paper.open_access && <span className="flag good">Open access</span>}
                <span className={`flag ${avail.tone}`.trim()}>{avail.text}</span>
              </div>
              {blurb && (
                <p className="cite-snippet">
                  {blurb.slice(0, 360)}
                  {blurb.length > 360 ? "…" : ""}
                </p>
              )}
            </div>
            <div className="paper-actions">
              {paperHref(paper) && (
                <a className="ghost" href={paperHref(paper)} target="_blank" rel="noreferrer">
                  Open paper
                </a>
              )}
              <button
                className="primary"
                disabled={importing === paper.paper_id || !canAdd(paper)}
                onClick={() => addPaper(paper)}
              >
                {importing === paper.paper_id ? "Adding…" : "Add to library"}
              </button>
            </div>
          </div>
        );
      })}
    </>
  );
}

function Library({ docs, onChange }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [uploadCount, setUploadCount] = useState(0);
  const [editingDoc, setEditingDoc] = useState(null);
  const [editTitle, setEditTitle] = useState("");
  const [editAuthors, setEditAuthors] = useState("");
  const [editYear, setEditYear] = useState("");
  const fileRef = useRef(null);
  const MAX_UPLOAD_FILES = 20;

  function startEdit(doc) {
    setEditingDoc(doc);
    setEditTitle(doc.title || doc.name || "");
    setEditAuthors((doc.authors || []).join(", "));
    setEditYear(doc.year ? String(doc.year) : "");
  }

  async function saveEdit(e) {
    if (e && e.preventDefault) e.preventDefault();
    if (!editingDoc) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const authors = editAuthors.split(",").map((a) => a.trim()).filter(Boolean);
      const parsedYear = editYear.trim() ? parseInt(editYear.trim(), 10) : null;
      await api.updateDocument(editingDoc.document_id, {
        title: editTitle.trim(),
        authors,
        year: isNaN(parsedYear) ? null : parsedYear,
      });
      setEditingDoc(null);
      await onChange();
      setMessage("Paper title and authors updated.");
    } catch (err) {
      setError(err.message || "Failed to update paper details.");
    } finally {
      setBusy(false);
    }
  }

  async function uploadFiles(list) {
    const files = Array.from(list || []);
    if (!files.length || busy) return;
    if (files.length > MAX_UPLOAD_FILES) {
      setError(`Select up to ${MAX_UPLOAD_FILES} files at once (you picked ${files.length}).`);
      return;
    }
    setBusy(true);
    setUploadCount(files.length);
    setError("");
    setMessage("");
    try {
      const result = await api.upload(files);
      await onChange();
      const n = result.documents?.length || 0;
      setMessage(`Indexed ${n} file${n === 1 ? "" : "s"} (${result.total_chunks} passages).`);
      if (result.warnings?.length) setError(result.warnings.join(" · "));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      setUploadCount(0);
    }
  }

  async function openPicker() {
    if (busy) return;
    if (typeof window.showOpenFilePicker === "function") {
      try {
        const handles = await window.showOpenFilePicker({
          multiple: true,
          types: [
            {
              description: "Papers",
              accept: {
                "application/pdf": [".pdf"],
                "text/plain": [".txt", ".md", ".markdown"],
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
              },
            },
          ],
        });
        const files = await Promise.all(handles.map((handle) => handle.getFile()));
        await uploadFiles(files);
        return;
      } catch (err) {
        if (err?.name === "AbortError") return;
      }
    }
    const input = fileRef.current;
    if (!input) return;
    input.multiple = true;
    input.value = "";
    input.click();
  }

  async function remove(id) {
    if (!window.confirm("Remove this source from the Library? Indexed passages will be deleted.")) {
      return;
    }
    setBusy(true);
    setError("");
    try {
      await api.deleteDocument(id);
      await onChange();
      setMessage("Source removed from Library.");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-title library-head">
        <div>
          <h2>Library</h2>
          <p>
            Sources Ask can use. Click Upload files and select several PDFs or text files in the same
            dialog. Citation metadata comes from the paper record or the file name.
          </p>
        </div>
        <button type="button" className="primary" disabled={busy} onClick={openPicker}>
          {busy ? `Indexing ${uploadCount || "…"}…` : "Upload files"}
        </button>
        <input
          ref={fileRef}
          className="upload-file-input"
          type="file"
          multiple
          accept=".pdf,.txt,.md,.markdown,.docx"
          tabIndex={-1}
          onChange={(event) => {
            const chosen = event.target.files;
            event.target.value = "";
            uploadFiles(chosen);
          }}
        />
      </div>
      <div className="spotlight spotlight-violet">
        <p className="spotlight-kicker">Your sources</p>
        <p>Everything Ask retrieves from lives here — imported papers and files you upload.</p>
      </div>
      {error && <StatusBanner tone="error" onDismiss={() => setError("")}>{error}</StatusBanner>}
      {message && <StatusBanner tone="success" onDismiss={() => setMessage("")}>{message}</StatusBanner>}
      {editingDoc && (
        <form className="card" onSubmit={saveEdit} style={{ marginBottom: 16, border: "1px solid var(--accent-orange)" }}>
          <h4 style={{ margin: "0 0 12px" }}>Edit Paper Citation Details ({editingDoc.name})</h4>
          <label style={{ display: "block", marginBottom: 8, fontSize: "0.88rem" }}>
            Paper Title:
            <input
              style={{ width: "100%", marginTop: 4 }}
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              placeholder="Full paper title"
              required
            />
          </label>
          <label style={{ display: "block", marginBottom: 8, fontSize: "0.88rem" }}>
            Authors (comma-separated):
            <input
              style={{ width: "100%", marginTop: 4 }}
              value={editAuthors}
              onChange={(e) => setEditAuthors(e.target.value)}
              placeholder="e.g. Ashish Vaswani, Noam Shazeer, Niki Parmar"
            />
          </label>
          <label style={{ display: "block", marginBottom: 12, fontSize: "0.88rem" }}>
            Year:
            <input
              style={{ width: 140, marginTop: 4 }}
              type="number"
              value={editYear}
              onChange={(e) => setEditYear(e.target.value)}
              placeholder="e.g. 2023"
            />
          </label>
          <div className="row" style={{ gap: 8 }}>
            <button className="primary" type="submit" disabled={busy || !editTitle.trim()}>
              Save Details
            </button>
            <button className="ghost" type="button" disabled={busy} onClick={() => setEditingDoc(null)}>
              Cancel
            </button>
          </div>
        </form>
      )}
      {!docs.length && (
        <EmptyState title="No sources yet">
          Click Upload files and select several at once, or search academic papers in{" "}
          <strong>Find papers</strong>.
        </EmptyState>
      )}
      {docs.length > 0 && (
        <div className="card">
          <table>
            <thead>
              <tr>
                <th>Citation</th>
                <th className="num">Passages</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {docs.map((doc) => (
                <tr key={doc.document_id}>
                  <td>
                    <div>
                      <strong>{doc.title || doc.name}</strong>
                      <div className="status" style={{ margin: 0 }}>
                        {(doc.authors || []).join(", ") || "Unknown author"}
                        {doc.year ? ` · ${doc.year}` : ""}
                        {doc.venue ? ` · ${doc.venue}` : ""}
                        <br />
                        {doc.name}
                      </div>
                    </div>
                  </td>
                  <td className="num">{doc.n_chunks}</td>
                  <td className="num" style={{ whiteSpace: "nowrap" }}>
                    <button className="ghost" style={{ marginRight: 6 }} disabled={busy} onClick={() => startEdit(doc)}>
                      Edit details
                    </button>
                    <button className="danger" disabled={busy} onClick={() => remove(doc.document_id)}>
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

const STRUCTURE_COLUMNS = ["objective", "method", "dataset", "metric", "result", "limitation"];

function ComparePapers({ docs, prefill, onPrefillConsumed }) {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [extractBusy, setExtractBusy] = useState(false);
  const [selectedIds, setSelectedIds] = useState([]);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [selectedEvidence, setSelectedEvidence] = useState(null);
  const [scopeLabel, setScopeLabel] = useState("");

  useEffect(() => {
    if (!prefill) return;
    if (Array.isArray(prefill.documentIds) && prefill.documentIds.length) {
      setSelectedIds(prefill.documentIds);
    }
    if (prefill.label) setScopeLabel(prefill.label);
    if (prefill.question) setQuestion(prefill.question);
    onPrefillConsumed?.();
  }, [prefill, onPrefillConsumed]);

  const load = useCallback(async () => {
    try {
      const payload = await api.extractions();
      setRows(payload.papers || []);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, docs]);

  function toggleSource(id) {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  async function refreshAll() {
    setExtractBusy(true);
    setError("");
    try {
      const payload = await api.refreshExtractions();
      setRows(payload.papers || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setExtractBusy(false);
    }
  }

  async function runCompare() {
    const text = question.trim();
    if (!text || selectedIds.length < 2) return;
    setBusy(true);
    setError("");
    setSelectedEvidence(null);
    try {
      const payload = await api.query({
        query: text,
        document_ids: selectedIds,
        include_trace: true,
      });
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const badge = result ? verificationBadge(result) : null;
  const canAsk = selectedIds.length >= 2 && question.trim() && !busy;

  return (
    <>
      <div className="page-title">
        <div>
          <h2>Compare papers</h2>
          <p>
            Side-by-side structure fields, plus a multi-paper question over sources you select.
          </p>
        </div>
        <button className="ghost" type="button" disabled={extractBusy || !docs.length} onClick={refreshAll}>
          {extractBusy ? "Extracting…" : "Re-extract library"}
        </button>
      </div>
      {scopeLabel ? (
        <div className="theme-chip-row" style={{ marginBottom: 8 }}>
          <span className="theme-chip">
            Comparing: {scopeLabel}
            <button
              type="button"
              className="theme-chip-clear"
              onClick={() => setScopeLabel("")}
              aria-label="Clear project compare scope label"
            >
              ×
            </button>
          </span>
        </div>
      ) : null}
      <div className="spotlight spotlight-forest">
        <p className="spotlight-kicker">Side by side</p>
        <p>Objective, method, dataset, metric, result, limitation — extracted once per paper, kept even when confidence is low.</p>
      </div>
      {error && <StatusBanner tone="error">{error}</StatusBanner>}
      {!docs.length && (
        <EmptyState title="No sources yet">
          Add papers in Library or Find papers, then compare structure or ask across two or more sources.
        </EmptyState>
      )}
      {docs.length > 0 && (
        <>
          <div className="card compare-qa">
            <h3>Ask across papers</h3>
            <p className="status" style={{ marginTop: 0 }}>
              Select at least two library papers, then ask about similarities, differences, methods, findings, or limitations.
            </p>
            <div className="write-sources">
              <div className="write-sources-head">
                <strong>Papers to compare</strong>
                <span className="status" style={{ margin: 0 }}>
                  {selectedIds.length ? `${selectedIds.length} selected` : "Select 2+"}
                </span>
              </div>
              <div className="chips write-source-chips">
                {docs.map((doc) => {
                  const id = doc.document_id;
                  const active = selectedIds.includes(id);
                  return (
                    <button
                      key={id}
                      type="button"
                      className={`chip${active ? " active" : ""}`}
                      onClick={() => toggleSource(id)}
                    >
                      {doc.title || doc.name}
                    </button>
                  );
                })}
              </div>
              {selectedIds.length > 0 && selectedIds.length < 2 && (
                <p className="status">Select at least two papers to run a comparison question.</p>
              )}
            </div>
            <label>
              Comparison question
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. How do these papers differ in method and limitations?"
                rows={3}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (canAsk) runCompare();
                  }
                }}
              />
            </label>
            <div className="row" style={{ marginTop: 10 }}>
              <button
                className={`primary${busy ? " is-busy" : ""}`}
                type="button"
                disabled={!canAsk}
                onClick={runCompare}
              >
                {busy ? "Comparing…" : "Compare"}
              </button>
              {badge && <span className={`flag ${badge.tone}`}>{badge.label}</span>}
            </div>
          </div>

          {result && (
            <CompareResultView
              result={result}
              selectedDocIds={selectedIds}
              docs={docs}
              selected={selectedEvidence}
              onSelect={setSelectedEvidence}
            />
          )}

          <div className="card compare-wrap">
            <h3>Structure table</h3>
            <table className="compare-papers">
              <thead>
                <tr>
                  <th>Paper</th>
                  {STRUCTURE_COLUMNS.map((col) => (
                    <th key={col}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {docs.map((doc) => {
                  const record = rows.find((row) => row.document_id === doc.document_id);
                  return (
                    <tr key={doc.document_id}>
                      <td>
                        <strong>{doc.title || doc.name}</strong>
                        <div className="status" style={{ margin: 0 }}>
                          {(doc.authors || []).slice(0, 4).join(", ") || "Unknown author"}
                          {doc.year ? ` · ${doc.year}` : ""}
                        </div>
                      </td>
                      {STRUCTURE_COLUMNS.map((col) => {
                        const field = record?.fields?.[col];
                        const value = (field?.value || "").trim();
                        const missing = !value;
                        const low = Boolean(value && field?.low_confidence);
                        return (
                          <td key={col} className={missing || low ? "field-low" : ""}>
                            <span>{value || "—"}</span>
                            {low && (
                              <span className="flag warn low-flag" title="Extracted, but not clearly supported by this paper's passages">
                                Low
                              </span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="status" style={{ marginBottom: 0 }}>
              Empty cells mean that field was not found in the paper. A Low tag means a value was extracted but failed a support check — it is still shown.
            </p>
          </div>
        </>
      )}
    </>
  );
}

function claimPaperLabels(claim, evidenceByCite) {
  const cites = claim.supporting_citations || [];
  const names = [];
  const seen = new Set();
  for (const cite of cites) {
    const item = evidenceByCite.get(cite);
    if (!item) continue;
    const label = item.title || item.document_name;
    if (!label || seen.has(label)) continue;
    seen.add(label);
    names.push(label);
  }
  return names;
}

function CompareResultView({ result, selectedDocIds, docs, selected, onSelect }) {
  const evidenceByCite = useMemo(() => {
    const map = new Map();
    for (const item of result.evidence || []) map.set(item.citation_id, item);
    return map;
  }, [result.evidence]);

  const evidenceByPaper = useMemo(() => {
    const groups = new Map();
    for (const item of result.evidence || []) {
      const key = item.document_id || item.document_name || "unknown";
      if (!groups.has(key)) {
        groups.set(key, {
          key,
          title: item.title || item.document_name || "Source",
          items: [],
        });
      }
      groups.get(key).items.push(item);
    }
    return [...groups.values()];
  }, [result.evidence]);

  const claims = result.claims || [];
  const usedIds = new Set((result.evidence || []).map((item) => item.document_id).filter(Boolean));
  const unusedSelected = docs.filter(
    (d) => selectedDocIds.includes(d.document_id) && !usedIds.has(d.document_id),
  );

  function selectCite(id) {
    const item = evidenceByCite.get(id);
    if (item) onSelect(item);
  }

  return (
    <div className="compare-result">
      <div className="card">
        <h3>Comparison answer</h3>
        {result.unrelated_to_sources && (
          <p className="status" style={{ marginTop: 0 }}>
            This question is not covered by the selected papers.
          </p>
        )}
        {!result.unrelated_to_sources &&
          (result.status === "INSUFFICIENT_EVIDENCE" || looksLikeDumpedJson(result.answer)) && (
            <p className="status" style={{ marginTop: 0 }}>
              The selected papers do not contain enough support for a reliable comparison.
            </p>
          )}
        <div className="answer">
          {!looksLikeDumpedJson(result.answer) && (result.answer || "").trim() && (
            <AnswerText text={result.answer} onCite={selectCite} />
          )}
        </div>
        {(result.status === "INSUFFICIENT_EVIDENCE" || result.unrelated_to_sources) &&
          result.mismatch_detail && (
          <p className="status" style={{ marginTop: 8 }}>
            {result.mismatch_detail}
          </p>
        )}
      </div>

      <div className="card">
        <h3>Claims by paper</h3>
        {!claims.length ? (
          <p className="status">No claims were extracted for this answer.</p>
        ) : (
          <ul className="claim-list compare-claim-list">
            {claims.map((claim) => {
              const papers = claimPaperLabels(claim, evidenceByCite);
              const tone =
                claim.status === "SUPPORTED"
                  ? "good"
                  : claim.status === "PARTIALLY_SUPPORTED"
                    ? "warn"
                    : "bad";
              return (
                <li key={claim.claim_id}>
                  <div className="compare-claim-head">
                    <span className={`flag ${tone}`}>{claim.status.replace(/_/g, " ")}</span>
                    {papers.length > 0 ? (
                      <span className="status" style={{ margin: 0 }}>
                        {papers.join(" · ")}
                      </span>
                    ) : (
                      <span className="status" style={{ margin: 0 }}>
                        No supporting paper cited
                      </span>
                    )}
                  </div>
                  <p className="compare-claim-text">{claim.text}</p>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="card">
        <h3>Evidence by paper</h3>
        {!evidenceByPaper.length ? (
          <p className="status">No passages were retrieved.</p>
        ) : (
          evidenceByPaper.map((group) => (
            <div key={group.key} className="compare-evidence-group">
              <h4>{group.title}</h4>
              <div className="evidence-list">
                {group.items.map((item) => (
                  <EvidenceCard
                    key={item.chunk_id}
                    item={item}
                    active={selected?.chunk_id === item.chunk_id}
                    expanded={selected?.chunk_id === item.chunk_id}
                    onSelect={() => onSelect(item)}
                  />
                ))}
              </div>
            </div>
          ))
        )}
        {unusedSelected.length > 0 && (
          <p className="status">
            Not represented in retrieved evidence:{" "}
            {unusedSelected.map((d) => d.title || d.name).join(", ")}
          </p>
        )}
      </div>

      {(result.hallucination?.flags || []).length > 0 && (
        <div className="card">
          <h3>Verification notes</h3>
          <p className="error">{result.hallucination.flags.join(" · ")}</p>
        </div>
      )}
    </div>
  );
}

function verificationBadge(result) {
  if (result.outcome === "premise_contradicted" || result.status === "PREMISE_CONTRADICTED") {
    return { label: "Premise contradicted", tone: "warn" };
  }
  const isMultiSummary = Boolean(result.per_paper_summary && result.per_paper_summary.length > 0);
  if (
    !isMultiSummary &&
    (result.outcome === "conflicting" ||
      result.status === "CONFLICTING_EVIDENCE" ||
      (result.conflicts && result.conflicts.length > 0) ||
      (result.contradictions && result.contradictions.length > 0))
  ) {
    return { label: "Sources disagree", tone: "warn" };
  }
  if (
    result.outcome === "not_enough_evidence" ||
    result.unrelated_to_sources ||
    result.abstained ||
    result.status === "INSUFFICIENT_EVIDENCE"
  ) {
    return { label: "Not enough evidence", tone: "bad" };
  }
  if (isMultiSummary) {
    return { label: "Multi-paper summary", tone: "good" };
  }
  if ((result.uncovered_aspects || []).length > 0) {
    return { label: "Partially supported", tone: "warn" };
  }
  const claims = result.claims || [];
  if (!claims.length) {
    if (result.status === "ANSWERED" || result.status === "NO_RETRIEVAL_NEEDED" || result.outcome === "answered") {
      return { label: "Supported", tone: "good" };
    }
    return { label: "Not enough evidence", tone: "bad" };
  }
  const statuses = claims.map((claim) => String(claim.status || "").toUpperCase());
  if (statuses.some((status) => status === "UNSUPPORTED" || status === "CONTRADICTED")) {
    return { label: "Unsupported claims present", tone: "bad" };
  }
  if (statuses.some((status) => status === "PARTIALLY_SUPPORTED")) {
    return { label: "Partially supported", tone: "warn" };
  }
  return { label: "Supported", tone: "good" };
}

function exportText(result) {
  const cites = (result.evidence || [])
    .map((item) => `[${item.citation_id}] ${item.apa || item.title || item.document_name}`)
    .join("\n");
  return cites ? `${result.answer}\n\n${cites}` : result.answer || "";
}

function sourceHoverLabel(source, span) {
  if (!source) return "No source chunk tied to this claim";
  const heading = `[${source.citation_id}] ${source.title || source.document_name}`;
  const page = source.page != null ? ` · p. ${source.page}` : "";
  const body = span || source.text || "";
  return `${heading}${page}\n${body}`;
}

function looksLikeDumpedJson(text) {
  const trimmed = String(text || "").trim();
  return trimmed.startsWith("{") && trimmed.includes('"answer"');
}

function ClaimCard({
  text,
  label,
  tone,
  confidence,
  source,
  span,
  onSelectSource,
  onCite,
  highlights = [],
}) {
  const [open, setOpen] = useState(false);
  const hover = sourceHoverLabel(source, span);

  return (
    <article className={`claim claim-card ${tone || ""}`}>
      <div className="claim-head">
        <span className={`flag ${tone || ""}`}>{label}</span>
        {confidence != null && (
          <span className="claim-conf" title="Verification confidence">
            {pct(confidence)}
          </span>
        )}
      </div>
      <div className="claim-text">
        {onCite ? (
          <AnswerText text={text} onCite={onCite} highlights={highlights} />
        ) : (
          text
        )}
      </div>
      <button
        type="button"
        className="claim-source-toggle"
        title={hover}
        onClick={() => {
          setOpen((value) => !value);
          if (source) onSelectSource?.(source);
        }}
      >
        {source ? `Source [${source.citation_id}]` : "No source chunk"}
      </button>
      {open && (
        <div className="claim-source">
          {source ? (
            <>
              <div className="cite">
                [{source.citation_id}] {source.title || source.document_name}
                {source.page != null ? ` · p. ${source.page}` : ""}
                {source.chunk_id ? ` · ${source.chunk_id}` : ""}
              </div>
              <p>{span || source.text}</p>
            </>
          ) : (
            <p className="status">No retrieved chunk is tied to this claim.</p>
          )}
        </div>
      )}
    </article>
  );
}

function AnswerView({ result, docs, selected, onSelect }) {
  const conf = result.confidence || {};
  const lastGate = (result.gate_decisions || []).at(-1);
  const [copied, setCopied] = useState("");
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [speakStarting, setSpeakStarting] = useState(false);
  const [speakError, setSpeakError] = useState("");
  const speakHandleRef = useRef(null);
  const canSpeak = speechSynthesisSupported();
  const supportBadge = verificationBadge(result);
  const showEvidence = Boolean(result.evidence && result.evidence.length > 0);
  const evidenceByCite = useMemo(() => {
    const map = new Map();
    for (const item of result.evidence || []) map.set(item.citation_id, item);
    return map;
  }, [result.evidence]);
  const usedIds = new Set((result.evidence || []).map((item) => item.document_id));
  const usedNames = new Set((result.evidence || []).map((item) => item.document_name));
  const unused = docs.filter((d) => !usedIds.has(d.document_id) && !usedNames.has(d.name));
  const contradictions = result.contradictions || [];
  const answerText = (result.answer || "").trim();
  const mainAnswerText = useMemo(() => {
    return answerText;
  }, [answerText]);
  const canReadAloud = canSpeak && answerText && !looksLikeDumpedJson(result.answer);

  useEffect(() => {
    warmSpeechVoices();
  }, []);

  useEffect(
    () => () => {
      speakHandleRef.current?.stop?.();
      speakHandleRef.current = null;
      stopSpeaking();
    },
    [],
  );

  useEffect(() => {
    speakHandleRef.current?.stop?.();
    speakHandleRef.current = null;
    stopSpeaking();
    setSpeaking(false);
    setSpeakStarting(false);
    setSpeakError("");
  }, [result?.query_id]);

  function selectCite(id) {
    const item = evidenceByCite.get(id);
    if (item) {
      setEvidenceOpen(true);
      onSelect(item);
      setTimeout(() => {
        const el = document.getElementById(`evidence-cite-${item.citation_id}`);
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      }, 50);
    }
  }

  async function copyAnswer() {
    const ok = await copyText(exportText(result));
    setCopied(ok ? "copy" : "failed");
    setTimeout(() => setCopied(""), 1600);
  }

  function saveAnswer() {
    const blob = new Blob([exportText(result)], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "self-rag-answer.txt";
    link.click();
    URL.revokeObjectURL(url);
  }

  async function toggleSpeak(event) {
    event?.preventDefault?.();
    event?.stopPropagation?.();
    if (!canReadAloud) return;
    warmSpeechVoices();
    unlockAudio();

    if (speaking || speakStarting) {
      speakHandleRef.current?.stop?.();
      speakHandleRef.current = null;
      stopSpeaking();
      setSpeaking(false);
      setSpeakStarting(false);
      return;
    }

    setSpeakError("");
    setSpeakStarting(true);
    setSpeaking(false);

    const onStart = () => {
      setSpeakStarting(false);
      setSpeaking(true);
    };
    const onEnd = () => {
      speakHandleRef.current = null;
      setSpeakStarting(false);
      setSpeaking(false);
    };
    const onError = (err) => {
      speakHandleRef.current = null;
      setSpeakStarting(false);
      setSpeaking(false);
      setSpeakError(
        err?.message || "Could not play audio. Check system volume and try again.",
      );
    };

    // Prefer server TTS (macOS say) — browser SpeechSynthesis often stays silent.
    try {
      const handle = await speakViaServer(result.answer, {
        onStart,
        onEnd,
        // Don't surface error yet — we fall back to browser speech first.
      });
      speakHandleRef.current = handle;
      return;
    } catch {
      /* fall through to browser speech */
    }

    const handle = speakText(result.answer, { onStart, onEnd, onError });
    if (!handle) {
      setSpeakStarting(false);
      setSpeaking(false);
      setSpeakError("Could not play audio. Check system volume and try Chrome or Safari.");
      return;
    }
    speakHandleRef.current = handle;
  }

  return (
    <div className="chat-answer">
      {(result.status === "PREMISE_CONTRADICTED" || result.outcome === "premise_contradicted") && (
        <p className="status chat-status-note">
          A premise in your question conflicts with the findings reported in the indexed sources.
        </p>
      )}
      {((result.status === "CONFLICTING_EVIDENCE" || result.outcome === "conflicting") && !hasPerPaper) && (
        <p className="status chat-status-note">Sources disagree. Both views stay visible below.</p>
      )}
      {result.unrelated_to_sources && (
        <p className="status chat-status-note">This question is not covered by your files.</p>
      )}
      {!result.unrelated_to_sources &&
        (result.status === "INSUFFICIENT_EVIDENCE" || looksLikeDumpedJson(result.answer)) && (
        <p className="status chat-status-note">
          The indexed files do not contain enough support for a reliable answer.
        </p>
      )}
      <div className="answer">
        {!looksLikeDumpedJson(result.answer) && mainAnswerText && (
          <AnswerText text={mainAnswerText} onCite={selectCite} />
        )}
      </div>
      {(result.status === "INSUFFICIENT_EVIDENCE" || result.unrelated_to_sources) &&
        result.mismatch_detail && (
        <p className="status chat-status-note">{result.mismatch_detail}</p>
      )}
      <div className="export-bar">
        <button type="button" className="ghost" onClick={copyAnswer}>
          {copied === "copy" ? "Copied" : "Copy"}
        </button>
        <button type="button" className="ghost" onClick={saveAnswer}>
          Save
        </button>
        {canReadAloud && (
          <button
            type="button"
            className={`ghost ask-speak${speaking || speakStarting ? " is-live" : ""}`}
            onClick={toggleSpeak}
            aria-pressed={speaking || speakStarting}
            aria-label={
              speaking || speakStarting ? "Stop reading aloud" : "Read answer aloud"
            }
          >
            {speaking || speakStarting ? "Stop" : "Read aloud"}
          </button>
        )}
        {copied === "failed" && <span className="error">Copy failed</span>}
      </div>
      {speakStarting && !speaking ? (
        <p className="ask-voice-hint">Starting speech…</p>
      ) : null}
      {speakError ? <p className="error ask-speak-error">{speakError}</p> : null}

      {result.per_paper_summary && result.per_paper_summary.length > 0 && (
        <details className="chat-panel" open>
          <summary>
            Per-paper summaries
            <span className="ask-sources-hint">
              {result.per_paper_summary.length} paper{result.per_paper_summary.length === 1 ? "" : "s"}
            </span>
          </summary>
          <div className="per-paper-grid">
            {result.per_paper_summary.map((p, idx) => (
              <div key={p.document_id || idx} className="paper-summary-card">
                <h4>{p.title || `Paper ${idx + 1}`}</h4>
                <div className="paper-field">
                  <span className="field-label">Research Problem:</span> <span>{p.problem}</span>
                </div>
                <div className="paper-field">
                  <span className="field-label">Method:</span> <span>{p.method}</span>
                </div>
                <div className="paper-field">
                  <span className="field-label">Dataset & Setup:</span> <span>{p.dataset}</span>
                </div>
                <div className="paper-field">
                  <span className="field-label">Main Findings:</span> <span>{p.findings}</span>
                </div>
                <div className="paper-field">
                  <span className="field-label">Limitations:</span> <span>{p.limitations}</span>
                </div>
                {p.citation_id && (
                  <div style={{ marginTop: 8 }}>
                    <button
                      type="button"
                      className="cite-link"
                      onClick={() => selectCite(p.citation_id)}
                    >
                      View Source [{p.citation_id}]
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </details>
      )}

      {showEvidence && contradictions.length > 0 && (
        <details className="chat-panel" open>
          <summary>Conflicting evidence</summary>
          <p className="status" style={{ marginTop: 0 }}>
            Sources disagree. Both sides stay visible.
          </p>
          <ul className="claim-list">
            {contradictions.map((pair, index) => (
              <li key={`${pair.citation_a}-${pair.citation_b}-${index}`}>
                <div className="conflict-pair">
                  <ClaimCard
                    text={pair.claim_a}
                    label="contradicted"
                    tone="warn"
                    source={evidenceByCite.get(pair.citation_a)}
                    onSelectSource={(item) => item && onSelect(item)}
                    onCite={selectCite}
                  />
                  <ClaimCard
                    text={pair.claim_b}
                    label="contradicted"
                    tone="warn"
                    source={evidenceByCite.get(pair.citation_b)}
                    onSelectSource={(item) => item && onSelect(item)}
                    onCite={selectCite}
                  />
                  {pair.explanation && <p className="status conflict-note">{pair.explanation}</p>}
                </div>
              </li>
            ))}
          </ul>
        </details>
      )}

      {result.claims && result.claims.length > 0 && (
        <details className="chat-panel">
          <summary>
            Claim verification breakdown
            <span className="ask-sources-hint">
              {result.claims.filter((c) => String(c.status || "").toLowerCase() === "supported").length}/{result.claims.length} supported
            </span>
          </summary>
          <div className="claim-counts-bar" style={{ display: "flex", flexWrap: "wrap", gap: "12px", padding: "8px 14px", fontSize: "12px", borderBottom: "1px solid rgba(255,255,255,0.08)", opacity: 0.9 }}>
            <span><strong style={{ color: "var(--leaf)" }}>{result.claims.filter((c) => String(c.status || "").toLowerCase() === "supported").length}</strong> Supported</span>
            <span><strong style={{ color: "var(--orange)" }}>{result.claims.filter((c) => String(c.status || "").toLowerCase() === "partially_supported").length}</strong> Partially supported</span>
            <span><strong style={{ color: "var(--crimson)" }}>{result.claims.filter((c) => String(c.status || "").toLowerCase() === "unsupported").length}</strong> Unsupported</span>
            {result.claims.filter((c) => String(c.status || "").toLowerCase() === "contradicted").length > 0 && (
              <span><strong style={{ color: "var(--red)" }}>{result.claims.filter((c) => String(c.status || "").toLowerCase() === "contradicted").length}</strong> Contradicted</span>
            )}
          </div>
          <div className="claim-table-wrapper">
            <table className="claim-table">
              <thead>
                <tr>
                  <th>Claim</th>
                  <th>Status</th>
                  <th>Sources</th>
                  <th>Verifier</th>
                </tr>
              </thead>
              <tbody>
                {result.claims.map((claim, idx) => {
                  const s = String(claim.status || "").toLowerCase();
                  const tone = s === "supported" ? "good" : s === "partially_supported" ? "warn" : "bad";
                  const label = s.replace(/_/g, " ");
                  return (
                    <tr key={claim.claim_id || idx}>
                      <td className="claim-table-text">{claim.text}</td>
                      <td>
                        <span className={`flag ${tone}`}>{label}</span>
                      </td>
                      <td>
                        {(claim.supporting_citations || []).length > 0 ? (
                          (claim.supporting_citations || []).map((cid) => (
                            <button
                              key={cid}
                              type="button"
                              className="cite-link"
                              style={{ marginRight: 4 }}
                              onClick={() => selectCite(cid)}
                            >
                              [{cid}]
                            </button>
                          ))
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td className="muted" style={{ fontSize: "0.85em" }}>
                        {claim.verifier || "evidence verifier"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {showEvidence && (
        <details className="chat-panel" open={evidenceOpen} onToggle={(e) => setEvidenceOpen(e.currentTarget.open)}>
          <summary>
            Evidence
            <span className="ask-sources-hint">
              {(result.evidence || []).length
                ? `${result.evidence.length} passage${result.evidence.length === 1 ? "" : "s"}`
                : "None cited"}
            </span>
          </summary>
          {(result.evidence || []).length === 0 ? (
            <p className="status">No passages were cited.</p>
          ) : (
            <div className="evidence-list">
              {(result.evidence || []).map((item) => (
                <EvidenceCard
                  key={item.chunk_id}
                  item={item}
                  active={selected?.chunk_id === item.chunk_id}
                  expanded={selected?.chunk_id === item.chunk_id}
                  onSelect={() => onSelect(item)}
                />
              ))}
            </div>
          )}
        </details>
      )}

      <details className="chat-panel">
        <summary>Details</summary>
        <div className="meta">
          <span>Confidence <strong>{pct(conf.confidence)}</strong></span>
          <span>Evidence <strong>{pct(lastGate?.evidence_score ?? conf.evidence_coverage)}</strong></span>
          <span>Claims <strong>{conf.claims_verified ?? result.claims?.filter((c) => String(c.status || "").toLowerCase() === "supported").length ?? 0}/{conf.claims_total || result.claims?.length || 0} supported</strong></span>
          <span>Checked in <strong>{ms(result.metrics?.latency_ms)}</strong></span>
        </div>
        {conf.caveat && <p className="status">{conf.caveat}</p>}
        {result.hallucination?.flags?.length > 0 && (
          <p className="error">{result.hallucination.flags.join(" · ")}</p>
        )}
        <h3 style={{ marginTop: 16 }}>Pipeline Execution & Verification</h3>
        <p className="status" style={{ fontSize: "12px", margin: "4px 0 10px" }}>
          The evidence support score measures claim-level entailment across retrieved passages. It is an empirical grounding metric, not a truth probability.
        </p>
        <div className="meta" style={{ marginBottom: 12 }}>
          <span>Support Score <strong>{result.details?.evidence_support_score != null ? `${result.details.evidence_support_score}%` : pct(conf.confidence)}</strong></span>
          <span>Retrieval Rounds <strong>{result.metrics?.retrieval_attempts || 1}</strong></span>
          <span>Verification <strong>{result.details?.verification_method || "lexical_baseline"}</strong></span>
        </div>
        {result.details?.unresolved_claims?.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <h4 style={{ fontSize: "13px", color: "var(--apricot-soft)" }}>Unresolved evidence gaps</h4>
            <ul className="plain-list" style={{ fontSize: "12.5px" }}>
              {result.details.unresolved_claims.map((claim, idx) => (
                <li key={idx} style={{ opacity: 0.85 }}>• {claim}</li>
              ))}
            </ul>
          </div>
        )}
        <h3 style={{ marginTop: 16 }}>Not used</h3>
        {unused.length === 0 ? (
          <p className="status">Every indexed file contributed, or the library is empty.</p>
        ) : (
          <ul className="plain-list">
            {unused.map((doc) => (
              <li key={doc.document_id}>{doc.title || doc.name}</li>
            ))}
          </ul>
        )}
        <h3 style={{ marginTop: 16 }}>Check trace</h3>
        <ul className="trace compact">
          {(result.trace || []).map((event) => (
            <li key={`${event.step}-${event.stage}`}>
              <span className={`dot ${event.status === "warn" ? "warn" : event.status === "skip" ? "skip" : ""}`} />
              <div>
                {event.label}
                {event.detail && <div className="detail">{event.detail}</div>}
              </div>
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

function AnswerText({ text, onCite }) {
  if (!text) return null;
  return <MarkdownView content={text} onCite={onCite} />;
}

function EvidenceCard({ item, active, expanded, onSelect }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState("");
  useEffect(() => {
    if (active || expanded) setOpen(true);
  }, [active, expanded]);
  const showFull = open;
  const authors = (item.authors || []).join(", ") || "Unknown author";
  const year = item.year || "n.d.";
  const snippet = item.text.slice(0, 160);
  const displayTitle = item.title || item.document_name;
  const isGeneric = item.document_name && item.document_name !== item.title;

  async function copy(kind) {
    const ok = await copyText(kind === "apa" ? item.apa : item.bibtex);
    setCopied(ok ? kind : "failed");
    setTimeout(() => setCopied(""), 1600);
  }

  return (
    <div
      id={`evidence-cite-${item.citation_id}`}
      className={`evidence-card ${active ? "active" : ""}`}
    >
      <button type="button" className="evidence-card-main" onClick={onSelect}>
        <div className="cite">[{item.citation_id}] {displayTitle}</div>
        <div className="status" style={{ margin: "4px 0 0" }}>
          {authors} · {year}
          {item.page != null ? ` · p. ${item.page}` : ""}
          {item.doi ? ` · ${item.doi}` : ""}
          {isGeneric ? ` · file: ${item.document_name}` : ""}
        </div>
        <p className="cite-snippet">{showFull ? item.text : `${snippet}${item.text.length > 160 ? "…" : ""}`}</p>
      </button>
      <div className="row" style={{ marginTop: 8 }}>
        <button
          type="button"
          className="ghost"
          onClick={() => setOpen((value) => !value)}
        >
          {showFull ? "View less" : "View more"}
        </button>
        {showFull && (
          <>
            <button type="button" className="ghost" onClick={() => copy("apa")}>
              {copied === "apa" ? "Copied APA" : "Copy APA"}
            </button>
            <button type="button" className="ghost" onClick={() => copy("bibtex")}>
              {copied === "bibtex" ? "Copied BibTeX" : "Copy BibTeX"}
            </button>
          </>
        )}
        {copied === "failed" && <span className="error">Copy failed</span>}
      </div>
    </div>
  );
}


