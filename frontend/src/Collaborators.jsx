import { useEffect, useState } from "react";
import { api } from "./api";
import { Logo } from "./Auth";
import { StatusBanner } from "./ui";

function CollabHistory({ story, canDelete, busy, onDeleteInvite, onDeleteClose }) {
  const invites = [...(story?.invites || [])].sort((a, b) =>
    String(b.created_at || "").localeCompare(String(a.created_at || ""))
  );
  const closedBy = (story?.work_closed_by_name || "").trim();
  if (!invites.length && !closedBy) return null;
  return (
    <div className="collab-history">
      <h5>History</h5>
      <ul className="collab-pending-mini">
        {invites.map((inv) => (
          <li key={inv.invite_id}>
            <span>
              {inv.recipient_name || inv.recipient_researcher_id || "Collaborator"} ·{" "}
              {inviteHistoryLabel(inv.status)}
            </span>
            {canDelete ? (
              <button
                type="button"
                className="ghost"
                disabled={busy}
                onClick={() => onDeleteInvite(inv.invite_id)}
              >
                Delete
              </button>
            ) : null}
          </li>
        ))}
        {closedBy ? (
          <li>
            <span>{closedBy} finished and closed the work.</span>
            {canDelete ? (
              <button type="button" className="ghost" disabled={busy} onClick={onDeleteClose}>
                Delete
              </button>
            ) : null}
          </li>
        ) : null}
      </ul>
    </div>
  );
}

function inviteHistoryLabel(status) {
  if (status === "accepted") return "accepted";
  if (status === "declined") return "declined";
  if (status === "revoked") return "deleted";
  return "invited";
}

const PANEL_NAV = [
  ["home", "Home"],
  ["shared", "Shared"],
  ["invites", "Invites"],
  ["papers", "Papers"],
  ["write", "Write"],
];

/**
 * Collaborators hub — centered glass panel matching the 3D glassmorphism reference.
 */
export default function CollaboratorsPage({ session, onOpenPaper, onNavigate }) {
  const [section, setSection] = useState("home");
  const [mine, setMine] = useState([]);
  const [shared, setShared] = useState([]);
  const [pendingInvites, setPendingInvites] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [story, setStory] = useState(null);
  const [inviteRid, setInviteRid] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [copied, setCopied] = useState(false);

  const myUserId = session?.user_id || session?.id || "";
  const myRid = session?.researcher_id || "";
  const isOwner = Boolean(
    story && (story.my_role === "owner" || (myUserId && story.author_user_id === myUserId))
  );
  const mySharedRole =
    (story?.collaborators || []).find((c) => c.user_id === myUserId)?.role || "viewer";

  async function refresh() {
    const [myStories, sharedStories, invites] = await Promise.all([
      api.listMyStories(),
      api.listSharedStories(),
      api.listStoryInvitesPending(),
    ]);
    setMine(myStories.stories || []);
    setShared(sharedStories.stories || []);
    setPendingInvites(invites.invites || []);
  }

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
  }, []);

  function goNav(id) {
    if (id === "write") {
      onNavigate?.("write");
      return;
    }
    setSection(id);
  }

  async function selectPaper(id, nextSection = "papers") {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const payload = await api.getStory(id);
      setSelectedId(id);
      setStory(payload);
      setSection(nextSection);
    } catch (err) {
      setError(err.message);
      setStory(null);
      setSelectedId(null);
    } finally {
      setBusy(false);
    }
  }

  async function copyRid() {
    if (!myRid) return;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(myRid);
        setCopied(true);
        setMessage(`Copied your Researcher ID: ${myRid}`);
        setTimeout(() => setCopied(false), 2000);
        return;
      }
    } catch {
      /* fall through */
    }
    window.prompt("Copy your Researcher ID:", myRid);
  }

  async function sendInvite() {
    if (!selectedId || !isOwner) return;
    const rid = inviteRid.trim();
    if (!rid) {
      setError("Enter a Researcher ID to invite.");
      return;
    }
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await api.inviteStoryCollaborator(selectedId, {
        researcher_id: rid,
        role: "editor",
      });
      const updated = await api.getStory(selectedId);
      setStory(updated);
      setInviteRid("");
      setMessage("Invite sent.");
      await refresh();
    } catch (err) {
      setError(err.message || "Could not send invite.");
    } finally {
      setBusy(false);
    }
  }

  async function deleteInviteHistory(inviteId) {
    if (!selectedId) return;
    setBusy(true);
    setError("");
    try {
      const updated = await api.deleteStoryInviteHistory(selectedId, inviteId);
      setStory(updated);
      setMessage("History deleted.");
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function deleteCloseHistory() {
    if (!selectedId) return;
    setBusy(true);
    setError("");
    try {
      const updated = await api.deleteStoryCloseHistory(selectedId);
      setStory(updated);
      setMessage("History deleted.");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function removeCollab(userId) {
    if (!selectedId) return;
    const leaving = userId === myUserId;
    if (
      !window.confirm(
        leaving
          ? "Leave this shared workspace?"
          : "Remove this collaborator from the paper?"
      )
    ) {
      return;
    }
    setBusy(true);
    setError("");
    try {
      await api.removeStoryCollaborator(selectedId, userId);
      if (leaving) {
        setSelectedId(null);
        setStory(null);
        setMessage("You left the shared workspace.");
      } else {
        const updated = await api.getStory(selectedId);
        setStory(updated);
        setMessage("Collaborator removed.");
      }
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function closeWork() {
    if (!selectedId) return;
    setBusy(true);
    setError("");
    try {
      const updated = await api.setStoryWorkStatus(selectedId, "completed");
      setStory(updated);
      setMessage("Work closed.");
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function acceptPending(inviteId) {
    setBusy(true);
    setError("");
    try {
      const accepted = await api.acceptStoryInvite(inviteId);
      setMessage("Joined shared workspace.");
      await refresh();
      await selectPaper(accepted.story_id, "shared");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function declinePending(inviteId) {
    setBusy(true);
    setError("");
    try {
      await api.declineStoryInvite(inviteId);
      setMessage("Invite declined.");
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const showWorkspace = section !== "home";

  return (
    <div className="collab-stage">
      <div className="collab-depth" aria-hidden="true">
        <span className="collab-orb collab-orb-back-a" />
        <span className="collab-orb collab-orb-back-b" />
        <span className="collab-ring collab-ring-back" />
        <span className="collab-orb collab-orb-mid" />
        <span className="collab-orb collab-orb-fore-a" />
        <span className="collab-ring collab-ring-fore" />
        <span className="collab-orb collab-orb-fore-b" />
      </div>

      <div className="collab-panel">
        <header className="collab-panel-bar">
          <Logo size="nav" />
          <nav className="collab-panel-nav" aria-label="Collaborators">
            {PANEL_NAV.map(([id, label], index) => (
              <span key={id} className="collab-panel-nav-item">
                {index > 0 ? <span className="collab-panel-nav-sep" aria-hidden="true" /> : null}
                <button
                  type="button"
                  className={section === id || (id === "home" && section === "home") ? "is-active" : ""}
                  onClick={() => goNav(id)}
                >
                  {label}
                </button>
              </span>
            ))}
          </nav>
        </header>

        {(error || message) && (
          <div className="collab-panel-alerts">
            {error ? <StatusBanner tone="error">{error}</StatusBanner> : null}
            {message ? <StatusBanner tone="success">{message}</StatusBanner> : null}
          </div>
        )}

        <div className={`collab-panel-body${showWorkspace ? " has-workspace" : " is-home"}`}>
          {showWorkspace ? (
            <div className="collab-workspace">
              {section === "shared" ? (
                <div className="collab-section">
                  <h3 className="collab-section-title">Shared with me</h3>
                  {!shared.length ? (
                    <p className="collab-quiet">No shared workspaces yet. Accept an invite to join.</p>
                  ) : (
                    <ul className="collab-link-list">
                      {shared.map((item) => (
                        <li key={item.story_id}>
                          <button type="button" onClick={() => selectPaper(item.story_id, "shared")}>
                            <strong>{item.title}</strong>
                            <span>
                              {item.my_role || "viewer"}
                              {item.author_name ? ` · ${item.author_name}` : ""}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                  {story && !isOwner ? (
                    <div className="collab-manage-pane">
                      <div className="collab-manage-head">
                        <div>
                          <p className="collab-mini-label">Shared paper</p>
                          <h4>{story.title}</h4>
                        </div>
                        {story.work_status !== "completed" && mySharedRole === "editor" ? (
                          <button type="button" className="primary" disabled={busy} onClick={closeWork}>
                            Close work
                          </button>
                        ) : null}
                      </div>
                      {story.work_status === "completed" ? (
                        <p className="collab-complete">
                          {story.work_closed_by_name
                            ? `${story.work_closed_by_name} finished and closed this paper.`
                            : "This paper is closed."}
                        </p>
                      ) : null}
                      <CollabHistory
                        story={story}
                        canDelete={Boolean(story.my_role)}
                        busy={busy}
                        onDeleteInvite={deleteInviteHistory}
                        onDeleteClose={deleteCloseHistory}
                      />
                    </div>
                  ) : null}
                </div>
              ) : null}

              {section === "invites" ? (
                <div className="collab-section">
                  <h3 className="collab-section-title">Pending invites</h3>
                  {!pendingInvites.length ? (
                    <p className="collab-quiet">You’re all caught up — no pending invites.</p>
                  ) : (
                    <ul className="collab-invite-list">
                      {pendingInvites.map((inv) => (
                        <li key={inv.invite_id}>
                          <div>
                            <strong>{inv.story_title || "Untitled paper"}</strong>
                            <span>
                              {inv.invited_by_name || "Someone"} · {inv.role}
                            </span>
                          </div>
                          <div className="collab-inline-actions">
                            <button
                              type="button"
                              className="primary"
                              disabled={busy}
                              onClick={() => acceptPending(inv.invite_id)}
                            >
                              Accept
                            </button>
                            <button
                              type="button"
                              className="ghost"
                              disabled={busy}
                              onClick={() => declinePending(inv.invite_id)}
                            >
                              Decline
                            </button>
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ) : null}

              {section === "papers" ? (
                <div className="collab-section collab-papers-grid">
                  <div>
                    <h3 className="collab-section-title">My papers</h3>
                    {!mine.length ? (
                      <p className="collab-quiet">Create a paper in Write, then invite teammates here.</p>
                    ) : (
                      <ul className="collab-link-list">
                        {mine.map((item) => (
                          <li key={item.story_id}>
                            <button
                              type="button"
                              className={selectedId === item.story_id ? "is-active" : ""}
                              onClick={() => selectPaper(item.story_id)}
                            >
                              <strong>{item.title}</strong>
                              <span>
                                {item.work_status === "completed" ? "completed" : item.status}
                              </span>
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  <div className="collab-manage-pane">
                    {!story ? (
                      <p className="collab-quiet">Select a paper to manage people.</p>
                    ) : (
                      <>
                        <div className="collab-manage-head">
                          <div>
                            <p className="collab-mini-label">
                              {isOwner ? "Owner workspace" : `Joined as ${mySharedRole}`}
                            </p>
                            <h4>{story.title}</h4>
                          </div>
                          <div className="row">
                            {story.work_status !== "completed" && (isOwner || mySharedRole === "editor") ? (
                              <button type="button" className="primary" disabled={busy} onClick={closeWork}>
                                Close work
                              </button>
                            ) : null}
                            <button
                              type="button"
                              className="ghost"
                              disabled={busy}
                              onClick={() => onOpenPaper?.(story.story_id)}
                            >
                              Open in Write
                            </button>
                          </div>
                        </div>

                        {story.work_status === "completed" ? (
                          <p className="collab-complete">
                            {story.work_closed_by_name
                              ? `${story.work_closed_by_name} finished and closed this paper.`
                              : "This paper is closed."}
                          </p>
                        ) : null}

                        <ul className="collab-people">
                          <li>
                            <span className="collab-avatar" aria-hidden="true">
                              {(isOwner ? "Y" : (story.author_name || "O").charAt(0)).toUpperCase()}
                            </span>
                            <div>
                              <strong>{isOwner ? "You" : story.author_name || "Owner"}</strong>
                              <em>owner</em>
                            </div>
                          </li>
                          {(story.collaborators || [])
                            .filter((c) => c.user_id !== myUserId)
                            .map((c) => (
                              <li key={c.user_id}>
                                <span className="collab-avatar" aria-hidden="true">
                                  {(c.name || c.researcher_id || "?").charAt(0).toUpperCase()}
                                </span>
                                <div>
                                  <strong>{c.name || c.researcher_id || "Collaborator"}</strong>
                                  <em>editing</em>
                                </div>
                                {isOwner ? (
                                  <button
                                    type="button"
                                    className="ghost"
                                    disabled={busy}
                                    onClick={() => removeCollab(c.user_id)}
                                  >
                                    Delete
                                  </button>
                                ) : null}
                              </li>
                            ))}
                        </ul>

                        {isOwner && story.work_status !== "completed" ? (
                          <div className="collab-invite-row-form">
                            <input
                              value={inviteRid}
                              onChange={(e) => setInviteRid(e.target.value)}
                              placeholder="Name or user ID"
                              disabled={busy}
                              aria-label="Collaborator name or user ID"
                            />
                            <button type="button" className="primary" disabled={busy} onClick={sendInvite}>
                              Invite
                            </button>
                          </div>
                        ) : null}

                        <CollabHistory
                          story={story}
                          canDelete={Boolean(story.my_role)}
                          busy={busy}
                          onDeleteInvite={deleteInviteHistory}
                          onDeleteClose={deleteCloseHistory}
                        />

                        <p className="collab-id-footnote">
                          Your ID: <code>{myRid || "—"}</code>
                        </p>
                      </>
                    )}
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            <div className="collab-hero-anchor">
              <p className="collab-hero-sub">Shared workspace</p>
              <h2 className="collab-hero-title">Collaborators</h2>
              <p className="collab-hero-copy">
                Type a name or Researcher ID to invite someone. They accept, edit, and close the paper when they are finished.
              </p>
              {myRid ? (
                <p className="collab-hero-rid">
                  <span>Your ID</span>
                  <code>{myRid}</code>
                </p>
              ) : null}
              <button
                type="button"
                className="collab-cta-outline"
                disabled={!myRid || busy}
                onClick={copyRid}
              >
                {copied ? "Copied" : "Copy your ID"}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
