const API_ROOT = import.meta.env.VITE_API_BASE_URL ? import.meta.env.VITE_API_BASE_URL.replace(/\/$/, "") : "";
const BASE = `${API_ROOT}/api`;

let authToken = typeof localStorage !== "undefined" ? localStorage.getItem("selfrag_token") || "" : "";

export function setAuthToken(token) {
  authToken = token || "";
  if (typeof localStorage !== "undefined") {
    if (token) {
      localStorage.setItem("selfrag_token", token);
    } else {
      localStorage.removeItem("selfrag_token");
    }
  }
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...options,
      headers,
      credentials: "include",
    });
  } catch (err) {
    const raw = String(err?.message || "");
    if (raw === "Failed to fetch" || err?.name === "TypeError") {
      throw new Error(
        "Could not reach the server. The backend may be waking up on Render (cold start takes ~30-50s) or rebuilding. Please wait a moment and refresh."
      );
    }
    throw err;
  }
  if (!response.ok) {
    if (
      response.status === 401 &&
      !path.startsWith("/auth") &&
      !path.startsWith("/stories/public")
    ) {
      window.dispatchEvent(new Event("selfrag-auth-lost"));
    }
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail?.message || body.detail || JSON.stringify(body);
      if (body.detail?.warnings && Array.isArray(body.detail.warnings) && body.detail.warnings.length) {
        detail = `${typeof detail === "string" ? detail : "Upload failed"}: ${body.detail.warnings.join("; ")}`;
      }
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (response.status === 204) return null;
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

export const api = {
  me: () => request("/auth/me"),
  lookupResearcher: (researcherId) =>
    request(`/auth/researchers/${encodeURIComponent(researcherId)}`),
  signup: async (payload) => {
    const res = await request("/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res?.token) setAuthToken(res.token);
    return res;
  },
  login: async (payload) => {
    const res = await request("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res?.token) setAuthToken(res.token);
    return res;
  },
  googleLogin: async (payload) => {
    const res = await request("/auth/google", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res?.token) setAuthToken(res.token);
    return res;
  },
  logout: async () => {
    setAuthToken("");
    return request("/auth/logout", { method: "POST" });
  },
  health: () => request("/health"),
  config: () => request("/config"),
  /**
   * Server-side TTS (macOS say). Returns a Blob of audio/wav or audio/aiff.
   */
  tts: async (text) => {
    const response = await fetch(`${BASE}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ text: String(text || "") }),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail?.message || body.detail || detail;
      } catch {
        /* ignore */
      }
      throw new Error(typeof detail === "string" ? detail : "Speech synthesis failed");
    }
    return response.blob();
  },
  documents: () => request("/documents"),
  clusters: () => request("/documents/clusters"),
  extractions: () => request("/documents/extractions"),
  refreshExtractions: () => request("/documents/extractions/refresh", { method: "POST" }),
  upload: (files) => {
    const data = new FormData();
    for (const file of files) data.append("files", file);
    return request("/documents/upload", { method: "POST", body: data });
  },
  deleteDocument: (id) => request(`/documents/${id}`, { method: "DELETE" }),
  searchPapers: (q, opts = {}) => {
    const limit = opts.limit ?? 10;
    const filter = opts.filter ?? "all";
    return request(
      `/papers/search?q=${encodeURIComponent(q)}&limit=${limit}&filter=${encodeURIComponent(filter)}`
    );
  },
  importPaper: (payload) =>
    request("/papers/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  query: (payload) =>
    request("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  recent: (limit = 12) => request(`/query/recent?limit=${limit}`),
  historyItem: (id) => request(`/query/history/${id}`),
  events: (params = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value != null && value !== "") qs.set(key, String(value));
    });
    const suffix = qs.toString() ? `?${qs}` : "";
    return request(`/events${suffix}`);
  },
  workspaceRecents: (limit = 24) => request(`/workspace/recents?limit=${limit}`),
  saveAskChat: (payload) =>
    request("/workspace/ask-chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  askChatItem: (id) => request(`/workspace/ask-chats/${encodeURIComponent(id)}`),
  savePaperSearch: (payload) =>
    request("/workspace/paper-searches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  paperSearchItem: (id) => request(`/workspace/paper-searches/${id}`),
  deleteRecentAsk: (id) => request(`/workspace/recents/ask/${encodeURIComponent(id)}`, { method: "DELETE" }),
  deleteRecentPapers: (id) =>
    request(`/workspace/recents/papers/${encodeURIComponent(id)}`, { method: "DELETE" }),
  listProjects: (opts = {}) => {
    const qs = new URLSearchParams();
    if (opts.limit != null) qs.set("limit", String(opts.limit));
    if (opts.include_archived != null) qs.set("include_archived", String(opts.include_archived));
    const suffix = qs.toString() ? `?${qs}` : "";
    // Trailing slash required: bare /projects 307-redirects to the API origin and drops the session cookie.
    return request(`/projects/${suffix}`);
  },
  createProject: (payload) =>
    request("/projects/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  getProject: (id) => request(`/projects/${encodeURIComponent(id)}`),
  updateProject: (id, payload) =>
    request(`/projects/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  archiveProject: (id) =>
    request(`/projects/${encodeURIComponent(id)}/archive`, { method: "POST" }),
  deleteProject: (id) =>
    request(`/projects/${encodeURIComponent(id)}`, { method: "DELETE" }),
  listPendingInvites: () => request("/projects/invites/pending"),
  acceptProjectInvite: (payload) =>
    request("/projects/invites/accept", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  rejectProjectInvite: (payload) =>
    request("/projects/invites/reject", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  createProjectInvite: (projectId, payload) =>
    request(`/projects/${encodeURIComponent(projectId)}/invites`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  revokeProjectInvite: (projectId, inviteId) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/invites/${encodeURIComponent(inviteId)}`,
      { method: "DELETE" }
    ),
  updateProjectMemberRole: (projectId, memberUserId, role) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/members/${encodeURIComponent(memberUserId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      }
    ),
  removeProjectMember: (projectId, memberUserId) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/members/${encodeURIComponent(memberUserId)}`,
      { method: "DELETE" }
    ),
  addProjectNote: (projectId, text, extras = {}) =>
    request(`/projects/${encodeURIComponent(projectId)}/notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, ...extras }),
    }),
  updateProjectNote: (projectId, noteId, payload) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/notes/${encodeURIComponent(noteId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),
  deleteProjectNote: (projectId, noteId) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/notes/${encodeURIComponent(noteId)}`,
      { method: "DELETE" }
    ),
  addProjectReview: (projectId, text, target = "project") =>
    request(`/projects/${encodeURIComponent(projectId)}/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, target }),
    }),
  deleteProjectReview: (projectId, reviewId) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/reviews/${encodeURIComponent(reviewId)}`,
      { method: "DELETE" }
    ),
  listProjectSources: (projectId) =>
    request(`/projects/${encodeURIComponent(projectId)}/sources`),
  linkProjectSource: (projectId, documentId) =>
    request(`/projects/${encodeURIComponent(projectId)}/sources`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: documentId }),
    }),
  importProjectSource: (projectId, payload) =>
    request(`/projects/${encodeURIComponent(projectId)}/sources/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  unlinkProjectSource: (projectId, documentId) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/sources/${encodeURIComponent(documentId)}`,
      { method: "DELETE" }
    ),
  listProjectEvidence: (projectId) =>
    request(`/projects/${encodeURIComponent(projectId)}/evidence`),
  addProjectEvidence: (projectId, payload) =>
    request(`/projects/${encodeURIComponent(projectId)}/evidence`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  removeProjectEvidence: (projectId, evidenceId) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/evidence/${encodeURIComponent(evidenceId)}`,
      { method: "DELETE" }
    ),
  getProjectManuscript: (projectId) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript`),
  createProjectManuscript: (projectId, payload = {}) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  updateProjectManuscript: (projectId, payload) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  updateManuscriptSections: (projectId, sections) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript/sections`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sections),
    }),
  addManuscriptSection: (projectId, payload) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript/sections`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  patchManuscriptSection: (projectId, sectionId, payload) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/manuscript/sections/${encodeURIComponent(sectionId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),
  reorderManuscriptSections: (projectId, sectionIds) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript/sections/order`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ section_ids: sectionIds }),
    }),
  deleteManuscriptSection: (projectId, sectionId) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/manuscript/sections/${encodeURIComponent(sectionId)}`,
      { method: "DELETE" }
    ),
  assignManuscriptSection: (projectId, key, assigneeUserId, sectionId = "") =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript/assign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        key: key || "",
        section_id: sectionId || "",
        assignee_user_id: assigneeUserId || "",
      }),
    }),
  addManuscriptSectionComment: (projectId, sectionId, text) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/manuscript/sections/${encodeURIComponent(sectionId)}/comments`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }
    ),
  citeManuscriptSource: (projectId, documentId, style = "apa") =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript/cite`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: documentId, style }),
    }),
  manuscriptAi: (projectId, payload) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript/ai`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  getManuscriptReview: (projectId) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript/review`),
  runManuscriptReview: (projectId, payload = {}) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  updateManuscriptReviewIssue: (projectId, issueId, payload) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/manuscript/review/${encodeURIComponent(issueId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),
  getManuscriptChecklist: (projectId) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript/checklist`),
  refreshManuscriptChecklist: (projectId) =>
    request(`/projects/${encodeURIComponent(projectId)}/manuscript/checklist/refresh`, {
      method: "POST",
    }),
  updateManuscriptChecklistItem: (projectId, itemKey, payload) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/manuscript/checklist/${encodeURIComponent(itemKey)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),
  createProjectTask: (projectId, payload) =>
    request(`/projects/${encodeURIComponent(projectId)}/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  updateProjectTask: (projectId, taskId, payload) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/tasks/${encodeURIComponent(taskId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),
  addProjectTaskComment: (projectId, taskId, text) =>
    request(
      `/projects/${encodeURIComponent(projectId)}/tasks/${encodeURIComponent(taskId)}/comments`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }
    ),
  listMyStories: () => request("/stories/mine"),
  listSharedStories: () => request("/stories/shared"),
  listPublicStories: (limit = 40) => request(`/stories/public?limit=${limit}`),
  getPublicStory: (slug) => request(`/stories/public/${encodeURIComponent(slug)}`),
  getStory: (id) => request(`/stories/${encodeURIComponent(id)}`),
  createStory: (payload) =>
    request("/stories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  updateStory: (id, payload) =>
    request(`/stories/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  setStoryWorkStatus: (id, status) =>
    request(`/stories/${encodeURIComponent(id)}/work-status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }),
  publishStory: (id) =>
    request(`/stories/${encodeURIComponent(id)}/publish`, { method: "POST" }),
  unpublishStory: (id) =>
    request(`/stories/${encodeURIComponent(id)}/unpublish`, { method: "POST" }),
  deleteStory: (id) =>
    request(`/stories/${encodeURIComponent(id)}`, { method: "DELETE" }),
  listStoryInvitesPending: () => request("/stories/invites/pending"),
  acceptStoryInvite: (inviteId) =>
    request(`/stories/invites/${encodeURIComponent(inviteId)}/accept`, { method: "POST" }),
  declineStoryInvite: (inviteId) =>
    request(`/stories/invites/${encodeURIComponent(inviteId)}/decline`, { method: "POST" }),
  inviteStoryCollaborator: (storyId, payload) =>
    request(`/stories/${encodeURIComponent(storyId)}/invites`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  revokeStoryInvite: (storyId, inviteId) =>
    request(
      `/stories/${encodeURIComponent(storyId)}/invites/${encodeURIComponent(inviteId)}`,
      { method: "DELETE" }
    ),
  deleteStoryInviteHistory: (storyId, inviteId) =>
    request(
      `/stories/${encodeURIComponent(storyId)}/history/invites/${encodeURIComponent(inviteId)}`,
      { method: "DELETE" }
    ),
  deleteStoryCloseHistory: (storyId) =>
    request(`/stories/${encodeURIComponent(storyId)}/history/close`, { method: "DELETE" }),
  updateStoryCollaboratorRole: (storyId, userId, role) =>
    request(
      `/stories/${encodeURIComponent(storyId)}/collaborators/${encodeURIComponent(userId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      }
    ),
  removeStoryCollaborator: (storyId, userId) =>
    request(
      `/stories/${encodeURIComponent(storyId)}/collaborators/${encodeURIComponent(userId)}`,
      { method: "DELETE" }
    ),
  downloadStoryDocx: async (id) => {
    const response = await fetch(`${BASE}/stories/${encodeURIComponent(id)}/docx`, {
      credentials: "include",
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch {
        /* ignore */
      }
      throw new Error(typeof detail === "string" ? detail : "Download failed");
    }
    const blob = await response.blob();
    const dispo = response.headers.get("Content-Disposition") || "";
    const match = /filename="?([^";]+)"?/i.exec(dispo);
    return { blob, filename: match?.[1] || "paper.docx" };
  },
  downloadPublicStoryDocx: async (slug) => {
    const response = await fetch(`${BASE}/stories/public/${encodeURIComponent(slug)}/docx`, {
      credentials: "include",
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch {
        /* ignore */
      }
      throw new Error(typeof detail === "string" ? detail : "Download failed");
    }
    const blob = await response.blob();
    const dispo = response.headers.get("Content-Disposition") || "";
    const match = /filename="?([^";]+)"?/i.exec(dispo);
    return { blob, filename: match?.[1] || "paper.docx" };
  },
};

export function pct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${Math.round(Number(value) * 100)}%`;
}

export function ms(value) {
  if (value == null) return "—";
  return `${Math.round(value)} ms`;
}
