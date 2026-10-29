"""Evidence-based paper workspace — manuscript auth, sections, versions, cites, AI."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.models.documents import Document
from app.services.auth.sessions import clear_sessions
from app.services.projects import store as project_store


def _prepare(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("SELFRAG_AUTH_REQUIRED", "true")
    monkeypatch.setenv("SELFRAG_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    clear_sessions()
    settings.auth_required = True
    settings.data_dir = tmp_path
    from app import config as config_mod

    config_mod.settings = settings
    root = tmp_path / "projects"
    monkeypatch.setattr(project_store, "_ROOT", root)
    monkeypatch.setattr(project_store, "_INVITES_ROOT", root / "_invites")
    monkeypatch.setattr(project_store, "_MANUSCRIPTS_ROOT", root / "manuscripts")


def _client():
    return TestClient(create_app())


def _signup(client: TestClient, email: str, name: str = "User"):
    resp = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "secret-pass", "name": name},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _invite_and_accept(owner, invitee, project_id: str, role: str, rid: str):
    created = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"researcher_id": rid, "role": role},
    )
    assert created.status_code == 200, created.text
    invite_id = created.json()["invite"]["invite_id"]
    accepted = invitee.post(
        "/api/projects/invites/accept",
        json={"invite_id": invite_id},
    )
    assert accepted.status_code == 200, accepted.text


def _fake_kb(docs: list[Document] | None = None):
    by_id = {d.document_id: d for d in (docs or [])}

    class Store:
        def get_document(self, document_id: str):
            return by_id.get(document_id)

        def list_documents(self):
            return list(by_id.values())

        def get_chunk(self, chunk_id: str):
            return None

        def all_chunks(self, document_ids=None):
            return []

    return SimpleNamespace(store=Store())


def test_manuscript_auth_section_crud_versions_comments(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    monkeypatch.setattr("app.api.projects.get_knowledge_base", lambda: _fake_kb())

    owner = _client()
    editor = _client()
    reviewer = _client()
    viewer = _client()
    _signup(owner, "ms2.owner@example.com", "Owner")
    editor_sess = _signup(editor, "ms2.editor@example.com", "Editor")
    reviewer_sess = _signup(reviewer, "ms2.reviewer@example.com", "Reviewer")
    viewer_sess = _signup(viewer, "ms2.viewer@example.com", "Viewer")
    project_id = owner.post("/api/projects/", json={"title": "Paper WS"}).json()["project_id"]
    _invite_and_accept(owner, editor, project_id, "editor", editor_sess["researcher_id"])
    _invite_and_accept(owner, reviewer, project_id, "reviewer", reviewer_sess["researcher_id"])
    _invite_and_accept(owner, viewer, project_id, "viewer", viewer_sess["researcher_id"])

    # Viewer cannot create
    assert (
        viewer.post(
            f"/api/projects/{project_id}/manuscript",
            json={"template": "ieee_research", "title": "Nope"},
        ).status_code
        == 403
    )

    created = owner.post(
        f"/api/projects/{project_id}/manuscript",
        json={
            "template": "ieee_research",
            "document_type": "ieee_research",
            "title": "Self-RAG Draft",
            "authors": ["A. Owner"],
        },
    )
    assert created.status_code == 200, created.text
    ms = created.json()
    assert ms["document_type"] == "ieee_research"
    assert ms["authors"] == ["A. Owner"]
    assert any(s["key"] == "keywords" for s in ms["sections"])
    assert any(s["key"] == "limitations" for s in ms["sections"])
    intro = next(s for s in ms["sections"] if s["key"] == "introduction")
    assert intro["section_id"]
    assert intro["status"] == "draft"

    # Editor saves body → version history
    saved = editor.patch(
        f"/api/projects/{project_id}/manuscript/sections/{intro['section_id']}",
        json={"body": "We study evidence-gated retrieval.", "save_version": True},
    )
    assert saved.status_code == 200, saved.text
    intro2 = next(s for s in saved.json()["sections"] if s["section_id"] == intro["section_id"])
    assert "evidence-gated" in intro2["body"]
    assert len(intro2["versions"]) >= 1
    assert intro2["versions"][0]["body"] == ""

    # Rename + status approved
    renamed = editor.patch(
        f"/api/projects/{project_id}/manuscript/sections/{intro['section_id']}",
        json={"title": "Introduction (revised)", "status": "in_review", "save_version": False},
    )
    assert renamed.status_code == 200
    intro3 = next(s for s in renamed.json()["sections"] if s["section_id"] == intro["section_id"])
    assert intro3["title"] == "Introduction (revised)"
    assert intro3["status"] == "in_review"

    # Add / reorder / delete custom section
    added = owner.post(
        f"/api/projects/{project_id}/manuscript/sections",
        json={"title": "Appendix", "after_section_id": intro["section_id"]},
    )
    assert added.status_code == 200, added.text
    appendix = next(s for s in added.json()["sections"] if s["title"] == "Appendix")
    ids = [s["section_id"] for s in added.json()["sections"]]
    # move appendix to end
    ids = [i for i in ids if i != appendix["section_id"]] + [appendix["section_id"]]
    ordered = owner.put(
        f"/api/projects/{project_id}/manuscript/sections/order",
        json={"section_ids": ids},
    )
    assert ordered.status_code == 200
    assert ordered.json()["sections"][-1]["section_id"] == appendix["section_id"]

    deleted = owner.delete(
        f"/api/projects/{project_id}/manuscript/sections/{appendix['section_id']}"
    )
    assert deleted.status_code == 200
    assert all(s["section_id"] != appendix["section_id"] for s in deleted.json()["sections"])

    # Reviewer can comment, cannot edit body
    comment = reviewer.post(
        f"/api/projects/{project_id}/manuscript/sections/{intro['section_id']}/comments",
        json={"text": "Please add a citation here."},
    )
    assert comment.status_code == 200, comment.text
    intro4 = next(s for s in comment.json()["sections"] if s["section_id"] == intro["section_id"])
    assert intro4["comments"][0]["text"].startswith("Please add")

    deny_edit = reviewer.patch(
        f"/api/projects/{project_id}/manuscript/sections/{intro['section_id']}",
        json={"body": "Hijack"},
    )
    assert deny_edit.status_code == 403

    deny_comment = viewer.post(
        f"/api/projects/{project_id}/manuscript/sections/{intro['section_id']}/comments",
        json={"text": "Nope"},
    )
    assert deny_comment.status_code == 403

    # Assign to owner — editor can no longer edit
    owner_user = owner.get("/api/auth/me").json()
    assigned = owner.post(
        f"/api/projects/{project_id}/manuscript/assign",
        json={"section_id": intro["section_id"], "assignee_user_id": owner_user["user_id"]},
    )
    assert assigned.status_code == 200
    deny_assigned = editor.patch(
        f"/api/projects/{project_id}/manuscript/sections/{intro['section_id']}",
        json={"body": "Still hijack"},
    )
    assert deny_assigned.status_code == 403


def test_manuscript_cite_and_ai_missing_evidence(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    doc = Document(
        document_id="doc-cite",
        name="paper.md",
        title="Cited Paper",
        authors=["Lewis"],
        year=2023,
        doi="10.1000/cite",
        source="upload",
    )
    monkeypatch.setattr("app.api.projects.get_knowledge_base", lambda: _fake_kb([doc]))

    owner = _client()
    _signup(owner, "cite.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Cite"}).json()["project_id"]
    link = owner.post(
        f"/api/projects/{project_id}/sources",
        json={"document_id": "doc-cite"},
    )
    assert link.status_code == 200, link.text

    ms = owner.post(
        f"/api/projects/{project_id}/manuscript",
        json={"template": "other", "document_type": "other", "title": "Other"},
    )
    assert ms.status_code == 200
    assert ms.json()["document_type"] == "other"

    cite = owner.post(
        f"/api/projects/{project_id}/manuscript/cite",
        json={"document_id": "doc-cite", "style": "apa"},
    )
    assert cite.status_code == 200, cite.text
    body = cite.json()
    assert "Lewis" in body["apa"]
    assert "2023" in body["apa"]
    assert body["citation"] == body["apa"]

    foreign = owner.post(
        f"/api/projects/{project_id}/manuscript/cite",
        json={"document_id": "not-linked"},
    )
    assert foreign.status_code in (400, 404)

    ai = owner.post(
        f"/api/projects/{project_id}/manuscript/ai",
        json={"action": "summarize_evidence", "section_id": ms.json()["sections"][0]["section_id"]},
    )
    assert ai.status_code == 200, ai.text
    assert ai.json()["missing_evidence"] is True
    assert "evidence" in " ".join(ai.json()["warnings"]).lower()
