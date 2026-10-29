"""Phase 3–6: typed notes, evidence, manuscript, tasks + RBAC."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.models.documents import Chunk, ChunkMetadata, Document
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


def _doc(document_id: str = "doc-ws") -> Document:
    return Document(
        document_id=document_id,
        name="paper.md",
        title="Workspace Paper",
        authors=["A", "B"],
        year=2024,
        doi="10.1000/ws",
        source="upload",
    )


def _chunk(document_id: str, text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            document_name="paper.md",
            page=2,
            chunk_index=0,
        ),
    )


def _fake_kb(docs: list[Document], chunks: list[Chunk] | None = None):
    by_id = {d.document_id: d for d in docs}
    chunk_map = {c.chunk_id: c for c in (chunks or [])}

    class Store:
        def get_document(self, document_id: str):
            return by_id.get(document_id)

        def list_documents(self):
            return list(by_id.values())

        def get_chunk(self, chunk_id: str):
            return chunk_map.get(chunk_id)

        def all_chunks(self, document_ids=None):
            if document_ids is None:
                return list(chunk_map.values())
            wanted = set(document_ids)
            return [c for c in chunk_map.values() if c.metadata.document_id in wanted]

    return SimpleNamespace(store=Store())


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


def test_typed_notes_and_own_delete(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    doc = _doc()
    monkeypatch.setattr("app.api.projects.get_knowledge_base", lambda: _fake_kb([doc]))

    owner = _client()
    editor = _client()
    owner_sess = _signup(owner, "ws.owner@example.com", "Owner")
    editor_sess = _signup(editor, "ws.editor@example.com", "Editor")
    project_id = owner.post("/api/projects/", json={"title": "Notes"}).json()["project_id"]
    owner.post(f"/api/projects/{project_id}/sources", json={"document_id": doc.document_id})
    _invite_and_accept(
        owner, editor, project_id, "editor", editor_sess["researcher_id"]
    )

    added = editor.post(
        f"/api/projects/{project_id}/notes",
        json={
            "text": "Method uses RRF.",
            "note_type": "methodology",
            "document_id": doc.document_id,
        },
    )
    assert added.status_code == 200, added.text
    note = added.json()["notes"][0]
    assert note["note_type"] == "methodology"
    assert note["document_id"] == doc.document_id

    owner_note = owner.post(
        f"/api/projects/{project_id}/notes",
        json={"text": "Owner note", "note_type": "finding"},
    ).json()["notes"][0]

    # Editor cannot delete owner's note
    deny = editor.delete(f"/api/projects/{project_id}/notes/{owner_note['note_id']}")
    assert deny.status_code == 403

    # Editor can delete own note
    ok = editor.delete(f"/api/projects/{project_id}/notes/{note['note_id']}")
    assert ok.status_code == 200
    assert all(n["note_id"] != note["note_id"] for n in ok.json()["notes"])

    # Owner can delete any
    wipe = owner.delete(f"/api/projects/{project_id}/notes/{owner_note['note_id']}")
    assert wipe.status_code == 200
    assert wipe.json()["notes"] == []
    _ = owner_sess


def test_evidence_quote_must_match_chunk(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    doc = _doc()
    chunk = _chunk(doc.document_id, "Hybrid retrieval combines dense vectors with BM25.", "c-ev")
    monkeypatch.setattr(
        "app.api.projects.get_knowledge_base", lambda: _fake_kb([doc], [chunk])
    )

    owner = _client()
    viewer = _client()
    _signup(owner, "ev.owner@example.com", "Owner")
    viewer_sess = _signup(viewer, "ev.viewer@example.com", "Viewer")
    project_id = owner.post("/api/projects/", json={"title": "Evidence"}).json()["project_id"]
    owner.post(f"/api/projects/{project_id}/sources", json={"document_id": doc.document_id})
    _invite_and_accept(
        owner, viewer, project_id, "viewer", viewer_sess["researcher_id"]
    )

    bad = owner.post(
        f"/api/projects/{project_id}/evidence",
        json={"document_id": doc.document_id, "quote": "This text is invented."},
    )
    assert bad.status_code == 400

    good = owner.post(
        f"/api/projects/{project_id}/evidence",
        json={
            "document_id": doc.document_id,
            "quote": "dense vectors with BM25",
            "kind": "user_marked",
        },
    )
    assert good.status_code == 200, good.text
    ev = good.json()["evidence"][0]
    assert "BM25" in ev["quote"]
    assert ev["chunk_id"] == "c-ev"
    assert ev["page"] == 2

    deny = viewer.post(
        f"/api/projects/{project_id}/evidence",
        json={"document_id": doc.document_id, "quote": "dense vectors with BM25"},
    )
    assert deny.status_code == 403

    listed = owner.get(f"/api/projects/{project_id}/evidence")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    removed = owner.delete(f"/api/projects/{project_id}/evidence/{ev['evidence_id']}")
    assert removed.status_code == 200
    assert removed.json()["evidence"] == []


def test_manuscript_create_edit_assign(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    monkeypatch.setattr("app.api.projects.get_knowledge_base", lambda: _fake_kb([]))

    owner = _client()
    editor = _client()
    reviewer = _client()
    _signup(owner, "ms.owner@example.com", "Owner")
    editor_sess = _signup(editor, "ms.editor@example.com", "Editor")
    reviewer_sess = _signup(reviewer, "ms.reviewer@example.com", "Reviewer")
    project_id = owner.post("/api/projects/", json={"title": "MS"}).json()["project_id"]
    _invite_and_accept(
        owner, editor, project_id, "editor", editor_sess["researcher_id"]
    )
    _invite_and_accept(
        owner, reviewer, project_id, "reviewer", reviewer_sess["researcher_id"]
    )

    created = owner.post(
        f"/api/projects/{project_id}/manuscript",
        json={"template": "ieee_research", "title": "Draft"},
    )
    assert created.status_code == 200, created.text
    ms = created.json()
    assert ms["template"] == "ieee_research"
    assert any(s["key"] == "introduction" for s in ms["sections"])

    # Duplicate create fails
    assert (
        owner.post(
            f"/api/projects/{project_id}/manuscript",
            json={"template": "generic_academic"},
        ).status_code
        == 400
    )

    # Editor can edit unassigned section
    patched = editor.patch(
        f"/api/projects/{project_id}/manuscript/sections",
        json=[{"key": "introduction", "body": "We study Self-RAG.", "status": "in_review"}],
    )
    assert patched.status_code == 200, patched.text
    intro = next(s for s in patched.json()["sections"] if s["key"] == "introduction")
    assert "Self-RAG" in intro["body"]
    assert intro["status"] == "in_review"
    assert len(intro.get("versions") or []) >= 1

    # Assign introduction to owner — editor can no longer edit
    editor_user = editor.get("/api/auth/me").json()
    owner_user = owner.get("/api/auth/me").json()
    assigned = owner.post(
        f"/api/projects/{project_id}/manuscript/assign",
        json={"key": "introduction", "assignee_user_id": owner_user["user_id"]},
    )
    assert assigned.status_code == 200, assigned.text

    deny = editor.patch(
        f"/api/projects/{project_id}/manuscript/sections",
        json=[{"key": "introduction", "body": "Hijack"}],
    )
    assert deny.status_code == 403

    # Reviewer cannot edit
    assert (
        reviewer.patch(
            f"/api/projects/{project_id}/manuscript/sections",
            json=[{"key": "abstract", "body": "Nope"}],
        ).status_code
        == 403
    )
    _ = editor_user


def test_tasks_rbac(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    monkeypatch.setattr("app.api.projects.get_knowledge_base", lambda: _fake_kb([]))

    owner = _client()
    editor = _client()
    reviewer = _client()
    viewer = _client()
    _signup(owner, "tk.owner@example.com", "Owner")
    editor_sess = _signup(editor, "tk.editor@example.com", "Editor")
    reviewer_sess = _signup(reviewer, "tk.reviewer@example.com", "Reviewer")
    viewer_sess = _signup(viewer, "tk.viewer@example.com", "Viewer")
    project_id = owner.post("/api/projects/", json={"title": "Tasks"}).json()["project_id"]
    _invite_and_accept(
        owner, editor, project_id, "editor", editor_sess["researcher_id"]
    )
    _invite_and_accept(
        owner, reviewer, project_id, "reviewer", reviewer_sess["researcher_id"]
    )
    _invite_and_accept(
        owner, viewer, project_id, "viewer", viewer_sess["researcher_id"]
    )

    reviewer_user = reviewer.get("/api/auth/me").json()
    created = owner.post(
        f"/api/projects/{project_id}/tasks",
        json={
            "title": "Review intro",
            "description": "Check claims",
            "assignee_user_id": reviewer_user["user_id"],
        },
    )
    assert created.status_code == 200, created.text
    task = created.json()["tasks"][0]
    assert task["status"] == "todo"
    assert task["assignee_user_id"] == reviewer_user["user_id"]

    # Viewer cannot create
    assert (
        viewer.post(
            f"/api/projects/{project_id}/tasks",
            json={"title": "Nope"},
        ).status_code
        == 403
    )

    # Reviewer can update assigned task status
    upd = reviewer.patch(
        f"/api/projects/{project_id}/tasks/{task['task_id']}",
        json={"status": "in_review"},
    )
    assert upd.status_code == 200, upd.text
    assert upd.json()["tasks"][0]["status"] == "in_review"

    # Editor can create self-assigned
    ed = editor.post(
        f"/api/projects/{project_id}/tasks",
        json={"title": "Draft methods"},
    )
    assert ed.status_code == 200, ed.text

    # Comment
    commented = editor.post(
        f"/api/projects/{project_id}/tasks/{ed.json()['tasks'][0]['task_id']}/comments",
        json={"text": "Started outline"},
    )
    assert commented.status_code == 200
    assert commented.json()["tasks"][0]["comments"][0]["text"] == "Started outline"
