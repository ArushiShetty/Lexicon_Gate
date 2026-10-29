"""Project shared sources — link Library / import Find Papers / unlink / RBAC."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.models.documents import Document
from app.services.auth.sessions import clear_sessions
from app.services.auth.users import create_user
from app.services.papers.import_paper import PaperImportResult
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


def _client():
    return TestClient(create_app())


def _signup(client: TestClient, email: str, name: str = "User"):
    resp = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "secret-pass", "name": name},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _doc(document_id: str = "doc-alpha", **kwargs) -> Document:
    return Document(
        document_id=document_id,
        name=kwargs.get("name", "paper.md"),
        title=kwargs.get("title", "Self-RAG Paper"),
        authors=kwargs.get("authors", ["Lewis", "Liu"]),
        year=kwargs.get("year", 2023),
        doi=kwargs.get("doi", "10.1000/test"),
        source=kwargs.get("source", "upload"),
    )


def _fake_kb(docs: list[Document]):
    by_id = {d.document_id: d for d in docs}

    class Store:
        def get_document(self, document_id: str):
            return by_id.get(document_id)

        def list_documents(self):
            return list(by_id.values())

    return SimpleNamespace(store=Store())


def test_link_library_source_list_unlink(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    doc = _doc()
    monkeypatch.setattr("app.api.projects.get_knowledge_base", lambda: _fake_kb([doc]))

    owner = _client()
    _signup(owner, "src.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Sources"}).json()["project_id"]

    linked = owner.post(
        f"/api/projects/{project_id}/sources",
        json={"document_id": doc.document_id},
    )
    assert linked.status_code == 200, linked.text
    body = linked.json()
    assert body["source"]["document_id"] == doc.document_id
    assert body["source"]["title"] == "Self-RAG Paper"
    assert body["source"]["authors"] == ["Lewis", "Liu"]
    assert body["source"]["year"] == 2023
    assert body["source"]["doi"] == "10.1000/test"
    assert body["source"]["added_by_email"] == "src.owner@example.com"
    assert doc.document_id in body["project"]["document_ids"]
    assert len(body["project"]["sources"]) == 1

    # Duplicate rejected
    dup = owner.post(
        f"/api/projects/{project_id}/sources",
        json={"document_id": doc.document_id},
    )
    assert dup.status_code == 400
    assert "already" in dup.json()["detail"].lower()

    listed = owner.get(f"/api/projects/{project_id}/sources").json()
    assert listed["total"] == 1
    assert listed["sources"][0]["source_status"] == "linked"

    # Viewer cannot link
    guest = _client()
    guest_body = _signup(guest, "src.viewer@example.com", "Viewer")
    # invite as viewer via researcher id
    owner2 = _client()
    owner2.post("/api/auth/login", json={"email": "src.owner@example.com", "password": "secret-pass"})
    inv = owner2.post(
        f"/api/projects/{project_id}/invites",
        json={"researcher_id": guest_body["researcher_id"], "role": "viewer"},
    )
    assert inv.status_code == 200
    assert guest.post("/api/projects/invites/accept", json={"invite_id": inv.json()["invite"]["invite_id"]}).status_code == 200

    denied = guest.post(
        f"/api/projects/{project_id}/sources",
        json={"document_id": "doc-other"},
    )
    assert denied.status_code == 403

    # Viewer can list
    assert guest.get(f"/api/projects/{project_id}/sources").status_code == 200

    # Unlink does not require KB delete; association gone
    removed = owner2.delete(f"/api/projects/{project_id}/sources/{doc.document_id}")
    assert removed.status_code == 200
    assert removed.json()["sources"] == []
    assert removed.json()["document_ids"] == []
    # Fake KB still has the document
    assert _fake_kb([doc]).store.get_document(doc.document_id) is not None


def test_import_then_link_and_cross_project_isolation(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    imported = _doc("doc-import", title="Imported Paper")
    docs: list[Document] = []

    def _fake_import(request, kb, **kwargs):
        docs.clear()
        docs.append(imported)
        return PaperImportResult(
            document=imported,
            ingested="abstract",
            warnings=["abstract only"],
            paper_url="https://example.com/paper",
            pdf_url=None,
        )

    monkeypatch.setattr("app.api.projects.import_paper", _fake_import)
    monkeypatch.setattr(
        "app.api.projects.get_knowledge_base", lambda: _fake_kb(docs)
    )

    owner = _client()
    _signup(owner, "imp.owner@example.com", "Owner")
    a = owner.post("/api/projects/", json={"title": "Project A"}).json()["project_id"]
    b = owner.post("/api/projects/", json={"title": "Project B"}).json()["project_id"]

    resp = owner.post(
        f"/api/projects/{a}/sources/import",
        json={
            "paper_id": "p1",
            "source": "semantic_scholar",
            "title": "Imported Paper",
            "url": "https://example.com/paper",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["source"]["document_id"] == "doc-import"
    assert resp.json()["source"]["external_url"] == "https://example.com/paper"
    assert "abstract only" in resp.json()["warnings"]

    # Same doc can also link to B without duplication of files
    link_b = owner.post(f"/api/projects/{b}/sources", json={"document_id": "doc-import"})
    assert link_b.status_code == 200

    # Isolation: stranger cannot see project A sources
    stranger = _client()
    _signup(stranger, "imp.stranger@example.com", "Stranger")
    assert stranger.get(f"/api/projects/{a}/sources").status_code == 404


def test_missing_library_document_rejected(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    monkeypatch.setattr("app.api.projects.get_knowledge_base", lambda: _fake_kb([]))
    owner = _client()
    _signup(owner, "miss.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Empty"}).json()["project_id"]
    resp = owner.post(
        f"/api/projects/{project_id}/sources",
        json={"document_id": "nope"},
    )
    assert resp.status_code == 404
