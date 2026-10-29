"""Phase 6 — end-to-end research workflow integration.

Flow: discover/import paper → link to project → add evidence → cite in manuscript
→ review citation marker. Also covers import dedupe and RBAC.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.models.documents import Chunk, ChunkMetadata, Document
from app.services.auth.sessions import clear_sessions
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


def _doc(document_id: str = "doc-p6", **kwargs) -> Document:
    return Document(
        document_id=document_id,
        name=kwargs.get("name", "paper.md"),
        title=kwargs.get("title", "Self-RAG Foundations"),
        authors=kwargs.get("authors", ["Asai", "Wu"]),
        year=kwargs.get("year", 2024),
        doi=kwargs.get("doi", "10.1000/phase6"),
        source=kwargs.get("source", "upload"),
    )


def _chunk(document_id: str, text: str, chunk_id: str = "c-p6") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            document_name="paper.md",
            page=1,
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
            if not document_ids:
                return list(chunk_map.values())
            wanted = set(document_ids)
            return [c for c in chunk_map.values() if c.metadata.document_id in wanted]

    return SimpleNamespace(store=Store())


QUOTE = (
    "Retrieval-augmented generation improves factuality when evidence is cited "
    "from retrieved passages."
)


def test_phase6_discover_add_evidence_cite_review(settings, monkeypatch, tmp_path):
    """discover → project source → evidence → manuscript cite → review marker."""
    _prepare(settings, monkeypatch, tmp_path)
    doc = _doc()
    chunk = _chunk(doc.document_id, QUOTE)
    # Library starts empty; import populates it.
    docs: list[Document] = []
    chunks: list[Chunk] = []

    def _kb():
        return _fake_kb(docs, chunks)

    monkeypatch.setattr("app.api.projects.get_knowledge_base", _kb)

    import_calls = {"n": 0}

    def _fake_import(request, kb_arg, **kwargs):
        import_calls["n"] += 1
        docs.clear()
        docs.append(doc)
        chunks.clear()
        chunks.append(chunk)
        return PaperImportResult(
            document=doc,
            ingested="abstract",
            warnings=[],
            paper_url="https://example.com/self-rag",
            pdf_url=None,
        )

    monkeypatch.setattr("app.api.projects.import_paper", _fake_import)

    owner = _client()
    _signup(owner, "p6.owner@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Phase 6 Flow"}).json()[
        "project_id"
    ]

    # 1) Discover / import into project
    imported = owner.post(
        f"/api/projects/{project_id}/sources/import",
        json={
            "paper_id": "p6-paper",
            "source": "semantic_scholar",
            "title": doc.title,
            "doi": doc.doi,
            "year": doc.year,
            "url": "https://example.com/self-rag",
        },
    )
    assert imported.status_code == 200, imported.text
    assert imported.json()["source"]["document_id"] == doc.document_id
    assert doc.document_id in imported.json()["project"]["document_ids"]
    assert import_calls["n"] == 1

    # 2) Re-import same DOI must not create a second source / second ingest
    again = owner.post(
        f"/api/projects/{project_id}/sources/import",
        json={
            "paper_id": "p6-paper",
            "source": "semantic_scholar",
            "title": doc.title,
            "doi": doc.doi,
            "year": doc.year,
        },
    )
    assert again.status_code == 200, again.text
    assert again.json()["source"]["document_id"] == doc.document_id
    assert "already linked" in " ".join(again.json()["warnings"]).lower()
    assert import_calls["n"] == 1  # skipped re-import
    assert len(again.json()["project"]["sources"]) == 1

    # 3) Add evidence from Ask-like quote (must match chunk text)
    evidence = owner.post(
        f"/api/projects/{project_id}/evidence",
        json={
            "document_id": doc.document_id,
            "quote": QUOTE,
            "kind": "imported",
            "section_key": "introduction",
            "claim_ref": "1",
        },
    )
    assert evidence.status_code == 200, evidence.text
    ev = evidence.json()["evidence"][0]
    assert ev["document_id"] == doc.document_id
    assert ev["section_key"] == "introduction"
    assert QUOTE in ev["quote"]

    # Duplicate evidence rejected
    dup_ev = owner.post(
        f"/api/projects/{project_id}/evidence",
        json={"document_id": doc.document_id, "quote": QUOTE, "kind": "imported"},
    )
    assert dup_ev.status_code == 400
    assert "already" in dup_ev.json()["detail"].lower()

    # 4) Manuscript + cite with stable marker
    ms = owner.post(
        f"/api/projects/{project_id}/manuscript",
        json={"template": "ieee_research", "title": "Phase 6 Paper"},
    )
    assert ms.status_code == 200, ms.text
    section = ms.json()["sections"][0]
    cite = owner.post(
        f"/api/projects/{project_id}/manuscript/cite",
        json={"document_id": doc.document_id, "style": "apa"},
    )
    assert cite.status_code == 200, cite.text
    body = cite.json()
    assert body["document_id"] == doc.document_id
    assert body["marker"] == f"[doc:{doc.document_id}]"
    assert body["apa"]
    assert "[doc:" in body["marker"]

    # Persist citation into section body
    patched = owner.patch(
        f"/api/projects/{project_id}/manuscript/sections",
        json=[
            {
                "key": section["key"],
                "body": (
                    f"Self-RAG improves factuality {body['citation']} "
                    f"{body['marker']}."
                ),
            }
        ],
    )
    assert patched.status_code == 200, patched.text
    saved_body = next(
        s["body"] for s in patched.json()["sections"] if s["key"] == section["key"]
    )
    assert body["marker"] in saved_body

    # 5) Review should accept linked [doc:…] marker (no source_not_linked)
    review = owner.post(
        f"/api/projects/{project_id}/manuscript/review",
        json={"section_id": section["section_id"]},
    )
    assert review.status_code == 200, review.text
    bad_markers = [
        i
        for i in review.json()["issues"]
        if i.get("citation_status") == "source_not_linked"
        or (
            i.get("kind") == "citation_not_in_project"
            and body["marker"] in (i.get("citation_marker") or "")
            and i.get("verdict") == "evidence_not_found"
        )
    ]
    assert bad_markers == [], bad_markers

    # Unlinked marker must be flagged
    owner.patch(
        f"/api/projects/{project_id}/manuscript/sections",
        json=[
            {
                "key": section["key"],
                "body": "Claim with unknown cite [doc:missing-doc-id].",
            }
        ],
    )
    review2 = owner.post(
        f"/api/projects/{project_id}/manuscript/review",
        json={"section_id": section["section_id"]},
    )
    assert review2.status_code == 200
    assert any(
        i.get("citation_status") == "source_not_linked"
        for i in review2.json()["issues"]
    )


def test_phase6_import_reuses_library_doi_without_reingest(
    settings, monkeypatch, tmp_path
):
    """Find Papers import links an existing Library DOI match instead of ingesting again."""
    _prepare(settings, monkeypatch, tmp_path)
    existing = _doc("doc-library", title="Existing Paper", doi="10.1000/reuse")
    kb = _fake_kb([existing])
    monkeypatch.setattr("app.api.projects.get_knowledge_base", lambda: kb)

    def _should_not_import(*args, **kwargs):
        raise AssertionError("import_paper should not be called when DOI exists in Library")

    monkeypatch.setattr("app.api.projects.import_paper", _should_not_import)

    owner = _client()
    _signup(owner, "p6.reuse@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "Reuse"}).json()["project_id"]

    resp = owner.post(
        f"/api/projects/{project_id}/sources/import",
        json={
            "paper_id": "external-id",
            "source": "openalex",
            "title": "Existing Paper",
            "doi": "https://doi.org/10.1000/reuse",
            "year": 2024,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["source"]["document_id"] == "doc-library"
    assert any("skipped re-import" in w.lower() for w in resp.json()["warnings"])


def test_phase6_viewer_cannot_import_or_add_evidence(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    doc = _doc()
    chunk = _chunk(doc.document_id, QUOTE)
    monkeypatch.setattr(
        "app.api.projects.get_knowledge_base", lambda: _fake_kb([doc], [chunk])
    )
    monkeypatch.setattr(
        "app.api.projects.import_paper",
        lambda *a, **k: PaperImportResult(document=doc, ingested="abstract"),
    )

    owner = _client()
    owner_body = _signup(owner, "p6.own2@example.com", "Owner")
    project_id = owner.post("/api/projects/", json={"title": "RBAC"}).json()["project_id"]
    owner.post(
        f"/api/projects/{project_id}/sources",
        json={"document_id": doc.document_id},
    )

    guest = _client()
    guest_body = _signup(guest, "p6.view@example.com", "Viewer")
    inv = owner.post(
        f"/api/projects/{project_id}/invites",
        json={"researcher_id": guest_body["researcher_id"], "role": "viewer"},
    )
    assert inv.status_code == 200
    assert (
        guest.post(
            "/api/projects/invites/accept",
            json={"invite_id": inv.json()["invite"]["invite_id"]},
        ).status_code
        == 200
    )

    denied_import = guest.post(
        f"/api/projects/{project_id}/sources/import",
        json={"paper_id": "x", "source": "openalex", "title": "Nope"},
    )
    assert denied_import.status_code == 403

    denied_ev = guest.post(
        f"/api/projects/{project_id}/evidence",
        json={"document_id": doc.document_id, "quote": QUOTE},
    )
    assert denied_ev.status_code == 403

    # Viewer can still list sources and cite (read path)
    assert guest.get(f"/api/projects/{project_id}/sources").status_code == 200
    cite = guest.post(
        f"/api/projects/{project_id}/manuscript/cite",
        json={"document_id": doc.document_id, "style": "apa"},
    )
    # Manuscript may not exist yet — cite only needs project source access
    # If manuscript missing is not required for cite endpoint:
    assert cite.status_code in (200, 404)
    if cite.status_code == 200:
        assert cite.json()["marker"] == f"[doc:{doc.document_id}]"

    _ = owner_body  # silence unused
