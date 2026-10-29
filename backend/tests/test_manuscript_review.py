"""Phase 4 — manuscript evidence & citation review."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.models.documents import Chunk, ChunkMetadata, Document
from app.models.query import Claim, ClaimStatus, EvidenceItem
from app.services.auth.sessions import clear_sessions
from app.services.projects import manuscript_review as review_mod
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
    assert (
        invitee.post(
            "/api/projects/invites/accept",
            json={"invite_id": invite_id},
        ).status_code
        == 200
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


def test_review_claim_link_missing_evidence_and_auth(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    doc = Document(
        document_id="doc-rev",
        name="paper.md",
        title="Evidence Paper",
        authors=["Author"],
        year=2022,
        doi="10.1000/rev",
        source="upload",
    )
    chunk = Chunk(
        chunk_id="c-rev",
        text="Hybrid retrieval combines dense vectors with BM25 via reciprocal rank fusion.",
        metadata=ChunkMetadata(
            document_id="doc-rev", document_name="paper.md", page=1, chunk_index=0
        ),
    )
    monkeypatch.setattr(
        "app.api.projects.get_knowledge_base", lambda: _fake_kb([doc], [chunk])
    )

    # Force lexical verifier path for determinism
    monkeypatch.setattr(
        "app.services.projects.manuscript_review.ClaimVerifier.verify",
        lambda self, claims, evidence, query="": _fake_verify(claims, evidence),
    )
    monkeypatch.setattr(
        "app.services.projects.manuscript_review.extract_claims",
        lambda answer, llm, max_claims=12: [
            Claim(claim_id="c1", text="Hybrid retrieval combines dense vectors with BM25."),
            Claim(claim_id="c2", text="Quantum teleportation scales linearly with moon phases."),
        ],
    )

    owner = _client()
    viewer = _client()
    reviewer = _client()
    _signup(owner, "rev.owner@example.com", "Owner")
    viewer_sess = _signup(viewer, "rev.viewer@example.com", "Viewer")
    reviewer_sess = _signup(reviewer, "rev.reviewer@example.com", "Reviewer")
    project_id = owner.post("/api/projects/", json={"title": "Review"}).json()["project_id"]
    _invite_and_accept(owner, viewer, project_id, "viewer", viewer_sess["researcher_id"])
    _invite_and_accept(
        owner, reviewer, project_id, "reviewer", reviewer_sess["researcher_id"]
    )

    assert (
        owner.post(
            f"/api/projects/{project_id}/sources",
            json={"document_id": "doc-rev"},
        ).status_code
        == 200
    )
    ev = owner.post(
        f"/api/projects/{project_id}/evidence",
        json={
            "document_id": "doc-rev",
            "quote": "dense vectors with BM25",
            "kind": "user_marked",
        },
    )
    assert ev.status_code == 200, ev.text

    ms = owner.post(
        f"/api/projects/{project_id}/manuscript",
        json={"template": "ieee_research", "title": "Draft"},
    )
    assert ms.status_code == 200
    intro = next(s for s in ms.json()["sections"] if s["key"] == "introduction")
    saved = owner.patch(
        f"/api/projects/{project_id}/manuscript/sections/{intro['section_id']}",
        json={
            "body": (
                "Hybrid retrieval combines dense vectors with BM25. "
                "Quantum teleportation scales linearly with moon phases."
            )
        },
    )
    assert saved.status_code == 200

    # Viewer cannot run review
    assert (
        viewer.post(
            f"/api/projects/{project_id}/manuscript/review",
            json={"section_id": intro["section_id"]},
        ).status_code
        == 403
    )

    ran = owner.post(
        f"/api/projects/{project_id}/manuscript/review",
        json={"section_id": intro["section_id"]},
    )
    assert ran.status_code == 200, ran.text
    body = ran.json()
    assert body["open_count"] >= 1
    assert body["limitations"]
    verdicts = {i["verdict"] for i in body["issues"] if i["kind"] == "claim_verification"}
    assert "supported" in verdicts or any(
        i["verdict"] == "supported" for i in body["issues"]
    )
    assert any(
        i["verdict"] in ("evidence_not_found", "needs_review", "possible_mismatch")
        for i in body["issues"]
        if "moon" in (i.get("claim_text") or "")
    )
    # Linked claim should expose evidence excerpt from stored quote only
    supported = next(
        (i for i in body["issues"] if i["verdict"] == "supported" and i.get("evidence_excerpt")),
        None,
    )
    if supported:
        assert "BM25" in supported["evidence_excerpt"]

    # Missing-source citation marker
    with_cite = owner.patch(
        f"/api/projects/{project_id}/manuscript/sections/{intro['section_id']}",
        json={"body": "A claim appears with a dangling marker [99].", "save_version": False},
    )
    assert with_cite.status_code == 200
    monkeypatch.setattr(
        "app.services.projects.manuscript_review.extract_claims",
        lambda answer, llm, max_claims=12: [
            Claim(claim_id="c3", text="A claim appears with a dangling marker [99].")
        ],
    )
    again = owner.post(
        f"/api/projects/{project_id}/manuscript/review",
        json={"section_id": intro["section_id"]},
    )
    assert again.status_code == 200
    assert any(i["kind"] == "citation_not_in_project" for i in again.json()["issues"])

    issue = again.json()["issues"][0]
    # Reviewer can mark manually checked
    patched = reviewer.patch(
        f"/api/projects/{project_id}/manuscript/review/{issue['issue_id']}",
        json={"state": "manually_checked", "resolution_note": "Checked"},
    )
    assert patched.status_code == 200, patched.text
    updated = next(
        i for i in patched.json()["issues"] if i["issue_id"] == issue["issue_id"]
    )
    assert updated["state"] == "manually_checked"
    assert updated["resolved_by_name"]

    # Viewer cannot update
    assert (
        viewer.patch(
            f"/api/projects/{project_id}/manuscript/review/{issue['issue_id']}",
            json={"state": "dismissed"},
        ).status_code
        == 403
    )


def _fake_verify(claims, evidence):
    from app.verification.claim_verifier import ClaimVerificationResult

    out = []
    for claim in claims:
        c = claim.model_copy(deep=True)
        text_l = claim.text.lower()
        if "bm25" in text_l and evidence:
            c.status = ClaimStatus.supported
            c.supporting_citations = [evidence[0].citation_id]
            c.best_evidence_span = evidence[0].text[:200]
            c.rationale = "Lexical overlap with project evidence."
            c.verifier = "test"
        elif "moon" in text_l:
            c.status = ClaimStatus.unsupported
            c.rationale = "No supporting evidence."
            c.verifier = "test"
        else:
            c.status = ClaimStatus.partially_supported
            c.rationale = "Partial overlap."
            c.verifier = "test"
        out.append(c)
    n_sup = sum(1 for c in out if c.status == ClaimStatus.supported)
    return ClaimVerificationResult(
        claims=out,
        support_rate=n_sup / max(1, len(out)),
        n_supported=n_sup,
        n_partial=sum(1 for c in out if c.status == ClaimStatus.partially_supported),
        n_unsupported=sum(1 for c in out if c.status == ClaimStatus.unsupported),
        n_contradicted=0,
        important_unsupported=[c for c in out if c.status == ClaimStatus.unsupported],
    )


def test_review_biblio_duplicate_and_map_status(settings, monkeypatch, tmp_path):
    _prepare(settings, monkeypatch, tmp_path)
    monkeypatch.setattr("app.api.projects.get_knowledge_base", lambda: _fake_kb([]))

    owner = _client()
    _signup(owner, "bib.owner@example.com", "Owner")
    # Direct unit checks for status mapping
    claim = Claim(claim_id="x", text="t", status=ClaimStatus.contradicted)
    verdict, kind = review_mod._map_claim_status(claim, has_evidence=True)
    assert verdict == "possible_mismatch"
    assert kind == "claim_evidence_mismatch"

    project_id = owner.post("/api/projects/", json={"title": "Bib"}).json()["project_id"]
    # Create manuscript empty then inject sources with duplicate DOI via store
    assert (
        owner.post(
            f"/api/projects/{project_id}/manuscript",
            json={"template": "other", "title": "T"},
        ).status_code
        == 200
    )
    # Use store to attach incomplete/duplicate sources without KB
    with project_store._store_lock():
        raw = project_store._read_raw(project_id)
        project = project_store._normalize_project(
            __import__("app.models.projects", fromlist=["ResearchProject"]).ResearchProject.model_validate(
                raw
            )
        )
        from app.models.projects import ProjectSource

        project.sources = [
            ProjectSource(document_id="a", title="", authors=[], year=None, doi="10.1/x"),
            ProjectSource(
                document_id="b",
                title="Same Title Paper",
                authors=["A"],
                year=2020,
                doi="10.1/x",
            ),
            ProjectSource(
                document_id="c",
                title="Same Title Paper",
                authors=["B"],
                year=2021,
                doi="",
            ),
        ]
        project.document_ids = ["a", "b", "c"]
        project_store._write_raw(project)

    issues = review_mod._biblio_and_duplicate_issues(project.sources, now="t")
    kinds = {i.kind for i in issues}
    assert "missing_biblio_fields" in kinds
    assert "possible_duplicate_reference" in kinds
