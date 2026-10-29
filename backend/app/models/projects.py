"""Research project models — membership, invites, notes, reviews, activity."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MemberRole = Literal["owner", "editor", "reviewer", "viewer"]
AssignableRole = Literal["editor", "reviewer", "viewer"]
ProjectStatus = Literal["active", "archived"]
InviteStatus = Literal["pending", "accepted", "revoked", "expired"]

MEMBER_ROLES: tuple[MemberRole, ...] = ("owner", "editor", "reviewer", "viewer")
ASSIGNABLE_ROLES: tuple[AssignableRole, ...] = ("editor", "reviewer", "viewer")
PROJECT_STATUSES: tuple[ProjectStatus, ...] = ("active", "archived")

ActivityType = Literal[
    "project_created",
    "project_updated",
    "project_archived",
    "member_invited",
    "member_joined",
    "member_removed",
    "role_changed",
    "note_added",
    "note_updated",
    "note_deleted",
    "review_added",
    "review_deleted",
    "source_linked",
    "source_unlinked",
    "evidence_added",
    "evidence_removed",
    "manuscript_created",
    "manuscript_updated",
    "task_created",
    "task_updated",
    "comment_added",
]

SourceStatus = Literal["linked", "missing"]
NoteType = Literal[
    "summary",
    "methodology",
    "finding",
    "limitation",
    "research_gap",
    "citation_note",
]
NOTE_TYPES: tuple[NoteType, ...] = (
    "summary",
    "methodology",
    "finding",
    "limitation",
    "research_gap",
    "citation_note",
)
EvidenceKind = Literal["imported", "extracted", "user_marked"]
SectionStatus = Literal["draft", "in_review", "approved"]
SECTION_STATUSES: tuple[SectionStatus, ...] = ("draft", "in_review", "approved")
ManuscriptTemplate = Literal[
    "ieee_research", "literature_review", "generic_academic", "other"
]
ManuscriptDocStatus = Literal["draft", "in_review", "approved"]
ManuscriptAiAction = Literal[
    "improve_grammar",
    "improve_clarity",
    "suggest_outline",
    "identify_citation_gaps",
    "suggest_evidence",
    "summarize_evidence",
]
MANUSCRIPT_AI_ACTIONS: tuple[ManuscriptAiAction, ...] = (
    "improve_grammar",
    "improve_clarity",
    "suggest_outline",
    "identify_citation_gaps",
    "suggest_evidence",
    "summarize_evidence",
)
# Evidence/citation review statuses (advisory — see manuscript_review module docstring).
ReviewVerdict = Literal[
    "supported",
    "needs_review",
    "evidence_not_found",
    "possible_mismatch",
]
REVIEW_VERDICTS: tuple[ReviewVerdict, ...] = (
    "supported",
    "needs_review",
    "evidence_not_found",
    "possible_mismatch",
)
ReviewIssueKind = Literal[
    "claim_verification",
    "claim_needs_citation",
    "citation_not_in_project",
    "claim_evidence_mismatch",
    "missing_biblio_fields",
    "possible_duplicate_reference",
    "section_may_need_references",
]
ReviewIssueState = Literal["open", "dismissed", "resolved", "manually_checked"]
ChecklistItemStatus = Literal["incomplete", "in_progress", "completed"]
CHECKLIST_ITEM_STATUSES: tuple[ChecklistItemStatus, ...] = (
    "incomplete",
    "in_progress",
    "completed",
)
TaskStatus = Literal["todo", "in_progress", "in_review", "completed"]
TASK_STATUSES: tuple[TaskStatus, ...] = ("todo", "in_progress", "in_review", "completed")


class ProjectMember(BaseModel):
    user_id: str
    email: str = ""
    name: str = ""
    role: MemberRole = "viewer"
    researcher_id: str = ""


class ProjectActivity(BaseModel):
    activity_id: str
    type: str
    actor_id: str = ""
    actor_email: str = ""
    actor_name: str = ""
    message: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class ProjectNote(BaseModel):
    note_id: str
    text: str
    note_type: NoteType = "summary"
    document_id: str = ""
    evidence_id: str = ""
    author_id: str = ""
    author_email: str = ""
    author_name: str = ""
    created_at: str = ""
    updated_at: str = ""


class ProjectEvidence(BaseModel):
    """User-marked or extracted evidence from a project source (never invented)."""

    evidence_id: str
    document_id: str
    chunk_id: str = ""
    quote: str = ""
    kind: EvidenceKind = "user_marked"
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str = ""
    page: int | None = None
    section_key: str = ""
    claim_ref: str = ""
    author_id: str = ""
    author_email: str = ""
    author_name: str = ""
    created_at: str = ""
    updated_at: str = ""


class SectionComment(BaseModel):
    comment_id: str
    text: str
    author_id: str = ""
    author_email: str = ""
    author_name: str = ""
    created_at: str = ""


class SectionVersion(BaseModel):
    version_id: str
    body: str = ""
    title: str = ""
    saved_by: str = ""
    saved_by_name: str = ""
    saved_at: str = ""
    summary: str = ""


class ManuscriptSection(BaseModel):
    section_id: str = ""
    key: str
    title: str
    body: str = ""
    order: int = 0
    assignee_user_id: str = ""
    assignee_name: str = ""
    status: SectionStatus = "draft"
    comments: list[SectionComment] = Field(default_factory=list)
    versions: list[SectionVersion] = Field(default_factory=list)
    updated_by: str = ""
    updated_by_name: str = ""
    updated_at: str = ""


class ManuscriptVersion(BaseModel):
    """Legacy whole-manuscript snapshot (kept for older files)."""

    version_id: str
    saved_at: str
    saved_by: str = ""
    saved_by_name: str = ""
    summary: str = ""
    sections: list[ManuscriptSection] = Field(default_factory=list)


class ManuscriptReviewIssue(BaseModel):
    """Advisory finding from evidence/citation review (never auto-edits the manuscript)."""

    issue_id: str
    kind: ReviewIssueKind
    verdict: ReviewVerdict = "needs_review"
    state: ReviewIssueState = "open"
    section_id: str = ""
    section_key: str = ""
    section_title: str = ""
    claim_text: str = ""
    document_id: str = ""
    source_title: str = ""
    evidence_id: str = ""
    evidence_excerpt: str = ""
    citation_marker: str = ""
    citation_status: str = ""
    verification_status: str = ""
    potential_issue: str = ""
    uncertainty: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    resolved_by: str = ""
    resolved_by_name: str = ""
    resolved_at: str = ""
    resolution_note: str = ""


class ManuscriptChecklistItem(BaseModel):
    """Submission readiness checklist row (human-owned; never an acceptance prediction)."""

    item_id: str
    key: str
    label: str
    description: str = ""
    status: ChecklistItemStatus = "incomplete"
    section_key: str = ""
    section_id: str = ""
    linked_issue_ids: list[str] = Field(default_factory=list)
    linked_issue_kinds: list[str] = Field(default_factory=list)
    warning: str = ""
    auto_hint: str = ""
    updated_by: str = ""
    updated_by_name: str = ""
    updated_at: str = ""


class ManuscriptChecklist(BaseModel):
    items: list[ManuscriptChecklistItem] = Field(default_factory=list)
    document_type: ManuscriptTemplate = "ieee_research"
    refreshed_at: str = ""
    disclaimer: str = (
        "Submission readiness is a researcher checklist only. "
        "It is not journal/conference compliance certification, not an acceptance "
        "prediction, and not a plagiarism-free claim. Always verify venue-specific "
        "instructions manually before submission."
    )


class ProjectManuscript(BaseModel):
    manuscript_id: str
    project_id: str
    template: ManuscriptTemplate = "ieee_research"
    document_type: ManuscriptTemplate = "ieee_research"
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    status: ManuscriptDocStatus = "draft"
    sections: list[ManuscriptSection] = Field(default_factory=list)
    versions: list[ManuscriptVersion] = Field(default_factory=list)
    review_issues: list[ManuscriptReviewIssue] = Field(default_factory=list)
    review_ran_at: str = ""
    review_disclaimer: str = (
        "Automated evidence/citation review is advisory only. "
        "It does not prove plagiarism-free status, scientific correctness, or novelty. "
        "Statuses are heuristic; always inspect evidence before changing claims."
    )
    checklist: ManuscriptChecklist | None = None
    created_at: str = ""
    updated_at: str = ""
    disclaimer: str = (
        "Human-edited draft. Not official or guaranteed IEEE compliance. "
        "AI suggestions are not plagiarism-free and must not invent references, "
        "evidence, data, or scientific findings."
    )


class TaskComment(BaseModel):
    comment_id: str
    text: str
    author_id: str = ""
    author_email: str = ""
    author_name: str = ""
    created_at: str = ""


class ProjectTask(BaseModel):
    task_id: str
    title: str
    description: str = ""
    assignee_user_id: str = ""
    assignee_name: str = ""
    creator_id: str = ""
    creator_name: str = ""
    status: TaskStatus = "todo"
    document_id: str = ""
    evidence_id: str = ""
    section_key: str = ""
    comments: list[TaskComment] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class ProjectReview(BaseModel):
    review_id: str
    text: str
    author_id: str = ""
    author_email: str = ""
    author_name: str = ""
    target: str = "project"  # e.g. section key later
    created_at: str = ""


class ProjectInvitePublic(BaseModel):
    """Invite metadata safe to return to project members (no token)."""

    invite_id: str
    project_id: str
    email: str = ""
    recipient_user_id: str = ""
    recipient_researcher_id: str = ""
    role: AssignableRole
    status: InviteStatus = "pending"
    invited_by: str = ""
    invited_by_email: str = ""
    invited_by_name: str = ""
    created_at: str = ""
    expires_at: str = ""


class ProjectSource(BaseModel):
    """Association between a shared KB document and a project (no file copy)."""

    document_id: str
    added_by: str = ""
    added_by_name: str = ""
    added_by_email: str = ""
    added_at: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str = ""
    external_url: str = ""
    source_status: SourceStatus = "linked"


class ProjectSourceLinkRequest(BaseModel):
    """Link an existing Library document to the project."""

    document_id: str = Field(min_length=1, max_length=200)


class ProjectSourceImportRequest(BaseModel):
    """Import a Find Papers hit into the Library, then link it to the project."""

    paper_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    summary: str | None = None
    doi: str | None = None
    pdf_url: str | None = None
    url: str | None = None
    result_kind: str | None = None


class ProjectSourceListResponse(BaseModel):
    sources: list[ProjectSource] = Field(default_factory=list)
    total: int = 0


class PendingInviteForUser(BaseModel):
    """Invite shown to the invitee in their inbox (no token)."""

    invite_id: str
    project_id: str
    project_title: str = ""
    email: str = ""
    role: AssignableRole
    invited_by_email: str = ""
    invited_by_name: str = ""
    created_at: str = ""
    expires_at: str = ""
    can_accept_in_app: bool = False


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    topic: str = Field(default="", max_length=500)
    description: str = Field(default="", max_length=4000)


class ProjectUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    topic: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=4000)
    status: ProjectStatus | None = None
    document_ids: list[str] | None = None


class ResearchProject(BaseModel):
    project_id: str
    title: str
    topic: str = ""
    description: str = ""
    owner_id: str
    members: list[ProjectMember] = Field(default_factory=list)
    status: ProjectStatus = "active"
    document_ids: list[str] = Field(default_factory=list)
    sources: list[ProjectSource] = Field(default_factory=list)
    evidence: list[ProjectEvidence] = Field(default_factory=list)
    evidence_notes: list[dict[str, Any]] = Field(default_factory=list)  # legacy unused
    notes: list[ProjectNote] = Field(default_factory=list)
    reviews: list[ProjectReview] = Field(default_factory=list)
    tasks: list[ProjectTask] = Field(default_factory=list)
    activity: list[ProjectActivity] = Field(default_factory=list)
    invites: list[ProjectInvitePublic] = Field(default_factory=list)
    manuscript_id: str | None = None
    created_at: str = ""
    updated_at: str = ""


class ProjectSourceMutationResponse(BaseModel):
    source: ProjectSource
    project: ResearchProject
    warnings: list[str] = Field(default_factory=list)


class ProjectListResponse(BaseModel):
    projects: list[ResearchProject] = Field(default_factory=list)
    total: int = 0


class InviteCreateRequest(BaseModel):
    """Invite by Researcher ID (preferred) or legacy email."""

    researcher_id: str | None = Field(default=None, max_length=32)
    email: str | None = Field(default=None, max_length=254)
    role: AssignableRole = "viewer"

    @field_validator("email")
    @classmethod
    def _norm_email(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = (value or "").strip().lower()
        return cleaned or None

    @field_validator("researcher_id")
    @classmethod
    def _norm_rid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = (value or "").strip().upper()
        return cleaned or None


class InviteCreateResponse(BaseModel):
    invite: ProjectInvitePublic
    """One-time token for legacy email invites only; empty for in-app Researcher ID invites."""
    token: str = ""
    project: ResearchProject


class InviteAcceptRequest(BaseModel):
    invite_id: str = Field(min_length=1)
    """Optional: required only for legacy email invites without recipient_user_id."""
    token: str | None = Field(
        default=None,
        description="Secret from legacy invite creation; send in POST body only. Not needed for in-app accepts.",
    )


class InviteRejectRequest(BaseModel):
    invite_id: str = Field(min_length=1)


class MemberRoleUpdate(BaseModel):
    role: AssignableRole


class NoteCreateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    note_type: NoteType = "summary"
    document_id: str = Field(default="", max_length=200)
    evidence_id: str = Field(default="", max_length=200)


class NoteUpdateRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=8000)
    note_type: NoteType | None = None
    document_id: str | None = Field(default=None, max_length=200)
    evidence_id: str | None = Field(default=None, max_length=200)


class EvidenceCreateRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    chunk_id: str = Field(default="", max_length=200)
    quote: str = Field(default="", max_length=8000)
    kind: EvidenceKind = "user_marked"
    section_key: str = Field(default="", max_length=80)
    claim_ref: str = Field(default="", max_length=200)


class EvidenceListResponse(BaseModel):
    evidence: list[ProjectEvidence] = Field(default_factory=list)
    total: int = 0


class ManuscriptCreateRequest(BaseModel):
    template: ManuscriptTemplate = "ieee_research"
    document_type: ManuscriptTemplate | None = None
    title: str = Field(default="", max_length=300)
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)


class ManuscriptUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=300)
    authors: list[str] | None = None
    affiliations: list[str] | None = None
    status: ManuscriptDocStatus | None = None
    document_type: ManuscriptTemplate | None = None


class ManuscriptSectionCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    key: str = Field(default="", max_length=80)
    after_section_id: str = Field(default="", max_length=64)


class ManuscriptSectionPatchRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    body: str | None = Field(default=None, max_length=100_000)
    status: SectionStatus | None = None
    save_version: bool = True
    version_summary: str = Field(default="", max_length=200)


class ManuscriptSectionOrderRequest(BaseModel):
    section_ids: list[str] = Field(min_length=1)


class ManuscriptSectionUpdate(BaseModel):
    """Legacy batch patch by key (still accepted)."""

    key: str = Field(min_length=1, max_length=80)
    body: str | None = Field(default=None, max_length=100_000)
    status: SectionStatus | None = None
    assignee_user_id: str | None = Field(default=None, max_length=64)


class ManuscriptAssignRequest(BaseModel):
    key: str = Field(default="", max_length=80)
    section_id: str = Field(default="", max_length=64)
    assignee_user_id: str = Field(default="", max_length=64)


class ManuscriptCiteRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    style: Literal["apa", "bibtex"] = "apa"


class ManuscriptCiteResponse(BaseModel):
    document_id: str
    citation: str
    apa: str
    bibtex: str
    marker: str = ""
    """Stable inline marker referencing the project source document_id."""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str = ""
    warnings: list[str] = Field(default_factory=list)


class ManuscriptAiRequest(BaseModel):
    action: ManuscriptAiAction
    section_id: str = Field(default="", max_length=64)
    selection: str = Field(default="", max_length=20_000)
    evidence_ids: list[str] = Field(default_factory=list)


class ManuscriptAiResponse(BaseModel):
    action: ManuscriptAiAction
    result_text: str = ""
    suggestions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_evidence: bool = False
    provider: str = ""


class ManuscriptReviewRunRequest(BaseModel):
    section_id: str = Field(default="", max_length=64)
    """If empty, review all sections with non-empty body."""


class ManuscriptReviewIssueUpdate(BaseModel):
    state: ReviewIssueState
    resolution_note: str = Field(default="", max_length=2000)


class ManuscriptReviewResponse(BaseModel):
    manuscript: ProjectManuscript
    issues: list[ManuscriptReviewIssue] = Field(default_factory=list)
    open_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ChecklistItemUpdateRequest(BaseModel):
    status: ChecklistItemStatus
    note: str = Field(default="", max_length=2000)


class ManuscriptChecklistResponse(BaseModel):
    checklist: ManuscriptChecklist
    manuscript_id: str
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SectionCommentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class TaskCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=8000)
    assignee_user_id: str = Field(default="", max_length=64)
    document_id: str = Field(default="", max_length=200)
    evidence_id: str = Field(default="", max_length=200)
    section_key: str = Field(default="", max_length=80)


class TaskUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    description: str | None = Field(default=None, max_length=8000)
    status: TaskStatus | None = None
    assignee_user_id: str | None = Field(default=None, max_length=64)
    document_id: str | None = Field(default=None, max_length=200)
    evidence_id: str | None = Field(default=None, max_length=200)
    section_key: str | None = Field(default=None, max_length=80)


class TaskCommentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ReviewCreateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    target: str = Field(default="project", max_length=120)


class PendingInvitesResponse(BaseModel):
    invites: list[PendingInviteForUser] = Field(default_factory=list)
    total: int = 0


_FULL_PAPER_SECTIONS: list[tuple[str, str]] = [
    ("title", "Title"),
    ("abstract", "Abstract"),
    ("keywords", "Keywords"),
    ("introduction", "Introduction"),
    ("related_work", "Related Work"),
    ("methodology", "Methodology"),
    ("results", "Results"),
    ("discussion", "Discussion"),
    ("limitations", "Limitations"),
    ("conclusion", "Conclusion"),
    ("references", "References"),
]

MANUSCRIPT_TEMPLATES: dict[ManuscriptTemplate, list[tuple[str, str]]] = {
    "ieee_research": list(_FULL_PAPER_SECTIONS),
    "generic_academic": list(_FULL_PAPER_SECTIONS),
    "other": list(_FULL_PAPER_SECTIONS),
    "literature_review": [
        ("title", "Title"),
        ("abstract", "Abstract"),
        ("keywords", "Keywords"),
        ("introduction", "Introduction"),
        ("search_strategy", "Search Strategy"),
        ("literature_review", "Literature Review"),
        ("themes", "Themes / Synthesis"),
        ("gaps", "Research Gaps"),
        ("limitations", "Limitations"),
        ("conclusion", "Conclusion"),
        ("references", "References"),
    ],
}
