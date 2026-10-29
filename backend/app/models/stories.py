"""Public Write / Stories models — freeform or IEEE research-paper format."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

StoryStatus = Literal["draft", "published"]
WorkStatus = Literal["open", "completed"]
StoryFormat = Literal["freeform", "ieee"]
StoryCollabRole = Literal["editor", "viewer"]
StoryInviteStatus = Literal["pending", "accepted", "declined", "revoked"]
StoryAccessRole = Literal["owner", "editor", "viewer"]

# IEEE-style research paper section order (keys → labels).
IEEE_SECTIONS: list[tuple[str, str]] = [
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

IEEE_SECTION_KEYS: tuple[str, ...] = tuple(key for key, _ in IEEE_SECTIONS)


def empty_ieee_sections() -> dict[str, str]:
    return {key: "" for key in IEEE_SECTION_KEYS}


def assemble_ieee_body(
    *,
    title: str = "",
    authors_line: str = "",
    affiliation: str = "",
    sections: dict[str, str] | None = None,
) -> str:
    """Build Markdown from IEEE section fields only.

    Title, authors, and affiliation are stored separately and rendered by the
    reader chrome — they must not be duplicated inside ``body_md``.
    """

    parts: list[str] = []
    sec = sections or {}
    for key, label in IEEE_SECTIONS:
        text = (sec.get(key) or "").strip()
        if not text:
            continue
        parts.append(f"## {label}\n\n{text}")
    return "\n\n".join(parts).strip()


class StoryCollaborator(BaseModel):
    user_id: str
    name: str = ""
    email: str = ""
    researcher_id: str = ""
    role: StoryCollabRole = "editor"
    added_at: str


class StoryInvite(BaseModel):
    invite_id: str
    recipient_user_id: str
    recipient_researcher_id: str = ""
    recipient_name: str = ""
    recipient_email: str = ""
    role: StoryCollabRole = "editor"
    status: StoryInviteStatus = "pending"
    invited_by: str
    invited_by_name: str = ""
    created_at: str


class StoryInviteCreate(BaseModel):
    researcher_id: str = Field(..., min_length=2, max_length=80)
    role: StoryCollabRole = "editor"

    @field_validator("researcher_id")
    @classmethod
    def rid_not_blank(cls, value: str) -> str:
        cleaned = (value or "").strip().upper()
        if not cleaned:
            raise ValueError("Researcher ID is required.")
        return cleaned


class StoryCollaboratorUpdate(BaseModel):
    role: StoryCollabRole


class WorkStatusUpdate(BaseModel):
    status: WorkStatus


class PendingStoryInvite(BaseModel):
    invite_id: str
    story_id: str
    story_title: str = ""
    role: StoryCollabRole
    invited_by: str
    invited_by_name: str = ""
    created_at: str


class PendingStoryInviteList(BaseModel):
    invites: list[PendingStoryInvite]


class Story(BaseModel):
    story_id: str
    slug: str
    title: str
    body_md: str = ""
    excerpt: str = ""
    author_user_id: str
    author_name: str = ""
    author_researcher_id: str = ""
    status: StoryStatus = "draft"
    work_status: WorkStatus = "open"
    work_closed_at: str | None = None
    work_closed_by_name: str = ""
    work_closed_by_user_id: str = ""
    format: StoryFormat = "ieee"
    authors_line: str = ""
    affiliation: str = ""
    sections: dict[str, str] = Field(default_factory=empty_ieee_sections)
    collaborators: list[StoryCollaborator] = Field(default_factory=list)
    invites: list[StoryInvite] = Field(default_factory=list)
    created_at: str
    updated_at: str
    published_at: str | None = None


class StoryDetail(Story):
    """Authenticated story payload including the caller's access role."""

    my_role: StoryAccessRole | None = None


class StoryCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body_md: str = Field(default="", max_length=200_000)
    format: StoryFormat = "ieee"
    authors_line: str = Field(default="", max_length=500)
    affiliation: str = Field(default="", max_length=500)
    sections: dict[str, str] | None = None

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("Title is required.")
        return cleaned


class StoryUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    body_md: str | None = Field(default=None, max_length=200_000)
    format: StoryFormat | None = None
    authors_line: str | None = Field(default=None, max_length=500)
    affiliation: str | None = Field(default=None, max_length=500)
    sections: dict[str, str] | None = None

    @field_validator("title")
    @classmethod
    def title_not_blank_if_set(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Title cannot be empty.")
        return cleaned


class StoryListItem(BaseModel):
    story_id: str
    slug: str
    title: str
    excerpt: str = ""
    author_user_id: str
    author_name: str = ""
    author_researcher_id: str = ""
    status: StoryStatus
    work_status: WorkStatus = "open"
    format: StoryFormat = "ieee"
    my_role: StoryAccessRole | None = None
    created_at: str
    updated_at: str
    published_at: str | None = None


class StoryListResponse(BaseModel):
    stories: list[StoryListItem]
    total: int


class PublicStory(BaseModel):
    """Published story payload safe for anonymous readers."""

    story_id: str
    slug: str
    title: str
    body_md: str
    excerpt: str = ""
    author_name: str = ""
    author_researcher_id: str = ""
    format: StoryFormat = "ieee"
    authors_line: str = ""
    affiliation: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    published_at: str | None = None
    updated_at: str
