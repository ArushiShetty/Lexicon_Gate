"""Event discovery models."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, computed_field

EventType = Literal["conference", "ieee", "research_cfp", "workshop", "hackathon", "meetup"]
EventTiming = Literal["live", "upcoming", "past"]


def is_registration_open(
    *,
    registration_deadline: date | None,
    cfp_deadline: date | None,
    today: date | None = None,
) -> bool:
    """True when registration or CFP apply window is still open."""
    day = today or date.today()
    if registration_deadline is not None:
        return registration_deadline >= day
    if cfp_deadline is not None:
        return cfp_deadline >= day
    return False


def event_timing(
    *,
    start_date: date,
    end_date: date,
    today: date | None = None,
) -> EventTiming:
    day = today or date.today()
    if end_date < day:
        return "past"
    if start_date <= day <= end_date:
        return "live"
    return "upcoming"


class Event(BaseModel):
    event_id: str
    name: str
    city: str
    country: str
    region: Literal["india", "worldwide"]
    event_type: EventType = "conference"
    venue: str = ""
    start_date: date
    end_date: date
    cfp_deadline: date | None = None
    registration_deadline: date | None = None
    apply_url: str = ""
    website: str = ""
    image_url: str = ""
    topics: list[str] = Field(default_factory=list)
    summary: str = ""

    @computed_field
    @property
    def registration_open(self) -> bool:
        return is_registration_open(
            registration_deadline=self.registration_deadline,
            cfp_deadline=self.cfp_deadline,
        )

    @computed_field
    @property
    def timing(self) -> EventTiming:
        return event_timing(start_date=self.start_date, end_date=self.end_date)


class EventListResponse(BaseModel):
    events: list[Event] = Field(default_factory=list)
    total: int = 0
    cities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
