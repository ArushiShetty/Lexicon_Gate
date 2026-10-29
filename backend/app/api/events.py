"""Event discovery API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..models.events import Event, EventListResponse
from ..services.events.catalog import (
    all_events,
    facet_cities,
    facet_topics,
    filter_events,
    get_event,
)

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=EventListResponse)
def list_events(
    q: str = "",
    region: str = Query(default="all", pattern="^(all|india|worldwide)$"),
    city: str = "",
    topic: str = "",
    event_type: str = Query(
        default="",
        pattern="^(|all|conference|ieee|research_cfp|workshop|hackathon|meetup)$",
    ),
    date_from: str | None = Query(default=None, alias="from"),
    date_to: str | None = Query(default=None, alias="to"),
    status: str = Query(
        default="active",
        pattern="^(all|active|live|upcoming|past|open|closed)$",
    ),
) -> EventListResponse:
    items = filter_events(
        q=q,
        region=region,
        city=city,
        topic=topic,
        event_type=event_type,
        date_from=date_from,
        date_to=date_to,
        status=status,
    )
    universe = all_events()
    return EventListResponse(
        events=items,
        total=len(items),
        cities=facet_cities(universe),
        topics=facet_topics(universe),
    )


@router.get("/{event_id}", response_model=Event)
def event_detail(event_id: str) -> Event:
    event = get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    return event
