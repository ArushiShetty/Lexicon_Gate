"""Load and filter the seeded event catalog."""

from __future__ import annotations

import json
from datetime import date, datetime
from functools import lru_cache
from typing import Iterable

from ...config import BACKEND_ROOT
from ...models.events import Event, event_timing, is_registration_open

_SEED = BACKEND_ROOT / "data" / "events" / "catalog.json"


@lru_cache(maxsize=4)
def _raw_catalog(mtime: float) -> tuple[dict, ...]:
    if not _SEED.exists():
        return tuple()
    payload = json.loads(_SEED.read_text(encoding="utf-8"))
    items = payload.get("events", []) if isinstance(payload, dict) else payload
    return tuple(items) if isinstance(items, list) else tuple()


def all_events() -> list[Event]:
    mtime = _SEED.stat().st_mtime if _SEED.exists() else 0.0
    return [Event.model_validate(item) for item in _raw_catalog(mtime)]


def get_event(event_id: str) -> Event | None:
    for item in all_events():
        if item.event_id == event_id:
            return item
    return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def filter_events(
    *,
    q: str = "",
    region: str = "all",
    city: str = "",
    topic: str = "",
    event_type: str = "",
    date_from: str | None = None,
    date_to: str | None = None,
    status: str = "active",
) -> list[Event]:
    today = datetime.now().date()
    query = (q or "").strip().lower()
    city_q = (city or "").strip().lower()
    topic_q = (topic or "").strip().lower()
    type_q = (event_type or "").strip().lower()
    start_bound = _parse_date(date_from)
    end_bound = _parse_date(date_to)
    region_q = (region or "all").strip().lower()
    status_q = (status or "active").strip().lower()

    out: list[Event] = []
    for event in all_events():
        if region_q in {"india", "worldwide"} and event.region != region_q:
            continue
        if type_q and type_q != "all" and event.event_type != type_q:
            continue
        if city_q and city_q not in event.city.lower():
            continue
        if topic_q and not any(topic_q in t.lower() for t in event.topics):
            continue
        if query:
            hay = " ".join(
                [
                    event.name,
                    event.city,
                    event.country,
                    event.venue,
                    event.summary,
                    event.event_type,
                    " ".join(event.topics),
                ]
            ).lower()
            if query not in hay:
                continue
        if start_bound and event.end_date < start_bound:
            continue
        if end_bound and event.start_date > end_bound:
            continue

        timing = event_timing(start_date=event.start_date, end_date=event.end_date, today=today)
        open_apply = is_registration_open(
            registration_deadline=event.registration_deadline,
            cfp_deadline=event.cfp_deadline,
            today=today,
        )

        # Default and primary views: only current (live) or future events.
        if status_q == "active" and timing == "past":
            continue
        if status_q == "live" and timing != "live":
            continue
        if status_q == "upcoming" and timing != "upcoming":
            continue
        if status_q == "past" and timing != "past":
            continue
        if status_q == "open":
            if timing == "past" or not open_apply:
                continue
        if status_q == "closed":
            if timing == "past" or open_apply:
                continue
        out.append(event)

    out.sort(
        key=lambda e: (
            0 if e.timing == "live" else 1 if e.timing == "upcoming" else 2,
            e.start_date,
            e.name.lower(),
        )
    )
    return out


def facet_cities(events: Iterable[Event]) -> list[str]:
    return sorted({e.city for e in events})


def facet_topics(events: Iterable[Event]) -> list[str]:
    topics: set[str] = set()
    for event in events:
        topics.update(event.topics)
    return sorted(topics)
