"""Event catalog tests."""

from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.events import Event, event_timing, is_registration_open
from app.services.auth.sessions import clear_sessions
from app.services.auth.users import create_user
from app.services.events.catalog import all_events, filter_events


def test_catalog_open_apply_and_images():
    today = date.today()
    events = all_events()
    assert events
    logo_hosts = set()
    for event in events:
        assert event.apply_url
        assert event.apply_url.startswith("https://")
        assert event.registration_open == is_registration_open(
            registration_deadline=event.registration_deadline,
            cfp_deadline=event.cfp_deadline,
            today=today,
        )
        assert event.timing == event_timing(
            start_date=event.start_date,
            end_date=event.end_date,
            today=today,
        )
        image = event.image_url or ""
        assert "icons.duckduckgo.com" not in image
        assert "favicon" not in image.lower()
        if image:
            assert image.startswith("https://")
            logo_hosts.add(urlparse(image).netloc)
    assert logo_hosts
    assert any(not (e.image_url or "").strip() for e in events)


def test_filter_active_excludes_past():
    today = date.today()
    active = filter_events(status="active")
    assert active
    assert all(e.end_date >= today for e in active)
    assert all(e.timing in {"live", "upcoming"} for e in active)


def test_filter_open_registration_only():
    today = date.today()
    open_events = filter_events(status="open")
    assert open_events
    assert all(e.registration_open for e in open_events)
    assert all(e.end_date >= today for e in open_events)


def test_registration_open_helper():
    today = date.today()
    assert is_registration_open(
        registration_deadline=today + timedelta(days=3),
        cfp_deadline=today - timedelta(days=10),
        today=today,
    )
    assert not is_registration_open(
        registration_deadline=today - timedelta(days=1),
        cfp_deadline=today - timedelta(days=10),
        today=today,
    )
    assert event_timing(
        start_date=today - timedelta(days=1),
        end_date=today + timedelta(days=1),
        today=today,
    ) == "live"
    assert event_timing(
        start_date=today + timedelta(days=2),
        end_date=today + timedelta(days=3),
        today=today,
    ) == "upcoming"
    closed = Event(
        event_id="closed-demo",
        name="Closed Demo",
        city="Remote",
        country="Worldwide",
        region="worldwide",
        start_date=today + timedelta(days=30),
        end_date=today + timedelta(days=31),
        registration_deadline=today - timedelta(days=2),
        apply_url="https://example.com/apply",
    )
    assert closed.registration_open is False
    assert closed.timing == "upcoming"


def test_filter_ieee():
    ieee = filter_events(event_type="ieee", status="all")
    assert ieee
    assert all(e.event_type == "ieee" for e in ieee)


def test_events_api(settings, monkeypatch):
    monkeypatch.setenv("SELFRAG_AUTH_REQUIRED", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    clear_sessions()
    user = create_user(settings, "events.open@example.com", "secret-pass", "Events")
    app = create_app()
    client = TestClient(app)
    login = client.post(
        "/api/auth/login",
        json={"email": "events.open@example.com", "password": "secret-pass"},
    )
    assert login.status_code == 200

    listed = client.get("/api/events")
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] > 0
    assert body["cities"]
    assert all(e["timing"] in {"live", "upcoming"} for e in body["events"])
    event = body["events"][0]
    assert event["apply_url"].startswith("https://")
    image = event.get("image_url") or ""
    assert "icons.duckduckgo.com" not in image
    assert "Watch" not in str(body)

    detail = client.get(f"/api/events/{event['event_id']}")
    assert detail.status_code == 200
    assert detail.json()["event_id"] == event["event_id"]
    assert "timing" in detail.json()

    assert client.get("/api/events/watch").status_code == 404
    assert client.get("/api/events/alerts").status_code == 404
    assert user["user_id"]
