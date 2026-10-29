"""Verify Google Identity Services ID tokens."""

from __future__ import annotations

from typing import Any

import httpx


def verify_google_id_token(credential: str, client_id: str, *, timeout: float = 12.0) -> dict[str, Any]:
    """
    Validate a GIS ID token with Google's tokeninfo endpoint.

    Raises ValueError with a short reason when the token is invalid.
    """
    token = (credential or "").strip()
    audience = (client_id or "").strip()
    if not token:
        raise ValueError("missing_credential")
    if not audience:
        raise ValueError("google_not_configured")

    try:
        response = httpx.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": token},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise ValueError("google_unreachable") from exc

    if response.status_code != 200:
        raise ValueError("invalid_google_token")

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("invalid_google_token") from exc

    if not isinstance(payload, dict):
        raise ValueError("invalid_google_token")

    aud = str(payload.get("aud") or "")
    if aud != audience:
        raise ValueError("audience_mismatch")

    if str(payload.get("email_verified") or "").lower() not in {"true", "1"}:
        raise ValueError("email_unverified")

    email = str(payload.get("email") or "").strip()
    sub = str(payload.get("sub") or "").strip()
    if not email or not sub:
        raise ValueError("incomplete_google_profile")

    return {
        "google_sub": sub,
        "email": email,
        "name": str(payload.get("name") or "").strip(),
    }
