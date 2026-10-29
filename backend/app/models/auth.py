"""Auth request and response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=256)
    name: str = Field(default="", max_length=80)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=256)


class GoogleAuthRequest(BaseModel):
    """Google Identity Services ID token from the client."""

    credential: str = Field(min_length=20, max_length=8192)


class UserPublic(BaseModel):
    user_id: str
    email: str
    name: str = ""
    researcher_id: str = ""


class ResearcherPublic(BaseModel):
    """Safe card for Find Researcher — no email or secrets."""

    researcher_id: str
    display_name: str = "Researcher"
