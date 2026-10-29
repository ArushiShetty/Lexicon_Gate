"""Server TTS helpers (macOS say)."""

from __future__ import annotations

import platform

import pytest

from app.services.tts import clean_tts_text, server_tts_available, synthesize_wav


def test_clean_tts_text_strips_citations():
    assert clean_tts_text("Hello [1] world [2].") == "Hello world."


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS say only")
def test_synthesize_wav_produces_riff():
    if not server_tts_available():
        pytest.skip("say not installed")
    audio = synthesize_wav("Lexicon Gate read aloud check.")
    assert audio[:4] == b"RIFF"
    assert len(audio) > 2000
