"""Local text-to-speech for Read aloud.

Uses macOS ``say`` when available so playback works even when the browser
SpeechSynthesis engine is silent. Returns WAV bytes the client can play with
HTMLAudioElement.
"""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_CHARS = 3500
_CITATION_RE = re.compile(r"\[\d+\]")


def clean_tts_text(text: str) -> str:
    cleaned = _CITATION_RE.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    if len(cleaned) > _MAX_CHARS:
        cleaned = cleaned[:_MAX_CHARS].rsplit(" ", 1)[0].strip()
    return cleaned


def server_tts_available() -> bool:
    if platform.system() != "Darwin":
        return False
    return shutil.which("say") is not None


def synthesize_wav(text: str) -> bytes:
    """Render ``text`` to WAV bytes. Raises RuntimeError on failure."""

    cleaned = clean_tts_text(text)
    if not cleaned:
        raise ValueError("Nothing to speak.")
    if not server_tts_available():
        raise RuntimeError("Server TTS is only available on macOS with the say command.")

    with tempfile.TemporaryDirectory(prefix="lexicon-tts-") as tmp:
        wav_path = Path(tmp) / "speech.wav"
        aiff_path = Path(tmp) / "speech.aiff"

        # Prefer direct WAV; fall back to AIFF + afconvert.
        direct = subprocess.run(
            [
                "say",
                "-o",
                str(wav_path),
                "--data-format=LEI16@22050",
                cleaned,
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if direct.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44:
            return wav_path.read_bytes()

        aiff = subprocess.run(
            ["say", "-o", str(aiff_path), cleaned],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if aiff.returncode != 0 or not aiff_path.exists():
            detail = (direct.stderr or aiff.stderr or "say failed").strip()
            logger.warning("macOS say failed: %s", detail)
            raise RuntimeError(detail or "macOS say failed")

        if shutil.which("afconvert"):
            converted = subprocess.run(
                [
                    "afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    "LEI16@22050",
                    str(aiff_path),
                    str(wav_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if converted.returncode == 0 and wav_path.exists():
                return wav_path.read_bytes()

        # Last resort: return AIFF (Chrome on macOS can play it).
        return aiff_path.read_bytes()
