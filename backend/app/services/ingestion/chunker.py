"""Section- and sentence-aware chunking.

Chunks never cross a page boundary, so every chunk can cite an exact page.
Within a page, splits prefer heading boundaries, then sentence boundaries, and
consecutive chunks share a configurable overlap to avoid severing facts that
straddle a split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...text_utils import split_sentences

_HEADING_PATTERNS = (
    re.compile(r"^\s{0,3}#{1,6}\s+(.{2,90})$"),
    re.compile(r"^\s{0,3}(\d+(?:\.\d+)*)[.)]?\s+([A-Z][^.!?]{3,80})$"),
    re.compile(r"^\s{0,3}([A-Z][A-Z0-9 \-/&']{4,60})\s*$"),
    re.compile(r"^\s{0,3}([A-Z][\w \-/&']{3,60}):\s*$"),
)


@dataclass(slots=True)
class TextChunk:
    text: str
    page: int
    section: str | None
    chunk_index: int
    char_start: int = 0
    char_end: int = 0
    extra: dict[str, str] = field(default_factory=dict)


def detect_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 95:
        return None
    if stripped.endswith((".", "!", "?", ",", ";")):
        return None
    for index, pattern in enumerate(_HEADING_PATTERNS):
        match = pattern.match(stripped)
        if not match:
            continue
        groups = [g for g in match.groups() if g]
        title = groups[-1].strip() if groups else stripped
        if index == 1 and match.group(1):
            title = f"{match.group(1)} {title}".strip()
        words = title.split()
        if len(words) > 14:
            return None
        return title
    return None


def split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split page text into ``(section_title, body)`` blocks."""

    sections: list[tuple[str | None, str]] = []
    current_title: str | None = None
    buffer: list[str] = []

    for line in text.split("\n"):
        heading = detect_heading(line)
        if heading is not None:
            if buffer and any(l.strip() for l in buffer):
                sections.append((current_title, "\n".join(buffer).strip()))
            current_title = heading
            buffer = []
            continue
        buffer.append(line)

    if buffer and any(l.strip() for l in buffer):
        sections.append((current_title, "\n".join(buffer).strip()))
    return sections or [(None, text.strip())]


def chunk_page(
    text: str,
    page: int,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    start_index: int = 0,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    index = start_index

    for section_title, body in split_into_sections(text):
        if not body.strip():
            continue
        for piece in _split_body(body, chunk_size, chunk_overlap):
            piece = piece.strip()
            if len(piece) < min_chunk_chars and chunks:
                # Fold a runt tail into the previous chunk instead of emitting
                # a chunk too small to be useful evidence.
                previous = chunks[-1]
                if previous.page == page and previous.section == section_title:
                    previous.text = f"{previous.text} {piece}".strip()
                    previous.char_end += len(piece) + 1
                    continue
            if not piece:
                continue
            chunks.append(
                TextChunk(
                    text=piece,
                    page=page,
                    section=section_title,
                    chunk_index=index,
                )
            )
            index += 1
    return chunks


def _split_body(body: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(body) <= chunk_size:
        return [body]

    sentences = split_sentences(body, min_chars=1) or [body]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current and current_len + sentence_len > chunk_size:
            chunks.append(" ".join(current).strip())
            current, current_len = _carry_overlap(current, chunk_overlap)
        if sentence_len > chunk_size:
            # A single oversized sentence (tables, dense PDF lines) is split on
            # whitespace as a last resort.
            for hard_piece in _hard_split(sentence, chunk_size):
                if current:
                    chunks.append(" ".join(current).strip())
                    current, current_len = [], 0
                chunks.append(hard_piece)
            continue
        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c.strip()]


def _carry_overlap(current: list[str], chunk_overlap: int) -> tuple[list[str], int]:
    if chunk_overlap <= 0:
        return [], 0
    carried: list[str] = []
    size = 0
    for sentence in reversed(current):
        if size + len(sentence) > chunk_overlap and carried:
            break
        carried.insert(0, sentence)
        size += len(sentence) + 1
    return carried, size


def _hard_split(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    buffer: list[str] = []
    size = 0
    for word in words:
        if size + len(word) + 1 > chunk_size and buffer:
            pieces.append(" ".join(buffer))
            buffer, size = [], 0
        buffer.append(word)
        size += len(word) + 1
    if buffer:
        pieces.append(" ".join(buffer))
    return pieces
