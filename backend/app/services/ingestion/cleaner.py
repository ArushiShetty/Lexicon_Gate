"""Text cleaning for extracted document pages."""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])-\n([a-z])")
_SOFT_WRAP_RE = re.compile(r"([a-z,;:])\n([a-z])")
_MULTI_SPACE_RE = re.compile(r"[ \t\u00a0]{2,}")
_BULLET_RE = re.compile(r"^[\s]*[•▪◦·‣∙*]\s*", re.M)
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.I | re.M)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_URL_SPACE_RE = re.compile(r"(https?://\S+)")


_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}
_SPLIT_WORD_RE = re.compile(r"\b([A-Z])\s+(olume|he|ethod|ntroduction|bstract|igure|esults|iscussion|onclusion|eference|able|lgorithm|ataset|ection)\b", re.I)


def clean_page(text: str) -> str:
    text = _CONTROL_RE.sub(" ", text or "")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "—")
    for lig, replacement in _LIGATURES.items():
        text = text.replace(lig, replacement)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _SOFT_WRAP_RE.sub(r"\1 \2", text)
    text = _SPLIT_WORD_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}", text)
    text = _BULLET_RE.sub("- ", text)
    text = _PAGE_NUMBER_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_repeated_lines(pages: Sequence[str], min_ratio: float = 0.6) -> list[str]:
    """Remove running headers and footers.

    A short line appearing on most pages is boilerplate, not content, and it
    pollutes both BM25 statistics and evidence scoring.
    """

    if len(pages) < 3:
        return list(pages)

    counts: Counter[str] = Counter()
    for page in pages:
        lines = {line.strip() for line in page.split("\n") if 0 < len(line.strip()) <= 90}
        counts.update(lines)

    threshold = max(2, int(len(pages) * min_ratio))
    boilerplate = {line for line, count in counts.items() if count >= threshold}
    if not boilerplate:
        return list(pages)

    cleaned: list[str] = []
    for page in pages:
        kept = [line for line in page.split("\n") if line.strip() not in boilerplate]
        cleaned.append("\n".join(kept).strip())
    return cleaned
