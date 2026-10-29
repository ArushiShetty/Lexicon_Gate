"""Claim extraction.

An answer is only as reliable as its individual assertions. This module turns a
draft answer into a list of atomic factual claims that can be independently
checked against the retrieved evidence.

When a hosted LLM is available it is asked to extract claims; otherwise (and
always as a fallback) we use the deterministic splitter in the extractive
engine. The two paths share the same ``Claim`` schema so the rest of the
pipeline does not care which extractor ran.
"""

from __future__ import annotations

from ..models.query import Claim, ClaimStatus
from ..services.llm.base import LLMProvider, LLMRequest, LLMTask
from ..services.llm import extractive_engine as engine
from ..text_utils import content_tokens


EXTRACT_PROMPT = """Extract every atomic factual claim from the answer below.

Rules:
- Each claim must be a standalone assertion that can be checked against source text.
- Do not invent claims that are not in the answer.
- Split compound sentences joined by "and", "but", "while".
- Keep inline citation numbers if present.

Return JSON: {{"claims": [{{"text": "...", "inline_citations": [1, 2]}}]}}

Answer:
{answer}
"""


def extract_claims(answer: str, llm: LLMProvider, max_claims: int = 12) -> list[Claim]:
    """Extract claims from a draft answer."""

    if not answer or not answer.strip():
        return []

    raw: list[dict] = []
    try:
        response = llm.generate(
            LLMRequest(
                task=LLMTask.extract_claims,
                prompt=EXTRACT_PROMPT.format(answer=answer.strip()),
                system="You extract atomic factual claims. Return JSON only.",
                payload={"answer": answer, "max_claims": max_claims},
            )
        )
        if response.structured and response.structured.get("claims"):
            raw = list(response.structured["claims"])
    except Exception:
        raw = []

    if not raw:
        raw = engine.extract_claims(answer, max_claims=max_claims)

    claims: list[Claim] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if isinstance(item, str):
            text = item.strip()
            inline: list[int] = []
        else:
            text = str(item.get("text", "")).strip()
            inline = [int(x) for x in item.get("inline_citations", []) if str(x).isdigit()]
        if not text or len(content_tokens(text)) < 3:
            continue
        key = " ".join(content_tokens(text))
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            Claim(
                claim_id=f"c{index}",
                text=text,
                status=ClaimStatus.unsupported,
                supporting_citations=inline,
            )
        )
        if len(claims) >= max_claims:
            break
    return claims
