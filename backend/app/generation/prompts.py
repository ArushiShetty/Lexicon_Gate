"""Generation prompts used by hosted LLM providers.

The extractive provider ignores these and uses the structured ``payload``
instead. Hosted providers receive both, so swapping the LLM does not require
rewriting the pipeline.
"""

from __future__ import annotations

from typing import Sequence

from ..models.query import EvidenceItem, QueryAnalysis


GROUNDED_SYSTEM = """You are a research assistant that answers ONLY from the provided sources.

Rules:
- Every factual sentence must be grounded in a source and cite it as [n].
- If the sources do not contain enough information, say so explicitly and refuse
  to invent an answer that is not in those sources.
- Do not fill gaps from your own knowledge.
- If sources contradict each other, report both views with citations.
- Do not invent page numbers, figures, or statistics.
"""

UNGROUNDED_SYSTEM = """You answer from your own knowledge. There are no retrieved sources.

Rules:
- Do not invent citations or document names.
- If you do not know, say you do not know.
"""


def format_evidence(evidence: Sequence[EvidenceItem]) -> str:
    blocks: list[str] = []
    for item in evidence:
        header = f"[{item.citation_id}] {item.citation_label}"
        if item.section:
            header += f" — {item.section}"
        blocks.append(f"{header}\n{item.text.strip()}")
    return "\n\n".join(blocks) if blocks else "(no evidence retrieved)"


def answer_prompt(query: str, evidence: Sequence[EvidenceItem], analysis: QueryAnalysis | None = None) -> str:
    from ..text_utils import question_aspects

    aspects = question_aspects(query)
    extra = ""
    if analysis and analysis.is_multi_hop:
        extra = "\nThis question likely requires combining evidence from more than one source."
    aspect_block = ""
    if len(aspects) >= 2:
        bullets = "\n".join(f"- {aspect}" for aspect in aspects)
        aspect_block = (
            "\nCover each of these aspects in an organized answer when the sources support it:\n"
            f"{bullets}\n"
            "If an aspect is not supported by the sources, say so explicitly under "
            "'Not covered by the indexed sources' rather than inventing content.\n"
        )
    return (
        f"Question: {query}\n"
        f"{extra}"
        f"{aspect_block}\n"
        f"Sources:\n{format_evidence(evidence)}\n\n"
        "Write a clear, evidence-grounded answer with inline citations like [1]. "
        "Prefer explaining mechanisms and roles over copying a single introductory sentence."
    )


def ungrounded_answer_prompt(query: str) -> str:
    return (
        f"Question: {query}\n\n"
        "Answer using only your own knowledge. Do not cite documents."
    )


def revise_prompt(
    query: str,
    evidence: Sequence[EvidenceItem],
    previous_answer: str,
    unsupported_claims: Sequence[str],
) -> str:
    claims = "\n".join(f"- {c}" for c in unsupported_claims) or "- (none listed)"
    return (
        f"Question: {query}\n\n"
        f"Previous draft:\n{previous_answer}\n\n"
        "The following claims were NOT supported by the sources and must be removed or rewritten:\n"
        f"{claims}\n\n"
        f"Sources:\n{format_evidence(evidence)}\n\n"
        "Write a revised answer that only asserts what the sources support. "
        "If that is not enough to answer the question, say so."
    )


def conflict_prompt(
    query: str,
    evidence: Sequence[EvidenceItem],
    conflict_explanations: Sequence[str],
) -> str:
    conflicts = "\n".join(f"- {c}" for c in conflict_explanations)
    return (
        f"Question: {query}\n\n"
        "The retrieved sources disagree:\n"
        f"{conflicts}\n\n"
        f"Sources:\n{format_evidence(evidence)}\n\n"
        "Write an answer that presents both views with citations. "
        "Do not pick a winner unless the sources themselves rank one as more reliable."
    )


INSUFFICIENT_ANSWER = (
    "I searched your indexed sources. They do not contain enough support to answer "
    "this question. I won't invent an answer that isn't grounded in those files. "
    "Try asking about a term that appears in the paper, or add a source that defines it."
)

UNRELATED_ANSWER = (
    "I searched your indexed sources. They do not contain enough support for this "
    "question, so no answer is given. I won't invent an answer that isn't grounded "
    "in those files. Try asking about a term that appears in the paper, or add a "
    "source that defines it."
)

CONFLICT_PREFIX = (
    "The retrieved sources provide conflicting information."
)
