#!/usr/bin/env python3
"""Run the four-system labeled eval over eval/test_set.json.

Does not invent gold labels or metric values. Unlabeled rows are skipped.
Citation precision and relevance scores are written as pending_manual_review
and as empty columns in the manual-review CSV.

Usage (repo root, backend venv, your papers already indexed):

    python eval/run_eval.py --dump-chunks
    python eval/run_eval.py
    python eval/run_eval.py --k 5 --out-dir eval/output
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
EVAL_DIR = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from metrics import (  # noqa: E402
    PENDING_MANUAL_REVIEW,
    format_metric,
    hallucination_rate_for_answer,
    mean_reciprocal_rank,
    mean_skip_none,
    precision_at_k,
    recall_at_k,
)

from app.pipelines.runner import (  # noqa: E402
    PipelineConfig,
    PipelineRunner,
    enhanced_config,
    no_rag_config,
    traditional_config,
    verify_only_config,
)
from app.services.llm.registry import get_llm_provider  # noqa: E402
from app.services.store.knowledge_base import get_knowledge_base  # noqa: E402
from app.verification.claim_extraction import extract_claims  # noqa: E402

SYSTEMS: list[tuple[str, str, Callable[[], PipelineConfig]]] = [
    ("base_llm_no_retrieval", "Base LLM (no retrieval)", no_rag_config),
    ("basic_rag", "Basic RAG (retrieve + generate)", traditional_config),
    ("rag_verify_no_retry", "RAG + verification (no retry)", verify_only_config),
    ("full_system", "Full system (verify + self-correct)", enhanced_config),
]


def load_test_set(path: Path) -> tuple[list[dict[str, Any]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    default_k = int(payload.get("default_k") or 5)
    entries = list(payload.get("entries") or [])
    return entries, default_k


def is_labeled(entry: dict[str, Any]) -> bool:
    query = str(entry.get("query") or "").strip()
    if not query:
        return False
    has_chunks = any(str(x).strip() for x in entry.get("relevant_chunk_ids") or [])
    has_facts = any(str(x).strip() for x in entry.get("expected_facts") or [])
    return has_chunks or has_facts


def dump_chunks(out_dir: Path) -> Path:
    kb = get_knowledge_base()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "index_chunks.csv"
    chunks = kb.store.all_chunks()
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chunk_id", "document_id", "document_name", "page", "text_preview"],
        )
        writer.writeheader()
        for chunk in chunks:
            preview = (chunk.text or "").replace("\n", " ").strip()[:240]
            writer.writerow(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.metadata.document_id,
                    "document_name": chunk.metadata.document_name,
                    "page": chunk.metadata.page if chunk.metadata.page is not None else "",
                    "text_preview": preview,
                }
            )
    return dest


def cited_ids(answer: str) -> list[int]:
    return [int(m) for m in re.findall(r"\[(\d+)\]", answer or "")]


def run_systems(
    labeled: list[dict[str, Any]],
    k: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    kb = get_knowledge_base()
    llm = get_llm_provider()
    if kb.is_empty():
        raise SystemExit(
            "Knowledge base is empty. Upload your papers first. "
            "This harness will not load the demo corpus."
        )

    per_system: dict[str, dict[str, list[float | None]]] = {
        key: {
            "recall": [],
            "precision": [],
            "mrr": [],
            "hallucination": [],
        }
        for key, _, _ in SYSTEMS
    }
    manual_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, Any]] = []

    for key, label, builder in SYSTEMS:
        config = builder()
        runner = PipelineRunner(kb, llm, config)
        print(f"Running {label} on {len(labeled)} labeled quer" + ("y" if len(labeled) == 1 else "ies"))
        for entry in labeled:
            query = str(entry["query"]).strip()
            query_id = str(entry.get("id") or query[:40])
            gold_chunks = [str(x) for x in (entry.get("relevant_chunk_ids") or []) if str(x).strip()]
            gold_facts = [str(x) for x in (entry.get("expected_facts") or []) if str(x).strip()]
            result = runner.run(query, top_k=k, query_id=f"eval-{key}-{query_id}")
            retrieved = [item.chunk_id for item in result.evidence]
            rec = recall_at_k(retrieved, gold_chunks, k)
            prec = precision_at_k(retrieved, gold_chunks, k)
            mrr = mean_reciprocal_rank(retrieved, gold_chunks)
            per_system[key]["recall"].append(rec)
            per_system[key]["precision"].append(prec)
            per_system[key]["mrr"].append(mrr)

            claims = list(result.claims)
            if not claims and result.answer:
                claims = extract_claims(result.answer, llm)
            hall = hallucination_rate_for_answer([c.text for c in claims], gold_facts)
            per_system[key]["hallucination"].append(hall)

            detail_rows.append(
                {
                    "system": key,
                    "query_id": query_id,
                    "query": query,
                    "n_retrieved": len(retrieved),
                    "recall_at_k": format_metric(rec),
                    "precision_at_k": format_metric(prec),
                    "mrr": format_metric(mrr),
                    "hallucination_rate": format_metric(hall),
                    "n_claims": len(claims),
                    "n_gold_chunks": len(gold_chunks),
                    "n_gold_facts": len(gold_facts),
                    "citation_precision": PENDING_MANUAL_REVIEW,
                    "avg_relevance_score": PENDING_MANUAL_REVIEW,
                    "abstained": str(bool(result.abstained)),
                }
            )
            manual_rows.append(
                {
                    "system": key,
                    "query_id": query_id,
                    "query": query,
                    "retrieved_chunk_ids": " | ".join(retrieved),
                    "cited_markers": ",".join(str(n) for n in cited_ids(result.answer)),
                    "answer_preview": (result.answer or "").replace("\n", " ")[:500],
                    "citation_precision": "",
                    "avg_relevance_score": "",
                    "reviewer_notes": "",
                    "instructions": (
                        "Fill citation_precision (0-1) and avg_relevance_score (0-1) by hand. "
                        "Do not use an LLM as a stand-in for this judgment."
                    ),
                }
            )

    table_rows: list[dict[str, Any]] = []
    for key, label, _ in SYSTEMS:
        bag = per_system[key]
        table_rows.append(
            {
                "system": key,
                "label": label,
                "recall@k": mean_skip_none(bag["recall"]),
                "precision@k": mean_skip_none(bag["precision"]),
                "mrr": mean_skip_none(bag["mrr"]),
                "citation_precision": PENDING_MANUAL_REVIEW,
                "hallucination_rate": mean_skip_none(bag["hallucination"]),
                "avg_relevance_score": PENDING_MANUAL_REVIEW,
                "n_queries_run": len(labeled),
                "n_queries_with_retrieval_gold": sum(
                    1 for v in bag["recall"] if v is not None
                ),
                "n_queries_with_fact_gold": sum(
                    1 for v in bag["hallucination"] if v is not None
                ),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(out_dir / "results.csv", table_rows, k)
    _write_detail_csv(out_dir / "per_query.csv", detail_rows)
    _write_manual_csv(out_dir / "manual_review.csv", manual_rows)
    _write_chart(out_dir / "hallucination_rate.png", table_rows)
    return table_rows, manual_rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]], k: int) -> None:
    fieldnames = [
        "system",
        "label",
        f"recall@{k}",
        f"precision@{k}",
        "mrr",
        "citation_precision",
        "hallucination_rate",
        "avg_relevance_score",
        "n_queries_run",
        "n_queries_with_retrieval_gold",
        "n_queries_with_fact_gold",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "system": row["system"],
                    "label": row["label"],
                    f"recall@{k}": format_metric(row["recall@k"]),
                    f"precision@{k}": format_metric(row["precision@k"]),
                    "mrr": format_metric(row["mrr"]),
                    "citation_precision": PENDING_MANUAL_REVIEW,
                    "hallucination_rate": format_metric(row["hallucination_rate"]),
                    "avg_relevance_score": PENDING_MANUAL_REVIEW,
                    "n_queries_run": row["n_queries_run"],
                    "n_queries_with_retrieval_gold": row["n_queries_with_retrieval_gold"],
                    "n_queries_with_fact_gold": row["n_queries_with_fact_gold"],
                }
            )


def _write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(
                handle,
                fieldnames=[
                    "system",
                    "query_id",
                    "query",
                    "note",
                ],
            ).writeheader()
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_manual_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "system",
        "query_id",
        "query",
        "retrieved_chunk_ids",
        "cited_markers",
        "answer_preview",
        "citation_precision",
        "avg_relevance_score",
        "reviewer_notes",
        "instructions",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if not rows:
            writer.writerow(
                {
                    "system": "",
                    "query_id": "",
                    "query": "",
                    "retrieved_chunk_ids": "",
                    "cited_markers": "",
                    "answer_preview": "",
                    "citation_precision": "",
                    "avg_relevance_score": "",
                    "reviewer_notes": "",
                    "instructions": (
                        "No labeled queries yet. Fill eval/test_set.json, re-run, then "
                        "score citation_precision and avg_relevance_score by hand."
                    ),
                }
            )
            return
        writer.writerows(rows)


def _write_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["system"] for row in rows]
    values = [row["hallucination_rate"] for row in rows]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        note = path.with_suffix(".PENDING.txt")
        note.write_text(
            "matplotlib is not installed. pip install matplotlib then re-run.\n"
            "No hallucination-rate chart was generated (not a fabricated figure).\n"
        )
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = [(lab, val) for lab, val in zip(labels, values) if val is not None]
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No hallucination_rate yet.\nLabel expected_facts in eval/test_set.json.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        names, heights = zip(*plotted)
        ax.bar(names, heights, color="#3b6ea5")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Hallucination rate")
        ax.set_title("Hallucination rate vs labeled expected_facts (not retrieved evidence)")
        ax.tick_params(axis="x", rotation=20)
        missing = [lab for lab, val in zip(labels, values) if val is None]
        if missing:
            ax.text(
                0.02,
                0.95,
                "n/a (no fact gold): " + ", ".join(missing),
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def print_table(rows: list[dict[str, Any]], k: int) -> None:
    headers = [
        "system",
        f"recall@{k}",
        f"precision@{k}",
        "mrr",
        "citation_precision",
        "hallucination_rate",
        "avg_relevance_score",
    ]
    display = []
    for row in rows:
        display.append(
            [
                row["system"],
                format_metric(row["recall@k"]),
                format_metric(row["precision@k"]),
                format_metric(row["mrr"]),
                PENDING_MANUAL_REVIEW,
                format_metric(row["hallucination_rate"]),
                PENDING_MANUAL_REVIEW,
            ]
        )
    widths = [max(len(headers[i]), max(len(r[i]) for r in display)) for i in range(len(headers))]
    def fmt(cols: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print()
    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in display:
        print(fmt(row))
    print()
    print(
        "citation_precision and avg_relevance_score are pending_manual_review "
        "(fill eval/output/manual_review.csv by hand; not estimated by an LLM)."
    )
    print("n/a means that row has no labeled gold for that metric yet.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-set",
        type=Path,
        default=EVAL_DIR / "test_set.json",
        help="Labeled queries (placeholders until you fill them).",
    )
    parser.add_argument("--k", type=int, default=None, help="Retrieval cutoff (default: test_set default_k).")
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR / "output")
    parser.add_argument(
        "--dump-chunks",
        action="store_true",
        help="Write index_chunks.csv from the current knowledge base and exit.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap labeled queries (debug).")
    args = parser.parse_args()

    if args.dump_chunks:
        dest = dump_chunks(args.out_dir)
        print(f"Wrote {dest}. Copy chunk_id values into eval/test_set.json relevant_chunk_ids.")
        return

    entries, default_k = load_test_set(args.test_set)
    k = int(args.k if args.k is not None else default_k)
    labeled = [e for e in entries if is_labeled(e)]
    if args.limit is not None:
        labeled = labeled[: max(0, args.limit)]

    print(f"Test set: {args.test_set}")
    print(f"Entries: {len(entries)} total, {len(labeled)} labeled (non-empty query + gold). k={k}")
    if not labeled:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        empty_rows = [
            {
                "system": key,
                "label": label,
                "recall@k": None,
                "precision@k": None,
                "mrr": None,
                "citation_precision": PENDING_MANUAL_REVIEW,
                "hallucination_rate": None,
                "avg_relevance_score": PENDING_MANUAL_REVIEW,
                "n_queries_run": 0,
                "n_queries_with_retrieval_gold": 0,
                "n_queries_with_fact_gold": 0,
            }
            for key, label, _ in SYSTEMS
        ]
        _write_summary_csv(args.out_dir / "results.csv", empty_rows, k)
        _write_detail_csv(args.out_dir / "per_query.csv", [])
        _write_manual_csv(args.out_dir / "manual_review.csv", [])
        _write_chart(args.out_dir / "hallucination_rate.png", empty_rows)
        print_table(empty_rows, k)
        print(
            "No labeled queries. Fill eval/test_set.json (use --dump-chunks for ids). "
            "CSV/PNG were written with n/a and pending_manual_review, not fabricated scores."
        )
        return

    table_rows, _ = run_systems(labeled, k, args.out_dir)
    print_table(table_rows, k)
    print(f"Wrote {args.out_dir / 'results.csv'}")
    print(f"Wrote {args.out_dir / 'per_query.csv'}")
    print(f"Wrote {args.out_dir / 'manual_review.csv'}  ← fill citation_precision / avg_relevance_score")
    print(f"Wrote {args.out_dir / 'hallucination_rate.png'}")


if __name__ == "__main__":
    main()
