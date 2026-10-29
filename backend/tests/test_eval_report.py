"""Labeled eval harness: retrieval + gold-fact hallucination; no fabricated scores."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "eval"))

import pytest

from metrics import (
    PENDING_MANUAL_REVIEW,
    format_metric,
    hallucination_rate_for_answer,
    mean_reciprocal_rank,
    mean_skip_none,
    precision_at_k,
    recall_at_k,
)
from run_eval import is_labeled, load_test_set, print_table


def test_retrieval_metrics_need_gold_ids():
    retrieved = ["a", "b", "c"]
    assert recall_at_k(retrieved, [], 5) is None
    assert precision_at_k(retrieved, [], 5) is None
    assert mean_reciprocal_rank(retrieved, []) is None
    assert recall_at_k(retrieved, ["c", "z"], 2) == 0.0
    assert recall_at_k(retrieved, ["c", "z"], 3) == 0.5
    assert precision_at_k(["a", "c"], ["c"], 2) == 0.5
    assert mean_reciprocal_rank(["x", "c"], ["c"]) == 0.5
    assert mean_reciprocal_rank(["x", "y"], ["c"]) == 0.0


def test_hallucination_uses_expected_facts_not_evidence():
    facts = ["Dropout reduces overfitting during training."]
    assert hallucination_rate_for_answer(
        ["Dropout reduces overfitting during training."],
        facts,
    ) == 0.0
    rate = hallucination_rate_for_answer(
        [
            "Dropout reduces overfitting during training.",
            "Dropout was invented on Titan in 1901.",
        ],
        facts,
    )
    assert rate == 0.5
    assert hallucination_rate_for_answer(["anything"], []) is None
    assert hallucination_rate_for_answer([], facts) is None


def test_test_set_scaffold_has_no_invented_gold():
    entries, default_k = load_test_set(ROOT / "eval" / "test_set.json")
    assert default_k == 5
    assert len(entries) >= 30
    assert len(entries) <= 50
    labeled = [e for e in entries if is_labeled(e)]
    assert labeled == []
    for entry in entries:
        assert entry["query"] == ""
        assert entry["relevant_chunk_ids"] == []
        assert entry["expected_facts"] == []


def test_pending_manual_review_never_looks_numeric():
    assert format_metric(None, pending=True) == PENDING_MANUAL_REVIEW
    assert format_metric(None) == "n/a"
    rows = [
        {
            "system": "basic_rag",
            "recall@k": None,
            "precision@k": None,
            "mrr": None,
            "hallucination_rate": None,
        }
    ]
    # print_table must not crash on empty gold
    print_table(
        [
            {
                **rows[0],
                "citation_precision": PENDING_MANUAL_REVIEW,
                "avg_relevance_score": PENDING_MANUAL_REVIEW,
            }
        ],
        k=5,
    )


def test_mean_skip_none():
    assert mean_skip_none([None, None]) is None
    assert mean_skip_none([0.2, None, 0.4]) == pytest.approx(0.3)
