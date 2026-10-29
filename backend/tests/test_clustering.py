"""Topic clustering: agglomerative grouping and library themes."""

from __future__ import annotations

import numpy as np

from app.services.clustering import agglomerative_groups, cluster_documents


def test_agglomerative_keeps_orthogonal_vectors_apart():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
            [0.0, 0.0, 1.0],
            [0.05, 0.0, 0.99],
        ],
        dtype=np.float32,
    )
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    groups = agglomerative_groups(vectors, min_similarity=0.7, max_themes=8)
    frozen = {frozenset(group) for group in groups}
    assert len(groups) == 2
    assert frozenset({0, 1}) in frozen
    assert frozenset({2, 3}) in frozen


def test_agglomerative_respects_max_themes():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(6, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    groups = agglomerative_groups(vectors, min_similarity=0.99, max_themes=2)
    assert len(groups) == 2
    assert sorted(i for group in groups for i in group) == list(range(6))


def test_empty_library_has_no_themes(kb):
    result = cluster_documents(kb)
    assert result.total_documents == 0
    assert result.clusters == []


def test_single_document_is_one_theme(kb):
    kb.ingest_bytes(
        "dropout.md",
        b"# Dropout in neural networks\n\nDropout reduces overfitting in neural networks.\n",
        title="Dropout in neural networks",
    )
    result = cluster_documents(kb)
    assert result.total_documents == 1
    assert len(result.clusters) == 1
    assert result.clusters[0].size == 1
    assert result.clusters[0].label


def test_related_papers_merge_unrelated_stay_apart(kb):
    kb.ingest_bytes(
        "dropout.md",
        b"# Dropout regularisation\n\n"
        b"Dropout randomly disables units in a neural network during training. "
        b"This regularisation method reduces overfitting in deep neural networks.\n",
        rebuild=False,
        title="Dropout regularisation in neural networks",
    )
    kb.ingest_bytes(
        "overfitting.md",
        b"# Overfitting in neural networks\n\n"
        b"Neural network overfitting is reduced by dropout regularisation. "
        b"Disabled units during training improve generalisation of deep networks.\n",
        rebuild=False,
        title="Overfitting in neural networks",
    )
    kb.ingest_bytes(
        "climate.md",
        b"# Climate change adaptation\n\n"
        b"Coastal cities plan climate change adaptation with seawalls, wetlands, "
        b"and managed retreat. Adaptation policy for flooding and heat is distinct "
        b"from greenhouse gas mitigation.\n",
        rebuild=True,
        title="Climate change adaptation in coastal cities",
    )
    result = cluster_documents(kb)
    assert result.total_documents == 3
    by_id = {doc.document_id: doc for doc in kb.store.list_documents()}
    dropout_ids = {
        doc.document_id
        for doc in by_id.values()
        if "neural" in (doc.title or "").lower() or "dropout" in (doc.title or "").lower()
        or "overfitting" in (doc.title or "").lower()
    }
    climate_ids = {
        doc.document_id
        for doc in by_id.values()
        if "climate" in (doc.title or "").lower()
    }
    membership = {
        doc_id: cluster.cluster_id
        for cluster in result.clusters
        for doc_id in cluster.document_ids
    }
    neural_cluster = {membership[i] for i in dropout_ids}
    assert len(neural_cluster) == 1
    assert membership[next(iter(climate_ids))] not in neural_cluster
    climate_cluster = next(c for c in result.clusters if next(iter(climate_ids)) in c.document_ids)
    joined = " ".join([climate_cluster.label, *climate_cluster.keywords]).lower()
    assert "climate" in joined or "adaptation" in joined
