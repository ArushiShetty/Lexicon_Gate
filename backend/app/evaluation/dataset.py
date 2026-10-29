"""Demo corpus and gold-labelled benchmark.

The documents are written so that *architecture*, not generation quality, is
what the metrics pick up:

* Traditional RAG always retrieves a fixed dense top-k and always generates.
  It therefore answers unanswerable questions, collapses conflicts into a
  fluent but one-sided answer, and misses lexical-only matches.
* Standard Self-RAG can skip retrieval and do whole-answer reflection, but it
  has no evidence gate, no hybrid retrieval and no claim-level check.
* Enhanced Self-RAG is the only variant that abstains, reports conflicts,
  rewrites failed queries and verifies claims.

Every benchmark question carries:

* a reference answer / key points (generation metrics)
* relevant chunk-id prefixes (retrieval metrics)
* ``should_abstain`` / ``has_conflict`` flags (hallucination metrics)
"""

from __future__ import annotations

from ..models.evaluation import BenchmarkDataset, BenchmarkQuestion, QuestionCategory
from ..services.ingestion.loaders import LoadedPage
from ..services.store.knowledge_base import KnowledgeBase

# Source-quality priors: peer-reviewed-style notes outrank a blog post, which
# outranks a leaked internal memo. The gate and the contradiction detector both
# use these numbers.
DOCUMENTS: list[dict] = [
    {
        "name": "Neural Regularization Handbook.md",
        "source": "handbook",
        "source_quality": 0.92,
        "tags": ["dropout", "regularization", "neural-nets"],
        "text": """
# Neural Regularization Handbook

## Dropout

Dropout is a regularization technique for neural networks. During training, dropout randomly sets a fraction of hidden units to zero. The typical dropout rate used in hidden layers is 0.5.

## Advantages of dropout

The main advantage of dropout is that it reduces overfitting by preventing co-adaptation of hidden units. Because each unit cannot rely on the presence of any particular peer, the network is forced to learn redundant, more robust representations. Empirically, dropout improves generalization performance on held-out data.

A second advantage is that dropout can be interpreted as training an exponential number of thinned networks that share parameters, which is a cheap approximation to model averaging.

## When dropout is applied

Dropout is applied only during training. At test time every unit is kept and outgoing weights are scaled by the keep probability so that the expected activation matches training.

## Batch normalization

Batch normalization is a different technique. It normalizes layer activations using mini-batch statistics. Unlike dropout, batch normalization does not drop units; it stabilizes the distribution of activations and often allows higher learning rates.

## Limitations of dropout

Dropout increases training time because many stochastic forward passes are needed for the same effective capacity. On very small datasets, an aggressive dropout rate can underfit rather than regularize.
""",
    },
    {
        "name": "Self-RAG Survey.md",
        "source": "survey",
        "source_quality": 0.88,
        "tags": ["self-rag", "retrieval", "architecture"],
        "text": """
# Self-RAG: A Survey of Adaptive Retrieval Generation

## Problem

Traditional retrieval-augmented generation (RAG) always retrieves a fixed number of documents and then generates an answer. This pipeline has three well-documented failure modes. First, it retrieves even when the question does not need external evidence. Second, it generates even when the retrieved passages are irrelevant. Third, it provides no claim-level check that the answer is entailed by those passages. The result is fluent hallucination with citations that look legitimate.

## Standard Self-RAG

Standard Self-RAG, introduced by Asai et al., trains a language model to emit reflection tokens. The Retrieve token decides whether retrieval is needed. The IsRel, IsSup and IsUse tokens critique relevance, support and usefulness of a generated segment. This is a genuine improvement over always-retrieve RAG: the model can skip retrieval and can rewrite a poorly supported continuation.

## Remaining gaps in standard Self-RAG

Standard Self-RAG still has important gaps. Reflection is coarse: a whole segment is labelled supported or not, rather than each factual claim. Retrieval is typically dense-only, so lexical matches (rare identifiers, exact figures) are missed. There is no explicit evidence gate before generation, so a weakly related passage can still enter the prompt. There is no contradiction detector, so two sources that disagree are silently collapsed into one fluent answer. Confidence, when reported, is a decoded token rather than a function of evidence.

## Proposed architecture

The proposed Enhanced Self-RAG architecture adds four components. An Evidence Gate scores retrieved chunks on relevance, coverage, source quality and consistency, and refuses to generate if the score is below a configurable threshold. Hybrid retrieval combines dense vectors with BM25 via Reciprocal Rank Fusion. Claim-level verification splits a draft answer into atomic claims and checks each one against the evidence. A contradiction detector surfaces source disagreement instead of picking a winner.

## Main advantage of the proposed architecture

The main advantage of the proposed architecture is that it answers only when retrieved evidence is sufficient, and it verifies every factual claim against that evidence before the answer is shown. This evidence-gated, confidence-aware loop is what reduces hallucinations relative to both traditional RAG and standard Self-RAG.

## Adaptive retrieval

Adaptive retrieval starts with a small top-k. If the evidence score is low, the system retrieves more documents or rewrites the query. If the evidence score is already high, retrieval stops early. The design is therefore both more accurate and, on easy questions, cheaper than always retrieving a large k.
""",
    },
    {
        "name": "Hybrid Retrieval Notes.md",
        "source": "notes",
        "source_quality": 0.8,
        "tags": ["retrieval", "bm25", "rrf"],
        "text": """
# Hybrid Retrieval Notes

## Dense retrieval

Dense retrieval embeds queries and passages into a shared vector space and ranks by cosine similarity. It captures paraphrases well: a question about "benefits of dropout" will match a passage about "advantages of dropout" even if those exact words do not appear. Dense retrieval fails on rare tokens, identifiers, and exact numeric literals, because those signals are washed out by the embedding.

## BM25

BM25 is a lexical ranking function. It scores a passage by term frequency, inverse document frequency and document-length normalisation. BM25 is excellent at exact keyword matches, including the identifier RRF-7F3 and the figure 0.5, which dense retrievers often miss. BM25 fails on paraphrases: it will not match "reduces overfitting" to "improves generalization" unless those words co-occur.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) combines two ranked lists without requiring their scores to share a scale. The RRF score of a document is the sum over lists of 1 / (k + rank). A typical k is 60. Because the method is rank-based, cosine similarities and BM25 scores can be fused as-is.

## Identifier RRF-7F3

The fusion constant used in the reference implementation is identified as RRF-7F3. Experiments in this note set k = 60 for RRF-7F3 and keep dense and BM25 weights equal at 0.5.

## When hybrid wins

Hybrid retrieval outperforms pure dense retrieval on questions that mix a paraphrase with an exact token. An example is "What k does RRF-7F3 use?": BM25 finds the identifier, dense retrieval finds the surrounding explanation, and RRF puts both near the top.
""",
    },
    {
        "name": "Evaluation Protocol.md",
        "source": "protocol",
        "source_quality": 0.85,
        "tags": ["evaluation", "metrics"],
        "text": """
# Evaluation Protocol for Retrieval-Augmented Generation

## Retrieval metrics

Precision@K is the fraction of the top-K retrieved chunks that are relevant. Recall@K is the fraction of all relevant chunks that appear in the top-K. Mean Reciprocal Rank (MRR) is the average of 1 / rank of the first relevant chunk. nDCG@K additionally rewards placing highly relevant chunks above marginally relevant ones.

## Generation metrics

Answer correctness is the fraction of gold keypoints present in the system answer. Faithfulness is the fraction of generated claims that are entailed by the retrieved evidence, independent of whether the answer is complete. Citation accuracy is the fraction of cited passages that actually support the sentence they are attached to.

## Hallucination metrics

Unsupported claim rate is the fraction of generated claims with no supporting passage. Contradiction rate is the fraction of questions on which the system asserts one side of a known source conflict without reporting the other. Abstention accuracy is the fraction of unanswerable questions on which the system refuses to answer, minus the fraction of answerable questions on which it incorrectly refuses.

## Why these metrics

A system can be fluent and still fail every one of these metrics. Fluency is therefore never used as a proxy for factual reliability in this protocol.
""",
    },
    {
        "name": "Lab Blog — Informal Notes.md",
        "source": "blog",
        "source_quality": 0.35,
        "tags": ["dropout", "opinion", "conflict"],
        "text": """
# Lab Blog — Informal Notes

## Hot take on dropout

Dropout is overrated. In our internal toy experiments dropout did not reduce overfitting at all. We observed that dropout increased overfitting on the 200-example toy set, which is the opposite of the usual claim. I would not recommend dropout for any new project.

## Confidence

I am extremely confident about this, even though this is a blog post and not a paper. The handbook is probably wrong.

## Random claim

Also, the fusion constant RRF-7F3 is definitely 12, not 60. Trust me.
""",
    },
    {
        "name": "Internal Memo — Conflicting Dropout Study.md",
        "source": "memo",
        "source_quality": 0.4,
        "tags": ["dropout", "conflict"],
        "text": """
# Internal Memo — Conflicting Dropout Study

## Finding

A small internal study on a 200-example toy dataset found that dropout increased overfitting rather than reducing it. The authors therefore conclude that dropout is harmful and should be avoided.

## Conflict with the literature

This finding directly contradicts the Neural Regularization Handbook, which states that dropout reduces overfitting and improves generalization. The memo does not offer a theoretical explanation; it reports only the toy-set observation.

## Status

This memo is unpublished and has not been peer reviewed. Treat it as a conflicting source, not as a replacement for the handbook.
""",
    },
    {
        "name": "Astronomy Mini-Encyclopedia.md",
        "source": "encyclopedia",
        "source_quality": 0.9,
        "tags": ["astronomy", "out-of-scope"],
        "text": """
# Astronomy Mini-Encyclopedia

## Saturn

Saturn is a gas giant. It is the sixth planet from the Sun. Saturn is best known for its prominent ring system, composed mainly of ice particles with smaller amounts of rock and dust.

## Titan

Titan is Saturn's largest moon and the only moon in the solar system with a dense atmosphere. Titan's atmosphere is mostly nitrogen.

## Out of scope

This encyclopedia covers solar-system bodies only. It does not discuss machine learning methods or information-retrieval algorithms.
""",
    },
]


QUESTIONS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        question_id="q_easy_dropout_advantage",
        question="What is the main advantage of dropout?",
        category=QuestionCategory.easy,
        reference_answer=(
            "The main advantage of dropout is that it reduces overfitting by preventing "
            "co-adaptation of hidden units, which improves generalization."
        ),
        answer_keypoints=["reduces overfitting", "co-adaptation", "generalization"],
        relevant_documents=["Neural Regularization Handbook.md"],
        notes="Single-document factual. All three systems should answer; enhanced should cite.",
    ),
    BenchmarkQuestion(
        question_id="q_easy_dropout_rate",
        question="What typical dropout rate is used in hidden layers?",
        category=QuestionCategory.easy,
        reference_answer="The typical dropout rate used in hidden layers is 0.5.",
        answer_keypoints=["0.5"],
        relevant_documents=["Neural Regularization Handbook.md"],
    ),
    BenchmarkQuestion(
        question_id="q_easy_test_time",
        question="Is dropout applied at test time?",
        category=QuestionCategory.easy,
        reference_answer=(
            "Dropout is applied only during training. At test time every unit is kept "
            "and outgoing weights are scaled by the keep probability."
        ),
        answer_keypoints=["only during training", "test time"],
        relevant_documents=["Neural Regularization Handbook.md"],
    ),
    BenchmarkQuestion(
        question_id="q_easy_precision",
        question="What is Precision@K in retrieval evaluation?",
        category=QuestionCategory.easy,
        reference_answer=(
            "Precision@K is the fraction of the top-K retrieved chunks that are relevant."
        ),
        answer_keypoints=["fraction", "top-K", "relevant"],
        relevant_documents=["Evaluation Protocol.md"],
    ),
    BenchmarkQuestion(
        question_id="q_multi_hop_selfrag_gaps",
        question=(
            "What gaps remain in standard Self-RAG and what does the proposed architecture add?"
        ),
        category=QuestionCategory.multi_hop,
        reference_answer=(
            "Standard Self-RAG still uses coarse reflection, typically dense-only retrieval, "
            "and has no evidence gate or contradiction detector. The proposed architecture "
            "adds an Evidence Gate, hybrid retrieval, claim-level verification and contradiction detection."
        ),
        answer_keypoints=[
            "coarse",
            "evidence gate",
            "hybrid",
            "claim",
            "contradiction",
        ],
        relevant_documents=["Self-RAG Survey.md"],
        notes="Requires combining two sections of the survey.",
    ),
    BenchmarkQuestion(
        question_id="q_multi_doc_hybrid",
        question=(
            "How does hybrid retrieval using RRF relate to the remaining gaps in standard Self-RAG?"
        ),
        category=QuestionCategory.multi_document,
        reference_answer=(
            "Standard Self-RAG typically uses dense-only retrieval, which misses rare tokens. "
            "Hybrid retrieval combines dense vectors with BM25 via Reciprocal Rank Fusion so "
            "both paraphrases and exact identifiers are ranked."
        ),
        answer_keypoints=["dense-only", "BM25", "Reciprocal Rank Fusion", "rare"],
        relevant_documents=["Self-RAG Survey.md", "Hybrid Retrieval Notes.md"],
    ),
    BenchmarkQuestion(
        question_id="q_multi_doc_advantage",
        question="What is the main advantage of the proposed architecture?",
        category=QuestionCategory.multi_document,
        reference_answer=(
            "The main advantage of the proposed architecture is that it answers only when "
            "retrieved evidence is sufficient and verifies every factual claim against that evidence."
        ),
        answer_keypoints=["sufficient", "verifies", "claim", "evidence"],
        relevant_documents=["Self-RAG Survey.md"],
    ),
    BenchmarkQuestion(
        question_id="q_rewrite_identifier",
        question="What k does RRF-7F3 use?",
        category=QuestionCategory.easy,
        reference_answer="The fusion constant RRF-7F3 uses k = 60.",
        answer_keypoints=["60", "RRF-7F3"],
        relevant_documents=["Hybrid Retrieval Notes.md"],
        notes=(
            "Lexical identifier. Dense-only retrieval is weak here; hybrid / rewrite recover it. "
            "The low-quality blog also claims k=12, so contradiction handling matters."
        ),
    ),
    BenchmarkQuestion(
        question_id="q_ambiguous_it",
        question="What does it improve?",
        category=QuestionCategory.ambiguous,
        reference_answer=(
            "The question is ambiguous. In the handbook, dropout improves generalization; "
            "in the survey, the proposed architecture reduces hallucinations."
        ),
        answer_keypoints=["dropout", "generalization"],
        relevant_documents=["Neural Regularization Handbook.md"],
        notes="Ambiguous pronoun. A good system should either disambiguate from evidence or hedge.",
    ),
    BenchmarkQuestion(
        question_id="q_unanswerable_titan_dropout",
        question="How does dropout affect Titan's nitrogen atmosphere?",
        category=QuestionCategory.unanswerable,
        reference_answer=(
            "I could not find sufficient evidence in the available sources to answer this reliably."
        ),
        answer_keypoints=[],
        relevant_documents=[],
        should_abstain=True,
        notes="The two topics exist in the corpus but never together. Must abstain.",
    ),
    BenchmarkQuestion(
        question_id="q_unanswerable_quantum",
        question="What is the quantum volume of the proposed architecture?",
        category=QuestionCategory.unanswerable,
        reference_answer=(
            "I could not find sufficient evidence in the available sources to answer this reliably."
        ),
        answer_keypoints=[],
        relevant_documents=[],
        should_abstain=True,
    ),
    BenchmarkQuestion(
        question_id="q_unanswerable_saturn_rag",
        question="How many rings does Reciprocal Rank Fusion have?",
        category=QuestionCategory.unanswerable,
        reference_answer=(
            "I could not find sufficient evidence in the available sources to answer this reliably."
        ),
        answer_keypoints=[],
        relevant_documents=[],
        should_abstain=True,
        notes="Combines two in-corpus entities that have no relation.",
    ),
    BenchmarkQuestion(
        question_id="q_conflict_dropout",
        question="Does dropout reduce overfitting?",
        category=QuestionCategory.conflicting,
        reference_answer=(
            "The retrieved sources provide conflicting information. The Neural Regularization "
            "Handbook states that dropout reduces overfitting. The Internal Memo and Lab Blog "
            "report that dropout increased overfitting on a 200-example toy set."
        ),
        answer_keypoints=["reduces overfitting", "increased overfitting"],
        relevant_documents=[
            "Neural Regularization Handbook.md",
            "Internal Memo — Conflicting Dropout Study.md",
        ],
        has_conflict=True,
    ),
    BenchmarkQuestion(
        question_id="q_conflict_rrf_k",
        question="What value of k does the fusion constant RRF-7F3 use?",
        category=QuestionCategory.conflicting,
        reference_answer=(
            "The Hybrid Retrieval Notes state that RRF-7F3 uses k = 60. The Lab Blog claims "
            "the constant is 12. The notes are the higher-quality source."
        ),
        answer_keypoints=["60"],
        relevant_documents=["Hybrid Retrieval Notes.md", "Lab Blog — Informal Notes.md"],
        has_conflict=True,
    ),
    BenchmarkQuestion(
        question_id="q_easy_batchnorm",
        question="How is batch normalization different from dropout?",
        category=QuestionCategory.easy,
        reference_answer=(
            "Batch normalization normalizes layer activations using mini-batch statistics "
            "and does not drop units, unlike dropout."
        ),
        answer_keypoints=["normalizes", "does not drop"],
        relevant_documents=["Neural Regularization Handbook.md"],
    ),
    BenchmarkQuestion(
        question_id="q_easy_faithfulness",
        question="What is faithfulness in the evaluation protocol?",
        category=QuestionCategory.easy,
        reference_answer=(
            "Faithfulness is the fraction of generated claims that are entailed by the "
            "retrieved evidence, independent of whether the answer is complete."
        ),
        answer_keypoints=["claims", "entailed", "evidence"],
        relevant_documents=["Evaluation Protocol.md"],
    ),
    BenchmarkQuestion(
        question_id="q_multi_hop_adaptive",
        question=(
            "When does adaptive retrieval stop early and why does that matter for cost?"
        ),
        category=QuestionCategory.multi_hop,
        reference_answer=(
            "Adaptive retrieval stops early when the evidence score is already high, so easy "
            "questions retrieve a small k and are cheaper than always retrieving a large k."
        ),
        answer_keypoints=["evidence score", "high", "cheaper"],
        relevant_documents=["Self-RAG Survey.md"],
    ),
    BenchmarkQuestion(
        question_id="q_enumerative_metrics",
        question="List the hallucination metrics used in the evaluation protocol.",
        category=QuestionCategory.easy,
        reference_answer=(
            "Unsupported claim rate, contradiction rate, and abstention accuracy."
        ),
        answer_keypoints=["unsupported claim rate", "contradiction rate", "abstention accuracy"],
        relevant_documents=["Evaluation Protocol.md"],
    ),
    BenchmarkQuestion(
        question_id="q_numerical_rrf_constant",
        question="What is the typical value of k used in Reciprocal Rank Fusion?",
        category=QuestionCategory.easy,
        reference_answer="A typical k is 60 in Reciprocal Rank Fusion.",
        answer_keypoints=["60", "k"],
        relevant_documents=["Hybrid Retrieval Notes.md"],
    ),
    BenchmarkQuestion(
        question_id="q_exact_formula_rrf",
        question="How is the Reciprocal Rank Fusion score of a document computed?",
        category=QuestionCategory.easy,
        reference_answer=(
            "The RRF score of a document is the sum over ranked lists of 1 / (k + rank)."
        ),
        answer_keypoints=["sum", "1 / (k + rank)"],
        relevant_documents=["Hybrid Retrieval Notes.md"],
    ),
    BenchmarkQuestion(
        question_id="q_contradicted_premise_dropout_rate",
        question="Why was the typical dropout rate set to 0.9 in hidden layers?",
        category=QuestionCategory.easy,
        reference_answer=(
            "The indexed paper states that the typical dropout rate used in hidden layers is 0.5, rather than 0.9."
        ),
        answer_keypoints=["0.5", "hidden layers"],
        relevant_documents=["Neural Regularization Handbook.md"],
    ),
    BenchmarkQuestion(
        question_id="q_unsupported_dataset_premise",
        question="Why did the authors evaluate dropout on the SVHN benchmark dataset?",
        category=QuestionCategory.unanswerable,
        should_abstain=True,
        reference_answer="The indexed papers do not provide evidence regarding evaluation on the SVHN dataset.",
        answer_keypoints=[],
        relevant_documents=[],
    ),
    BenchmarkQuestion(
        question_id="q_unsupported_model_claim",
        question="What was the exact 99.8% compression ratio achieved by the model?",
        category=QuestionCategory.unanswerable,
        should_abstain=True,
        reference_answer="The indexed text does not mention or establish a 99.8% compression ratio.",
        answer_keypoints=[],
        relevant_documents=[],
    ),
    BenchmarkQuestion(
        question_id="q_method_evidence_gate",
        question="What four components does the Evidence Gate evaluate when scoring retrieved chunks?",
        category=QuestionCategory.easy,
        reference_answer=(
            "The Evidence Gate scores retrieved chunks on relevance, coverage, source quality and consistency."
        ),
        answer_keypoints=["relevance", "coverage", "source quality", "consistency"],
        relevant_documents=["Self-RAG Survey.md"],
    ),
    BenchmarkQuestion(
        question_id="q_multi_paper_summary_regularization",
        question="Summarize the research problem and proposed method across both the neural regularization handbook and Self-RAG survey.",
        category=QuestionCategory.multi_document,
        reference_answer=(
            "The handbook addresses neural network overfitting via dropout regularization, while the survey addresses retrieval-augmented generation failure modes via an Evidence Gate and claim verification."
        ),
        answer_keypoints=["overfitting", "dropout", "retrieval", "evidence gate"],
        relevant_documents=["Neural Regularization Handbook.md", "Self-RAG Survey.md"],
    ),
    BenchmarkQuestion(
        question_id="q_limitation_dropout_small_data",
        question="Under what condition can an aggressive dropout rate harm model performance?",
        category=QuestionCategory.easy,
        reference_answer=(
            "On very small datasets, an aggressive dropout rate can underfit rather than regularize."
        ),
        answer_keypoints=["small datasets", "underfit", "aggressive"],
        relevant_documents=["Neural Regularization Handbook.md"],
    ),
    BenchmarkQuestion(
        question_id="q_dense_retrieval_ranking",
        question="How does dense retrieval rank passages in vector space?",
        category=QuestionCategory.easy,
        reference_answer=(
            "Dense retrieval embeds queries and passages into a shared vector space and ranks by cosine similarity."
        ),
        answer_keypoints=["shared vector space", "cosine similarity"],
        relevant_documents=["Hybrid Retrieval Notes.md"],
    ),
    BenchmarkQuestion(
        question_id="q_bm25_paraphrase_limitation",
        question="Why does BM25 fail on semantic paraphrases?",
        category=QuestionCategory.easy,
        reference_answer=(
            "BM25 fails on paraphrases because it relies on exact token matching and will not match terms unless those words co-occur."
        ),
        answer_keypoints=["exact token", "co-occur", "paraphrases"],
        relevant_documents=["Hybrid Retrieval Notes.md"],
    ),
    BenchmarkQuestion(
        question_id="q_unanswerable_quantum_computing",
        question="What quantum computing framework is used for fault tolerance in the system?",
        category=QuestionCategory.unanswerable,
        should_abstain=True,
        reference_answer="I could not find sufficient evidence in the available sources to answer this reliably.",
        answer_keypoints=[],
        relevant_documents=[],
    ),
    BenchmarkQuestion(
        question_id="q_unanswerable_imagenet_top1",
        question="What was the top-1 accuracy achieved on the ImageNet benchmark?",
        category=QuestionCategory.unanswerable,
        should_abstain=True,
        reference_answer="I could not find sufficient evidence in the available sources to answer this reliably.",
        answer_keypoints=[],
        relevant_documents=[],
    ),
    BenchmarkQuestion(
        question_id="q_memo_contradiction_finding",
        question="What was the dropout finding reported in the internal memo on toy datasets?",
        category=QuestionCategory.easy,
        has_conflict=True,
        reference_answer="The internal memo reported that dropout was ineffective or increased error on a small toy dataset.",
        answer_keypoints=["ineffective", "error", "toy"],
        relevant_documents=["Internal Memo — Conflicting Dropout Study.md"],
    ),
]


def benchmark_dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        name="enhanced-self-rag-demo-v1",
        description=(
            "Gold-labelled questions over the bundled demo corpus. Categories cover easy, "
            "multi-hop, multi-document, ambiguous, unanswerable and conflicting cases."
        ),
        questions=list(QUESTIONS),
    )


def load_demo_corpus(kb: KnowledgeBase, reset: bool = True) -> dict:
    """Index the demo documents. Idempotent if ``reset`` is true."""

    if reset:
        kb.reset()
    indexed = []
    for spec in DOCUMENTS:
        document, chunks = kb.ingest_pages(
            name=spec["name"],
            pages=[LoadedPage(page=1, text=spec["text"])],
            source=spec["source"],
            source_quality=float(spec["source_quality"]),
            tags=spec["tags"],
            rebuild=False,
        )
        indexed.append(
            {
                "document_id": document.document_id,
                "name": document.name,
                "n_chunks": len(chunks),
                "source_quality": document.source_quality,
            }
        )
    kb.rebuild_indexes()
    _annotate_relevant_chunks(kb)
    return {
        "documents": indexed,
        "chunks": kb.store.chunk_count(),
        "questions": len(QUESTIONS),
    }


def _annotate_relevant_chunks(kb: KnowledgeBase) -> None:
    """Fill ``relevant_chunk_ids`` from document names + lexical overlap with the question.

    Done after indexing so gold retrieval labels refer to real chunk ids.
    """

    from ..text_utils import content_tokens, stem

    for question in QUESTIONS:
        if question.should_abstain:
            question.relevant_chunk_ids = []
            continue
        wanted_names = set(question.relevant_documents)
        q_stems = {stem(t) for t in content_tokens(question.question)}
        scored: list[tuple[int, str]] = []
        for chunk in kb.store.all_chunks():
            if wanted_names and chunk.metadata.document_name not in wanted_names:
                continue
            overlap = len(q_stems & {stem(t) for t in content_tokens(chunk.text)})
            if overlap <= 0 and wanted_names:
                # Still mark the document's chunks so recall is defined, but
                # rank ones that actually overlap first.
                overlap = 0
            if wanted_names or overlap > 0:
                scored.append((overlap, chunk.chunk_id))
        scored.sort(key=lambda pair: -pair[0])
        # Keep the chunks that actually overlap; if none do, keep the top few
        # from the named documents so Recall@K is not vacuously zero.
        relevant = [cid for overlap, cid in scored if overlap > 0][:8]
        if not relevant:
            relevant = [cid for _overlap, cid in scored[:4]]
        question.relevant_chunk_ids = relevant
