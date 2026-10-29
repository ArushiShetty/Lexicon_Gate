# Architecture

## Contribution (one sentence)

An evidence-gated and confidence-aware Self-RAG architecture that performs adaptive retrieval, claim-level verification, contradiction detection, and self-correction to reduce hallucinations and improve factual reliability.

## Module map

```
                    ┌──────────── FastAPI (app/api) ────────────┐
User ──► React UI ──┤ /query  /query/compare  /evaluate         │
                    └───────────────┬───────────────────────────┘
                                    │
                    PipelineRunner (feature flags)
                                    │
     ┌──────────────┬───────────────┼───────────────┬──────────────┐
     ▼              ▼               ▼               ▼              ▼
 Query analysis  HybridRetriever  EvidenceGate  Generator    Verifiers
 (retrieve?)     dense+BM25+RRF   score+action  LLMProvider  claims
                                                    │         contradictions
                                                    │         hallucination
                                                    ▼         confidence
                                              PipelineResult
```

## Pipelines as flags

Traditional RAG, Standard Self-RAG, Enhanced Self-RAG and every ablation variant are **the same runner** with different `PipelineConfig` flags (`backend/app/pipelines/runner.py`).

| Flag | Traditional | Self-RAG | Enhanced |
| --- | --- | --- | --- |
| Always retrieve | yes | no (decision) | no (decision) |
| Retrieval | dense | dense | hybrid RRF |
| Evidence gate | no | no | yes |
| Adaptive k / rewrite | no | no | yes |
| Whole-answer reflection | no | yes | no |
| Claim verification | no | no | yes |
| Contradiction detection | no | no | yes |
| Bounded self-correction | no | on `NO_SUPPORT` | on unsupported claims |

This is what makes the comparison a controlled experiment: the generator is identical.

## Provider abstraction

```
LLMProvider
  ├── ExtractiveProvider     (offline, deterministic, default)
  ├── OpenAIProvider         (any OpenAI-compatible /chat/completions)
  └── LocalModelProvider     (Ollama)

EmbeddingProvider
  ├── LsaEmbeddingProvider   (offline TF-IDF + SVD)
  ├── SentenceTransformerProvider
  └── OpenAIEmbeddingProvider

VectorStore
  ├── NumpyVectorStore       (exact cosine, persisted)
  ├── ChromaVectorStore
  └── PineconeVectorStore
```

`SELFRAG_*_PROVIDER=auto` picks the best provider that is actually available and always falls back to the offline path. Keys stay in `.env`.

## Evidence score

For each chunk:

```
relevance    = 0.55·semantic + 0.32·keyword_idf + 0.13·entity
chunk_score  = 0.58·relevance + 0.14·source_quality + 0.18·retrieval − 0.12·contradiction + length prior
```

For the attempt:

```
evidence_score = w_r·relevance + w_q·source_quality + w_c·coverage + w_s·consistency
```

Coverage is **not** the union of terms across unrelated documents. It blends union, best-single-passage and best-single-document coverage. If distinctive query terms never co-occur in any source document, the gate abstains rather than answering a Frankenstein mix.

Actions: `proceed` | `expand` | `rewrite` | `conflict` | `abstain`. Loops are capped (`SELFRAG_MAX_RETRIEVAL_ATTEMPTS`, `SELFRAG_MAX_CORRECTION_LOOPS`).

## Confidence

```
confidence = w_e·evidence + w_s·claim_support + w_a·source_agreement + w_c·coverage
```

This is a **system** metric. The UI states that it is not a guarantee of truth. Calibration against actual correctness (ECE) is measured in the harness.

## Data flow for citations

Upload → parse (PDF page-aware, markdown section-aware) → clean → chunk **without crossing page boundaries** → metadata `{document, page, section, chunk_id, source_quality}` → dense index + BM25. Citations render as `[n] Document — page p`.
