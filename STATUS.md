# Pipeline audit

Checked that each stage is implemented and wired (API/runner → backend → tests or UI), not that a route name exists. Default runtime: extractive LLM, LSA or sentence-transformer embeddings, numpy vectors (`SELFRAG_VECTOR_STORE=numpy`).

## PDF ingestion / chunking

**Working** — Upload and paper-import parse PDFs with pypdf, chunk by page/section, persist chunks, then rebuild indexes.

- `backend/app/api/routes.py` — `upload_documents`, `papers_import`
- `backend/app/services/ingestion/loaders.py` — `load_bytes`, `load_document`, `_load_pdf`, `_pages_from_reader`
- `backend/app/services/ingestion/chunker.py` — `chunk_page`, `split_into_sections`
- `backend/app/services/ingestion/cleaner.py` — `clean_page`
- `backend/app/services/store/knowledge_base.py` — `ingest_bytes`, `ingest_pages`, `rebuild_indexes`
- `backend/app/services/papers/import_paper.py` — `import_paper` (PDF bytes or abstract fallback)
- Tests: `backend/tests/test_ingestion.py`, `backend/tests/test_api.py` (`test_upload_and_query`)

Caveat: text-layer PDFs only (no OCR). Empty extract → no chunks.

## Embedding generation

**Working** — After ingest, every chunk is encoded and upserted; query encoding is used for dense search.

- `backend/app/services/store/knowledge_base.py` — `rebuild_indexes`, `dense_search`
- `backend/app/services/embeddings/registry.py` — `get_embedding_provider`
- `backend/app/services/embeddings/lsa_provider.py` — `LsaEmbeddingProvider` (always available)
- `backend/app/services/embeddings/sentence_transformer_provider.py` — optional
- `backend/app/services/embeddings/openai_provider.py` — optional HTTP embeddings
- Wired from `KnowledgeBase.__init__` via `get_embedding_provider()`

No embedding UI; providers are env-selected (`SELFRAG_EMBEDDING_PROVIDER`).

## Pinecone indexing

**Partial** — Real upsert/query class exists and is selectable, but it is not the default path, not a required dependency, and has no tests.

- `backend/app/services/vectorstore/pinecone_store.py` — `PineconeVectorStore.upsert`, `search`, `delete_document`, `clear`
- `backend/app/services/vectorstore/registry.py` — `get_vector_store` maps `"pinecone"`
- `backend/app/config.py` — `pinecone_api_key`, `pinecone_index`, `vector_store` (default `"numpy"`)
- Production default: `NumpyVectorStore` (`backend/app/services/vectorstore/numpy_store.py`)
- `pinecone` is not in `backend/requirements.txt`; `.env.example` does not set Pinecone
- `is_available()` requires the SDK **and** an API key; otherwise registry falls back to numpy

End-to-end Pinecone only if `SELFRAG_VECTOR_STORE=pinecone` plus key plus installed SDK.

## Retrieval

**Working** — Ask always runs Enhanced Self-RAG, which hybrid-retrieves (dense + BM25 + RRF) inside the runner.

- `backend/app/api/routes.py` — `query` → `_runner(PipelineName.enhanced)`
- `backend/app/pipelines/runner.py` — `PipelineRunner.run` (retrieve loop), `enhanced_config`
- `backend/app/retrieval/hybrid.py` — `HybridRetriever.retrieve`
- `backend/app/retrieval/bm25.py` — `BM25Index.search`
- `backend/app/retrieval/fusion.py` — `reciprocal_rank_fusion`
- `backend/app/services/store/knowledge_base.py` — `dense_search`, `lexical_search`
- Tests: `backend/tests/test_retrieval.py`, `backend/tests/test_pipelines.py`

## LLM answer generation with citations

**Working** — Generation uses retrieved chunks; answers get `[n]` markers and APA/BibTeX on evidence items.

- `backend/app/pipelines/runner.py` — `_generate`, `_llm`
- `backend/app/services/llm/registry.py` — `get_llm_provider` (default extractive if no hosted key)
- `backend/app/services/llm/extractive_provider.py` — `ExtractiveProvider`
- `backend/app/services/llm/extractive_engine.py` — `compose_answer`, `synthesise_lead`
- `backend/app/services/llm/openai_provider.py` / `ollama_provider.py` — optional hosted
- `backend/app/generation/prompts.py` — `answer_prompt`, `GROUNDED_SYSTEM`
- `backend/app/pipelines/common.py` — `number_citations`
- `backend/app/services/citations.py` — `attach_cite_strings`, `format_apa`
- Tests: `backend/tests/test_pipelines.py` (`test_enhanced_answers_supported_question_with_citations`), `backend/tests/test_citations.py`

Default generator is extractive (span + lead), not a large abstractive LM, unless OpenAI/Ollama is configured.

## Claim extraction

**Working** — After a draft answer, claims are extracted (LLM JSON or deterministic split) and attached to the result.

- `backend/app/pipelines/runner.py` — `run` calls `extract_claims` when `claim_verification` or `hallucination_detection` is on (Enhanced: both true)
- `backend/app/verification/claim_extraction.py` — `extract_claims`
- `backend/app/services/llm/extractive_engine.py` — `extract_claims` (fallback splitter)
- Tests: `backend/tests/test_verification.py` (`test_claim_extraction_and_verification`), `backend/tests/test_pipelines.py`

## Claim verification

**Working** — Each claim is scored against retrieved sentences (lexical / polarity / numeric, not NLI) and statuses are returned to the API.

- `backend/app/pipelines/runner.py` — `_verify_and_correct`
- `backend/app/verification/claim_verifier.py` — `ClaimVerifier.verify`, `_verify_one`
- `backend/app/pipelines/runner.py` — `enhanced_config` sets `claim_verification=True`
- Tests: `backend/tests/test_verification.py`

Not a trained NLI model; still wired and used on the product Ask path.

## Self-correction / retry loop

**Working** — Unsupported important claims trigger revise + re-verify up to `max_correction_loops`, then unsupported sentences can be dropped.

- `backend/app/pipelines/runner.py` — `_verify_and_correct` (while `self_correction`), `_revise`, `_enforce_supported_answer`
- `backend/app/pipelines/common.py` — `drop_unsupported_sentences`
- `backend/app/config.py` — `max_correction_loops` (default 2)
- Adaptive **retrieval** retry (deeper k / rewrite) is separate: `PipelineRunner.run` + `backend/app/retrieval/adaptive.py` (`AdaptiveController.plan`)
- Tests exercise Enhanced abstain/correct behaviour in `backend/tests/test_pipelines.py`; verify-only asserts `correction_loops == 0`

## Frontend UI for the above

**Partial** — Library + Ask are wired to ingest and the full Enhanced pipeline; embeddings and Pinecone have no UI; retry is visible only as trace events.

| Stage | UI | Wiring |
| --- | --- | --- |
| PDF ingest | Library upload, Find papers import | `frontend/src/App.jsx` `Library`, `FindPapers`; `frontend/src/api.js` `upload`, `importPaper` → `/documents/upload`, `/papers/import` |
| Embeddings | None | Backend only |
| Pinecone | None | Backend env only |
| Retrieval / answer / citations | Ask | `Ask` → `api.query` → `POST /query`; `AnswerView`, `AnswerText` (clickable `[n]`), evidence cards |
| Claims / verification | Ask | `verificationBadge`, claims list, hallucination flags |
| Self-correction | Trace only | `include_trace: true`; `Check trace` lists `correction` / `verify` events — no retry button |

Product Ask ignores any `pipeline` field and always uses Enhanced (`routes.query`).
