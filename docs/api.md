# HTTP API

Base URL: `http://127.0.0.1:8000/api`  
OpenAPI: `http://127.0.0.1:8000/docs`

All successful bodies are JSON. Errors use FastAPI's `{ "detail": ... }` shape. The frontend never receives provider API keys.

When `SELFRAG_AUTH_REQUIRED` is true (the product default), product routes need a session cookie from login or signup. Pytest turns this off. `GET /health` and `GET /config` stay public.

## Auth

### `POST /auth/signup`

```json
{ "email": "you@example.com", "password": "at-least-8", "name": "Optional" }
```

Creates an account, sets an HttpOnly `selfrag_session` cookie, and returns `{ user_id, email, name }`. Duplicate email → 409.

### `POST /auth/login`

Same email/password body. Wrong credentials → 401.

### `POST /auth/logout`

Clears the session cookie.

### `GET /auth/me`

Current user, or 401.

## System

### `GET /health`

Liveness plus which LLM / embedding / vector-store providers are active and which are available.

### `GET /config`

Public runtime knobs (thresholds, k, loop caps). No secrets.

## Documents

### `POST /documents/upload`

`multipart/form-data` field `files`. Parses, chunks, embeds, rebuilds BM25.

### `GET /documents`

Indexed documents, chunk counts, active providers.

### `GET /documents/extractions`

Per-paper structured records (`objective`, `method`, `dataset`, `metric`, `result`, `limitation`) extracted at ingest. Independent of Ask. Each field includes `low_confidence` when NLI/overlap checks fail; values are not dropped.

### `POST /documents/extractions/refresh`

Re-run extraction for every library paper.

### `GET /documents/{document_id}/extraction`

### `POST /documents/{document_id}/extract`

Re-run extraction for one paper.

### `GET /documents/clusters`

Theme groups of library papers (agglomerative cosine clustering of document embeddings). Each cluster has a label, keywords, and `document_ids` for scoped Ask.

### `DELETE /documents/{document_id}`

### `POST /documents/demo`

Reload the bundled research corpus. Used by pytest and `/evaluate`, not by the product UI.

## Query

Product Ask always runs **Enhanced Self-RAG**. `pipeline` on `/query` is ignored if sent.

### `POST /query`

```json
{
  "query": "What is the main advantage of dropout?",
  "top_k": 5,
  "document_ids": null,
  "evidence_threshold": null,
  "include_trace": true
}
```

Response is a `PipelineResult`: answer, status, evidence[], claims[], contradictions[], confidence, hallucination, trace[], metrics.

### `GET /query/history/{id}`

Full saved result for a recent question (in-memory session history).

### `POST /query/compare`

Harness-only: runs selected pipelines on one question.

### `GET /query/{id}/trace`

Structured system events for a recent query.

### `GET /query/recent`

## Evaluation (harness, not the product UI)

### `POST /evaluate`

```json
{
  "pipelines": ["traditional_rag", "self_rag", "enhanced_self_rag"],
  "include_ablation": true,
  "include_headline": false,
  "limit": null,
  "categories": null,
  "k": 5,
  "persist": true
}
```

Returns a full `EvaluationRun` (several seconds on the demo set). Set `include_headline` to score the four-system table (no-RAG, Traditional RAG, RAG+verify, Enhanced) instead of the three research pipelines. Product Ask is unchanged.

### `GET /evaluation/results`

Latest persisted run, headline table, and per-metric comparison rows for the dashboard.

### `GET /evaluation/runs`

### `GET /evaluation/benchmark`

Gold questions.

### `GET /evaluation/demo-cases`

Curated questions for a live demo.
