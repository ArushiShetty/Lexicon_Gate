# Evaluation

The harness lives in `backend/app/evaluation/`. It loads the gold set, runs each selected pipeline, scores every question, aggregates, tests significance, and optionally runs the ablation suite. Results are written to `backend/data/results/<run_id>.json`.

The **Ask** UI always runs Enhanced Self-RAG. The four-system headline table is produced only by this harness (pytest or `POST /api/evaluate`), not by the product UI.

## How to run

```bash
cd backend
.venv/bin/python -m pytest tests/test_evaluation.py -q
```

Optional HTTP (not shown in Ask):

```bash
# Three-system research run (Traditional / Self-RAG / Enhanced) plus ablation
curl -X POST http://127.0.0.1:8000/api/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"include_ablation": true, "persist": true}'

# Four-system headline table (no-RAG, basic RAG, verify-only, full)
curl -X POST http://127.0.0.1:8000/api/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"include_headline": true, "persist": true}'
```

`GET /api/evaluation/results` returns the latest persisted run, `table` (wide comparison), and `headline_table` (compact viva rows).

## Dataset

`enhanced-self-rag-demo-v1` — 18 questions over 7 documents.

| Category | n | Intent |
| --- | --- | --- |
| easy | 8 | Single-source facts |
| multi-hop | 2 | Combine sections |
| multi-document | 2 | Combine files |
| ambiguous | 1 | Unresolved pronoun |
| unanswerable | 3 | Must abstain |
| conflicting | 2 | Must report disagreement |

Each item has a reference answer, keypoints, relevant document names, and flags `should_abstain` / `has_conflict`.

## Headline four-system table

`EvaluationHarness.run_headline()` scores the gold set on four configs. Numbers are computed; they are not the example 82 / 88 / 94 figures from a brief.

| System | Config | What it does |
| --- | --- | --- |
| Base LLM (no retrieval) | `no_rag_config()` | Never retrieves. Hosted LLMs get an ungrounded prompt; the default extractive provider has no parametric knowledge, so it typically abstains or fails closed. |
| Basic RAG | `traditional_config()` | Dense retrieve + generate, no verification or retry. |
| RAG + verification (no retry) | `verify_only_config()` | Retrieve, label claims, no self-correction, no evidence gate, no query rewrite. |
| Full system | `enhanced_config()` | Hybrid retrieve, evidence gate, claim verify, self-correct. |

Compact columns (`headline_table`):

- **Hallucination rate** — unanswerable-but-answered, missed conflict, or unsupported-claim rate ≥ 0.34 on an answerable item
- **Citation accuracy** — cited passages overlap the answer; **no-RAG is 0** when the answer is not an abstention (there is no evidence to cite). Abstentions score 1 on this metric.
- **Answer accuracy** — keypoint recall (stand-in for average relevance); unanswerable items score 1 iff the system abstained
- **Faithfulness** — fraction of generated claims supported by retrieved evidence

On the **extractive** default provider, do not expect no-RAG to have the **highest** hallucination rate. It cannot invent corpus facts the way Gemini can; it mostly refuses. Hosted parametric models are the setting where a no-retrieval row typically hallucinates more.

On this gold set, Enhanced Self-RAG is expected to have hallucination rate **≤** Traditional RAG.

## Retrieval metrics

Computed on answerable questions (unanswerable items would otherwise zero-out precision for every system that retrieves at all, which is the correct first step even when the eventual action is to abstain).

- Precision@K, Recall@K, MRR, nDCG@K
- Context relevance (query-term overlap of retrieved texts)

## Generation metrics

- **Answer correctness** — keypoint recall with a small exact-match bonus. Unanswerable items score 1 iff the system abstained.
- **Faithfulness** — fraction of generated claims supported by retrieved evidence (independent of completeness).
- **Citation accuracy** — cited passages actually overlap the answer.
- **Claim support rate**

## Hallucination metrics

A question counts as hallucinated when:

- it is unanswerable and the system answered, or
- it is conflicting and the system did not report the conflict, or
- unsupported-claim rate ≥ 0.34 on an answerable item.

Also reported: unsupported-claim rate, contradiction rate (missed conflicts), abstention accuracy (F1 of refuse-on-unanswerable vs over-refusal).

## System metrics

Latency, retrieval attempts, correction loops, token estimates, mean confidence, evidence coverage. Confidence calibration: ECE over five bins of (confidence, correctness).

## Reading a local run

On the deterministic extractive provider, a typical laptop **three-system** run looks like this (re-run to replace these figures — they are not baked into the UI):

| Metric | Traditional RAG | Self-RAG | Enhanced Self-RAG |
| --- | --- | --- | --- |
| Answer accuracy | 0.57 | 0.57 | **0.65** |
| Faithfulness | 0.93 | ~0.93 | **0.98** |
| Hallucination rate | 0.28 | 0.28 | **0.00** |
| Abstention accuracy | 0.00 | 0.00 | **0.86** |
| Conflict detection | 0.00 | 0.00 | **1.00** |
| Precision@K | 0.52 | 0.52 | **0.61** |

Paired tests on this 18-item set: hallucination drop vs Traditional RAG is significant (p ≈ 0.02); the accuracy lift is not (p ≈ 0.49). That is an honest outcome on a small sample, and it is exactly what a persisted JSON run will show.

## Ablation (same three-system run)

The evidence gate is the component that moves hallucination. Hybrid retrieval alone does not; claim verification without a gate helps less than the gate; the full system is the only row at hallucination = 0.

The JSON on disk is the source of truth.
