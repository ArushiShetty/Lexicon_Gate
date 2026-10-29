# Methodology

## Why Traditional RAG fails

The pipeline `query → top-k → generate` has no notion of *evidence sufficiency*. If the retriever returns Saturn when you asked about dropout, the model still writes a paragraph and often cites those chunks. Fluency is not faithfulness.

Failure modes this project targets:

1. Generating when no relevant passage exists.
2. Missing lexical identifiers that dense retrieval ranks poorly.
3. Collapsing two contradictory sources into one confident sentence.
4. Asserting numbers and entities that never appear in the evidence.
5. Reporting a made-up confidence percentage.

## Why standard Self-RAG is not enough

Self-RAG adds a retrieval decision and coarse reflection tokens. That helps on questions that need no retrieval and on continuations that are obviously unsupported. It does not:

- score the retrieved set *before* generation,
- fuse lexical and dense rankings,
- check each factual claim,
- surface source disagreement,
- derive confidence from evidence.

Those are the increments evaluated here.

## Experimental design

- **Shared generator.** All systems call the same `LLMProvider`. Default: deterministic extractive engine, so a laptop with no API key still produces reproducible numbers.
- **Shared corpus and questions.** The harness never gives Enhanced Self-RAG a different index or a different gold set.
- **Ablation.** Baseline RAG → Self-RAG → + hybrid → + evidence gate → + claim verification → full system. Each row flips one concern.
- **Metrics** are defined in `docs/evaluation.md`. They are functions of (prediction, gold), not of which pipeline produced the prediction.
- **Statistics.** Paired t-tests and Cohen's d on per-question scores. A non-significant accuracy delta on 18 items is reported as such; a significant hallucination drop is also reported as such. We do not round either into a marketing number.

## Demo behaviours (what to show)

| # | Question | Expected Enhanced behaviour |
| --- | --- | --- |
| 1 | “What is the main advantage of dropout?” | Answer + citations from the handbook. |
| 2 | Hybrid retrieval vs Self-RAG gaps | Combine survey + retrieval notes. |
| 3 | “What k does RRF-7F3 use?” | Hybrid / rewrite recovers `k = 60`. |
| 4 | “How does dropout affect Titan's nitrogen atmosphere?” | Refuse. Traditional RAG typically does not. |
| 5 | “Does dropout reduce overfitting?” | Report handbook vs memo/blog conflict. |
| 6 | Comparison mode | Side-by-side answers, scores, latency. |

## Threats to validity

- Small, author-constructed benchmark (content validity high, external validity limited).
- Offline generator under-states the damage a large LM can do when the gate is off, and under-states how much a large LM can recover when the gate is on.
- Gold relevant-chunk labels are lexical-overlap heuristics over named documents, not expert span annotations.
