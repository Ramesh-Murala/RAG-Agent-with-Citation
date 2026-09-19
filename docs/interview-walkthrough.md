# Interview walkthrough

## 60-second explanation

I built this project to separate three questions that RAG demos often combine: did retrieval find the right evidence, did the model return valid citations, and does the evidence actually support the answer? The repository benchmarks five offline retrieval configurations on 72 labeled questions. Hybrid retrieval with a transparent reranker reached 88.3% Recall@3 on the synthetic fixture, compared with 80.0% for TF-IDF. After retrieval, the generation layer forces a Pydantic schema and validates each citation's chunk ID, source, and verbatim quote. Invalid output receives bounded corrective retries; missing context causes abstention. I also include a counterexample proving that citation integrity alone cannot establish entailment.

## Design decisions

**Why keep TF-IDF?** It is a fast, deterministic baseline. An embedding system is only persuasive when it beats a simpler method on the same labeled data.

**Why BM25 plus dense LSA?** BM25 handles rare terms and document length well. LSA adds a low-dimensional signal using only local dependencies. Reciprocal-rank fusion combines their ordering without pretending their raw scores are calibrated alike.

**Why a simple reranker?** It gives an inspectable baseline for candidate reranking. The README calls it a heuristic. A production experiment would compare it with a pinned cross-encoder on a held-out set.

**Why validate quotes?** Schema validation alone only proves the output has fields. Checking source, chunk ID, and verbatim text closes common fabricated-citation failure paths.

**Why abstain?** Returning a clear unsupported result is safer than generating when retrieval has no matching evidence. A production threshold would require calibration against false accepts and false rejects.

## Evaluation story

- Dataset: 20 synthetic policy chunks, 60 answerable questions, 12 unanswerable questions.
- Query mix: direct wording and paraphrases, with one relevant chunk per answerable question.
- Metrics: Top-1 accuracy, Recall@3, MRR@3, per-query-kind breakdown, and local latency.
- Result: hybrid + rerank reached 88.3% Recall@3 and 77.2% MRR@3.
- Interpretation: the approach improved candidate coverage on this fixture. The data size and synthetic construction limit generalization.

## Failure modes I would discuss

1. A correct quote can be attached to a false or contradictory answer.
2. Corpus-trained LSA cannot understand unseen language like a pretrained embedding model.
3. Raw retrieval scores are not comparable across strategies.
4. Prompt instructions reduce risk but do not guarantee prompt-injection resistance.
5. Provider timeouts, cost, and token usage need separate live-model evaluation.
6. The in-memory index does not address persistence, access control, or multi-tenant isolation.

## Production extension

I would version documents and embeddings, filter retrieval by tenant and authorization metadata, compare a pinned embedding model and cross-encoder against these baselines, calibrate abstention on held-out data, evaluate claim-to-evidence entailment, and add tracing for retrieval latency, generation latency, token use, validation retries, and fallbacks. I would keep the offline fixture in CI as a regression gate.

## Demo sequence

1. Run the benchmark summary and explain why Recall@3 matters for downstream generation.
2. Launch `demo_app.py` and search for “How soon must missing equipment be reported?”
3. Switch among TF-IDF, BM25, dense LSA, and hybrid + rerank.
4. Open the returned source and score fields to show inspectability.
5. Run the test for a fabricated quote and the counterexample where a real quote accompanies a false answer.
6. Close with the next experiment: neural embeddings and an entailment evaluator on held-out data.
