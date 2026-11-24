### Retrieval (n=33 questions, top-3 documents)

| Mode | hit@3 | MRR |
|---|---|---|
| bm25 | 1.000 | 0.960 |
| dense | 0.939 | 0.934 |
| hybrid | 0.970 | 0.942 |

Hybrid minus BM25 MRR: -0.018 (95% paired-bootstrap CI -0.051 to +0.000).

### Agent (n=13 questions, offline rule-based planner)

| Metric | Value |
|---|---|
| Correct tool sequence | 1.000 |
| Answer contains expected facts | 1.000 |
| Run completed (status ok) | 1.000 |
