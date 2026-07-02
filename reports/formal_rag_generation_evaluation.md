# Formal Local RAG Generation Evaluation

## Evaluation status

- Completed cases: 40
- Supported cases completed: 30
- Unsupported cases completed: 10
- Model: `qwen3.5:4b`
- Retrieval mode: global hybrid retrieval without metadata filters.
- The frozen 30-case benchmark was not used for retriever tuning.

## Supported questions

| Metric | Result |
|---|---:|
| Answered | 30 / 30 |
| Unexpected abstentions | 0 |
| Valid citations | 1.0 |
| Expected chunk retrieved | 1.0 |
| Expected chunk supplied to model | 1.0 |
| Expected chunk cited | 1.0 |
| Mean semantic similarity | 0.8617 |
| Mean reference-token recall | 0.8311 |
| Mean numeric-fact recall | 0.9627 |
| Mean grounded-claim rate | 0.9544 |

Generation modes:

- `context_extractive_fallback`: 1
- `ollama`: 28
- `ollama_repair`: 1

Automated quality flags:

- `pass`: 23
- `review`: 7

## Unsupported questions

| Metric | Result |
|---|---:|
| Correct abstentions | 10 / 10 |
| Incorrect answers | 0 |
| Abstentions before generation | 10 |
| Provider calls avoided | 10 |
| Valid citation state | 1.0 |

## Latency

| Metric | Seconds |
|---|---:|
| Mean wall time | 19.403 |
| Median wall time | 21.455 |
| 95th percentile wall time | 35.105 |
| Mean Ollama provider time | 21.891 |

## Manual review

The CSV review file contains blank columns for human scoring of correctness, completeness, groundedness and citation quality. Automated similarity and overlap metrics are diagnostics, not substitutes for human judgement.

## Files

- Raw checkpoint: `reports\formal_rag_generation_raw.jsonl`
- JSON report: `reports\formal_rag_generation_evaluation.json`
- Review CSV: `reports\formal_rag_generation_review.csv`
