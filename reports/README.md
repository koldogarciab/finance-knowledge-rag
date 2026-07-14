# Evaluation reports

This directory contains the evaluation outputs and audit artefacts generated during the development of Finance Knowledge RAG.

## Canonical final outputs

These files contain the final results used in the project documentation and presentation:

- `formal_hybrid_retriever_evaluation.json`  
  Final evaluation of the hybrid MiniLM and BM25 retriever.

- `formal_rag_generation_evaluation.json`  
  Structured results of the final RAG generation benchmark.

- `formal_rag_generation_evaluation.md`  
  Human-readable summary of the final generation evaluation.

- `formal_rag_generation_manual_review.md`  
  Manual review of answer correctness, grounding and abstention behaviour.

## Intermediate and audit artefacts

These files preserve the detailed execution and review trail:

- `formal_rag_generation_raw.jsonl`  
  Raw answers generated during the formal benchmark.

- `formal_rag_generation_review.csv`  
  Initial review template.

- `formal_rag_generation_review_completed.csv`  
  Completed manual review records.

## Main reported results

- 30 of 30 supported questions answered with retrieved evidence
- 10 of 10 unsupported questions correctly rejected
- 95.44% mean groundedness
- Hybrid retriever MRR: 0.8107
- Hybrid retriever Hit@5: 90%

Intermediate files are retained for reproducibility and auditability. The canonical final outputs listed above should be used when quoting project results.
