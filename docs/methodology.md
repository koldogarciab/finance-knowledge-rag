# Methodology and design decisions

## 1. Synthetic finance corpus

A synthetic corpus was used to make the project reproducible and safe to publish. The documents represent realistic finance and FP&A workflows without exposing confidential employer data.

The corpus contains monthly performance data, management commentary, policies, KPI definitions and meeting decisions across five file formats.

## 2. Common extraction schema

Each format is parsed into a common record structure containing:

- stable identifiers;
- source path and file type;
- document title;
- extracted text;
- format-specific metadata;
- citation metadata.

This makes downstream chunking and retrieval format-agnostic.

## 3. Token-aware chunking

Chunking balances two competing needs:

- enough context to preserve financial meaning;
- enough granularity to retrieve the precise fact requested.

The final corpus contains 316 chunks from 305 extracted records.

## 4. Dense baseline

The first retriever used MiniLM embeddings only. This established a measurable baseline and exposed weaknesses on exact month, department, policy and finance-term matching.

## 5. Frozen benchmark

The formal retriever benchmark contains 30 questions, with six questions per source format.

The benchmark was frozen before selecting the final hybrid parameters. A separate 10-question development set was used for tuning.

This separation is important because tuning directly against the formal set would inflate the reported result.

## 6. Hybrid retrieval

BM25 was added to complement semantic similarity. Weighted reciprocal rank fusion was chosen because it combines rankings without requiring the raw dense and lexical scores to share a scale.

The final weights favour lexical search:

```text
dense = 0.20
lexical = 0.80
```

This was empirically selected on the development set and then evaluated once on the frozen benchmark.

## 7. Grounded generation

The local model is not allowed to browse the entire corpus. It receives only the evidence approved by the grounding layer.

The generation pipeline supports:

- normal Ollama generation;
- repair generation;
- guarded extractive fallback;
- pre-generation abstention.

## 8. Evaluation strategy

### Retriever evaluation

Metrics:

- Mean Reciprocal Rank;
- Hit@1, Hit@3, Hit@5 and Hit@10;
- mean, median, best and worst rank;
- case-level wins, ties and losses;
- results by file type.

### Generation evaluation

The formal generation set contains:

- 30 supported questions;
- 10 unsupported questions.

Automated metrics include:

- answer and abstention counts;
- citation validity;
- expected evidence retrieved, sent and cited;
- semantic similarity;
- reference-token recall;
- numeric-fact recall;
- grounded-claim rate;
- latency;
- quality flags.

The seven automated-review cases were then manually assessed for correctness, completeness, groundedness and citation quality.

## 9. Iterative failure analysis

Generation quality was improved through explicit diagnosis rather than blind prompt editing.

Examples of identified failure modes:

- month and year mismatch;
- aggregated CSV row not selected;
- relevant source retrieved but not sent to the model;
- invented or missing citations;
- favourable/unfavourable polarity errors;
- threshold-based approvals omitted;
- generic tax references incorrectly treated as a requested tax-rate metric;
- citation markers parsed as finance numbers;
- generic “led by” language confused with a discussion lead.

Each correction was paired with a regression test.

## 10. Stopping criterion

Generation v5.3 was frozen after:

- 30/30 supported questions answered;
- 10/10 unsupported questions correctly rejected;
- 100% evidence retrieval, model supply and citation;
- 0 automatic failures;
- 29 manually accepted answers and one conservative partial answer.

Further benchmark-specific tuning was deliberately avoided to reduce overfitting.
