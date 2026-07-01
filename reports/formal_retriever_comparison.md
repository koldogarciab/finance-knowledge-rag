# Baseline vs Hybrid Retriever

## Frozen evaluation

- Formal queries: 30
- Corpus chunks: 316
- Dataset SHA-256: `894144e2eee345fac4efc80474e6cd970d3912c98b7cd78c1d4c3f8adaff18d6`
- The formal benchmark was not used for parameter tuning.
- Parameters were selected using the separate 10-case development set.

## Selected hybrid configuration

- Dense weight: 0.20
- Lexical BM25 weight: 0.80
- Weighted RRF k: 10
- BM25 k1: 1.5
- BM25 b: 0.75

## Global retrieval

| Metric | Dense baseline | Hybrid | Change |
|---|---:|---:|---:|
| MRR | 0.7117 | 0.8107 | +0.0990 |
| Hit@1 | 66.67% | 76.67% | +10.00% |
| Hit@3 | 70.00% | 80.00% | +10.00% |
| Hit@5 | 73.33% | 90.00% | +16.67% |
| Hit@10 | 90.00% | 90.00% | +0.00% |
| Mean rank | 18.3 | 3.97 | -14.33 |
| Median rank | 1.0 | 1.0 | +0.00 |

## Retrieval with metadata filters

| Metric | Dense baseline | Hybrid | Change |
|---|---:|---:|---:|
| MRR | 0.8486 | 0.9031 | +0.0545 |
| Hit@1 | 76.67% | 86.67% | +10.00% |
| Hit@3 | 93.33% | 90.00% | -3.33% |
| Hit@5 | 93.33% | 96.67% | +3.34% |
| Hit@10 | 100.00% | 100.00% | +0.00% |

## Case-level comparison

- Global wins: 7
- Global ties: 18
- Global losses: 5
- Filtered wins: 6
- Filtered ties: 22
- Filtered losses: 2

## Largest global rank improvements

- `formal_pdf_march_opex_drivers`: 220 → 1 (improvement 219).
- `formal_pdf_expense_variance_convention`: 193 → 1 (improvement 192).
- `formal_pdf_high_severity_risk`: 75 → 1 (improvement 74).
- `formal_markdown_closing_priorities`: 7 → 1 (improvement 6).
- `formal_csv_retail_operations_october`: 6 → 4 (improvement 2).
- `formal_markdown_forecast_action_deadlines`: 4 → 2 (improvement 2).
- `formal_pdf_q4_priorities_and_owners`: 3 → 1 (improvement 2).
- `formal_csv_finance_september`: 1 → 1 (improvement 0).
- `formal_csv_people_culture_july`: 1 → 1 (improvement 0).
- `formal_docx_accrual_policy`: 1 → 1 (improvement 0).

## Global regressions

- `formal_csv_marketing_january`: 7 → 34 (change -27).
- `formal_csv_supply_chain_december`: 8 → 30 (change -22).
- `formal_csv_it_august`: 6 → 17 (change -11).
- `formal_markdown_freight_forecast_decision`: 1 → 5 (change -4).
- `formal_markdown_ecommerce_performance`: 1 → 4 (change -3).

## Results by file type

| Type | Baseline MRR | Hybrid MRR | Baseline Hit@5 | Hybrid Hit@5 |
|---|---:|---:|---:|---:|
| csv | 0.4335 | 0.3953 | 33.33% | 50.00% |
| docx | 1.0000 | 1.0000 | 100.00% | 100.00% |
| json | 1.0000 | 1.0000 | 100.00% | 100.00% |
| markdown | 0.7321 | 0.6583 | 83.33% | 100.00% |
| pdf | 0.3927 | 1.0000 | 50.00% | 100.00% |

## Method

- Dense retrieval uses the existing normalised MiniLM embeddings.
- Lexical retrieval uses an in-memory BM25 index over `retrieval_text`.
- Results are combined using weighted reciprocal rank fusion.
- No corpus, chunk, embedding or formal evaluation case was modified.
