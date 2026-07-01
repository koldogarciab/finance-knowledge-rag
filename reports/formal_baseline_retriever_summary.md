# Formal Baseline Retriever Evaluation

## Evaluation dataset

- Formal queries: 30
- File formats: 6 CSV, 6 DOCX, 6 JSON, 6 Markdown and 6 PDF
- Relevant chunks: 31
- Dataset SHA-256: `894144e2eee345fac4efc80474e6cd970d3912c98b7cd78c1d4c3f8adaff18d6`
- Primary mode: global retrieval without metadata filters
- Secondary mode: retrieval with the case metadata filters

## Results

| Metric | Global | Filtered |
|---|---:|---:|
| MRR | 0.7117 | 0.8486 |
| Hit@1 | 66.67% | 76.67% |
| Hit@3 | 70.00% | 93.33% |
| Hit@5 | 73.33% | 93.33% |
| Hit@10 | 90.00% | 100.00% |
| Mean rank | 18.3 | 1.63 |
| Median rank | 1.0 | 1.0 |

## Main findings

- DOCX policies and JSON KPI definitions are generally retrieved at rank 1.
- PDF retrieval contains the largest outliers, especially where generic financial language competes with repetitive CSV records.
- CSV summaries remain difficult when several months and departments use nearly identical templates.
- Metadata filters materially improve ranking, raising Hit@3 from 70.00% to 93.33%.
- The global median rank is 1, but a small number of severe outliers increase the mean rank to 18.3.

## Hardest global cases

1. `formal_pdf_march_opex_drivers` ? global rank 220, filtered rank 1.
2. `formal_pdf_expense_variance_convention` ? global rank 193, filtered rank 2.
3. `formal_pdf_high_severity_risk` ? global rank 75, filtered rank 2.
4. `formal_csv_supply_chain_december` ? global rank 8, filtered rank 8.
5. `formal_csv_marketing_january` ? global rank 7, filtered rank 1.
6. `formal_markdown_closing_priorities` ? global rank 7, filtered rank 3.
7. `formal_csv_it_august` ? global rank 6, filtered rank 2.
8. `formal_csv_retail_operations_october` ? global rank 6, filtered rank 6.
9. `formal_markdown_forecast_action_deadlines` ? global rank 4, filtered rank 1.
10. `formal_pdf_q4_priorities_and_owners` ? global rank 3, filtered rank 3.

## Benchmark policy

- The 30-case formal dataset is frozen and must not be edited while developing the improved retriever.
- Retriever parameters should be developed using the separate 10-case acceptance set.
- Final baseline and improved-retriever comparisons must use the same dataset hash.
