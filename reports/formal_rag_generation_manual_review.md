# Formal RAG Generation Evaluation — Final Manual Review

## Final automated results

- Supported questions answered: **30/30**
- Unexpected abstentions: **0**
- Valid citation rate: **100%**
- Expected chunk retrieved / sent to model / cited: **100% / 100% / 100%**
- Mean semantic similarity: **0.8617**
- Mean reference-token recall: **0.8311**
- Mean numeric-fact recall: **0.9627**
- Mean grounded-claim rate: **0.9544**
- Automated quality flags: **23 pass, 7 review, 0 fail**
- Unsupported questions correctly abstained: **10/10**
- False answers on unsupported questions: **0**
- Provider calls avoided on unsupported questions: **10/10**

## Manual review of the seven flagged cases

| Case | Verdict | Correctness | Completeness | Groundedness | Citation quality | Conclusion |
|---|---|---:|---:|---:|---:|---|
| `formal_csv_it_august` | ACCEPT_WITH_FORMATTING_NOTE | 2/2 | 2/2 | 2/2 | 1/2 | Substantively correct and complete: budget AUD 390,000, actual AUD 382,200, favourable variance AUD 7,800 / 2.00%, forecast AUD 383,760 and AUD 1,560 below forecast. The first citation is placed after the sentence-ending full stop, so the automated claim splitter did not attach it to the first claim. Extra category and manager details were not requested. |
| `formal_csv_supply_chain_december` | ACCEPT | 2/2 | 2/2 | 2/2 | 2/2 | Correct comparison against budget and forecast, with the responsible manager identified. The 6.00% variance from the reference answer is omitted, but the question did not explicitly require the percentage. |
| `formal_docx_capex_approval_300k` | ACCEPT | 2/2 | 2/2 | 2/2 | 2/2 | Correct and complete: Finance Director, Chief Executive Officer and Board approval, plus all required business-case components. AUD 300,000 is supplied by the question, which explains the automated numeric-grounding warning. |
| `formal_json_capex_utilisation` | ACCEPT | 2/2 | 2/2 | 2/2 | 2/2 | Formula, 90%-100% target range and KPI owner are correct. The answer expresses the target as 0.9 to 1.0, which is numerically equivalent but less natural for a business user. |
| `formal_json_net_working_capital` | PARTIAL | 1/2 | 1/2 | 2/2 | 2/2 | Formula and AUD 18.5 million limit are correct and grounded. The answer does not fully address why the preferred direction is context dependent; it should explain that excessively high and excessively low working capital can signal different operational or liquidity issues. This is a conservative omission rather than a hallucination. |
| `formal_markdown_closing_priorities` | ACCEPT | 2/2 | 2/2 | 2/2 | 2/2 | All four priorities are reproduced correctly. The automated review is a semantic-similarity false positive caused by minor wording differences. |
| `formal_pdf_march_opex_drivers` | ACCEPT | 2/2 | 2/2 | 2/2 | 2/2 | Correctly states the AUD 393,000 total overspend and the Marketing and Supply Chain contributions of AUD 185,000 and AUD 150,000. The omitted combined AUD 335,000 is derivable and was not explicitly requested. |

## Final assessment

- **Six of the seven automated-review cases are accepted as substantively correct.**
- **One case is partially complete:** `formal_json_net_working_capital` answers the formula and limit correctly but does not provide the requested operational/liquidity rationale.
- Combining the 23 automated passes with the six manually accepted cases gives **29 accepted supported answers out of 30**, with **one conservative partial answer and no materially incorrect or hallucinated answer**.
- All ten unsupported questions were rejected before model generation, with no false answers.
- Generation v5.3 is suitable as the final evaluated version. The net-working-capital omission should be documented as a known limitation rather than addressed through another benchmark-specific rule.

## Recommendation

Freeze Generation v5.3, retain the benchmark and final reports, and proceed to the interface and final project documentation. Avoid further benchmark-specific prompt or heuristic tuning, which would risk overfitting the frozen evaluation set.
