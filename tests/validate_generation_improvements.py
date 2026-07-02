from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_retrieve import expand_lexical_query
from src.ollama_rag import (
    OllamaGroundedRAG,
    build_context_extractive_fallback,
    build_evidence_context_bundle,
    build_ollama_messages,
    remap_citations,
    validate_financial_consistency,
    validate_requested_completeness,
    validate_sentence_citations,
)
from src.rag_answer import (
    ABSTENTION_MESSAGE,
    DeterministicGroundedGenerator,
    requested_intents,
    specific_metric_phrases,
)


def source(
    number: int,
    chunk_id: str,
    content: str,
    *,
    citation: str | None = None,
    file_type: str = "markdown",
    granularity: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    if granularity:
        metadata["granularity"] = granularity

    return {
        "source_number": number,
        "rank": number,
        "score": 0.1,
        "chunk_id": chunk_id,
        "record_id": f"record-{number}",
        "document_id": "TEST_DOC",
        "document_title": "Synthetic test source",
        "document_name": "synthetic",
        "file_type": file_type,
        "citation": citation or f"test source {number}",
        "citation_locator": citation or f"test source {number}",
        "source_path": "tests/synthetic",
        "content": content,
        "retrieval_text": content,
        "hybrid_score": 0.1,
        "dense_score": 0.5,
        "lexical_score": 1.0,
        "dense_rank": number,
        "lexical_rank": number,
        "metadata": metadata,
        "granularity": granularity,
    }


def bundle(
    query: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "query": query,
        "filters": {},
        "source_count": len(sources),
        "context_character_count": 0,
        "context": "",
        "sources": sources,
    }


class StubRetriever:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.results[:top_k]


class SequenceAdapter:
    def __init__(self, responses: list[str]) -> None:
        self.model = "qwen3.5:4b"
        self.responses = list(responses)
        self.call_count = 0

    def chat(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        self.call_count += 1

        if not self.responses:
            raise AssertionError("No stub response remains.")

        return {
            "content": self.responses.pop(0),
            "model": self.model,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 20,
            "total_duration": 1_000_000,
        }


def main() -> None:
    assert "2025-08" in expand_lexical_query(
        "Information Technology in August 2025"
    )
    assert "2026-01" in expand_lexical_query(
        "Marketing in January 2026"
    )

    accrual_query = (
        "When must an accrual be recognised, what is the minimum "
        "individual threshold, and how must it be supported and "
        "subsequently reviewed?"
    )

    accrual_intents = requested_intents(accrual_query)

    assert "condition" in accrual_intents
    assert "documentation" in accrual_intents
    assert "deadline" not in accrual_intents

    accrual_content = (
        "An accrual is required when goods or services have been "
        "received but the supplier invoice has not been recorded. "
        "The minimum individual threshold is AUD 5,000. It must be "
        "supported by a purchase order, contract, supplier estimate "
        "or another reasonable calculation, reviewed monthly and "
        "reversed when the related invoice is recorded."
    )

    generator = DeterministicGroundedGenerator()

    accrual_result = generator.generate(
        accrual_query,
        bundle(
            accrual_query,
            [source(1, "accrual:01", accrual_content)],
        ),
    )

    assert accrual_result["abstained"] is False
    assert accrual_result["selected_evidence"]

    reconciliation_query = (
        "When must balance-sheet reconciliations be prepared and reviewed, "
        "and what is required for reconciling items older than 60 days?"
    )
    reconciliation_content = (
        "Preparers must complete balance-sheet reconciliations by business "
        "day 7 and reviewers by business day 10. Reconciling items older "
        "than 60 days must have an owner and a documented resolution date."
    )
    reconciliation_result = generator.generate(
        reconciliation_query,
        bundle(
            reconciliation_query,
            [
                source(
                    1,
                    "DOC_DOCX_001:section:10:reconciliation:chunk:01",
                    reconciliation_content,
                    file_type="docx",
                    granularity="section",
                )
            ],
        ),
    )
    assert reconciliation_result["abstained"] is False
    assert reconciliation_result["selected_evidence"]

    model_bundle = build_evidence_context_bundle(
        context_bundle=bundle(
            accrual_query,
            [source(1, "accrual:01", accrual_content)],
        ),
        selected_evidence=accrual_result["selected_evidence"],
        selected_source_numbers=accrual_result[
            "selected_source_numbers"
        ],
    )

    assert "AUD 5,000" in model_bundle["context"]
    assert "reviewed monthly" in model_bundle["context"]
    assert model_bundle["source_count"] == 1

    csv_query = (
        "For Information Technology in August 2025, how far was actual "
        "expenditure below budget and below the pre-close forecast?"
    )
    csv_sources = [
        source(
            1,
            "DOC_CSV_001:row:BVA_113:chunk:01",
            (
                "2025-08 Information Technology software. Budget AUD "
                "117,000, actual AUD 114,660 and forecast AUD 115,128."
            ),
            citation=(
                "monthly_budget_vs_actual_fy2026.csv - 2025-08, "
                "Information Technology, software"
            ),
            file_type="csv",
            granularity="account_category_row",
        ),
        source(
            2,
            "DOC_CSV_001:row:BVA_114:chunk:01",
            (
                "2025-08 Information Technology licences. Budget AUD "
                "130,000, actual AUD 127,400 and forecast AUD 128,000."
            ),
            citation=(
                "monthly_budget_vs_actual_fy2026.csv - 2025-08, "
                "Information Technology, licences"
            ),
            file_type="csv",
            granularity="account_category_row",
        ),
        source(
            3,
            "DOC_CSV_001:summary:2025-08:information-technology:chunk:01",
            (
                "2025-08 Information Technology department summary. "
                "Budget AUD 390,000, actual AUD 382,200, favourable "
                "variance AUD 7,800 or 2.00%. Pre-close forecast AUD "
                "383,760; actual was AUD 1,560 below forecast."
            ),
            citation=(
                "monthly_budget_vs_actual_fy2026.csv - 2025-08, "
                "Information Technology, department summary"
            ),
            file_type="csv",
            granularity="monthly_department_summary",
        ),
    ]

    csv_result = generator.generate(
        csv_query,
        bundle(csv_query, csv_sources),
    )

    assert csv_result["abstained"] is False
    assert csv_result["selected_source_numbers"] == [3]

    csv_model_bundle = build_evidence_context_bundle(
        context_bundle=bundle(csv_query, csv_sources),
        selected_evidence=csv_result["selected_evidence"],
        selected_source_numbers=csv_result[
            "selected_source_numbers"
        ],
    )

    assert csv_model_bundle["source_count"] == 1
    assert csv_model_bundle["sources"][0]["chunk_id"].startswith(
        "DOC_CSV_001:summary:"
    )
    assert csv_model_bundle["sources"][0]["source_number"] == 1
    assert csv_model_bundle["source_number_map"] == {3: 1}

    it_query = (
        "How did March Information Technology expenditure perform "
        "against budget, and what caused the variance?"
    )
    it_sources = [
        source(
            1,
            "DOC_CSV_001:row:BVA_111:chunk:01",
            (
                "March Information Technology software actual AUD "
                "120,000 and budget AUD 118,000."
            ),
            citation="monthly CSV account row",
            file_type="csv",
        ),
        source(
            2,
            (
                "DOC_MD_001:section:08:"
                "information-technology-expenditure:chunk:01"
            ),
            (
                "March Information Technology expenditure was AUD "
                "45,000 below budget because a planned vendor "
                "implementation milestone was delayed."
            ),
            citation=(
                "fp_and_a_forecast_meeting_2026-04-10.md - section 8: "
                "Information Technology expenditure"
            ),
            granularity="section",
        ),
        source(
            3,
            "DOC_PDF_001:page:05:chunk:02",
            (
                "March departmental cost performance. Information "
                "Technology budget AUD 410,000 and actual AUD 365,000, "
                "a favourable variance of AUD 45,000 due to a delayed "
                "implementation."
            ),
            citation="Q3 report page 5: departmental cost performance",
            file_type="pdf",
            granularity="page",
        ),
    ]

    it_result = generator.generate(
        it_query,
        bundle(it_query, it_sources),
    )

    assert it_result["abstained"] is False
    assert it_result["selected_source_numbers"] == [2]

    bad_it_answer = (
        "Actual spend was AUD 410,000 compared with a budgeted amount "
        "of AUD 365,000, so expenditure was below budget and favourable."
    )

    assert validate_financial_consistency(
        it_query,
        bad_it_answer,
    )

    valid_it_answer = (
        "Actual expenditure was AUD 365,000 against a budgeted amount "
        "of AUD 410,000, so expenditure was AUD 45,000 below budget "
        "and favourable [1]."
    )
    assert validate_financial_consistency(
        it_query,
        valid_it_answer,
    ) == []

    forecast_answer = (
        "Actual expenditure was AUD 1,505,200 against a budget of AUD "
        "1,420,000 and above the pre-close forecast of AUD 1,488,160, "
        "which was a favourable forecast variance [1]."
    )
    assert validate_financial_consistency(
        "Compare expenditure with budget and forecast.",
        forecast_answer,
    )

    assert validate_sentence_citations(
        "The first factual sentence has no citation. The second does [1]."
    )
    assert validate_sentence_citations(
        "The first factual sentence is supported [1]. The second is too [1]."
    ) == []

    assert remap_citations(
        "Evidence [2] and [5].",
        {2: 1, 5: 2},
    ) == "Evidence [1] and [2]."

    tax_query = (
        "What effective corporate tax rate was forecast for Harbour "
        "Retail Group for FY2025/26?"
    )
    tax_sources = [
        source(
            1,
            "front-matter:01",
            "Corporate finance policies include taxation and reporting.",
            citation="Finance policy front matter",
            file_type="docx",
        ),
        source(
            2,
            "appendix:01",
            "The appendix contains forecast definitions and tax notes.",
            citation="Management report appendix",
            file_type="pdf",
        ),
    ]
    assert "effective corporate tax rate" in specific_metric_phrases(
        tax_query
    )
    tax_result = generator.generate(
        tax_query,
        bundle(tax_query, tax_sources),
    )
    assert tax_result["abstained"] is True

    supported_query = (
        "How is gross margin calculated and what target applies?"
    )
    supported_source = source(
        1,
        "gross-margin:01",
        (
            "Gross margin is calculated as revenue minus cost of goods "
            "sold, divided by revenue. The target is at least 43.1%."
        ),
        citation="Gross margin KPI",
        file_type="json",
        granularity="kpi",
    )

    abstention_adapter = SequenceAdapter(
        [ABSTENTION_MESSAGE + " [1]"]
    )
    abstention_rag = OllamaGroundedRAG(
        retriever=StubRetriever([supported_source]),
        adapter=abstention_adapter,
        top_k=5,
    )
    abstention_result = abstention_rag.answer(
        supported_query
    )

    assert abstention_result["abstained"] is True
    assert abstention_result["answer"] == ABSTENTION_MESSAGE
    assert abstention_result["generation_mode"] == "ollama_abstention"
    assert abstention_result[
        "citation_validation"
    ]["used_citations"] == []

    repair_adapter = SequenceAdapter(
        [
            "Gross margin has a target of 43.1%. [9]",
            (
                "Gross margin is calculated as revenue minus cost of "
                "goods sold, divided by revenue, and the target is at "
                "least 43.1%. [1]"
            ),
        ]
    )
    repair_rag = OllamaGroundedRAG(
        retriever=StubRetriever([supported_source]),
        adapter=repair_adapter,
        top_k=5,
    )
    repair_result = repair_rag.answer(
        supported_query
    )

    assert repair_adapter.call_count == 2
    assert repair_result["generation_mode"] == "ollama_repair"
    assert repair_result["abstained"] is False
    assert repair_result["citation_validation"]["valid"] is True

    discussion_query = (
        "How strongly did E-commerce sales grow, what concern accompanied "
        "that growth, and who led the discussion?"
    )
    discussion_sources = [
        source(
            1,
            "DOC_PDF_001:page:02:chunk:01",
            (
                "E-commerce sales grew strongly by 14.8%. Growth was led "
                "by the online channel. E-commerce performance, revenue, "
                "online conversion and management discussion were covered "
                "in the report. Revenue remained below forecast because "
                "physical stores were softer."
            ),
            citation=(
                "Q3 management report page 2: E-commerce performance, "
                "growth and management discussion"
            ),
            file_type="pdf",
            granularity="page",
        ),
        source(
            2,
            "DOC_MD_001:section:09:e-commerce-performance:chunk:01",
            (
                "E-commerce sales grew by 14.8%, but paid acquisition "
                "costs increased faster than online conversion. "
                "Discussion lead: Mia Thompson, E-commerce Director."
            ),
            citation="Forecast meeting section 9: E-commerce performance",
            granularity="section",
        ),
    ]
    discussion_result = generator.generate(
        discussion_query,
        bundle(discussion_query, discussion_sources),
    )
    assert discussion_result["selected_source_numbers"] == [2]

    drivers_query = (
        "How far above budget was March operating expenditure, and which "
        "two departments accounted for most of the unfavourable variance?"
    )
    drivers_sources = [
        source(
            1,
            "DOC_PDF_001:page:05:chunk:01",
            (
                "March operating expenditure was AUD 393,000 above budget."
            ),
            citation="Q3 report page 5",
            file_type="pdf",
            granularity="page",
        ),
        source(
            2,
            "DOC_PDF_001:page:05:chunk:02",
            (
                "Marketing contributed AUD 185,000 and Supply Chain "
                "AUD 150,000, a combined AUD 335,000."
            ),
            citation="Q3 report page 5",
            file_type="pdf",
            granularity="page",
        ),
    ]
    drivers_result = generator.generate(
        drivers_query,
        bundle(drivers_query, drivers_sources),
    )
    assert set(drivers_result["selected_source_numbers"]) == {1, 2}

    it_context = {
        "context": (
            "[1]\nContent: Budget AUD 390,000, actual expenditure AUD "
            "382,200, favourable variance AUD 7,800 or 2.00%. "
            "Pre-close forecast AUD 383,760; actual was AUD 1,560 "
            "below forecast."
        )
    }
    incomplete_it = (
        "Actual expenditure was AUD 382,200 against budget AUD 390,000 "
        "and forecast AUD 383,760 [1]."
    )
    assert validate_requested_completeness(
        csv_query,
        incomplete_it,
        it_context,
    )
    complete_it = (
        "Actual expenditure was AUD 7,800 below budget and AUD 1,560 "
        "below the pre-close forecast [1]."
    )
    assert validate_requested_completeness(
        csv_query,
        complete_it,
        it_context,
    ) == []

    capex_query = (
        "A capital expenditure project is expected to cost AUD 300,000. "
        "Who must approve it, and what supporting documentation is required?"
    )
    capex_context = {
        "context": (
            "[1]\nContent: Projects above AUD 250,000 require approval "
            "from the Finance Director, Chief Executive Officer and Board "
            "of Directors. A documented business case is required."
        )
    }
    assert validate_requested_completeness(
        capex_query,
        (
            "The Finance Director and Chief Executive Officer must "
            "approve the project [1]."
        ),
        capex_context,
    )
    assert validate_requested_completeness(
        capex_query,
        (
            "The Finance Director, Chief Executive Officer and Board of "
            "Directors must approve the project [1]."
        ),
        capex_context,
    ) == []

    nwc_query = (
        "How is net working capital calculated, what limit applies, and why "
        "does the KPI not have a universally higher-or-lower preferred direction?"
    )
    nwc_context = {
        "context": (
            "[1]\nContent: Better direction: Context dependent because "
            "excessively high and excessively low working capital can "
            "indicate different operational or liquidity issues."
        )
    }
    assert validate_requested_completeness(
        nwc_query,
        "The direction is context dependent [1].",
        nwc_context,
    )
    assert validate_requested_completeness(
        nwc_query,
        (
            "The direction is context dependent because high or low values "
            "can indicate operational or liquidity issues [1]."
        ),
        nwc_context,
    ) == []

    discussion_context = {
        "context": (
            "[1]\nContent: Discussion lead: Mia Thompson, "
            "E-commerce Director."
        )
    }
    assert validate_requested_completeness(
        discussion_query,
        "The discussion was led by management [1].",
        discussion_context,
    )
    assert validate_requested_completeness(
        discussion_query,
        "The discussion was led by Mia Thompson [1].",
        discussion_context,
    ) == []

    equal_budget_answer = (
        "Actual expenditure was AUD 500,000 against a budget of "
        "AUD 500,000, which was favourable [1]."
    )
    assert validate_financial_consistency(
        "How did expenditure compare with budget?",
        equal_budget_answer,
    )

    variance_definition_answer = (
        "Expense variance equals actual expense less budget [1]. "
        "A positive value is unfavourable because actual expenditure "
        "exceeded budget [1]."
    )
    assert validate_financial_consistency(
        (
            "How is expense variance calculated, and why is a positive "
            "expense variance classified as unfavourable?"
        ),
        variance_definition_answer,
    ) == []

    extractive_bundle = {
        "sources": [
            {
                **csv_model_bundle["sources"][0],
                "content": (
                    "Budget AUD 390,000. Actual expenditure AUD 382,200. "
                    "The favourable actual-versus-budget difference was "
                    "AUD 7,800 or 2.00%. Pre-close forecast was AUD 383,760. "
                    "Actual expenditure was AUD 1,560 below forecast."
                ),
            }
        ]
    }
    extractive_answer = build_context_extractive_fallback(
        extractive_bundle
    )
    assert "AUD 7,800" in extractive_answer
    assert "AUD 1,560" in extractive_answer
    assert extractive_answer.count("[1]") >= 5
    assert validate_sentence_citations(extractive_answer) == []

    fallback_adapter = SequenceAdapter(
        [
            (
                "Actual expenditure was AUD 382,200 against budget AUD "
                "390,000, with no citation."
            ),
            ABSTENTION_MESSAGE,
        ]
    )
    fallback_rag = OllamaGroundedRAG(
        retriever=StubRetriever([csv_sources[2]]),
        adapter=fallback_adapter,
        top_k=5,
    )

    class IncompleteFallbackGate:
        def generate(
            self,
            query: str,
            context_bundle: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                "answer": (
                    "The pre-close forecast was AUD 383,760 and actual "
                    "was AUD 1,560 below forecast. [1]"
                ),
                "abstained": False,
                "reason": None,
                "selected_evidence": [
                    {
                        "source_number": 1,
                        "text": context_bundle["sources"][0]["content"],
                    }
                ],
                "selected_source_numbers": [1],
            }

    fallback_rag.evidence_gate = IncompleteFallbackGate()
    fallback_result = fallback_rag.answer(csv_query)
    assert fallback_result["abstained"] is False
    assert fallback_result["generation_mode"] == (
        "context_extractive_fallback"
    )
    assert "AUD 7,800" in fallback_result["answer"]
    assert "AUD 1,560" in fallback_result["answer"]

    messages = build_ollama_messages(
        query=accrual_query,
        context_bundle=model_bundle,
    )

    system_prompt = messages[0]["content"]

    assert "Answer every component" in system_prompt
    assert "below budget is favourable" in system_prompt
    assert "aggregate department summary" in system_prompt
    assert "0.032 is 3.2%" in system_prompt
    assert "above forecast as favourable" in system_prompt
    assert "operational or liquidity reason" in system_prompt
    assert "exact named discussion lead" in system_prompt
    assert "list every approver" in system_prompt
    assert "state each requested difference amount" in system_prompt

    print("=" * 80)
    print("GENERATION IMPROVEMENT VALIDATION PASSED")
    print("=" * 80)
    print("Month/year lexical expansion: PASS")
    print("Aggregate CSV summary selection: PASS")
    print("Complete selected sources sent to model: PASS")
    print("Contiguous model citation numbering: PASS")
    print("Question-specific source selection: PASS")
    print("Single-source sufficiency stop: PASS")
    print("Heuristic intent-mismatch evidence fallback: PASS")
    print("Specific metric phrase abstention: PASS")
    print("Expense polarity consistency check: PASS")
    print("Forecast polarity consistency check: PASS")
    print("Sentence-level citation validation: PASS")
    print("Model abstention normalisation: PASS")
    print("Invalid citation repair retry: PASS")
    print("Explicit discussion-lead hard constraint: PASS")
    print("Generic led-by discussion decoy rejection: PASS")
    print("Full-context extractive fallback: PASS")
    print("Sibling multi-chunk selection: PASS")
    print("Requested difference completeness: PASS")
    print("Approval completeness: PASS")
    print("Context-dependent reason completeness: PASS")
    print("Zero-variance polarity check: PASS")
    print("Non-numeric variance-definition consistency: PASS")
    print("Expanded grounded prompt rules: PASS")


if __name__ == "__main__":
    main()
