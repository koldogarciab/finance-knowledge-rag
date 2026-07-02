from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1]),
    )

from src.hybrid_retrieve import HybridRetriever
from src.ollama_adapter import (
    OllamaAdapterError,
    OllamaChatAdapter,
)
from src.rag_answer import (
    ABSTENTION_MESSAGE,
    CITATION_PATTERN,
    DeterministicGroundedGenerator,
    RAGContextBuilder,
    build_llm_messages,
    validate_answer_citations,
)


def normalise_text(text: str) -> str:
    return " ".join(text.casefold().split()).rstrip(".")


def is_abstention(answer: str) -> bool:
    normalised_answer = normalise_text(answer)
    normalised_expected = normalise_text(
        ABSTENTION_MESSAGE
    )

    return (
        normalised_answer == normalised_expected
        or (
            "not have enough information"
            in normalised_answer
            and "retrieved finance sources"
            in normalised_answer
        )
    )


MONEY_VALUE_BODY = (
    r"(?:AUD|\$)?\s*"
    r"(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"\s*(?P<scale>million|billion|thousand|m|bn|k)?"
)


def _money_value(raw_number: str, scale: str | None) -> float:
    value = float(raw_number.replace(",", ""))
    factors = {
        "thousand": 1_000.0,
        "k": 1_000.0,
        "million": 1_000_000.0,
        "m": 1_000_000.0,
        "billion": 1_000_000_000.0,
        "bn": 1_000_000_000.0,
    }
    return value * factors.get((scale or "").casefold(), 1.0)


def _labelled_money_value(text: str, label_pattern: str) -> float | None:
    """Return a value appearing shortly after an explicit financial label."""
    pattern = re.compile(
        rf"(?is)\b{label_pattern}\b"
        rf"[^.\n]{{0,40}}?"
        rf"{MONEY_VALUE_BODY}"
    )
    match = pattern.search(text)

    if not match:
        return None

    return _money_value(
        match.group("number"),
        match.group("scale"),
    )


def _first_labelled_money_value(
    text: str,
    label_patterns: tuple[str, ...],
) -> float | None:
    for label_pattern in label_patterns:
        value = _labelled_money_value(text, label_pattern)
        if value is not None:
            return value
    return None


def validate_financial_consistency(
    query: str,
    answer: str,
) -> list[str]:
    """Detect clear expense actual-versus-budget polarity contradictions."""
    lowered_query = query.casefold()

    if not any(
        term in lowered_query
        for term in (
            "expenditure",
            "expense",
            "cost",
            "spend",
            "budget",
            "variance",
        )
    ):
        return []

    numeric_answer = CITATION_PATTERN.sub("", answer)

    actual = _first_labelled_money_value(
        numeric_answer,
        (
            r"(?:actual\s+)?(?:expenditure|expense|spend|cost)\s+(?:of|was|at)",
            r"actual\s+(?:expenditure|expense|spend|cost|amount)",
            r"actual",
        ),
    )
    budget = _first_labelled_money_value(
        numeric_answer,
        (
            r"budgeted\s+amount",
            r"budget\s+(?:amount|of|was|at)",
            r"budget",
        ),
    )
    forecast = _first_labelled_money_value(
        numeric_answer,
        (
            r"pre-close\s+forecast",
            r"forecast\s+(?:amount|of|was|at)",
            r"forecast",
        ),
    )

    if actual is None or budget is None:
        return []

    lowered_answer = answer.casefold()
    errors: list[str] = []

    if actual > budget:
        if "below budget" in lowered_answer:
            errors.append(
                "The answer says actual expenditure was below budget, "
                "but the stated actual amount is greater than budget."
            )
        if (
            "favourable" in lowered_answer
            and "unfavourable" not in lowered_answer
        ):
            errors.append(
                "The answer classifies an expense above budget as favourable."
            )

    if actual < budget:
        if "above budget" in lowered_answer:
            errors.append(
                "The answer says actual expenditure was above budget, "
                "but the stated actual amount is lower than budget."
            )
        if (
            "unfavourable" in lowered_answer
            and "favourable" not in lowered_answer
        ):
            errors.append(
                "The answer classifies an expense below budget as unfavourable."
            )

    if (
        actual is not None
        and budget is not None
        and actual == budget
    ):
        if (
            "favourable" in lowered_answer
            or "unfavourable" in lowered_answer
        ):
            errors.append(
                "The answer applies a favourable or unfavourable label "
                "when actual expenditure equals budget; describe it as on budget."
            )

    if forecast is not None:
        forecast_sentences = [
            sentence.casefold()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer)
            if "forecast" in sentence.casefold()
        ]
        forecast_text = " ".join(forecast_sentences)

        if actual > forecast:
            if "below" in forecast_text:
                errors.append(
                    "The answer says actual expenditure was below forecast, "
                    "but the stated actual amount is greater than forecast."
                )
            if (
                "favourable" in forecast_text
                and "unfavourable" not in forecast_text
            ):
                errors.append(
                    "The answer classifies expense above forecast as favourable."
                )

        if actual < forecast:
            if "above" in forecast_text or "higher" in forecast_text:
                errors.append(
                    "The answer says actual expenditure was above forecast, "
                    "but the stated actual amount is lower than forecast."
                )
            if (
                "unfavourable" in forecast_text
                and "favourable" not in forecast_text
            ):
                errors.append(
                    "The answer classifies expense below forecast as unfavourable."
                )

        if (
            actual is not None
            and forecast is not None
            and actual == forecast
            and (
                "favourable" in forecast_text
                or "unfavourable" in forecast_text
            )
        ):
            errors.append(
                "The answer applies a favourable or unfavourable label "
                "when actual expenditure equals forecast; describe it as on forecast."
            )

    return errors


def validate_sentence_citations(answer: str) -> list[str]:
    """Require a citation immediately within every factual sentence."""
    errors: list[str] = []

    for sentence in re.split(
        r"(?<=[.!?])\s+(?!\[\d+\])|\n+",
        answer,
    ):
        cleaned = sentence.strip()
        if not cleaned:
            continue
        if not re.search(r"[A-Za-z0-9]", cleaned):
            continue
        if not CITATION_PATTERN.search(cleaned):
            errors.append(
                "Every factual sentence must contain an immediate valid citation."
            )
            break

    return errors


def _all_money_values(text: str) -> list[float]:
    pattern = re.compile(MONEY_VALUE_BODY, re.IGNORECASE)
    return [
        _money_value(match.group("number"), match.group("scale"))
        for match in pattern.finditer(text)
    ]


def _contains_money_value(
    text: str,
    expected: float,
    tolerance: float = 1.0,
) -> bool:
    return any(
        abs(value - expected) <= tolerance
        for value in _all_money_values(text)
    )


def _explicit_discussion_lead(context: str) -> list[str]:
    patterns = (
        r"(?i)discussion\s+lead\s*:\s*"
        r"([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3})",
        r"(?i)discussion\s+was\s+led\s+by\s+"
        r"([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3})",
    )

    for pattern in patterns:
        match = re.search(pattern, context)
        if match:
            return [
                token.casefold()
                for token in re.findall(
                    r"[A-Za-z'’-]+",
                    match.group(1),
                )
            ]

    return []


def validate_requested_completeness(
    query: str,
    answer: str,
    context_bundle: dict[str, Any],
) -> list[str]:
    """Validate high-risk components explicitly requested by the user."""
    lowered_query = query.casefold()
    lowered_answer = answer.casefold()
    context = str(context_bundle.get("context", ""))
    lowered_context = context.casefold()
    errors: list[str] = []

    if (
        "how far" in lowered_query
        and "budget" in lowered_query
        and "forecast" in lowered_query
    ):
        actual = _first_labelled_money_value(
            context,
            (
                r"(?:actual\s+)?(?:expenditure|expense|spend|cost)\s+(?:of|was|at)",
                r"actual\s+(?:expenditure|expense|spend|cost|amount)",
                r"actual",
            ),
        )
        budget = _first_labelled_money_value(
            context,
            (
                r"budgeted\s+amount",
                r"budget\s+(?:amount|of|was|at)",
                r"budget",
            ),
        )
        forecast = _first_labelled_money_value(
            context,
            (
                r"pre-close\s+forecast",
                r"forecast\s+(?:amount|of|was|at)",
                r"forecast",
            ),
        )

        if actual is not None and budget is not None:
            budget_difference = abs(budget - actual)
            if (
                budget_difference > 0
                and not _contains_money_value(
                    answer,
                    budget_difference,
                )
            ):
                errors.append(
                    "The answer must state the requested actual-versus-budget "
                    "difference amount."
                )

        if actual is not None and forecast is not None:
            forecast_difference = abs(forecast - actual)
            if (
                forecast_difference > 0
                and not _contains_money_value(
                    answer,
                    forecast_difference,
                )
            ):
                errors.append(
                    "The answer must state the requested actual-versus-forecast "
                    "difference amount."
                )

    if (
        any(term in lowered_query for term in ("approve", "approval"))
        and "board of directors" in lowered_context
    ):
        query_values = _all_money_values(query)
        query_amount = max(query_values) if query_values else None
        board_index = lowered_context.find("board of directors")
        board_window = context[
            max(0, board_index - 220):
            min(len(context), board_index + 220)
        ]
        thresholds = _all_money_values(board_window)

        board_applies = (
            query_amount is not None
            and (
                not thresholds
                or query_amount >= max(thresholds)
            )
        )

        if board_applies and "board" not in lowered_answer:
            errors.append(
                "The answer omits the Board approval required for the "
                "project amount stated in the question."
            )

    if (
        "why" in lowered_query
        and any(
            phrase in lowered_query
            for phrase in (
                "context dependent",
                "preferred direction",
                "higher-or-lower",
                "higher or lower",
            )
        )
    ):
        required_reason_terms = [
            term
            for term in ("operational", "liquidity")
            if term in lowered_context
        ]
        missing_reason_terms = [
            term
            for term in required_reason_terms
            if term not in lowered_answer
        ]

        if missing_reason_terms:
            errors.append(
                "The answer must include the source's operational or liquidity "
                "reason for the context-dependent direction."
            )

    if any(
        phrase in lowered_query
        for phrase in (
            "who led the discussion",
            "who led discussion",
            "discussion lead",
        )
    ):
        lead_tokens = _explicit_discussion_lead(context)

        if lead_tokens and not all(
            token in lowered_answer
            for token in lead_tokens[:2]
        ):
            errors.append(
                "The answer must name the explicit discussion lead from "
                "the selected source."
            )

    return errors


def remap_citations(
    answer: str,
    source_number_map: dict[int, int],
) -> str:
    """Map retrieval citation numbers to contiguous model numbers."""
    def replace(match: re.Match[str]) -> str:
        original = int(match.group(1))
        mapped = source_number_map.get(original)
        return f"[{mapped}]" if mapped is not None else match.group(0)

    return CITATION_PATTERN.sub(replace, answer)


def build_repair_messages(
    original_messages: list[dict[str, str]],
    previous_answer: str,
    allowed_citations: list[int],
    errors: list[str],
) -> list[dict[str, str]]:
    allowed = ", ".join(
        f"[{number}]" for number in allowed_citations
    )
    error_text = "; ".join(errors) or "The answer failed validation."

    return [
        *original_messages,
        {
            "role": "assistant",
            "content": previous_answer,
        },
        {
            "role": "user",
            "content": (
                "Rewrite the answer so it is fully supported and internally "
                "consistent. Address every requested component. "
                f"Use only these citation markers: {allowed}. "
                f"Validation issues: {error_text} "
                "Re-read the numbered sources and copy the relevant actual, "
                "budget, forecast, variance, owner, approver, discussion lead "
                "and deadline values exactly. State every requested difference "
                "amount and every applicable approver. "
                "Do not abstain merely because the previous draft was "
                "inconsistent when the numbered sources explicitly contain all "
                "requested facts. If the sources genuinely do not support the "
                "complete answer, return "
                f"exactly: {ABSTENTION_MESSAGE}"
            ),
        },
    ]


def build_ollama_messages(
    query: str,
    context_bundle: dict[str, Any],
) -> list[dict[str, str]]:
    messages = build_llm_messages(
        query=query,
        context_bundle=context_bundle,
    )

    messages[0]["content"] += (
        "\n\nOutput requirements:\n"
        "- Answer every component explicitly requested by the question.\n"
        "- Use up to five concise sentences when several facts are requested.\n"
        "- Answer only the question asked; do not add unrelated commentary.\n"
        "- Reproduce names, dates, formulas and monetary amounts exactly "
        "from the sources.\n"
        "- When a percentage KPI is stored as a decimal, express it as the "
        "equivalent percentage (for example, 0.032 is 3.2%).\n"
        "- For expense performance, below budget is favourable and above "
        "budget is unfavourable unless the source explicitly states otherwise.\n"
        "- If actual equals budget or forecast, describe the result as on budget "
        "or on forecast, not as favourable or unfavourable.\n"
        "- Do not describe an expense above forecast as favourable; if the "
        "source gives only the amount above or below forecast, report that "
        "direction without inventing a favourable label.\n"
        "- When the question asks why a direction is context dependent, include "
        "the source's stated operational or liquidity reason, not only the "
        "words 'context dependent'.\n"
        "- When the question asks who led a discussion, use the exact named "
        "discussion lead and role from the source; do not substitute a team or "
        "the company.\n"
        "- When the question asks who must approve a transaction, list every "
        "approver whose threshold applies, including the Board where required.\n"
        "- When the question asks how far actual was above or below budget or "
        "forecast, state each requested difference amount explicitly.\n"
        "- When the question asks which departments drove a variance and the "
        "source gives department amounts, include each department's amount.\n"
        "- Match the scope of the question. For department-level monthly "
        "totals, prefer an aggregate department summary over one account row.\n"
        "- Before answering an expense comparison, verify that the stated "
        "actual and budget amounts agree with the favourable or unfavourable "
        "direction.\n"
        "- Use the fewest sources needed to support the complete answer.\n"
        "- Use only citation markers that are explicitly present in the "
        "numbered context.\n"
        "- Put a valid citation such as [1] immediately after each factual "
        "sentence.\n"
        "- Do not include a separate references or sources list.\n"
        "- Do not cite sources that do not directly support the claim.\n"
        "- If any requested component is unsupported, respond exactly and "
        "without a citation:\n"
        f"{ABSTENTION_MESSAGE}"
    )

    messages[1]["content"] += (
        "\n\nReturn only the final answer. "
        "Do not describe your reasoning."
    )

    return messages


def public_sources(
    sources: list[dict[str, Any]],
    used_citations: list[int],
) -> list[dict[str, Any]]:
    cited_numbers = set(used_citations)

    return [
        {
            "source_number": source["source_number"],
            "chunk_id": source["chunk_id"],
            "document_title": source[
                "document_title"
            ],
            "file_type": source["file_type"],
            "citation": source["citation"],
            "source_path": source["source_path"],
        }
        for source in sources
        if source["source_number"] in cited_numbers
    ]


def build_context_extractive_fallback(
    context_bundle: dict[str, Any],
    max_units: int = 16,
) -> str:
    """Build a fully cited extractive answer from approved model sources.

    This is a last-resort path used only after both Ollama attempts and the
    concise deterministic fallback fail validation. It preserves all factual
    units from the already-approved evidence context, attaches an immediate
    citation to every unit, and avoids abstaining on an otherwise supported
    question merely because a small local model omitted one requested value.
    """
    units: list[str] = []

    for source in context_bundle.get("sources", []):
        source_number = int(source["source_number"])
        content = str(source.get("content", ""))
        raw_units = re.split(r"(?<=[.!?])\s+|\n+", content)

        for unit in raw_units:
            cleaned = " ".join(unit.split()).strip()

            # Preserve short structured financial fields such as
            # "Budget AUD 390,000." and "Actual AUD 382,200.".
            if not cleaned or (len(cleaned) < 12 and not any(
                character.isdigit() for character in cleaned
            )):
                continue

            units.append(f"{cleaned} [{source_number}]")

            if len(units) >= max_units:
                return " ".join(units)

    return " ".join(units)



def build_evidence_context_bundle(
    context_bundle: dict[str, Any],
    selected_evidence: list[dict[str, Any]],
    selected_source_numbers: list[int] | None = None,
) -> dict[str, Any]:
    """
    Build the exact context sent to the LLM.

    Complete selected chunks are supplied and renumbered contiguously from
    one. Contiguous numbering reduces citation mistakes when the grounding
    gate selects retrieval sources such as 1, 3 and 5.
    """
    if selected_source_numbers is None:
        selected_numbers = {
            int(evidence["source_number"])
            for evidence in selected_evidence
        }
    else:
        selected_numbers = {
            int(number)
            for number in selected_source_numbers
        }

    selected_sources: list[dict[str, Any]] = []
    context_blocks: list[str] = []
    source_number_map: dict[int, int] = {}

    for source in context_bundle["sources"]:
        retrieval_number = int(source["source_number"])

        if retrieval_number not in selected_numbers:
            continue

        evidence_text = str(source["content"]).strip()

        if not evidence_text:
            continue

        model_number = len(selected_sources) + 1
        source_number_map[retrieval_number] = model_number

        selected_source = dict(source)
        selected_source["retrieval_source_number"] = retrieval_number
        selected_source["source_number"] = model_number
        selected_source["content"] = evidence_text
        selected_sources.append(selected_source)

        context_blocks.append(
            f"[{model_number}]\n"
            f"Title: {source['document_title']}\n"
            f"Citation: {source['citation']}\n"
            f"File type: {source['file_type']}\n"
            f"Content: {evidence_text}"
        )

    context = "\n\n".join(context_blocks)

    return {
        "query": context_bundle["query"],
        "filters": context_bundle["filters"],
        "source_count": len(selected_sources),
        "context_character_count": len(context),
        "context": context,
        "sources": selected_sources,
        "source_number_map": source_number_map,
    }


class OllamaGroundedRAG:
    """
    Grounded local RAG using Ollama.

    Retrieval and evidence sufficiency are checked before generation.
    Invalid model citations trigger a deterministic grounded fallback.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        adapter: OllamaChatAdapter,
        top_k: int = 5,
        max_context_chars: int = 8_000,
    ) -> None:
        self.adapter = adapter

        self.context_builder = RAGContextBuilder(
            retriever=retriever,
            top_k=top_k,
            max_context_chars=max_context_chars,
        )

        self.evidence_gate = (
            DeterministicGroundedGenerator()
        )

    def _build_result(
        self,
        *,
        query: str,
        context_bundle: dict[str, Any],
        answer: str,
        abstained: bool,
        generation_mode: str,
        citation_validation: dict[str, Any],
        provider_response: dict[str, Any] | None,
        provider_error: str | None,
        fallback_reason: str | None,
        model_context_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if model_context_bundle is not None:
            active_model_context = model_context_bundle
        elif generation_mode == "retrieval_abstention":
            active_model_context = {
                "source_count": 0,
                "sources": [],
                "context": "",
            }
        else:
            active_model_context = context_bundle

        cited_sources = public_sources(
            sources=active_model_context["sources"],
            used_citations=citation_validation[
                "used_citations"
            ],
        )

        return {
            "query": query,
            "answer": answer,
            "abstained": abstained,
            "generation_mode": generation_mode,
            "model": (
                provider_response.get("model")
                if provider_response
                else self.adapter.model
            ),
            "citation_validation": (
                citation_validation
            ),
            "source_count": context_bundle[
                "source_count"
            ],
            "model_source_count": active_model_context[
                "source_count"
            ],
            "cited_source_count": len(
                cited_sources
            ),
            "cited_sources": cited_sources,
            "sources": [
                {
                    "source_number": source[
                        "source_number"
                    ],
                    "chunk_id": source["chunk_id"],
                    "document_title": source[
                        "document_title"
                    ],
                    "file_type": source[
                        "file_type"
                    ],
                    "citation": source[
                        "citation"
                    ],
                    "source_path": source[
                        "source_path"
                    ],
                }
                for source in context_bundle[
                    "sources"
                ]
            ],
            "model_sources": [
                {
                    "source_number": source[
                        "source_number"
                    ],
                    "chunk_id": source["chunk_id"],
                    "document_title": source[
                        "document_title"
                    ],
                    "file_type": source[
                        "file_type"
                    ],
                    "citation": source["citation"],
                    "source_path": source[
                        "source_path"
                    ],
                }
                for source in active_model_context[
                    "sources"
                ]
            ],
            "provider_error": provider_error,
            "fallback_reason": fallback_reason,
            "provider_metrics": (
                {
                    "prompt_eval_count": (
                        provider_response.get(
                            "prompt_eval_count"
                        )
                    ),
                    "eval_count": (
                        provider_response.get(
                            "eval_count"
                        )
                    ),
                    "total_duration": (
                        provider_response.get(
                            "total_duration"
                        )
                    ),
                    "done_reason": (
                        provider_response.get(
                            "done_reason"
                        )
                    ),
                }
                if provider_response
                else None
            ),
            "context": context_bundle["context"],
            "model_context": active_model_context[
                "context"
            ],
        }

    def answer(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "The query cannot be empty."
            )

        context_bundle = self.context_builder.build(
            query=clean_query,
            filters=filters,
        )

        deterministic = (
            self.evidence_gate.generate(
                query=clean_query,
                context_bundle=context_bundle,
            )
        )

        # Do not call the model when retrieval cannot support the query.
        if deterministic["abstained"]:
            validation = validate_answer_citations(
                answer=ABSTENTION_MESSAGE,
                sources=context_bundle["sources"],
                abstained=True,
            )

            return self._build_result(
                query=clean_query,
                context_bundle=context_bundle,
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                generation_mode=(
                    "retrieval_abstention"
                ),
                citation_validation=validation,
                provider_response=None,
                provider_error=None,
                fallback_reason=deterministic[
                    "reason"
                ],
            )

        generation_context_bundle = (
            build_evidence_context_bundle(
                context_bundle=context_bundle,
                selected_evidence=deterministic[
                    "selected_evidence"
                ],
                selected_source_numbers=deterministic.get(
                    "selected_source_numbers"
                ),
            )
        )

        if (
            generation_context_bundle[
                "source_count"
            ]
            == 0
        ):
            raise ValueError(
                "The evidence gate approved the query "
                "but produced no model context."
            )

        messages = build_ollama_messages(
            query=clean_query,
            context_bundle=(
                generation_context_bundle
            ),
        )

        def validate_provider_answer(
            response: dict[str, Any],
        ) -> tuple[str, bool, dict[str, Any]]:
            answer = str(response["content"]).strip()
            abstained = is_abstention(answer)

            # An abstention is a control decision, not a factual answer.
            # Normalise it and remove any citations the model appended.
            if abstained:
                answer = ABSTENTION_MESSAGE

            validation = validate_answer_citations(
                answer=answer,
                sources=generation_context_bundle["sources"],
                abstained=abstained,
            )

            sentence_citation_errors = (
                []
                if abstained
                else validate_sentence_citations(answer)
            )
            consistency_errors = (
                []
                if abstained
                else validate_financial_consistency(
                    clean_query,
                    answer,
                )
            )
            completeness_errors = (
                []
                if abstained
                else validate_requested_completeness(
                    clean_query,
                    answer,
                    generation_context_bundle,
                )
            )

            validation_errors = [
                *sentence_citation_errors,
                *consistency_errors,
                *completeness_errors,
            ]

            if validation_errors:
                validation = dict(validation)
                validation["valid"] = False
                validation["errors"] = [
                    *validation["errors"],
                    *validation_errors,
                ]

            return answer, abstained, validation

        provider_response: dict[str, Any] | None
        provider_error: str | None

        try:
            provider_response = self.adapter.chat(messages)
            (
                model_answer,
                model_abstained,
                model_validation,
            ) = validate_provider_answer(
                provider_response
            )
            provider_error = None

        except OllamaAdapterError as exc:
            provider_response = None
            model_answer = ""
            model_abstained = False
            model_validation = {
                "valid": False,
                "errors": [str(exc)],
                "allowed_citations": [],
                "used_citations": [],
                "unknown_citations": [],
            }
            provider_error = str(exc)

        if model_abstained:
            return self._build_result(
                query=clean_query,
                context_bundle=context_bundle,
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                generation_mode="ollama_abstention",
                citation_validation=model_validation,
                provider_response=provider_response,
                provider_error=None,
                fallback_reason=None,
                model_context_bundle=generation_context_bundle,
            )

        if model_validation["valid"]:
            return self._build_result(
                query=clean_query,
                context_bundle=context_bundle,
                answer=model_answer,
                abstained=False,
                generation_mode="ollama",
                citation_validation=model_validation,
                provider_response=provider_response,
                provider_error=None,
                fallback_reason=None,
                model_context_bundle=generation_context_bundle,
            )

        # Give the local model one constrained opportunity to repair invalid
        # citations or a clear actual-versus-budget contradiction.
        if provider_response is not None:
            repair_messages = build_repair_messages(
                original_messages=messages,
                previous_answer=model_answer,
                allowed_citations=[
                    source["source_number"]
                    for source in generation_context_bundle["sources"]
                ],
                errors=model_validation["errors"],
            )

            try:
                repair_response = self.adapter.chat(
                    repair_messages
                )
                (
                    repair_answer,
                    repair_abstained,
                    repair_validation,
                ) = validate_provider_answer(
                    repair_response
                )

            except OllamaAdapterError as exc:
                provider_error = str(exc)

            else:
                if repair_abstained:
                    # The deterministic evidence gate has already established
                    # that the question is supported. A repair-time abstention
                    # should therefore continue to the grounded fallback paths
                    # rather than override available evidence.
                    repair_validation = dict(repair_validation)
                    repair_validation["valid"] = False
                    repair_validation["errors"] = [
                        *repair_validation["errors"],
                        (
                            "The repair response abstained despite approved "
                            "retrieved evidence."
                        ),
                    ]

                if repair_validation["valid"]:
                    return self._build_result(
                        query=clean_query,
                        context_bundle=context_bundle,
                        answer=repair_answer,
                        abstained=False,
                        generation_mode="ollama_repair",
                        citation_validation=repair_validation,
                        provider_response=repair_response,
                        provider_error=None,
                        fallback_reason=(
                            "Initial Ollama output was repaired after: "
                            + "; ".join(model_validation["errors"])
                        ),
                        model_context_bundle=generation_context_bundle,
                    )

                model_validation = repair_validation
                provider_response = repair_response

        # Safe deterministic fallback. Retrieval citation numbers are mapped
        # to the contiguous numbering used in the model context.
        fallback_answer = remap_citations(
            deterministic["answer"],
            generation_context_bundle.get(
                "source_number_map",
                {},
            ),
        )
        fallback_abstained = deterministic["abstained"]

        fallback_validation = validate_answer_citations(
            answer=fallback_answer,
            sources=generation_context_bundle["sources"],
            abstained=fallback_abstained,
        )

        fallback_sentence_citation_errors = (
            []
            if fallback_abstained
            else validate_sentence_citations(
                fallback_answer
            )
        )
        fallback_consistency_errors = (
            []
            if fallback_abstained
            else validate_financial_consistency(
                clean_query,
                fallback_answer,
            )
        )
        fallback_completeness_errors = (
            []
            if fallback_abstained
            else validate_requested_completeness(
                clean_query,
                fallback_answer,
                generation_context_bundle,
            )
        )

        fallback_errors = [
            *fallback_sentence_citation_errors,
            *fallback_consistency_errors,
            *fallback_completeness_errors,
        ]

        if fallback_errors:
            fallback_validation = dict(fallback_validation)
            fallback_validation["valid"] = False
            fallback_validation["errors"] = [
                *fallback_validation["errors"],
                *fallback_errors,
            ]

        fallback_reason = (
            "Ollama output failed validation: "
            + "; ".join(model_validation["errors"])
        )

        if not fallback_validation["valid"]:
            # A concise fallback can omit a requested numeric component even
            # though the complete approved chunk contains it. Before giving up
            # on a supported question, expose the approved evidence units
            # extractively, with an immediate citation on every factual unit.
            context_fallback_answer = build_context_extractive_fallback(
                generation_context_bundle
            )
            context_fallback_validation = validate_answer_citations(
                answer=context_fallback_answer,
                sources=generation_context_bundle["sources"],
                abstained=False,
            )
            context_fallback_errors = [
                *validate_sentence_citations(context_fallback_answer),
                *validate_financial_consistency(
                    clean_query,
                    context_fallback_answer,
                ),
                *validate_requested_completeness(
                    clean_query,
                    context_fallback_answer,
                    generation_context_bundle,
                ),
            ]

            if context_fallback_errors:
                context_fallback_validation = dict(
                    context_fallback_validation
                )
                context_fallback_validation["valid"] = False
                context_fallback_validation["errors"] = [
                    *context_fallback_validation["errors"],
                    *context_fallback_errors,
                ]

            if context_fallback_validation["valid"]:
                return self._build_result(
                    query=clean_query,
                    context_bundle=context_bundle,
                    answer=context_fallback_answer,
                    abstained=False,
                    generation_mode="context_extractive_fallback",
                    citation_validation=context_fallback_validation,
                    provider_response=provider_response,
                    provider_error=provider_error,
                    fallback_reason=(
                        fallback_reason
                        + "; deterministic fallback also failed: "
                        + "; ".join(fallback_validation["errors"])
                    ),
                    model_context_bundle=generation_context_bundle,
                )

            safe_validation = validate_answer_citations(
                answer=ABSTENTION_MESSAGE,
                sources=generation_context_bundle["sources"],
                abstained=True,
            )

            return self._build_result(
                query=clean_query,
                context_bundle=context_bundle,
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                generation_mode="safe_fallback_abstention",
                citation_validation=safe_validation,
                provider_response=provider_response,
                provider_error=provider_error,
                fallback_reason=(
                    fallback_reason
                    + "; deterministic fallback also failed: "
                    + "; ".join(fallback_validation["errors"])
                    + "; full context extractive fallback also failed: "
                    + "; ".join(
                        context_fallback_validation["errors"]
                    )
                ),
                model_context_bundle=generation_context_bundle,
            )

        return self._build_result(
            query=clean_query,
            context_bundle=context_bundle,
            answer=fallback_answer,
            abstained=fallback_abstained,
            generation_mode="deterministic_fallback",
            citation_validation=fallback_validation,
            provider_response=provider_response,
            provider_error=provider_error,
            fallback_reason=fallback_reason,
            model_context_bundle=generation_context_bundle,
        )



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local grounded finance RAG using Ollama."
        )
    )

    parser.add_argument(
        "query",
        help="Natural-language finance question.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks. Default: 5.",
    )
    parser.add_argument(
        "--model",
        help="Override the configured Ollama model.",
    )
    parser.add_argument(
        "--document-id",
        help="Optional document_id filter.",
    )
    parser.add_argument(
        "--file-type",
        help="Optional file type filter.",
    )
    parser.add_argument(
        "--granularity",
        help="Optional granularity filter.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Display the complete numbered context.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Return the complete result as JSON.",
    )

    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()

    filters: dict[str, Any] = {}

    if args.document_id:
        filters["document_id"] = (
            args.document_id
        )

    if args.file_type:
        filters["file_type"] = args.file_type

    if args.granularity:
        filters["granularity"] = (
            args.granularity
        )

    retriever = HybridRetriever()

    adapter = OllamaChatAdapter(
        model=args.model,
    )

    rag = OllamaGroundedRAG(
        retriever=retriever,
        adapter=adapter,
        top_k=args.top_k,
    )

    result = rag.answer(
        query=args.query,
        filters=filters,
    )

    if args.json:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("=" * 88)
    print("LOCAL OLLAMA FINANCE RAG")
    print("=" * 88)
    print(f"Question: {result['query']}")
    print()
    print(f"Answer: {result['answer']}")
    print()
    print(
        f"Generation mode: "
        f"{result['generation_mode']}"
    )
    print(f"Model: {result['model']}")
    print(f"Abstained: {result['abstained']}")
    print(
        "Citation validation: "
        + (
            "PASSED"
            if result[
                "citation_validation"
            ]["valid"]
            else "FAILED"
        )
    )

    if result["fallback_reason"]:
        print(
            f"Fallback reason: "
            f"{result['fallback_reason']}"
        )

    print()

    if result["abstained"]:
        print(
            "Sources: none cited because the "
            "retrieved evidence was insufficient."
        )
    else:
        print("Cited sources:")

        for source in result[
            "cited_sources"
        ]:
            print(
                f"[{source['source_number']}] "
                f"{source['citation']}"
            )

    if args.show_context:
        print()
        print("=" * 88)
        print("MODEL EVIDENCE CONTEXT")
        print("=" * 88)
        print(result["model_context"])


if __name__ == "__main__":
    main()
