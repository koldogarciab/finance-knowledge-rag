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

from src.hybrid_retrieve import (
    HybridRetriever,
    expand_lexical_query,
    lexical_tokens,
)


CITATION_PATTERN = re.compile(r"\[(\d+)\]")

SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+|\n+"
)

ABSTENTION_MESSAGE = (
    "I do not have enough information in the retrieved "
    "finance sources to answer this question."
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
}

DOMAIN_STOPWORDS = {
    "harbour",
    "retail",
    "group",
    "company",
    "finance",
    "financial",
    "fy2025",
    "fy2026",
}


SYSTEM_PROMPT = """
You are a finance knowledge assistant.

Answer only from the supplied numbered sources.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts, calculations, dates, owners or policies.
3. Cite every factual claim using one or more source markers such as [1].
4. Only use citation numbers that appear in the supplied context.
5. If the context is insufficient, state that there is not enough
   information in the retrieved finance sources.
6. Keep the answer direct and distinguish favourable and unfavourable
   financial variances carefully.
""".strip()


def compact_text(
    text: str,
    max_chars: int | None = None,
) -> str:
    value = " ".join(text.split())

    if max_chars is None or len(value) <= max_chars:
        return value

    return value[: max_chars - 3].rstrip() + "..."


def meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in lexical_tokens(text)
        if len(token) >= 2
        and token not in STOPWORDS
        and token not in DOMAIN_STOPWORDS
    }


def candidate_units(text: str) -> list[str]:
    units: list[str] = []

    for unit in SENTENCE_SPLIT_PATTERN.split(text):
        cleaned = compact_text(unit)

        if len(cleaned) < 20:
            continue

        units.append(cleaned)

    if not units:
        cleaned = compact_text(text)

        if cleaned:
            units.append(cleaned)

    return units


class RAGContextBuilder:
    """Retrieve and format numbered evidence for grounded answers."""

    def __init__(
        self,
        retriever: HybridRetriever,
        top_k: int = 5,
        max_context_chars: int = 8_000,
        max_chars_per_source: int = 2_000,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if max_context_chars < 1:
            raise ValueError(
                "max_context_chars must be positive."
            )

        if max_chars_per_source < 1:
            raise ValueError(
                "max_chars_per_source must be positive."
            )

        self.retriever = retriever
        self.top_k = int(top_k)
        self.max_context_chars = int(
            max_context_chars
        )
        self.max_chars_per_source = int(
            max_chars_per_source
        )

    def build(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        results = self.retriever.search(
            query=query,
            top_k=self.top_k,
            filters=filters,
        )

        sources: list[dict[str, Any]] = []
        context_blocks: list[str] = []
        used_characters = 0
        seen_chunk_ids: set[str] = set()

        for result in results:
            chunk_id = result["chunk_id"]

            if chunk_id in seen_chunk_ids:
                continue

            source_number = len(sources) + 1

            content = compact_text(
                result["content"],
                max_chars=self.max_chars_per_source,
            )

            block = (
                f"[{source_number}]\n"
                f"Title: {result['document_title']}\n"
                f"Citation: {result['citation']}\n"
                f"File type: {result['file_type']}\n"
                f"Content: {content}"
            )

            separator_length = 2 if context_blocks else 0

            if (
                used_characters
                + separator_length
                + len(block)
                > self.max_context_chars
            ):
                break

            sources.append(
                {
                    "source_number": source_number,
                    "chunk_id": chunk_id,
                    "document_id": result[
                        "document_id"
                    ],
                    "document_title": result[
                        "document_title"
                    ],
                    "file_type": result["file_type"],
                    "citation": result["citation"],
                    "source_path": result[
                        "source_path"
                    ],
                    "content": result["content"],
                    "hybrid_score": result[
                        "hybrid_score"
                    ],
                    "dense_score": result[
                        "dense_score"
                    ],
                    "lexical_score": result[
                        "lexical_score"
                    ],
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),
                    "granularity": (
                        result.get("metadata", {}).get(
                            "granularity"
                        )
                    ),
                    "citation_locator": result.get(
                        "citation_locator"
                    ),
                }
            )

            context_blocks.append(block)
            used_characters += (
                separator_length + len(block)
            )
            seen_chunk_ids.add(chunk_id)

        return {
            "query": query,
            "filters": filters or {},
            "source_count": len(sources),
            "context_character_count": (
                used_characters
            ),
            "context": "\n\n".join(
                context_blocks
            ),
            "sources": sources,
        }


GENERIC_QUERY_TOKENS = {
    "applied",
    "calculate",
    "calculated",
    "calculation",
    "ended",
    "formula",
    "give",
    "how",
    "months",
    "nine",
    "period",
    "please",
    "target",
    "tell",
    "what",
    "when",
    "which",
    "who",
    "why",
    "2025",
    "2026",
}

WEAK_ANCHOR_TOKENS = {
    "actual",
    "amount",
    "answer",
    "applies",
    "apply",
    "approved",
    "budget",
    "change",
    "compare",
    "compared",
    "data",
    "department",
    "discussion",
    "document",
    "expenditure",
    "finance",
    "forecast",
    "information",
    "included",
    "manager",
    "measure",
    "measured",
    "owner",
    "performance",
    "policy",
    "provided",
    "rate",
    "recent",
    "required",
    "requirement",
    "responsible",
    "result",
    "role",
    "source",
    "value",
    "year",
}

INTENT_PHRASES: dict[str, tuple[str, ...]] = {
    "formula": (
        "formula",
        "calculat",
        "how is",
        "how are",
    ),
    "target": (
        "target",
        "limit",
        "range",
        "threshold",
    ),
    "owner": (
        "who",
        "owner",
        "responsible",
        "led the discussion",
        "discussion lead",
    ),
    "deadline": (
        "deadline",
        "deadlines",
        "due by",
        "by what date",
        "by what time",
        "business day",
        "respective deadlines",
    ),
    "cause": (
        "why",
        "cause",
        "caused",
        "driver",
        "reason",
    ),
    "variance": (
        "variance",
        "against budget",
        "above budget",
        "below budget",
        "perform against budget",
        "compare with budget",
    ),
    "forecast": (
        "forecast",
        "pre-close forecast",
        "above forecast",
        "below forecast",
        "compare with forecast",
    ),
    "definition": (
        "definition",
        "defined",
        "what does",
        "what is",
    ),
    "documentation": (
        "documentation",
        "supported",
        "supporting",
        "business case",
    ),
    "exemption": (
        "exempt",
        "exemption",
        "exceptions",
    ),
    "frequency": (
        "frequency",
        "frequently",
        "how often",
        "weekly",
        "monthly",
    ),
    "system": (
        "which systems",
        "data source",
        "provide the data",
        "systems provide",
    ),
    "category": (
        "categories",
        "category",
        "types of expenditure",
    ),
    "approval": (
        "approve",
        "approval",
        "must approve",
    ),
    "condition": (
        "when must",
        "when is",
        "recognised",
        "recognized",
    ),
}


def requested_intents(query: str) -> list[str]:
    """Identify factual components requested by a question."""
    lowered = query.casefold()
    intents = [
        intent
        for intent, phrases in INTENT_PHRASES.items()
        if any(phrase in lowered for phrase in phrases)
    ]
    return list(dict.fromkeys(intents))


def evidence_intents(text: str) -> set[str]:
    """Identify factual components explicitly represented in evidence."""
    lowered = text.casefold()
    intents: set[str] = set()

    evidence_patterns: dict[str, tuple[str, ...]] = {
        "formula": (
            "formula:",
            "calculated as",
            "calculated by",
            "equals ",
            "divided by",
            "multiplied by",
        ),
        "target": (
            "target:",
            "target ",
            "threshold",
            "limit",
            "range",
            "greater than or equal",
            "less than or equal",
        ),
        "owner": (
            "owner:",
            "owners:",
            "owned by",
            "responsible manager",
            "responsible for",
            "discussion lead:",
            "discussion was led by",
            "led by",
            "policy owner",
        ),
        "deadline": (
            "deadline",
            "business day",
            "due by",
            "by 20",
        ),
        "cause": (
            "because",
            "caused",
            "driver",
            "reason",
            "reflecting",
            "due to",
            "context dependent",
            "operational or liquidity",
        ),
        "variance": (
            "variance",
            "above budget",
            "below budget",
            "favourable",
            "unfavourable",
        ),
        "forecast": (
            "forecast",
            "pre-close forecast",
            "above forecast",
            "below forecast",
            "actual-versus-forecast",
        ),
        "definition": (
            "definition:",
            "defined as",
            "measures ",
            "is earnings before",
        ),
        "documentation": (
            "supporting documentation",
            "supported by",
            "business case",
            "purchase order",
            "contract",
            "supplier estimate",
        ),
        "exemption": (
            "exempt",
            "exemption",
            "payroll",
            "regulated utilities",
            "emergency expenditure",
        ),
        "frequency": (
            "weekly",
            "monthly",
            "daily",
            "frequency:",
            "measured ",
        ),
        "system": (
            "system",
            "module",
            "data source",
            "general ledger",
            "erp",
        ),
        "category": (
            "categories",
            "category",
            "included:",
        ),
        "approval": (
            "approval",
            "approved by",
            "must approve",
            "requires approval",
        ),
        "condition": (
            "required when",
            "recognised when",
            "recognized when",
            "goods or services have been received",
        ),
    }

    for intent, phrases in evidence_patterns.items():
        if any(phrase in lowered for phrase in phrases):
            intents.add(intent)

    return intents


def query_token_sets(query: str) -> tuple[set[str], set[str], set[str]]:
    """
    Return all, core and anchor query tokens.

    Month/year aliases are included so natural-language dates such as
    ``August 2025`` match structured periods such as ``2025-08``.
    """
    expanded_query = expand_lexical_query(query)
    all_tokens = meaningful_tokens(expanded_query)
    core_tokens = {
        token
        for token in all_tokens
        if token not in GENERIC_QUERY_TOKENS
    }
    anchor_tokens = {
        token
        for token in core_tokens
        if token not in WEAK_ANCHOR_TOKENS
    }
    return all_tokens, core_tokens, anchor_tokens


def distinctive_query_phrases(query: str) -> set[str]:
    """Return adjacent distinctive token phrases from the question."""
    expanded = expand_lexical_query(query)
    ordered = [
        token
        for token in lexical_tokens(expanded)
        if (
            len(token) >= 2
            and token not in STOPWORDS
            and token not in DOMAIN_STOPWORDS
            and token not in GENERIC_QUERY_TOKENS
            and token not in WEAK_ANCHOR_TOKENS
        )
    ]

    phrases: set[str] = set()

    for size in (3, 2):
        for index in range(len(ordered) - size + 1):
            phrase = " ".join(ordered[index:index + size])
            phrases.add(phrase)

    return phrases


def specific_metric_phrases(query: str) -> set[str]:
    """Return explicit KPI or metric noun phrases from the question."""
    tokens = [
        token.casefold()
        for token in lexical_tokens(expand_lexical_query(query))
        if len(token) >= 2 and token not in STOPWORDS
    ]
    metric_heads = {
        "rate",
        "margin",
        "days",
        "utilisation",
        "utilization",
        "value",
        "capital",
        "ebitda",
    }
    phrases: set[str] = set()

    for index, token in enumerate(tokens):
        if token not in metric_heads:
            continue

        for width in (2, 3, 4):
            start = index - width + 1
            if start < 0:
                continue
            phrase = " ".join(tokens[start:index + 1])
            if phrase:
                phrases.add(phrase)

    return phrases


def query_may_require_multiple_sources(query: str) -> bool:
    """Identify questions whose requested evidence may span chunks."""
    lowered = query.casefold()
    return any(
        phrase in lowered
        for phrase in (
            "which two departments",
            "two departments",
            "combined aud",
            "combined variance",
            "across the two",
        )
    )


def query_requests_aggregate_summary(query: str) -> bool:
    """
    Identify department-level period comparisons that benefit from an
    aggregate summary rather than one account-category row.
    """
    lowered = query.casefold()
    expanded = expand_lexical_query(query)

    has_period_alias = expanded != query
    has_budget_scope = "budget" in lowered
    has_actual_or_variance = any(
        phrase in lowered
        for phrase in (
            "actual",
            "expenditure",
            "variance",
            "on budget",
            "performance",
        )
    )
    has_aggregate_signal = any(
        phrase in lowered
        for phrase in (
            "both budget",
            "below budget and",
            "above budget and",
            "pre-close forecast",
            "total variance",
            "included categories",
            "categories were included",
            "responsible manager",
            "who was responsible",
            "summarise",
            "on budget",
        )
    )

    return bool(
        has_period_alias
        and has_budget_scope
        and has_actual_or_variance
        and has_aggregate_signal
    )


def source_scope_text(source: dict[str, Any]) -> str:
    """Combine content and traceability fields for relevance scoring."""
    parts = [
        str(source.get("document_title", "")),
        str(source.get("citation", "")),
        str(source.get("chunk_id", "")),
        str(source.get("granularity", "")),
        str(source.get("content", "")),
    ]
    return " ".join(part for part in parts if part).casefold()


def source_group_key(source: dict[str, Any]) -> str:
    """Group sibling chunks from the same page or section."""
    chunk_id = str(source.get("chunk_id", ""))
    return chunk_id.split(":chunk:", 1)[0]


def explicit_owner_match(query: str, source_text: str) -> bool:
    """Require a true discussion-lead statement for leader questions."""
    lowered_query = query.casefold()

    if not any(
        phrase in lowered_query
        for phrase in (
            "who led the discussion",
            "who led discussion",
            "discussion lead",
        )
    ):
        return True

    lowered_source = source_text.casefold()

    # Do not accept the generic phrase "led by": financial reports often
    # use it for channel, revenue or growth drivers (for example,
    # "growth was led by E-commerce"). Only explicit discussion-lead
    # formulations qualify as evidence for a question asking who led the
    # discussion.
    return any(
        phrase in lowered_source
        for phrase in (
            "discussion lead:",
            "discussion was led by",
            "led the discussion",
        )
    )


def explicit_reason_match(query: str, source_text: str) -> bool:
    """Prefer sources that contain the requested contextual explanation."""
    lowered_query = query.casefold()

    if not (
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
        return True

    lowered_source = source_text.casefold()
    return any(
        phrase in lowered_source
        for phrase in (
            "operational",
            "liquidity",
            "excessively high",
            "excessively low",
        )
    )


class DeterministicGroundedGenerator:
    """
    Reproducible evidence gate and extractive fallback.

    The gate verifies subject-level overlap before generation, ranks
    sources using distinctive query anchors and requested components,
    and selects a small set of relevant sources. Missing heuristic
    intent labels do not by themselves force abstention: the complete
    selected chunks are later supplied to the LLM.
    """

    def __init__(
        self,
        max_evidence_units: int = 6,
        max_selected_sources: int = 3,
        minimum_core_overlap: int = 1,
        max_unit_chars: int = 650,
    ) -> None:
        self.max_evidence_units = int(max_evidence_units)
        self.max_selected_sources = int(max_selected_sources)
        self.minimum_core_overlap = int(minimum_core_overlap)
        self.max_unit_chars = int(max_unit_chars)

    @staticmethod
    def _required_anchor_overlap(anchor_count: int) -> int:
        if anchor_count == 0:
            return 0
        if anchor_count <= 2:
            return 1
        return 2

    def generate(
        self,
        query: str,
        context_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        query_tokens, core_query_tokens, anchor_query_tokens = (
            query_token_sets(query)
        )
        query_phrases = distinctive_query_phrases(query)
        metric_phrases = specific_metric_phrases(query)
        aggregate_summary_requested = (
            query_requests_aggregate_summary(query)
        )
        multiple_sources_may_be_required = (
            query_may_require_multiple_sources(query)
        )
        intents = requested_intents(query)

        source_profiles: dict[int, dict[str, Any]] = {}

        for source in context_bundle["sources"]:
            source_number = int(source["source_number"])
            scope_text = source_scope_text(source)
            source_tokens = meaningful_tokens(scope_text)
            source_intents = evidence_intents(source["content"])

            traceability_text = " ".join(
                (
                    str(source.get("document_title", "")),
                    str(source.get("citation", "")),
                    str(source.get("chunk_id", "")),
                )
            )
            traceability_tokens = meaningful_tokens(
                traceability_text
            )

            core_overlap = core_query_tokens & source_tokens
            anchor_overlap = anchor_query_tokens & source_tokens
            total_overlap = query_tokens & source_tokens
            traceability_overlap = (
                core_query_tokens & traceability_tokens
            )
            phrase_overlap = {
                phrase
                for phrase in query_phrases
                if phrase in scope_text
            }
            metric_phrase_overlap = {
                phrase
                for phrase in metric_phrases
                if phrase in scope_text
            }
            matched_intents = set(intents) & source_intents
            owner_match = explicit_owner_match(query, scope_text)
            reason_match = explicit_reason_match(query, scope_text)

            explicit_request_bonus = 0.0

            if (
                "owner" in intents
                and owner_match
                and any(
                    phrase in query.casefold()
                    for phrase in (
                        "who led the discussion",
                        "who led discussion",
                        "discussion lead",
                    )
                )
            ):
                explicit_request_bonus += 18.0

            if (
                "cause" in intents
                and reason_match
                and any(
                    phrase in query.casefold()
                    for phrase in (
                        "context dependent",
                        "preferred direction",
                        "higher-or-lower",
                        "higher or lower",
                    )
                )
            ):
                explicit_request_bonus += 12.0

            granularity = str(
                source.get("granularity")
                or source.get("metadata", {}).get(
                    "granularity",
                    "",
                )
            ).casefold()
            chunk_id = str(source.get("chunk_id", "")).casefold()

            aggregate_bonus = 0.0

            if (
                aggregate_summary_requested
                and (
                    granularity == "monthly_department_summary"
                    or ":summary:" in chunk_id
                    or "department summary" in scope_text
                )
            ):
                aggregate_bonus = 14.0

            source_score = (
                6.0 * len(anchor_overlap)
                + 2.0 * len(core_overlap)
                + 4.0 * len(traceability_overlap)
                + 5.0 * len(phrase_overlap)
                + 7.0 * len(metric_phrase_overlap)
                + 3.5 * len(matched_intents)
                + aggregate_bonus
                + explicit_request_bonus
                + len(total_overlap) / max(len(query_tokens), 1)
                + 0.25 / source_number
            )

            source_profiles[source_number] = {
                "source": source,
                "source_tokens": source_tokens,
                "source_intents": source_intents,
                "core_overlap": core_overlap,
                "anchor_overlap": anchor_overlap,
                "total_overlap": total_overlap,
                "traceability_overlap": traceability_overlap,
                "phrase_overlap": phrase_overlap,
                "metric_phrase_overlap": metric_phrase_overlap,
                "matched_intents": matched_intents,
                "aggregate_bonus": aggregate_bonus,
                "explicit_request_bonus": explicit_request_bonus,
                "owner_match": owner_match,
                "reason_match": reason_match,
                "source_group_key": source_group_key(source),
                "source_score": source_score,
            }

        if not source_profiles:
            return {
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "reason": "No sources were retrieved.",
                "selected_evidence": [],
            }

        required_anchor_overlap = self._required_anchor_overlap(
            len(anchor_query_tokens)
        )
        best_anchor_overlap = max(
            len(profile["anchor_overlap"])
            for profile in source_profiles.values()
        )
        best_core_overlap = max(
            len(profile["core_overlap"])
            for profile in source_profiles.values()
        )
        best_metric_phrase_overlap = max(
            len(profile["metric_phrase_overlap"])
            for profile in source_profiles.values()
        )

        if metric_phrases and best_metric_phrase_overlap == 0:
            return {
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "reason": (
                    "The retrieved sources do not contain the specific "
                    "metric phrase requested by the question."
                ),
                "selected_evidence": [],
            }

        if (
            required_anchor_overlap > 0
            and best_anchor_overlap < required_anchor_overlap
        ):
            return {
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "reason": (
                    "The retrieved sources do not contain enough "
                    "distinctive subject terms from the question."
                ),
                "selected_evidence": [],
            }

        if (
            not anchor_query_tokens
            and core_query_tokens
            and best_core_overlap < self.minimum_core_overlap
        ):
            return {
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "reason": (
                    "The retrieved sources do not contain the subject "
                    "of the question."
                ),
                "selected_evidence": [],
            }

        ranked_profiles = sorted(
            source_profiles.values(),
            key=lambda profile: (
                -profile["source_score"],
                profile["source"]["source_number"],
            ),
        )

        # When the user explicitly asks who led a discussion, a general
        # management report is not an acceptable substitute for a source
        # that contains a named discussion lead. Treat this as an evidence
        # eligibility rule rather than only a scoring bonus.
        explicit_discussion_lead_requested = any(
            phrase in query.casefold()
            for phrase in (
                "who led the discussion",
                "who led discussion",
                "discussion lead",
            )
        )

        if explicit_discussion_lead_requested:
            lead_profiles = [
                profile
                for profile in ranked_profiles
                if profile["owner_match"]
            ]

            if lead_profiles:
                ranked_profiles = lead_profiles
                best_profile_score = ranked_profiles[0]["source_score"]

        selected_source_numbers: list[int] = []
        covered_query_tokens: set[str] = set()
        covered_intents: set[str] = set()
        covered_phrases: set[str] = set()

        best_profile_score = ranked_profiles[0]["source_score"]

        for profile in ranked_profiles:
            same_group_needed = False

            if (
                multiple_sources_may_be_required
                and len(selected_source_numbers) == 1
            ):
                first_profile = source_profiles[
                    selected_source_numbers[0]
                ]
                same_group_needed = (
                    profile["source_group_key"]
                    == first_profile["source_group_key"]
                )

            if (
                anchor_query_tokens
                and not profile["anchor_overlap"]
                and not same_group_needed
            ):
                continue

            minimum_score_ratio = (
                0.0 if same_group_needed else 0.55
            )

            if (
                selected_source_numbers
                and profile["source_score"]
                < minimum_score_ratio * best_profile_score
            ):
                continue

            new_tokens = profile["total_overlap"] - covered_query_tokens
            new_intents = profile["matched_intents"] - covered_intents
            new_phrases = profile["phrase_overlap"] - covered_phrases

            if (
                selected_source_numbers
                and not new_tokens
                and not new_intents
                and not new_phrases
                and not same_group_needed
            ):
                continue

            source_number = int(
                profile["source"]["source_number"]
            )
            selected_source_numbers.append(source_number)
            covered_query_tokens.update(profile["total_overlap"])
            covered_intents.update(profile["matched_intents"])
            covered_phrases.update(profile["phrase_overlap"])

            strong_subject_match = bool(
                profile["phrase_overlap"]
                or profile["metric_phrase_overlap"]
                or len(profile["anchor_overlap"]) >= 2
                or len(profile["traceability_overlap"]) >= 2
            )
            source_covers_requested_intents = (
                not intents
                or set(intents).issubset(covered_intents)
            )

            if "owner" in intents:
                source_covers_requested_intents = (
                    source_covers_requested_intents
                    and profile["owner_match"]
                )

            if (
                "cause" in intents
                and any(
                    phrase in query.casefold()
                    for phrase in (
                        "context dependent",
                        "preferred direction",
                        "higher-or-lower",
                        "higher or lower",
                    )
                )
            ):
                source_covers_requested_intents = (
                    source_covers_requested_intents
                    and profile["reason_match"]
                )

            if (
                len(selected_source_numbers) == 1
                and source_covers_requested_intents
                and strong_subject_match
                and not multiple_sources_may_be_required
            ):
                break

            # A monthly department summary already contains the account
            # rows, totals, forecast comparison, categories and owner.
            if profile["aggregate_bonus"] > 0:
                break

            if len(selected_source_numbers) >= self.max_selected_sources:
                break

        if not selected_source_numbers:
            return {
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "reason": (
                    "No retrieved source met the subject-level "
                    "grounding requirements."
                ),
                "selected_evidence": [],
            }

        candidates: list[dict[str, Any]] = []

        for source_number in selected_source_numbers:
            profile = source_profiles[source_number]

            for unit in candidate_units(
                profile["source"]["content"]
            ):
                unit_tokens = meaningful_tokens(unit)
                overlap = query_tokens & unit_tokens
                unit_intents = evidence_intents(unit)
                matched_intents = set(intents) & unit_intents

                numeric_bonus = (
                    0.35
                    if any(character.isdigit() for character in unit)
                    else 0.0
                )

                score = (
                    profile["source_score"]
                    + 2.0 * len(
                        anchor_query_tokens & unit_tokens
                    )
                    + len(overlap)
                    + 3.0 * len(matched_intents)
                    + numeric_bonus
                )

                candidates.append(
                    {
                        "score": score,
                        "source_number": source_number,
                        "text": compact_text(
                            unit,
                            max_chars=self.max_unit_chars,
                        ),
                        "matched_intents": sorted(matched_intents),
                        "all_intents": sorted(unit_intents),
                        "overlap_tokens": sorted(overlap),
                    }
                )

        candidates.sort(
            key=lambda item: (
                -item["score"],
                item["source_number"],
                item["text"],
            )
        )

        selected: list[dict[str, Any]] = []
        used_texts: set[str] = set()
        covered_fallback_intents: set[str] = set()

        for intent in intents:
            matching_candidates = [
                candidate
                for candidate in candidates
                if (
                    intent in candidate["all_intents"]
                    and candidate["text"] not in used_texts
                )
            ]
            if not matching_candidates:
                continue

            chosen = matching_candidates[0]
            selected.append(chosen)
            used_texts.add(chosen["text"])
            covered_fallback_intents.update(chosen["all_intents"])

            if len(selected) >= self.max_evidence_units:
                break

        missing_fallback_intents = (
            set(intents) - covered_fallback_intents
        )

        for candidate in candidates:
            if len(selected) >= self.max_evidence_units:
                break
            if candidate["text"] in used_texts:
                continue
            if (
                query_tokens
                and not candidate["overlap_tokens"]
            ):
                continue

            candidate_intents = set(candidate["all_intents"])

            if intents and not (
                candidate_intents & missing_fallback_intents
            ):
                continue

            selected.append(candidate)
            used_texts.add(candidate["text"])
            covered_fallback_intents.update(candidate_intents)
            missing_fallback_intents = (
                set(intents) - covered_fallback_intents
            )

            if intents and not missing_fallback_intents:
                break

        if not selected:
            # Intent labels are heuristic. A source may be strongly relevant
            # even when its wording does not trigger the exact intent label
            # inferred from the question (for example, a reconciliation
            # timetable answering a question phrased with "when must").
            # Once subject-level source selection has passed, keep the best
            # overlapping evidence unit rather than abstaining solely because
            # of an intent-label mismatch. The complete selected chunk is sent
            # to the model later, so this fallback only opens the generation
            # path; it does not broaden the approved source set.
            fallback_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if (
                        not query_tokens
                        or candidate["overlap_tokens"]
                    )
                ),
                None,
            )

            if fallback_candidate is not None:
                selected.append(fallback_candidate)

        if not selected:
            return {
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "reason": (
                    "No retrieved evidence unit met the grounding "
                    "requirements."
                ),
                "selected_evidence": [],
            }

        answer = " ".join(
            f"{candidate['text']} [{candidate['source_number']}]"
            for candidate in selected
        )

        return {
            "answer": answer,
            "abstained": False,
            "reason": None,
            "selected_evidence": selected,
            "selected_source_numbers": selected_source_numbers,
            "covered_intents": sorted(covered_intents),
        }

def validate_answer_citations(
    answer: str,
    sources: list[dict[str, Any]],
    abstained: bool,
) -> dict[str, Any]:
    allowed_citations = {
        source["source_number"]
        for source in sources
    }

    used_citations = {
        int(match)
        for match in CITATION_PATTERN.findall(
            answer
        )
    }

    unknown_citations = (
        used_citations - allowed_citations
    )

    errors: list[str] = []

    if not answer.strip():
        errors.append("The answer is empty.")

    if unknown_citations:
        errors.append(
            "The answer contains unknown citations: "
            + ", ".join(
                f"[{citation}]"
                for citation in sorted(
                    unknown_citations
                )
            )
        )

    if abstained and used_citations:
        errors.append(
            "An abstained answer should not contain citations."
        )

    if (
        not abstained
        and not used_citations
    ):
        errors.append(
            "A factual answer must contain at least one citation."
        )

    return {
        "valid": not errors,
        "allowed_citations": sorted(
            allowed_citations
        ),
        "used_citations": sorted(
            used_citations
        ),
        "unknown_citations": sorted(
            unknown_citations
        ),
        "errors": errors,
    }


def build_llm_messages(
    query: str,
    context_bundle: dict[str, Any],
) -> list[dict[str, str]]:
    """Build provider-neutral chat messages for a future LLM adapter."""
    user_message = (
        f"Question:\n{query}\n\n"
        f"Numbered sources:\n"
        f"{context_bundle['context']}\n\n"
        "Provide a grounded answer using only these sources."
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


class GroundedRAG:
    """End-to-end local RAG pipeline with citation validation."""

    def __init__(
        self,
        retriever: HybridRetriever,
        top_k: int = 5,
        max_context_chars: int = 8_000,
    ) -> None:
        self.context_builder = RAGContextBuilder(
            retriever=retriever,
            top_k=top_k,
            max_context_chars=max_context_chars,
        )

        self.generator = (
            DeterministicGroundedGenerator()
        )

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

        context_bundle = (
            self.context_builder.build(
                query=clean_query,
                filters=filters,
            )
        )

        generation = self.generator.generate(
            query=clean_query,
            context_bundle=context_bundle,
        )

        citation_validation = (
            validate_answer_citations(
                answer=generation["answer"],
                sources=context_bundle[
                    "sources"
                ],
                abstained=generation[
                    "abstained"
                ],
            )
        )

        if not citation_validation["valid"]:
            raise ValueError(
                "Generated answer failed citation "
                "validation: "
                + "; ".join(
                    citation_validation[
                        "errors"
                    ]
                )
            )

        cited_numbers = set(
            citation_validation["used_citations"]
        )

        cited_sources = [
            {
                "source_number": source["source_number"],
                "chunk_id": source["chunk_id"],
                "document_title": source["document_title"],
                "file_type": source["file_type"],
                "citation": source["citation"],
                "source_path": source["source_path"],
            }
            for source in context_bundle["sources"]
            if source["source_number"] in cited_numbers
        ]

        return {
            "query": clean_query,
            "answer": generation["answer"],
            "abstained": generation[
                "abstained"
            ],
            "abstention_reason": generation[
                "reason"
            ],
            "citation_validation": (
                citation_validation
            ),
            "source_count": context_bundle[
                "source_count"
            ],
            "cited_source_count": len(cited_sources),
            "cited_sources": cited_sources,
            "sources": [
                {
                    "source_number": source[
                        "source_number"
                    ],
                    "chunk_id": source[
                        "chunk_id"
                    ],
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
            "context": context_bundle[
                "context"
            ],
            "llm_messages": build_llm_messages(
                query=clean_query,
                context_bundle=context_bundle,
            ),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grounded finance RAG answer generation "
            "with numbered citations."
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
        help=(
            "Number of retrieved chunks used as "
            "context. Default: 5."
        ),
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
        help="Display the numbered context.",
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
        filters["file_type"] = (
            args.file_type
        )

    if args.granularity:
        filters["granularity"] = (
            args.granularity
        )

    retriever = HybridRetriever()
    rag = GroundedRAG(
        retriever=retriever,
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
    print("GROUNDED FINANCE RAG")
    print("=" * 88)
    print(f"Question: {result['query']}")
    print()
    print(f"Answer: {result['answer']}")
    print()
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

    print()

    if result["abstained"]:
        print(
            "Sources: none cited because the retrieved "
            "evidence was insufficient."
        )
    else:
        print("Cited sources:")

        for source in result["cited_sources"]:
            print(
                f"[{source['source_number']}] "
                f"{source['citation']}"
            )

    if args.show_context:
        print()
        print("=" * 88)
        print("NUMBERED CONTEXT")
        print("=" * 88)
        print(result["context"])


if __name__ == "__main__":
    main()
