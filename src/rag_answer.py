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

from src.hybrid_retrieve import HybridRetriever, lexical_tokens


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
    "march",
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


def requested_intents(query: str) -> list[str]:
    """Identify the factual components requested by the question."""
    lowered = query.casefold()
    intents: list[str] = []

    if (
        "formula" in lowered
        or "calculat" in lowered
    ):
        intents.append("formula")

    if "target" in lowered:
        intents.append("target")

    if any(
        phrase in lowered
        for phrase in (
            "who",
            "owner",
            "responsible",
        )
    ):
        intents.append("owner")

    if any(
        phrase in lowered
        for phrase in (
            "deadline",
            "by what time",
            "business day",
            "when must",
            "when was",
        )
    ):
        intents.append("deadline")

    if any(
        phrase in lowered
        for phrase in (
            "why",
            "cause",
            "caused",
            "driver",
            "reason",
        )
    ):
        intents.append("cause")

    if any(
        phrase in lowered
        for phrase in (
            "variance",
            "against budget",
            "above budget",
            "below budget",
            "perform against budget",
        )
    ):
        intents.append("variance")

    if any(
        phrase in lowered
        for phrase in (
            "definition",
            "defined",
            "what does",
        )
    ):
        intents.append("definition")

    return list(dict.fromkeys(intents))


def evidence_intents(text: str) -> set[str]:
    """Identify which requested components an evidence unit supports."""
    lowered = text.casefold()
    intents: set[str] = set()

    if any(
        phrase in lowered
        for phrase in (
            "formula:",
            "calculated as",
            "calculated by",
            "equals ",
        )
    ):
        intents.add("formula")

    if "target:" in lowered:
        intents.add("target")

    if any(
        phrase in lowered
        for phrase in (
            "owner:",
            "owned by",
            "responsible manager",
            "discussion lead:",
            "policy owner",
        )
    ):
        intents.add("owner")

    if any(
        phrase in lowered
        for phrase in (
            "deadline",
            "business day",
            "due by",
        )
    ):
        intents.add("deadline")

    if any(
        phrase in lowered
        for phrase in (
            "because",
            "caused",
            "driver",
            "reason",
            "reflecting",
            "due to",
        )
    ):
        intents.add("cause")

    if any(
        phrase in lowered
        for phrase in (
            "variance",
            "above budget",
            "below budget",
            "favourable",
            "unfavourable",
        )
    ):
        intents.add("variance")

    if any(
        phrase in lowered
        for phrase in (
            "definition:",
            "defined as",
            "measures ",
        )
    ):
        intents.add("definition")

    return intents


class DeterministicGroundedGenerator:
    """
    Reproducible extractive generator used for local validation.

    Evidence is selected by subject overlap and by the factual components
    requested in the question. Several complementary units may come from
    the same source.
    """

    def __init__(
        self,
        max_evidence_units: int = 4,
        minimum_core_overlap: int = 1,
        max_unit_chars: int = 550,
    ) -> None:
        self.max_evidence_units = int(
            max_evidence_units
        )
        self.minimum_core_overlap = int(
            minimum_core_overlap
        )
        self.max_unit_chars = int(
            max_unit_chars
        )

    def generate(
        self,
        query: str,
        context_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        query_tokens = meaningful_tokens(query)

        core_query_tokens = {
            token
            for token in query_tokens
            if token not in GENERIC_QUERY_TOKENS
        }

        intents = requested_intents(query)

        source_profiles: dict[
            int,
            dict[str, Any],
        ] = {}

        for source in context_bundle["sources"]:
            source_number = source["source_number"]
            source_tokens = meaningful_tokens(
                source["content"]
            )

            core_overlap = (
                core_query_tokens & source_tokens
            )
            total_overlap = (
                query_tokens & source_tokens
            )

            source_score = (
                2.0 * len(core_overlap)
                + (
                    len(total_overlap)
                    / max(len(query_tokens), 1)
                )
                + 1.0 / source_number
            )

            source_profiles[source_number] = {
                "core_overlap": core_overlap,
                "total_overlap": total_overlap,
                "source_score": source_score,
            }

        if core_query_tokens:
            best_core_overlap = max(
                (
                    len(profile["core_overlap"])
                    for profile
                    in source_profiles.values()
                ),
                default=0,
            )

            if (
                best_core_overlap
                < self.minimum_core_overlap
            ):
                return {
                    "answer": ABSTENTION_MESSAGE,
                    "abstained": True,
                    "reason": (
                        "The retrieved sources do not "
                        "contain the subject of the question."
                    ),
                    "selected_evidence": [],
                }

        candidates: list[dict[str, Any]] = []

        for source in context_bundle["sources"]:
            source_number = source["source_number"]
            profile = source_profiles[source_number]

            if (
                core_query_tokens
                and not profile["core_overlap"]
            ):
                continue

            for unit in candidate_units(
                source["content"]
            ):
                unit_tokens = meaningful_tokens(unit)
                overlap = query_tokens & unit_tokens

                unit_intents = evidence_intents(unit)
                matched_intents = (
                    set(intents) & unit_intents
                )

                numeric_bonus = (
                    0.35
                    if any(
                        character.isdigit()
                        for character in unit
                    )
                    else 0.0
                )

                score = (
                    profile["source_score"]
                    + len(overlap)
                    + (
                        len(overlap)
                        / max(len(query_tokens), 1)
                    )
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
                        "matched_intents": sorted(
                            matched_intents
                        ),
                        "all_intents": sorted(
                            unit_intents
                        ),
                        "overlap_tokens": sorted(
                            overlap
                        ),
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
        covered_intents: set[str] = set()

        if intents:
            for intent in intents:
                if intent in covered_intents:
                    continue

                matching_candidates = [
                    candidate
                    for candidate in candidates
                    if (
                        intent
                        in candidate["all_intents"]
                        and candidate["text"]
                        not in used_texts
                    )
                ]

                if not matching_candidates:
                    continue

                chosen = matching_candidates[0]
                selected.append(chosen)
                used_texts.add(chosen["text"])
                covered_intents.update(
                    chosen["all_intents"]
                )

                if (
                    len(selected)
                    >= self.max_evidence_units
                ):
                    break

            missing_intents = (
                set(intents) - covered_intents
            )

            if missing_intents:
                return {
                    "answer": ABSTENTION_MESSAGE,
                    "abstained": True,
                    "reason": (
                        "The retrieved evidence does not "
                        "support every requested component: "
                        + ", ".join(
                            sorted(missing_intents)
                        )
                    ),
                    "selected_evidence": selected,
                }

        else:
            for candidate in candidates:
                if candidate["text"] in used_texts:
                    continue

                if (
                    not candidate["overlap_tokens"]
                    and core_query_tokens
                ):
                    continue

                selected.append(candidate)
                used_texts.add(candidate["text"])

                if len(selected) >= 2:
                    break

        if not selected:
            return {
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "reason": (
                    "No retrieved evidence met the "
                    "grounding requirements."
                ),
                "selected_evidence": [],
            }

        answer = " ".join(
            (
                f"{candidate['text']} "
                f"[{candidate['source_number']}]"
            )
            for candidate in selected
        )

        return {
            "answer": answer,
            "abstained": False,
            "reason": None,
            "selected_evidence": selected,
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
