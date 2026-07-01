from __future__ import annotations

import argparse
import json
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
        "- Answer in no more than three concise sentences.\n"
        "- Answer only the question asked; do not add related metrics,\n"
        "  results, causes or commentary unless explicitly requested.\n"
        "- Use the fewest sources needed to support the answer.\n"
        "- Put a valid citation such as [1] immediately after "
        "each factual sentence.\n"
        "- Do not include a separate references or sources list.\n"
        "- Do not cite sources that do not directly support the claim.\n"
        "- If any requested component is unsupported, respond exactly:\n"
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



def build_evidence_context_bundle(
    context_bundle: dict[str, Any],
    selected_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the exact context sent to the LLM.

    Only evidence units selected by the deterministic grounding gate are
    included. Original source numbers are preserved so citations remain
    traceable to the retrieved results.
    """
    evidence_by_source: dict[int, list[str]] = {}

    for evidence in selected_evidence:
        source_number = int(
            evidence["source_number"]
        )
        evidence_text = str(
            evidence["text"]
        ).strip()

        if not evidence_text:
            continue

        evidence_by_source.setdefault(
            source_number,
            [],
        )

        if (
            evidence_text
            not in evidence_by_source[source_number]
        ):
            evidence_by_source[
                source_number
            ].append(evidence_text)

    selected_sources: list[dict[str, Any]] = []
    context_blocks: list[str] = []

    for source in context_bundle["sources"]:
        source_number = source["source_number"]

        if source_number not in evidence_by_source:
            continue

        evidence_text = " ".join(
            evidence_by_source[source_number]
        )

        selected_source = dict(source)
        selected_source["content"] = evidence_text

        selected_sources.append(selected_source)

        context_blocks.append(
            f"[{source_number}]\n"
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
            sources=context_bundle["sources"],
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

        try:
            provider_response = (
                self.adapter.chat(messages)
            )
            model_answer = provider_response[
                "content"
            ].strip()
            model_abstained = is_abstention(
                model_answer
            )

            model_validation = (
                validate_answer_citations(
                    answer=model_answer,
                    sources=(
                        generation_context_bundle[
                            "sources"
                        ]
                    ),
                    abstained=model_abstained,
                )
            )

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

        else:
            provider_error = None

        if model_validation["valid"]:
            return self._build_result(
                query=clean_query,
                context_bundle=context_bundle,
                answer=model_answer,
                abstained=model_abstained,
                generation_mode="ollama",
                citation_validation=(
                    model_validation
                ),
                provider_response=(
                    provider_response
                ),
                provider_error=None,
                fallback_reason=None,
                model_context_bundle=(
                    generation_context_bundle
                ),
            )

        # Safe fallback: never expose an answer with invalid citations.
        fallback_answer = deterministic[
            "answer"
        ]
        fallback_abstained = deterministic[
            "abstained"
        ]

        fallback_validation = (
            validate_answer_citations(
                answer=fallback_answer,
                sources=(
                    generation_context_bundle[
                        "sources"
                    ]
                ),
                abstained=fallback_abstained,
            )
        )

        if not fallback_validation["valid"]:
            raise ValueError(
                "Both Ollama generation and the "
                "deterministic fallback failed "
                "citation validation."
            )

        fallback_reason = (
            "Ollama output failed validation: "
            + "; ".join(
                model_validation["errors"]
            )
        )

        return self._build_result(
            query=clean_query,
            context_bundle=context_bundle,
            answer=fallback_answer,
            abstained=fallback_abstained,
            generation_mode=(
                "deterministic_fallback"
            ),
            citation_validation=(
                fallback_validation
            ),
            provider_response=provider_response,
            provider_error=provider_error,
            fallback_reason=fallback_reason,
            model_context_bundle=(
                generation_context_bundle
            ),
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
