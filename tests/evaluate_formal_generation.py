from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_retrieve import HybridRetriever
from src.ollama_adapter import OllamaChatAdapter
from src.ollama_rag import OllamaGroundedRAG
from src.rag_answer import (
    ABSTENTION_MESSAGE,
    CITATION_PATTERN,
    meaningful_tokens,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "hybrid_retriever.json"
SUPPORTED_CASES_PATH = PROJECT_ROOT / "tests" / "formal_evaluation_cases.json"
UNSUPPORTED_CASES_PATH = (
    PROJECT_ROOT / "tests" / "formal_generation_unsupported_cases.json"
)
MANIFEST_PATH = PROJECT_ROOT / "reports" / "formal_evaluation_manifest.json"

RAW_RESULTS_PATH = (
    PROJECT_ROOT / "reports" / "formal_rag_generation_raw.jsonl"
)
JSON_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "formal_rag_generation_evaluation.json"
)
MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "formal_rag_generation_evaluation.md"
)
REVIEW_CSV_PATH = (
    PROJECT_ROOT / "reports" / "formal_rag_generation_review.csv"
)

NUMBER_PATTERN = re.compile(
    r"(?ix)"
    r"(?<![A-Za-z0-9])"
    r"(?:AUD|\$)?\s*"
    r"(?P<number>[-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
    r"\s*(?P<scale>%|percent(?:age)?|million|billion|thousand|bn|m|k)?"
    r"(?![A-Za-z])"
)
CONTEXT_BLOCK_PATTERN = re.compile(
    r"(?ms)^\[(\d+)\]\n"
    r"Title:.*?\n"
    r"Citation:.*?\n"
    r"File type:.*?\n"
    r"Content:\s*(.*?)"
    r"(?=\n\n\[\d+\]\n|\Z)"
)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")

SEMANTIC_REVIEW_THRESHOLD = 0.72
CLAIM_SUPPORT_THRESHOLD = 0.35


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def calculate_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {path} at line {line_number}."
                )

            rows.append(row)

    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False))
        file.write("\n")


def canonical_numeric_value(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(value)))

    return f"{value:.10f}".rstrip("0").rstrip(".")


def numeric_tokens(text: str) -> set[str]:
    """
    Extract canonical numeric facts while recognising equivalent formats.

    Examples treated as equivalent include ``AUD 12.2 million`` and
    ``12,200,000``, or ``90%`` and ``0.9``. Citation markers are removed
    before extraction so source numbers do not become financial facts.
    """
    cleaned = CITATION_PATTERN.sub("", text)
    values: set[str] = set()

    scale_factors = {
        "thousand": 1_000.0,
        "k": 1_000.0,
        "million": 1_000_000.0,
        "m": 1_000_000.0,
        "billion": 1_000_000_000.0,
        "bn": 1_000_000_000.0,
    }

    for match in NUMBER_PATTERN.finditer(cleaned):
        raw_number = match.group("number").replace(",", "")
        scale = (match.group("scale") or "").casefold()

        try:
            value = float(raw_number)
        except ValueError:
            continue

        if scale in scale_factors:
            values.add(
                canonical_numeric_value(value * scale_factors[scale])
            )
            continue

        if scale in {"%", "percent", "percentage"}:
            values.add(canonical_numeric_value(value))
            values.add(canonical_numeric_value(value / 100.0))
            continue

        values.add(canonical_numeric_value(value))

        if 0.0 < abs(value) <= 1.0:
            values.add(canonical_numeric_value(value * 100.0))

    return values


def token_recall(reference: str, answer: str) -> float | None:
    reference_tokens = meaningful_tokens(reference)

    if not reference_tokens:
        return None

    answer_tokens = meaningful_tokens(answer)

    return len(reference_tokens & answer_tokens) / len(reference_tokens)


def numeric_recall(reference: str, answer: str) -> float | None:
    reference_numbers = numeric_tokens(reference)

    if not reference_numbers:
        return None

    answer_numbers = numeric_tokens(answer)

    return len(reference_numbers & answer_numbers) / len(reference_numbers)


def semantic_similarity(
    model: Any,
    reference: str,
    answer: str,
) -> float | None:
    if not reference.strip() or not answer.strip():
        return None

    vectors = model.encode(
        [reference, answer],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    return float(vectors[0] @ vectors[1])


def parse_numbered_context(context: str) -> dict[int, str]:
    return {
        int(match.group(1)): " ".join(match.group(2).split())
        for match in CONTEXT_BLOCK_PATTERN.finditer(context)
    }


def split_claims(answer: str) -> list[str]:
    raw_units = [
        unit.strip()
        for unit in SENTENCE_SPLIT_PATTERN.split(answer.strip())
        if unit.strip()
    ]

    claims: list[str] = []

    for unit in raw_units:
        if (
            claims
            and re.fullmatch(r"(?:\[\d+\]\s*)+", unit)
        ):
            claims[-1] = f"{claims[-1]} {unit}".strip()
        else:
            claims.append(unit)

    return claims


def citation_claim_support(
    *,
    answer: str,
    model_context: str,
    model: Any,
    abstained: bool,
) -> dict[str, Any]:
    if abstained:
        return {
            "claim_count": 0,
            "supported_claim_count": 0,
            "grounded_claim_rate": None,
            "claims": [],
        }

    source_text = parse_numbered_context(model_context)
    claim_rows: list[dict[str, Any]] = []

    for claim in split_claims(answer):
        citations = sorted(
            {
                int(value)
                for value in CITATION_PATTERN.findall(claim)
            }
        )
        claim_text = CITATION_PATTERN.sub("", claim).strip()

        if not claim_text:
            continue

        cited_texts = [
            source_text[number]
            for number in citations
            if number in source_text
        ]
        combined_evidence = " ".join(cited_texts)

        similarity = None

        if combined_evidence:
            similarity = semantic_similarity(
                model=model,
                reference=combined_evidence,
                answer=claim_text,
            )

        claim_numbers = numeric_tokens(claim_text)
        evidence_numbers = numeric_tokens(combined_evidence)

        numeric_consistent = (
            claim_numbers <= evidence_numbers
            if claim_numbers
            else True
        )

        known_citations = (
            bool(citations)
            and all(number in source_text for number in citations)
        )

        claim_tokens = meaningful_tokens(claim_text)
        evidence_tokens = meaningful_tokens(combined_evidence)
        token_support = (
            len(claim_tokens & evidence_tokens) / len(claim_tokens)
            if claim_tokens
            else 0.0
        )

        supported = bool(
            known_citations
            and numeric_consistent
            and (
                (
                    similarity is not None
                    and similarity >= CLAIM_SUPPORT_THRESHOLD
                )
                or token_support >= 0.60
            )
        )

        claim_rows.append(
            {
                "claim": claim,
                "claim_text": claim_text,
                "citations": citations,
                "known_citations": known_citations,
                "numeric_tokens": sorted(claim_numbers),
                "numeric_consistent": numeric_consistent,
                "semantic_support": (
                    round(similarity, 4)
                    if similarity is not None
                    else None
                ),
                "token_support": round(token_support, 4),
                "supported_auto": supported,
            }
        )

    supported_count = sum(
        row["supported_auto"] for row in claim_rows
    )

    grounded_rate = (
        supported_count / len(claim_rows)
        if claim_rows
        else 0.0
    )

    return {
        "claim_count": len(claim_rows),
        "supported_claim_count": supported_count,
        "grounded_claim_rate": round(grounded_rate, 4),
        "claims": claim_rows,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def rounded_mean(values: list[float]) -> float | None:
    return (
        round(statistics.mean(values), 4)
        if values
        else None
    )


def public_source_chunk_ids(
    sources: list[dict[str, Any]],
) -> set[str]:
    return {
        str(source["chunk_id"])
        for source in sources
    }


def automated_quality_flag(
    *,
    should_abstain: bool,
    result: dict[str, Any],
    query: str,
    reference_answer: str,
    answer: str,
    expected_chunk_model_hit: bool | None,
    expected_chunk_cited: bool | None,
    semantic_score: float | None,
    lexical_score: float | None,
    numeric_score: float | None,
    grounded_claim_rate: float | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if should_abstain:
        if not result["abstained"]:
            return (
                "fail",
                ["The system answered a case that should be abstained."],
            )

        if result["generation_mode"] != "retrieval_abstention":
            reasons.append(
                "Abstention did not occur before model generation."
            )

        if result["citation_validation"]["used_citations"]:
            reasons.append(
                "The abstention unexpectedly contained citations."
            )

        return (
            "pass" if not reasons else "review",
            reasons,
        )

    if result["abstained"]:
        return (
            "fail",
            ["The system abstained on a supported question."],
        )

    if not result["citation_validation"]["valid"]:
        reasons.append("Citation validation failed.")

    if expected_chunk_model_hit is False:
        reasons.append(
            "The expected evidence chunk was not supplied to the model."
        )

    if expected_chunk_cited is False:
        reasons.append(
            "The answer did not cite an expected evidence chunk."
        )

    lower_reference = reference_answer.casefold()
    lower_answer = answer.casefold()

    polarity_conflict = (
        (
            "below budget" in lower_reference
            and "unfavourable" in lower_answer
        )
        or (
            "above budget" in lower_reference
            and "favourable" in lower_answer
            and "unfavourable" not in lower_answer
        )
    )

    if polarity_conflict:
        reasons.append(
            "The answer appears to reverse the favourable/unfavourable "
            "direction stated in the reference."
        )

    hard_failure = any(
        phrase in " ".join(reasons)
        for phrase in (
            "Citation validation failed",
            "reverse the favourable",
        )
    )

    if hard_failure:
        return "fail", reasons

    if numeric_score is not None and numeric_score < 1.0:
        reasons.append(
            "At least one reference numeric fact requires review."
        )

        if numeric_score < 0.5:
            return "fail", reasons

    if (
        any(
            phrase in query.casefold()
            for phrase in ("formula", "calculated", "how is", "how are")
        )
        and lexical_score is not None
        and lexical_score < 0.5
    ):
        reasons.append(
            "Formula or calculation coverage requires review."
        )

    if (
        semantic_score is not None
        and semantic_score < SEMANTIC_REVIEW_THRESHOLD
    ):
        reasons.append(
            "Answer/reference semantic similarity is below the review threshold."
        )

    if (
        grounded_claim_rate is not None
        and grounded_claim_rate < 1.0
    ):
        reasons.append(
            "At least one cited claim requires grounding review."
        )

    return (
        "pass" if not reasons else "review",
        reasons,
    )


def evaluate_case(
    *,
    case: dict[str, Any],
    rag: OllamaGroundedRAG,
    embedding_model: Any,
) -> dict[str, Any]:
    started = time.perf_counter()

    result = rag.answer(case["query"])

    wall_seconds = time.perf_counter() - started

    should_abstain = bool(case["should_abstain"])
    expected_chunk_ids = set(case.get("expected_chunk_ids", []))

    retrieved_ids = public_source_chunk_ids(
        result.get("sources", [])
    )
    model_ids = public_source_chunk_ids(
        result.get("model_sources", [])
    )
    cited_ids = public_source_chunk_ids(
        result.get("cited_sources", [])
    )

    expected_chunk_retrieved = (
        bool(expected_chunk_ids & retrieved_ids)
        if expected_chunk_ids
        else None
    )
    expected_chunk_model_hit = (
        bool(expected_chunk_ids & model_ids)
        if expected_chunk_ids
        else None
    )
    expected_chunk_cited = (
        bool(expected_chunk_ids & cited_ids)
        if expected_chunk_ids
        else None
    )

    reference_answer = str(
        case.get("reference_answer", "")
    ).strip()

    answer = str(result["answer"]).strip()

    semantic_score = (
        semantic_similarity(
            model=embedding_model,
            reference=reference_answer,
            answer=answer,
        )
        if reference_answer and not result["abstained"]
        else None
    )
    lexical_recall = (
        token_recall(reference_answer, answer)
        if reference_answer and not result["abstained"]
        else None
    )
    number_recall = (
        numeric_recall(reference_answer, answer)
        if reference_answer and not result["abstained"]
        else None
    )

    claim_support = citation_claim_support(
        answer=answer,
        model_context=result.get("model_context", ""),
        model=embedding_model,
        abstained=bool(result["abstained"]),
    )

    quality_flag, quality_reasons = automated_quality_flag(
        should_abstain=should_abstain,
        result=result,
        query=case["query"],
        reference_answer=reference_answer,
        answer=answer,
        expected_chunk_model_hit=expected_chunk_model_hit,
        expected_chunk_cited=expected_chunk_cited,
        semantic_score=semantic_score,
        lexical_score=lexical_recall,
        numeric_score=number_recall,
        grounded_claim_rate=claim_support[
            "grounded_claim_rate"
        ],
    )

    provider_metrics = result.get("provider_metrics")
    provider_seconds = None

    if provider_metrics:
        total_duration = provider_metrics.get(
            "total_duration"
        )

        if isinstance(total_duration, (int, float)):
            provider_seconds = float(total_duration) / 1_000_000_000

    return {
        "case_id": case["case_id"],
        "case_group": case["case_group"],
        "should_abstain": should_abstain,
        "file_type": case.get("file_type"),
        "document_id": case.get("document_id"),
        "difficulty": case.get("difficulty"),
        "query_type": case.get("query_type"),
        "tags": case.get("tags", []),
        "query": case["query"],
        "reference_answer": reference_answer or None,
        "expected_chunk_ids": sorted(expected_chunk_ids),
        "answer": answer,
        "abstained": bool(result["abstained"]),
        "generation_mode": result["generation_mode"],
        "model": result["model"],
        "citation_valid": bool(
            result["citation_validation"]["valid"]
        ),
        "used_citations": result[
            "citation_validation"
        ]["used_citations"],
        "citation_errors": result[
            "citation_validation"
        ]["errors"],
        "source_count": result["source_count"],
        "model_source_count": result["model_source_count"],
        "cited_source_count": result["cited_source_count"],
        "retrieved_chunk_ids": sorted(retrieved_ids),
        "model_chunk_ids": sorted(model_ids),
        "cited_chunk_ids": sorted(cited_ids),
        "expected_chunk_retrieved": expected_chunk_retrieved,
        "expected_chunk_model_hit": expected_chunk_model_hit,
        "expected_chunk_cited": expected_chunk_cited,
        "semantic_similarity": (
            round(semantic_score, 4)
            if semantic_score is not None
            else None
        ),
        "reference_token_recall": (
            round(lexical_recall, 4)
            if lexical_recall is not None
            else None
        ),
        "numeric_fact_recall": (
            round(number_recall, 4)
            if number_recall is not None
            else None
        ),
        "claim_support": claim_support,
        "automated_quality_flag": quality_flag,
        "automated_review_reasons": quality_reasons,
        "provider_error": result.get("provider_error"),
        "fallback_reason": result.get("fallback_reason"),
        "provider_metrics": provider_metrics,
        "wall_seconds": round(wall_seconds, 3),
        "provider_seconds": (
            round(provider_seconds, 3)
            if provider_seconds is not None
            else None
        ),
    }


def grouped_supported_summary(
    evaluations: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in evaluations:
        groups[str(row.get(field))].append(row)

    summary: dict[str, Any] = {}

    for group, rows in sorted(groups.items()):
        semantic_values = [
            float(row["semantic_similarity"])
            for row in rows
            if row["semantic_similarity"] is not None
        ]

        summary[group] = {
            "case_count": len(rows),
            "answered_count": sum(not row["abstained"] for row in rows),
            "unexpected_abstention_count": sum(
                row["abstained"] for row in rows
            ),
            "citation_valid_rate": round(
                sum(row["citation_valid"] for row in rows)
                / len(rows),
                4,
            ),
            "expected_chunk_model_rate": round(
                sum(
                    row["expected_chunk_model_hit"] is True
                    for row in rows
                )
                / len(rows),
                4,
            ),
            "expected_chunk_cited_rate": round(
                sum(
                    row["expected_chunk_cited"] is True
                    for row in rows
                )
                / len(rows),
                4,
            ),
            "mean_semantic_similarity": rounded_mean(
                semantic_values
            ),
            "quality_flags": dict(
                sorted(
                    Counter(
                        row["automated_quality_flag"]
                        for row in rows
                    ).items()
                )
            ),
        }

    return summary


def build_summary(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    supported = [
        row
        for row in evaluations
        if not row["should_abstain"]
    ]
    unsupported = [
        row
        for row in evaluations
        if row["should_abstain"]
    ]

    semantic_values = [
        float(row["semantic_similarity"])
        for row in supported
        if row["semantic_similarity"] is not None
    ]
    token_values = [
        float(row["reference_token_recall"])
        for row in supported
        if row["reference_token_recall"] is not None
    ]
    numeric_values = [
        float(row["numeric_fact_recall"])
        for row in supported
        if row["numeric_fact_recall"] is not None
    ]
    grounding_values = [
        float(row["claim_support"]["grounded_claim_rate"])
        for row in supported
        if row["claim_support"]["grounded_claim_rate"] is not None
    ]
    wall_values = [
        float(row["wall_seconds"])
        for row in evaluations
    ]
    provider_values = [
        float(row["provider_seconds"])
        for row in evaluations
        if row["provider_seconds"] is not None
    ]

    supported_count = len(supported)
    unsupported_count = len(unsupported)

    return {
        "completed_case_count": len(evaluations),
        "supported": {
            "case_count": supported_count,
            "answered_count": sum(
                not row["abstained"] for row in supported
            ),
            "unexpected_abstention_count": sum(
                row["abstained"] for row in supported
            ),
            "generation_modes": dict(
                sorted(
                    Counter(
                        row["generation_mode"]
                        for row in supported
                    ).items()
                )
            ),
            "citation_valid_rate": (
                round(
                    sum(row["citation_valid"] for row in supported)
                    / supported_count,
                    4,
                )
                if supported_count
                else None
            ),
            "expected_chunk_retrieved_rate": (
                round(
                    sum(
                        row["expected_chunk_retrieved"] is True
                        for row in supported
                    )
                    / supported_count,
                    4,
                )
                if supported_count
                else None
            ),
            "expected_chunk_model_rate": (
                round(
                    sum(
                        row["expected_chunk_model_hit"] is True
                        for row in supported
                    )
                    / supported_count,
                    4,
                )
                if supported_count
                else None
            ),
            "expected_chunk_cited_rate": (
                round(
                    sum(
                        row["expected_chunk_cited"] is True
                        for row in supported
                    )
                    / supported_count,
                    4,
                )
                if supported_count
                else None
            ),
            "mean_semantic_similarity": rounded_mean(
                semantic_values
            ),
            "mean_reference_token_recall": rounded_mean(
                token_values
            ),
            "mean_numeric_fact_recall": rounded_mean(
                numeric_values
            ),
            "mean_grounded_claim_rate": rounded_mean(
                grounding_values
            ),
            "quality_flags": dict(
                sorted(
                    Counter(
                        row["automated_quality_flag"]
                        for row in supported
                    ).items()
                )
            ),
            "by_file_type": grouped_supported_summary(
                supported,
                "file_type",
            ),
            "by_difficulty": grouped_supported_summary(
                supported,
                "difficulty",
            ),
        },
        "unsupported": {
            "case_count": unsupported_count,
            "correct_abstention_count": sum(
                row["abstained"] for row in unsupported
            ),
            "false_answer_count": sum(
                not row["abstained"] for row in unsupported
            ),
            "pre_generation_abstention_count": sum(
                row["generation_mode"] == "retrieval_abstention"
                for row in unsupported
            ),
            "provider_call_avoided_count": sum(
                row["provider_metrics"] is None
                for row in unsupported
            ),
            "citation_valid_rate": (
                round(
                    sum(row["citation_valid"] for row in unsupported)
                    / unsupported_count,
                    4,
                )
                if unsupported_count
                else None
            ),
            "quality_flags": dict(
                sorted(
                    Counter(
                        row["automated_quality_flag"]
                        for row in unsupported
                    ).items()
                )
            ),
        },
        "latency": {
            "wall_seconds_mean": (
                round(statistics.mean(wall_values), 3)
                if wall_values
                else None
            ),
            "wall_seconds_median": (
                round(statistics.median(wall_values), 3)
                if wall_values
                else None
            ),
            "wall_seconds_p95": (
                round(percentile(wall_values, 0.95), 3)
                if wall_values
                else None
            ),
            "provider_seconds_mean": (
                round(statistics.mean(provider_values), 3)
                if provider_values
                else None
            ),
        },
    }


def build_markdown(
    *,
    report: dict[str, Any],
) -> str:
    summary = report["summary"]
    supported = summary["supported"]
    unsupported = summary["unsupported"]
    latency = summary["latency"]

    lines = [
        "# Formal Local RAG Generation Evaluation",
        "",
        "## Evaluation status",
        "",
        f"- Completed cases: {summary['completed_case_count']}",
        f"- Supported cases completed: {supported['case_count']}",
        f"- Unsupported cases completed: {unsupported['case_count']}",
        f"- Model: `{report['configuration']['ollama_model']}`",
        "- Retrieval mode: global hybrid retrieval without metadata filters.",
        "- The frozen 30-case benchmark was not used for retriever tuning.",
        "",
        "## Supported questions",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Answered | {supported['answered_count']} / {supported['case_count']} |",
        f"| Unexpected abstentions | {supported['unexpected_abstention_count']} |",
        f"| Valid citations | {supported['citation_valid_rate'] if supported['citation_valid_rate'] is not None else 'n/a'} |",
        f"| Expected chunk retrieved | {supported['expected_chunk_retrieved_rate'] if supported['expected_chunk_retrieved_rate'] is not None else 'n/a'} |",
        f"| Expected chunk supplied to model | {supported['expected_chunk_model_rate'] if supported['expected_chunk_model_rate'] is not None else 'n/a'} |",
        f"| Expected chunk cited | {supported['expected_chunk_cited_rate'] if supported['expected_chunk_cited_rate'] is not None else 'n/a'} |",
        f"| Mean semantic similarity | {supported['mean_semantic_similarity'] if supported['mean_semantic_similarity'] is not None else 'n/a'} |",
        f"| Mean reference-token recall | {supported['mean_reference_token_recall'] if supported['mean_reference_token_recall'] is not None else 'n/a'} |",
        f"| Mean numeric-fact recall | {supported['mean_numeric_fact_recall'] if supported['mean_numeric_fact_recall'] is not None else 'n/a'} |",
        f"| Mean grounded-claim rate | {supported['mean_grounded_claim_rate'] if supported['mean_grounded_claim_rate'] is not None else 'n/a'} |",
        "",
        "Generation modes:",
        "",
    ]

    for mode, count in supported["generation_modes"].items():
        lines.append(f"- `{mode}`: {count}")

    lines.extend(
        [
            "",
            "Automated quality flags:",
            "",
        ]
    )

    for flag, count in supported["quality_flags"].items():
        lines.append(f"- `{flag}`: {count}")

    lines.extend(
        [
            "",
            "## Unsupported questions",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Correct abstentions | {unsupported['correct_abstention_count']} / {unsupported['case_count']} |",
            f"| Incorrect answers | {unsupported['false_answer_count']} |",
            f"| Abstentions before generation | {unsupported['pre_generation_abstention_count']} |",
            f"| Provider calls avoided | {unsupported['provider_call_avoided_count']} |",
            f"| Valid citation state | {unsupported['citation_valid_rate'] if unsupported['citation_valid_rate'] is not None else 'n/a'} |",
            "",
            "## Latency",
            "",
            "| Metric | Seconds |",
            "|---|---:|",
            f"| Mean wall time | {latency['wall_seconds_mean'] if latency['wall_seconds_mean'] is not None else 'n/a'} |",
            f"| Median wall time | {latency['wall_seconds_median'] if latency['wall_seconds_median'] is not None else 'n/a'} |",
            f"| 95th percentile wall time | {latency['wall_seconds_p95'] if latency['wall_seconds_p95'] is not None else 'n/a'} |",
            f"| Mean Ollama provider time | {latency['provider_seconds_mean'] if latency['provider_seconds_mean'] is not None else 'n/a'} |",
            "",
            "## Manual review",
            "",
            "The CSV review file contains blank columns for human scoring of correctness, completeness, groundedness and citation quality. Automated similarity and overlap metrics are diagnostics, not substitutes for human judgement.",
            "",
            "## Files",
            "",
            f"- Raw checkpoint: `{RAW_RESULTS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- JSON report: `{JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- Review CSV: `{REVIEW_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )

    return "\n".join(lines)


def write_review_csv(
    path: Path,
    evaluations: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "case_group",
        "file_type",
        "difficulty",
        "query_type",
        "query",
        "reference_answer",
        "answer",
        "should_abstain",
        "abstained",
        "generation_mode",
        "citation_valid",
        "expected_chunk_retrieved",
        "expected_chunk_model_hit",
        "expected_chunk_cited",
        "semantic_similarity",
        "reference_token_recall",
        "numeric_fact_recall",
        "grounded_claim_rate",
        "automated_quality_flag",
        "automated_review_reasons",
        "wall_seconds",
        "manual_correctness_0_2",
        "manual_completeness_0_2",
        "manual_groundedness_0_2",
        "manual_citation_quality_0_2",
        "manual_abstention_correct_0_1",
        "manual_notes",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in evaluations:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "case_group": row["case_group"],
                    "file_type": row.get("file_type"),
                    "difficulty": row.get("difficulty"),
                    "query_type": row.get("query_type"),
                    "query": row["query"],
                    "reference_answer": row.get(
                        "reference_answer"
                    ),
                    "answer": row["answer"],
                    "should_abstain": row["should_abstain"],
                    "abstained": row["abstained"],
                    "generation_mode": row["generation_mode"],
                    "citation_valid": row["citation_valid"],
                    "expected_chunk_retrieved": row.get(
                        "expected_chunk_retrieved"
                    ),
                    "expected_chunk_model_hit": row.get(
                        "expected_chunk_model_hit"
                    ),
                    "expected_chunk_cited": row.get(
                        "expected_chunk_cited"
                    ),
                    "semantic_similarity": row.get(
                        "semantic_similarity"
                    ),
                    "reference_token_recall": row.get(
                        "reference_token_recall"
                    ),
                    "numeric_fact_recall": row.get(
                        "numeric_fact_recall"
                    ),
                    "grounded_claim_rate": row[
                        "claim_support"
                    ].get("grounded_claim_rate"),
                    "automated_quality_flag": row[
                        "automated_quality_flag"
                    ],
                    "automated_review_reasons": " | ".join(
                        row["automated_review_reasons"]
                    ),
                    "wall_seconds": row["wall_seconds"],
                    "manual_correctness_0_2": "",
                    "manual_completeness_0_2": "",
                    "manual_groundedness_0_2": "",
                    "manual_citation_quality_0_2": "",
                    "manual_abstention_correct_0_1": "",
                    "manual_notes": "",
                }
            )


def build_report(
    *,
    evaluations: list[dict[str, Any]],
    config: dict[str, Any],
    supported_hash: str,
    unsupported_hash: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "status": "partial" if len(evaluations) < 40 else "completed",
        "evaluation_name": "formal_local_rag_generation_evaluation",
        "datasets": {
            "supported": {
                "path": str(
                    SUPPORTED_CASES_PATH.relative_to(PROJECT_ROOT)
                ),
                "sha256": supported_hash,
                "case_count": 30,
            },
            "unsupported": {
                "path": str(
                    UNSUPPORTED_CASES_PATH.relative_to(PROJECT_ROOT)
                ),
                "sha256": unsupported_hash,
                "case_count": 10,
            },
        },
        "configuration": {
            "retriever": config,
            "ollama_model": model_name,
            "top_k": 5,
            "semantic_review_threshold": SEMANTIC_REVIEW_THRESHOLD,
            "claim_support_threshold": CLAIM_SUPPORT_THRESHOLD,
            "evaluation_filters": {},
        },
        "summary": build_summary(evaluations),
        "cases": evaluations,
    }


def prepare_cases(
    supported_cases: list[dict[str, Any]],
    unsupported_cases: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    supported = [
        {
            **case,
            "case_group": "supported",
            "should_abstain": False,
        }
        for case in supported_cases
    ]

    unsupported = [
        {
            **case,
            "case_group": "unsupported",
            "should_abstain": True,
        }
        for case in unsupported_cases
    ]

    if mode == "supported":
        return supported

    if mode == "unsupported":
        return unsupported

    return supported + unsupported


def validate_frozen_dataset(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    supported_cases: list[dict[str, Any]],
) -> str:
    if len(supported_cases) != 30:
        raise ValueError(
            f"Expected 30 supported cases, found {len(supported_cases)}."
        )

    dataset_hash = calculate_sha256(SUPPORTED_CASES_PATH)

    expected_hashes = {
        config["formal_dataset_sha256"],
        manifest["dataset_sha256"],
    }

    if len(expected_hashes) != 1 or dataset_hash not in expected_hashes:
        raise ValueError(
            "The supported benchmark no longer matches the frozen hash."
        )

    if config["formal_benchmark_used_for_tuning"]:
        raise ValueError(
            "The configuration incorrectly states that the formal "
            "benchmark was used for tuning."
        )

    return dataset_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the complete local Ollama RAG on supported and "
            "unsupported formal questions. Results are checkpointed "
            "after every case and runs resume automatically."
        )
    )

    parser.add_argument(
        "--mode",
        choices=("all", "supported", "unsupported"),
        default="all",
        help="Cases to run. Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Maximum number of incomplete cases to run in this invocation."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete prior raw results and start again.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help=(
            "Run only a specific case_id. Repeat the option for multiple "
            "cases."
        ),
    )

    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    config = load_json(CONFIG_PATH)
    supported_cases = load_json(SUPPORTED_CASES_PATH)
    unsupported_cases = load_json(UNSUPPORTED_CASES_PATH)
    manifest = load_json(MANIFEST_PATH)

    supported_hash = validate_frozen_dataset(
        config=config,
        manifest=manifest,
        supported_cases=supported_cases,
    )

    if len(unsupported_cases) != 10:
        raise ValueError(
            f"Expected 10 unsupported cases, found {len(unsupported_cases)}."
        )

    unsupported_ids = [
        case["case_id"] for case in unsupported_cases
    ]

    if len(unsupported_ids) != len(set(unsupported_ids)):
        raise ValueError(
            "Unsupported evaluation case_id values must be unique."
        )

    unsupported_hash = calculate_sha256(
        UNSUPPORTED_CASES_PATH
    )

    if args.reset:
        for path in (
            RAW_RESULTS_PATH,
            JSON_REPORT_PATH,
            MARKDOWN_REPORT_PATH,
            REVIEW_CSV_PATH,
        ):
            if path.exists():
                path.unlink()

    cases = prepare_cases(
        supported_cases=supported_cases,
        unsupported_cases=unsupported_cases,
        mode=args.mode,
    )

    if args.case_id:
        requested = set(args.case_id)
        available = {case["case_id"] for case in cases}
        missing = requested - available

        if missing:
            raise ValueError(
                "Unknown case_id values: "
                + ", ".join(sorted(missing))
            )

        cases = [
            case
            for case in cases
            if case["case_id"] in requested
        ]

    existing_rows = load_jsonl(RAW_RESULTS_PATH)
    completed_ids = {
        row["case_id"] for row in existing_rows
    }

    pending = [
        case
        for case in cases
        if case["case_id"] not in completed_ids
    ]

    if args.limit is not None:
        pending = pending[: args.limit]

    print("=" * 96)
    print("FORMAL LOCAL RAG GENERATION EVALUATION")
    print("=" * 96)
    print(f"Completed before this run: {len(existing_rows)}")
    print(f"Pending selected for this run: {len(pending)}")
    print(f"Mode: {args.mode}")
    print()

    retriever = HybridRetriever(
        project_root=PROJECT_ROOT,
        dense_weight=float(config["dense_weight"]),
        rrf_k=int(config["rrf_k"]),
        bm25_k1=float(config["bm25_k1"]),
        bm25_b=float(config["bm25_b"]),
    )

    adapter = OllamaChatAdapter()
    adapter.ensure_model_available()

    rag = OllamaGroundedRAG(
        retriever=retriever,
        adapter=adapter,
        top_k=5,
    )

    total_selected = len(pending)

    for index, case in enumerate(pending, start=1):
        print(
            f"[{index}/{total_selected}] {case['case_id']} "
            f"({case['case_group']})"
        )

        try:
            evaluation = evaluate_case(
                case=case,
                rag=rag,
                embedding_model=retriever.semantic.model,
            )
        except Exception as exc:
            evaluation = {
                "case_id": case["case_id"],
                "case_group": case["case_group"],
                "should_abstain": case["should_abstain"],
                "file_type": case.get("file_type"),
                "document_id": case.get("document_id"),
                "difficulty": case.get("difficulty"),
                "query_type": case.get("query_type"),
                "tags": case.get("tags", []),
                "query": case["query"],
                "reference_answer": case.get("reference_answer"),
                "answer": "",
                "abstained": False,
                "generation_mode": "error",
                "model": adapter.model,
                "citation_valid": False,
                "used_citations": [],
                "citation_errors": [str(exc)],
                "source_count": 0,
                "model_source_count": 0,
                "cited_source_count": 0,
                "retrieved_chunk_ids": [],
                "model_chunk_ids": [],
                "cited_chunk_ids": [],
                "expected_chunk_retrieved": None,
                "expected_chunk_model_hit": None,
                "expected_chunk_cited": None,
                "semantic_similarity": None,
                "reference_token_recall": None,
                "numeric_fact_recall": None,
                "claim_support": {
                    "claim_count": 0,
                    "supported_claim_count": 0,
                    "grounded_claim_rate": None,
                    "claims": [],
                },
                "automated_quality_flag": "fail",
                "automated_review_reasons": [
                    f"Evaluation error: {exc}"
                ],
                "provider_error": str(exc),
                "fallback_reason": None,
                "provider_metrics": None,
                "wall_seconds": 0.0,
                "provider_seconds": None,
            }

        append_jsonl(RAW_RESULTS_PATH, evaluation)

        print(
            "  "
            f"mode={evaluation['generation_mode']} | "
            f"abstained={evaluation['abstained']} | "
            f"flag={evaluation['automated_quality_flag']} | "
            f"time={evaluation['wall_seconds']}s"
        )

    evaluations = load_jsonl(RAW_RESULTS_PATH)
    evaluations.sort(
        key=lambda row: (
            0 if row["case_group"] == "supported" else 1,
            row["case_id"],
        )
    )

    report = build_report(
        evaluations=evaluations,
        config=config,
        supported_hash=supported_hash,
        unsupported_hash=unsupported_hash,
        model_name=adapter.model,
    )

    JSON_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with JSON_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    MARKDOWN_REPORT_PATH.write_text(
        build_markdown(report=report),
        encoding="utf-8",
    )

    write_review_csv(
        REVIEW_CSV_PATH,
        evaluations,
    )

    print()
    print("=" * 96)
    print("CURRENT SUMMARY")
    print("=" * 96)
    print(
        json.dumps(
            report["summary"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print(
        "Raw checkpoint: "
        f"{RAW_RESULTS_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "JSON report: "
        f"{JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Markdown report: "
        f"{MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Manual review CSV: "
        f"{REVIEW_CSV_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
