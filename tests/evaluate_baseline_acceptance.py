from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieve import SemanticRetriever


CASES_PATH = PROJECT_ROOT / "tests" / "baseline_acceptance_cases.json"
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "baseline_retriever_acceptance.json"
)

TOP_K_VALUES = (1, 3, 5)


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open("r", encoding="utf-8-sig") as file:
        cases = json.load(file)

    case_ids = [case["case_id"] for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case_id values found.")

    return cases


def first_relevant_rank(
    results: list[dict[str, Any]],
    expected_chunk_ids: set[str],
) -> int | None:
    ranks = [
        result["rank"]
        for result in results
        if result["chunk_id"] in expected_chunk_ids
    ]

    return min(ranks) if ranks else None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0

    return 1.0 / rank


def compact_results(
    results: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": result["rank"],
            "score": round(result["score"], 4),
            "chunk_id": result["chunk_id"],
            "file_type": result["file_type"],
            "citation": result["citation"],
        }
        for result in results[:limit]
    ]


def summarise(
    evaluations: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    ranks = [
        evaluation[f"{prefix}_rank"]
        for evaluation in evaluations
    ]

    summary: dict[str, Any] = {
        "case_count": len(evaluations),
        "mrr": round(
            sum(reciprocal_rank(rank) for rank in ranks)
            / len(ranks),
            4,
        ),
        "mean_rank": round(
            sum(rank for rank in ranks if rank is not None)
            / sum(rank is not None for rank in ranks),
            2,
        )
        if any(rank is not None for rank in ranks)
        else None,
    }

    for k in TOP_K_VALUES:
        hit_count = sum(
            rank is not None and rank <= k
            for rank in ranks
        )

        summary[f"hit_at_{k}"] = hit_count
        summary[f"hit_rate_at_{k}"] = round(
            hit_count / len(ranks),
            4,
        )

    return summary


def main() -> None:
    cases = load_cases()
    retriever = SemanticRetriever(project_root=PROJECT_ROOT)

    available_chunk_ids = {
        chunk["chunk_id"]
        for chunk in retriever.rows
    }

    evaluations: list[dict[str, Any]] = []

    for case in cases:
        expected_chunk_ids = set(case["expected_chunk_ids"])

        missing_chunks = expected_chunk_ids - available_chunk_ids

        if missing_chunks:
            raise ValueError(
                f"{case['case_id']} refers to missing chunks: "
                f"{sorted(missing_chunks)}"
            )

        global_results = retriever.search(
            query=case["query"],
            top_k=len(retriever.rows),
        )

        filtered_results = retriever.search(
            query=case["query"],
            top_k=len(retriever.rows),
            filters=case["filters"],
        )

        global_rank = first_relevant_rank(
            global_results,
            expected_chunk_ids,
        )
        filtered_rank = first_relevant_rank(
            filtered_results,
            expected_chunk_ids,
        )

        evaluations.append(
            {
                "case_id": case["case_id"],
                "file_type": case["file_type"],
                "query": case["query"],
                "reference_answer": case["reference_answer"],
                "expected_chunk_ids": sorted(expected_chunk_ids),
                "filters": case["filters"],
                "global_rank": global_rank,
                "filtered_rank": filtered_rank,
                "global_hit_at_5": (
                    global_rank is not None
                    and global_rank <= 5
                ),
                "filtered_hit_at_5": (
                    filtered_rank is not None
                    and filtered_rank <= 5
                ),
                "global_top5": compact_results(global_results),
                "filtered_top5": compact_results(filtered_results),
            }
        )

    by_file_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for evaluation in evaluations:
        by_file_type[evaluation["file_type"]].append(evaluation)

    report = {
        "status": "completed",
        "retriever": {
            "type": "dense_exact_cosine",
            "model": retriever.model_name,
            "corpus_size": len(retriever.rows),
            "embedding_dimensions": retriever.embeddings.shape[1],
        },
        "case_count": len(evaluations),
        "global_summary": summarise(evaluations, "global"),
        "filtered_summary": summarise(evaluations, "filtered"),
        "summary_by_file_type": {
            file_type: {
                "global": summarise(items, "global"),
                "filtered": summarise(items, "filtered"),
            }
            for file_type, items in sorted(by_file_type.items())
        },
        "cases": evaluations,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    print("=" * 96)
    print("BASELINE RETRIEVER ACCEPTANCE EVALUATION")
    print("=" * 96)
    print(
        f"{'CASE':42} {'TYPE':9} "
        f"{'GLOBAL':12} {'FILTERED':12}"
    )
    print("-" * 96)

    for evaluation in evaluations:
        global_label = (
            str(evaluation["global_rank"])
            if evaluation["global_rank"] is not None
            else "MISS"
        )
        filtered_label = (
            str(evaluation["filtered_rank"])
            if evaluation["filtered_rank"] is not None
            else "MISS"
        )

        print(
            f"{evaluation['case_id'][:42]:42} "
            f"{evaluation['file_type']:9} "
            f"{global_label:12} "
            f"{filtered_label:12}"
        )

    print()
    print("=" * 96)
    print("GLOBAL SUMMARY")
    print("=" * 96)
    print(
        json.dumps(
            report["global_summary"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("=" * 96)
    print("FILTERED SUMMARY")
    print("=" * 96)
    print(
        json.dumps(
            report["filtered_summary"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(f"Report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
