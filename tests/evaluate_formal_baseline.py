from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieve import SemanticRetriever


CASES_PATH = (
    PROJECT_ROOT
    / "tests"
    / "formal_evaluation_cases.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "formal_baseline_retriever_evaluation.json"
)

TOP_K_VALUES = (1, 3, 5, 10)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def first_relevant_rank(
    results: list[dict[str, Any]],
    expected_chunk_ids: set[str],
) -> int | None:
    relevant_ranks = [
        result["rank"]
        for result in results
        if result["chunk_id"] in expected_chunk_ids
    ]

    if not relevant_ranks:
        return None

    return min(relevant_ranks)


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0

    return 1.0 / rank


def compact_results(
    results: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": result["rank"],
            "score": round(result["score"], 4),
            "chunk_id": result["chunk_id"],
            "document_id": result["document_id"],
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

    found_ranks = [
        rank
        for rank in ranks
        if rank is not None
    ]

    case_count = len(evaluations)

    summary: dict[str, Any] = {
        "case_count": case_count,
        "retrieved_case_count": len(found_ranks),
        "miss_count": case_count - len(found_ranks),
        "mrr": round(
            sum(
                reciprocal_rank(rank)
                for rank in ranks
            )
            / case_count,
            4,
        ),
        "mean_rank": (
            round(
                sum(found_ranks) / len(found_ranks),
                2,
            )
            if found_ranks
            else None
        ),
        "median_rank": (
            round(
                float(statistics.median(found_ranks)),
                2,
            )
            if found_ranks
            else None
        ),
        "best_rank": min(found_ranks) if found_ranks else None,
        "worst_rank": max(found_ranks) if found_ranks else None,
    }

    for k in TOP_K_VALUES:
        hit_count = sum(
            rank is not None and rank <= k
            for rank in ranks
        )

        summary[f"hit_at_{k}"] = hit_count
        summary[f"hit_rate_at_{k}"] = round(
            hit_count / case_count,
            4,
        )

    return summary


def grouped_summaries(
    evaluations: list[dict[str, Any]],
    grouping_field: str,
) -> dict[str, Any]:
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for evaluation in evaluations:
        groups[str(evaluation[grouping_field])].append(
            evaluation
        )

    return {
        group_name: {
            "global": summarise(items, "global"),
            "filtered": summarise(items, "filtered"),
        }
        for group_name, items in sorted(groups.items())
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cases = load_json(CASES_PATH)

    if not isinstance(cases, list):
        raise ValueError(
            "formal_evaluation_cases.json must contain a list."
        )

    if len(cases) != 30:
        raise ValueError(
            f"Expected 30 cases, found {len(cases)}."
        )

    retriever = SemanticRetriever(
        project_root=PROJECT_ROOT,
    )

    available_chunk_ids = {
        chunk["chunk_id"]
        for chunk in retriever.rows
    }

    evaluations: list[dict[str, Any]] = []

    for case_number, case in enumerate(cases, start=1):
        expected_chunk_ids = set(
            case["expected_chunk_ids"]
        )

        missing_chunk_ids = (
            expected_chunk_ids - available_chunk_ids
        )

        if missing_chunk_ids:
            raise ValueError(
                f"{case['case_id']} references missing chunks: "
                f"{sorted(missing_chunk_ids)}"
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

        rank_improvement = None

        if (
            global_rank is not None
            and filtered_rank is not None
        ):
            rank_improvement = (
                global_rank - filtered_rank
            )

        evaluations.append(
            {
                "case_number": case_number,
                "case_id": case["case_id"],
                "file_type": case["file_type"],
                "document_id": case["document_id"],
                "difficulty": case["difficulty"],
                "query_type": case["query_type"],
                "tags": case["tags"],
                "query": case["query"],
                "reference_answer": case[
                    "reference_answer"
                ],
                "expected_chunk_ids": sorted(
                    expected_chunk_ids
                ),
                "filters": case["filters"],
                "global_rank": global_rank,
                "filtered_rank": filtered_rank,
                "rank_improvement_with_filters": (
                    rank_improvement
                ),
                "global_hit_at_1": (
                    global_rank is not None
                    and global_rank <= 1
                ),
                "global_hit_at_3": (
                    global_rank is not None
                    and global_rank <= 3
                ),
                "global_hit_at_5": (
                    global_rank is not None
                    and global_rank <= 5
                ),
                "global_hit_at_10": (
                    global_rank is not None
                    and global_rank <= 10
                ),
                "filtered_hit_at_1": (
                    filtered_rank is not None
                    and filtered_rank <= 1
                ),
                "filtered_hit_at_3": (
                    filtered_rank is not None
                    and filtered_rank <= 3
                ),
                "filtered_hit_at_5": (
                    filtered_rank is not None
                    and filtered_rank <= 5
                ),
                "filtered_hit_at_10": (
                    filtered_rank is not None
                    and filtered_rank <= 10
                ),
                "global_top10": compact_results(
                    global_results,
                    limit=10,
                ),
                "filtered_top10": compact_results(
                    filtered_results,
                    limit=10,
                ),
            }
        )

    global_summary = summarise(
        evaluations,
        "global",
    )

    filtered_summary = summarise(
        evaluations,
        "filtered",
    )

    hardest_global_cases = sorted(
        evaluations,
        key=lambda item: (
            item["global_rank"] is None,
            -(
                item["global_rank"]
                if item["global_rank"] is not None
                else len(retriever.rows) + 1
            ),
        ),
    )[:10]

    report = {
        "status": "completed",
        "evaluation_name": (
            "formal_baseline_retriever_evaluation"
        ),
        "retriever": {
            "type": "dense_exact_cosine",
            "model": retriever.model_name,
            "corpus_size": len(retriever.rows),
            "embedding_dimensions": int(
                retriever.embeddings.shape[1]
            ),
            "normalised_embeddings": True,
        },
        "dataset": {
            "path": str(
                CASES_PATH.relative_to(PROJECT_ROOT)
            ),
            "case_count": len(cases),
            "expected_chunk_count": len(
                {
                    chunk_id
                    for case in cases
                    for chunk_id in case[
                        "expected_chunk_ids"
                    ]
                }
            ),
        },
        "global_summary": global_summary,
        "filtered_summary": filtered_summary,
        "summary_by_file_type": grouped_summaries(
            evaluations,
            "file_type",
        ),
        "summary_by_difficulty": grouped_summaries(
            evaluations,
            "difficulty",
        ),
        "summary_by_query_type": grouped_summaries(
            evaluations,
            "query_type",
        ),
        "hardest_global_cases": [
            {
                "case_id": evaluation["case_id"],
                "file_type": evaluation["file_type"],
                "difficulty": evaluation["difficulty"],
                "query_type": evaluation["query_type"],
                "global_rank": evaluation[
                    "global_rank"
                ],
                "filtered_rank": evaluation[
                    "filtered_rank"
                ],
                "query": evaluation["query"],
            }
            for evaluation in hardest_global_cases
        ],
        "cases": evaluations,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
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

    print("=" * 112)
    print("FORMAL BASELINE RETRIEVER EVALUATION")
    print("=" * 112)
    print(
        f"{'CASE':46} "
        f"{'TYPE':10} "
        f"{'DIFFICULTY':11} "
        f"{'GLOBAL':10} "
        f"{'FILTERED':10}"
    )
    print("-" * 112)

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
            f"{evaluation['case_id'][:46]:46} "
            f"{evaluation['file_type']:10} "
            f"{evaluation['difficulty']:11} "
            f"{global_label:10} "
            f"{filtered_label:10}"
        )

    print()
    print("=" * 112)
    print("GLOBAL SUMMARY")
    print("=" * 112)
    print(
        json.dumps(
            global_summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("=" * 112)
    print("FILTERED SUMMARY")
    print("=" * 112)
    print(
        json.dumps(
            filtered_summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("=" * 112)
    print("TEN HARDEST GLOBAL CASES")
    print("=" * 112)

    for position, evaluation in enumerate(
        hardest_global_cases,
        start=1,
    ):
        print(
            f"{position:>2}. "
            f"rank={evaluation['global_rank']} | "
            f"{evaluation['case_id']} | "
            f"{evaluation['file_type']} | "
            f"{evaluation['difficulty']}"
        )

    print()
    print(
        "Report saved to: "
        f"{REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
