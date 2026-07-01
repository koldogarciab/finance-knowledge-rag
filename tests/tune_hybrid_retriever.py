from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_retrieve import HybridRetriever


CASES_PATH = (
    PROJECT_ROOT
    / "tests"
    / "baseline_acceptance_cases.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "hybrid_retriever_tuning.json"
)

DENSE_WEIGHTS = [
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
]

RRF_VALUES = [
    10,
    30,
    60,
]

TOP_K_VALUES = (
    1,
    3,
    5,
    10,
)


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError(
            "The acceptance dataset must contain a list."
        )

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


def summarise(
    ranks: list[int | None],
) -> dict[str, Any]:
    found_ranks = [
        rank
        for rank in ranks
        if rank is not None
    ]

    case_count = len(ranks)

    summary: dict[str, Any] = {
        "case_count": case_count,
        "mrr": round(
            sum(
                0.0 if rank is None else 1.0 / rank
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
        "worst_rank": (
            max(found_ranks)
            if found_ranks
            else None
        ),
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


def selection_key(
    result: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    summary = result["summary"]

    return (
        summary["hit_rate_at_5"],
        summary["hit_rate_at_3"],
        summary["mrr"],
        summary["hit_rate_at_1"],
        -result["dense_weight"],
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cases = load_cases()

    configurations: list[dict[str, Any]] = []

    for dense_weight in DENSE_WEIGHTS:
        for rrf_k in RRF_VALUES:
            print(
                "Evaluating "
                f"dense_weight={dense_weight:.2f}, "
                f"rrf_k={rrf_k}"
            )

            retriever = HybridRetriever(
                project_root=PROJECT_ROOT,
                dense_weight=dense_weight,
                rrf_k=rrf_k,
            )

            ranks: list[int | None] = []
            case_results: list[dict[str, Any]] = []

            for case in cases:
                expected_chunk_ids = set(
                    case["expected_chunk_ids"]
                )

                results = retriever.search(
                    query=case["query"],
                    top_k=len(retriever.rows),
                )

                rank = first_relevant_rank(
                    results,
                    expected_chunk_ids,
                )

                ranks.append(rank)

                case_results.append(
                    {
                        "case_id": case["case_id"],
                        "file_type": case["file_type"],
                        "rank": rank,
                        "top_5_chunk_ids": [
                            result["chunk_id"]
                            for result in results[:5]
                        ],
                    }
                )

            configurations.append(
                {
                    "dense_weight": dense_weight,
                    "lexical_weight": round(
                        1.0 - dense_weight,
                        2,
                    ),
                    "rrf_k": rrf_k,
                    "summary": summarise(ranks),
                    "cases": case_results,
                }
            )

    configurations.sort(
        key=selection_key,
        reverse=True,
    )

    best = configurations[0]

    report = {
        "status": "completed",
        "development_dataset": str(
            CASES_PATH.relative_to(PROJECT_ROOT)
        ),
        "development_case_count": len(cases),
        "formal_benchmark_used_for_tuning": False,
        "selection_priority": [
            "Hit@5",
            "Hit@3",
            "MRR",
            "Hit@1",
            "lower dense weight as final tie-break",
        ],
        "best_configuration": best,
        "configurations": configurations,
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

    print()
    print("=" * 104)
    print("HYBRID RETRIEVER DEVELOPMENT TUNING")
    print("=" * 104)
    print(
        f"{'DENSE':>7} "
        f"{'LEXICAL':>8} "
        f"{'RRF_K':>7} "
        f"{'HIT@1':>7} "
        f"{'HIT@3':>7} "
        f"{'HIT@5':>7} "
        f"{'HIT@10':>8} "
        f"{'MRR':>8} "
        f"{'MEAN':>8}"
    )
    print("-" * 104)

    for configuration in configurations:
        summary = configuration["summary"]

        print(
            f"{configuration['dense_weight']:>7.2f} "
            f"{configuration['lexical_weight']:>8.2f} "
            f"{configuration['rrf_k']:>7} "
            f"{summary['hit_rate_at_1']:>7.2f} "
            f"{summary['hit_rate_at_3']:>7.2f} "
            f"{summary['hit_rate_at_5']:>7.2f} "
            f"{summary['hit_rate_at_10']:>8.2f} "
            f"{summary['mrr']:>8.4f} "
            f"{summary['mean_rank']:>8}"
        )

    print()
    print("=" * 104)
    print("SELECTED CONFIGURATION")
    print("=" * 104)
    print(
        json.dumps(
            {
                "dense_weight": best["dense_weight"],
                "lexical_weight": best["lexical_weight"],
                "rrf_k": best["rrf_k"],
                "summary": best["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "Report saved to: "
        f"{REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
