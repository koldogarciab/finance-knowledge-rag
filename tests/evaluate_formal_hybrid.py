from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_retrieve import HybridRetriever


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "hybrid_retriever.json"
)

CASES_PATH = (
    PROJECT_ROOT
    / "tests"
    / "formal_evaluation_cases.json"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "formal_evaluation_manifest.json"
)

BASELINE_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "formal_baseline_retriever_evaluation.json"
)

HYBRID_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "formal_hybrid_retriever_evaluation.json"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "formal_retriever_comparison.md"
)

TOP_K_VALUES = (1, 3, 5, 10)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def calculate_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    return 0.0 if rank is None else 1.0 / rank


def effective_rank(
    rank: int | None,
    corpus_size: int,
) -> int:
    return (
        rank
        if rank is not None
        else corpus_size + 1
    )


def compact_results(
    results: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": result["rank"],
            "hybrid_score": round(
                result["hybrid_score"],
                6,
            ),
            "dense_score": round(
                result["dense_score"],
                4,
            ),
            "lexical_score": round(
                result["lexical_score"],
                4,
            ),
            "dense_rank": result["dense_rank"],
            "lexical_rank": result["lexical_rank"],
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
        "best_rank": (
            min(found_ranks)
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


def grouped_summaries(
    evaluations: list[dict[str, Any]],
    grouping_field: str,
) -> dict[str, Any]:
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for evaluation in evaluations:
        groups[
            str(evaluation[grouping_field])
        ].append(evaluation)

    return {
        group_name: {
            "global": summarise(items, "global"),
            "filtered": summarise(items, "filtered"),
        }
        for group_name, items in sorted(groups.items())
    }


def metric_comparison(
    baseline: dict[str, Any],
    hybrid: dict[str, Any],
) -> dict[str, Any]:
    metrics = [
        "mrr",
        "mean_rank",
        "median_rank",
        "hit_rate_at_1",
        "hit_rate_at_3",
        "hit_rate_at_5",
        "hit_rate_at_10",
    ]

    comparison: dict[str, Any] = {}

    for metric in metrics:
        baseline_value = baseline.get(metric)
        hybrid_value = hybrid.get(metric)

        delta = None

        if (
            baseline_value is not None
            and hybrid_value is not None
        ):
            delta = round(
                hybrid_value - baseline_value,
                4,
            )

        comparison[metric] = {
            "baseline": baseline_value,
            "hybrid": hybrid_value,
            "delta": delta,
        }

    return comparison


def format_rank(rank: int | None) -> str:
    return "MISS" if rank is None else str(rank)


def build_markdown(
    config: dict[str, Any],
    dataset_hash: str,
    baseline_report: dict[str, Any],
    hybrid_report: dict[str, Any],
) -> str:
    baseline_global = baseline_report[
        "global_summary"
    ]
    baseline_filtered = baseline_report[
        "filtered_summary"
    ]

    hybrid_global = hybrid_report[
        "global_summary"
    ]
    hybrid_filtered = hybrid_report[
        "filtered_summary"
    ]

    comparison = hybrid_report["comparison"]

    lines = [
        "# Baseline vs Hybrid Retriever",
        "",
        "## Frozen evaluation",
        "",
        "- Formal queries: 30",
        "- Corpus chunks: 316",
        f"- Dataset SHA-256: `{dataset_hash}`",
        "- The formal benchmark was not used for parameter tuning.",
        "- Parameters were selected using the separate 10-case development set.",
        "",
        "## Selected hybrid configuration",
        "",
        (
            f"- Dense weight: "
            f"{config['dense_weight']:.2f}"
        ),
        (
            f"- Lexical BM25 weight: "
            f"{config['lexical_weight']:.2f}"
        ),
        f"- Weighted RRF k: {config['rrf_k']}",
        f"- BM25 k1: {config['bm25_k1']}",
        f"- BM25 b: {config['bm25_b']}",
        "",
        "## Global retrieval",
        "",
        "| Metric | Dense baseline | Hybrid | Change |",
        "|---|---:|---:|---:|",
        (
            f"| MRR | {baseline_global['mrr']:.4f} | "
            f"{hybrid_global['mrr']:.4f} | "
            f"{hybrid_global['mrr'] - baseline_global['mrr']:+.4f} |"
        ),
        (
            f"| Hit@1 | "
            f"{baseline_global['hit_rate_at_1']:.2%} | "
            f"{hybrid_global['hit_rate_at_1']:.2%} | "
            f"{hybrid_global['hit_rate_at_1'] - baseline_global['hit_rate_at_1']:+.2%} |"
        ),
        (
            f"| Hit@3 | "
            f"{baseline_global['hit_rate_at_3']:.2%} | "
            f"{hybrid_global['hit_rate_at_3']:.2%} | "
            f"{hybrid_global['hit_rate_at_3'] - baseline_global['hit_rate_at_3']:+.2%} |"
        ),
        (
            f"| Hit@5 | "
            f"{baseline_global['hit_rate_at_5']:.2%} | "
            f"{hybrid_global['hit_rate_at_5']:.2%} | "
            f"{hybrid_global['hit_rate_at_5'] - baseline_global['hit_rate_at_5']:+.2%} |"
        ),
        (
            f"| Hit@10 | "
            f"{baseline_global['hit_rate_at_10']:.2%} | "
            f"{hybrid_global['hit_rate_at_10']:.2%} | "
            f"{hybrid_global['hit_rate_at_10'] - baseline_global['hit_rate_at_10']:+.2%} |"
        ),
        (
            f"| Mean rank | "
            f"{baseline_global['mean_rank']} | "
            f"{hybrid_global['mean_rank']} | "
            f"{hybrid_global['mean_rank'] - baseline_global['mean_rank']:+.2f} |"
        ),
        (
            f"| Median rank | "
            f"{baseline_global['median_rank']} | "
            f"{hybrid_global['median_rank']} | "
            f"{hybrid_global['median_rank'] - baseline_global['median_rank']:+.2f} |"
        ),
        "",
        "## Retrieval with metadata filters",
        "",
        "| Metric | Dense baseline | Hybrid | Change |",
        "|---|---:|---:|---:|",
        (
            f"| MRR | {baseline_filtered['mrr']:.4f} | "
            f"{hybrid_filtered['mrr']:.4f} | "
            f"{hybrid_filtered['mrr'] - baseline_filtered['mrr']:+.4f} |"
        ),
        (
            f"| Hit@1 | "
            f"{baseline_filtered['hit_rate_at_1']:.2%} | "
            f"{hybrid_filtered['hit_rate_at_1']:.2%} | "
            f"{hybrid_filtered['hit_rate_at_1'] - baseline_filtered['hit_rate_at_1']:+.2%} |"
        ),
        (
            f"| Hit@3 | "
            f"{baseline_filtered['hit_rate_at_3']:.2%} | "
            f"{hybrid_filtered['hit_rate_at_3']:.2%} | "
            f"{hybrid_filtered['hit_rate_at_3'] - baseline_filtered['hit_rate_at_3']:+.2%} |"
        ),
        (
            f"| Hit@5 | "
            f"{baseline_filtered['hit_rate_at_5']:.2%} | "
            f"{hybrid_filtered['hit_rate_at_5']:.2%} | "
            f"{hybrid_filtered['hit_rate_at_5'] - baseline_filtered['hit_rate_at_5']:+.2%} |"
        ),
        (
            f"| Hit@10 | "
            f"{baseline_filtered['hit_rate_at_10']:.2%} | "
            f"{hybrid_filtered['hit_rate_at_10']:.2%} | "
            f"{hybrid_filtered['hit_rate_at_10'] - baseline_filtered['hit_rate_at_10']:+.2%} |"
        ),
        "",
        "## Case-level comparison",
        "",
        (
            f"- Global wins: "
            f"{comparison['global_outcomes']['wins']}"
        ),
        (
            f"- Global ties: "
            f"{comparison['global_outcomes']['ties']}"
        ),
        (
            f"- Global losses: "
            f"{comparison['global_outcomes']['losses']}"
        ),
        (
            f"- Filtered wins: "
            f"{comparison['filtered_outcomes']['wins']}"
        ),
        (
            f"- Filtered ties: "
            f"{comparison['filtered_outcomes']['ties']}"
        ),
        (
            f"- Filtered losses: "
            f"{comparison['filtered_outcomes']['losses']}"
        ),
        "",
        "## Largest global rank improvements",
        "",
    ]

    for item in comparison["largest_global_improvements"]:
        lines.append(
            f"- `{item['case_id']}`: "
            f"{format_rank(item['baseline_rank'])} → "
            f"{format_rank(item['hybrid_rank'])} "
            f"(improvement {item['rank_improvement']})."
        )

    lines.extend(
        [
            "",
            "## Global regressions",
            "",
        ]
    )

    regressions = comparison["global_regressions"]

    if not regressions:
        lines.append(
            "- No global rank regressions were observed."
        )
    else:
        for item in regressions:
            lines.append(
                f"- `{item['case_id']}`: "
                f"{format_rank(item['baseline_rank'])} → "
                f"{format_rank(item['hybrid_rank'])} "
                f"(change {item['rank_improvement']})."
            )

    lines.extend(
        [
            "",
            "## Results by file type",
            "",
            "| Type | Baseline MRR | Hybrid MRR | Baseline Hit@5 | Hybrid Hit@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    baseline_by_type = baseline_report[
        "summary_by_file_type"
    ]
    hybrid_by_type = hybrid_report[
        "summary_by_file_type"
    ]

    for file_type in sorted(hybrid_by_type):
        baseline_item = baseline_by_type[
            file_type
        ]["global"]
        hybrid_item = hybrid_by_type[
            file_type
        ]["global"]

        lines.append(
            f"| {file_type} | "
            f"{baseline_item['mrr']:.4f} | "
            f"{hybrid_item['mrr']:.4f} | "
            f"{baseline_item['hit_rate_at_5']:.2%} | "
            f"{hybrid_item['hit_rate_at_5']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Dense retrieval uses the existing normalised MiniLM embeddings.",
            "- Lexical retrieval uses an in-memory BM25 index over `retrieval_text`.",
            "- Results are combined using weighted reciprocal rank fusion.",
            "- No corpus, chunk, embedding or formal evaluation case was modified.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_json(CONFIG_PATH)
    cases = load_json(CASES_PATH)
    manifest = load_json(MANIFEST_PATH)
    baseline_report = load_json(
        BASELINE_REPORT_PATH
    )

    dataset_hash = calculate_sha256(CASES_PATH)

    expected_hashes = {
        config["formal_dataset_sha256"],
        manifest["dataset_sha256"],
    }

    if dataset_hash not in expected_hashes:
        raise ValueError(
            "The formal dataset hash does not match "
            "the frozen configuration."
        )

    if len(expected_hashes) != 1:
        raise ValueError(
            "Configuration and manifest contain "
            "different formal dataset hashes."
        )

    if config["formal_benchmark_used_for_tuning"]:
        raise ValueError(
            "The configuration incorrectly states that "
            "the formal benchmark was used for tuning."
        )

    if len(cases) != 30:
        raise ValueError(
            f"Expected 30 formal cases, found {len(cases)}."
        )

    lexical_weight = 1.0 - float(
        config["dense_weight"]
    )

    if not abs(
        lexical_weight - float(
            config["lexical_weight"]
        )
    ) < 1e-9:
        raise ValueError(
            "Dense and lexical weights do not sum to 1."
        )

    retriever = HybridRetriever(
        project_root=PROJECT_ROOT,
        dense_weight=float(
            config["dense_weight"]
        ),
        rrf_k=int(config["rrf_k"]),
        bm25_k1=float(config["bm25_k1"]),
        bm25_b=float(config["bm25_b"]),
    )

    available_chunk_ids = {
        chunk["chunk_id"]
        for chunk in retriever.rows
    }

    baseline_cases = {
        item["case_id"]: item
        for item in baseline_report["cases"]
    }

    evaluations: list[dict[str, Any]] = []

    for case_number, case in enumerate(
        cases,
        start=1,
    ):
        expected_chunk_ids = set(
            case["expected_chunk_ids"]
        )

        missing_chunks = (
            expected_chunk_ids - available_chunk_ids
        )

        if missing_chunks:
            raise ValueError(
                f"{case['case_id']} references missing "
                f"chunks: {sorted(missing_chunks)}"
            )

        if case["case_id"] not in baseline_cases:
            raise ValueError(
                f"Baseline report is missing "
                f"{case['case_id']}."
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

        baseline_case = baseline_cases[
            case["case_id"]
        ]

        baseline_global_rank = baseline_case[
            "global_rank"
        ]
        baseline_filtered_rank = baseline_case[
            "filtered_rank"
        ]

        global_improvement = (
            effective_rank(
                baseline_global_rank,
                len(retriever.rows),
            )
            - effective_rank(
                global_rank,
                len(retriever.rows),
            )
        )

        filtered_improvement = (
            effective_rank(
                baseline_filtered_rank,
                len(retriever.rows),
            )
            - effective_rank(
                filtered_rank,
                len(retriever.rows),
            )
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
                "baseline_global_rank": (
                    baseline_global_rank
                ),
                "global_rank": global_rank,
                "global_rank_improvement": (
                    global_improvement
                ),
                "baseline_filtered_rank": (
                    baseline_filtered_rank
                ),
                "filtered_rank": filtered_rank,
                "filtered_rank_improvement": (
                    filtered_improvement
                ),
                "global_top10": compact_results(
                    global_results,
                ),
                "filtered_top10": compact_results(
                    filtered_results,
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

    global_wins = sum(
        item["global_rank_improvement"] > 0
        for item in evaluations
    )
    global_ties = sum(
        item["global_rank_improvement"] == 0
        for item in evaluations
    )
    global_losses = sum(
        item["global_rank_improvement"] < 0
        for item in evaluations
    )

    filtered_wins = sum(
        item["filtered_rank_improvement"] > 0
        for item in evaluations
    )
    filtered_ties = sum(
        item["filtered_rank_improvement"] == 0
        for item in evaluations
    )
    filtered_losses = sum(
        item["filtered_rank_improvement"] < 0
        for item in evaluations
    )

    largest_improvements = sorted(
        evaluations,
        key=lambda item: (
            -item["global_rank_improvement"],
            item["case_id"],
        ),
    )[:10]

    regressions = sorted(
        [
            item
            for item in evaluations
            if item["global_rank_improvement"] < 0
        ],
        key=lambda item: (
            item["global_rank_improvement"],
            item["case_id"],
        ),
    )

    hardest_hybrid_cases = sorted(
        evaluations,
        key=lambda item: (
            -effective_rank(
                item["global_rank"],
                len(retriever.rows),
            ),
            item["case_id"],
        ),
    )[:10]

    comparison = {
        "global_metrics": metric_comparison(
            baseline_report["global_summary"],
            global_summary,
        ),
        "filtered_metrics": metric_comparison(
            baseline_report["filtered_summary"],
            filtered_summary,
        ),
        "global_outcomes": {
            "wins": global_wins,
            "ties": global_ties,
            "losses": global_losses,
        },
        "filtered_outcomes": {
            "wins": filtered_wins,
            "ties": filtered_ties,
            "losses": filtered_losses,
        },
        "largest_global_improvements": [
            {
                "case_id": item["case_id"],
                "file_type": item["file_type"],
                "baseline_rank": item[
                    "baseline_global_rank"
                ],
                "hybrid_rank": item["global_rank"],
                "rank_improvement": item[
                    "global_rank_improvement"
                ],
            }
            for item in largest_improvements
        ],
        "global_regressions": [
            {
                "case_id": item["case_id"],
                "file_type": item["file_type"],
                "baseline_rank": item[
                    "baseline_global_rank"
                ],
                "hybrid_rank": item["global_rank"],
                "rank_improvement": item[
                    "global_rank_improvement"
                ],
            }
            for item in regressions
        ],
    }

    report = {
        "status": "completed",
        "evaluation_name": (
            "formal_hybrid_retriever_evaluation"
        ),
        "dataset": {
            "path": str(
                CASES_PATH.relative_to(PROJECT_ROOT)
            ),
            "sha256": dataset_hash,
            "case_count": len(cases),
        },
        "configuration": config,
        "retriever": {
            "type": (
                "hybrid_dense_bm25_weighted_rrf"
            ),
            "dense_model": retriever.model_name,
            "corpus_size": len(retriever.rows),
            "embedding_dimensions": int(
                retriever.embeddings.shape[1]
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
        "comparison": comparison,
        "hardest_hybrid_cases": [
            {
                "case_id": item["case_id"],
                "file_type": item["file_type"],
                "difficulty": item["difficulty"],
                "query_type": item["query_type"],
                "global_rank": item["global_rank"],
                "filtered_rank": item["filtered_rank"],
            }
            for item in hardest_hybrid_cases
        ],
        "cases": evaluations,
    }

    HYBRID_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with HYBRID_REPORT_PATH.open(
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

    COMPARISON_PATH.write_text(
        build_markdown(
            config=config,
            dataset_hash=dataset_hash,
            baseline_report=baseline_report,
            hybrid_report=report,
        ),
        encoding="utf-8",
    )

    print("=" * 118)
    print("FORMAL HYBRID RETRIEVER EVALUATION")
    print("=" * 118)
    print(
        f"{'CASE':44} "
        f"{'TYPE':9} "
        f"{'BASE G':>7} "
        f"{'HYBRID G':>8} "
        f"{'CHANGE':>7} "
        f"{'BASE F':>7} "
        f"{'HYBRID F':>8}"
    )
    print("-" * 118)

    for item in evaluations:
        print(
            f"{item['case_id'][:44]:44} "
            f"{item['file_type']:9} "
            f"{format_rank(item['baseline_global_rank']):>7} "
            f"{format_rank(item['global_rank']):>8} "
            f"{item['global_rank_improvement']:>+7} "
            f"{format_rank(item['baseline_filtered_rank']):>7} "
            f"{format_rank(item['filtered_rank']):>8}"
        )

    print()
    print("=" * 118)
    print("HYBRID GLOBAL SUMMARY")
    print("=" * 118)
    print(
        json.dumps(
            global_summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("=" * 118)
    print("HYBRID FILTERED SUMMARY")
    print("=" * 118)
    print(
        json.dumps(
            filtered_summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("=" * 118)
    print("BASELINE VS HYBRID")
    print("=" * 118)
    print(
        json.dumps(
            {
                "global_metrics": comparison[
                    "global_metrics"
                ],
                "global_outcomes": comparison[
                    "global_outcomes"
                ],
                "filtered_outcomes": comparison[
                    "filtered_outcomes"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "Hybrid report: "
        f"{HYBRID_REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Comparison summary: "
        f"{COMPARISON_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
