from __future__ import annotations

import hashlib
import json
import sys
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

DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "tests"
    / "baseline_acceptance_cases.json"
)

FORMAL_PATH = (
    PROJECT_ROOT
    / "tests"
    / "formal_evaluation_cases.json"
)

TUNING_PATH = (
    PROJECT_ROOT
    / "reports"
    / "hybrid_retriever_tuning.json"
)

FORMAL_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "formal_hybrid_retriever_evaluation.json"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


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


def main() -> None:
    config = load_json(CONFIG_PATH)
    development_cases = load_json(DEVELOPMENT_PATH)
    tuning = load_json(TUNING_PATH)
    formal_report = load_json(FORMAL_REPORT_PATH)

    formal_hash = hashlib.sha256(
        FORMAL_PATH.read_bytes()
    ).hexdigest()

    assert formal_hash == config["formal_dataset_sha256"]
    assert formal_hash == formal_report["dataset"]["sha256"]

    assert config["selection_status"] == (
        "formally_evaluated_without_retuning"
    )

    assert config["formal_benchmark_used_for_tuning"] is False

    assert abs(
        config["dense_weight"]
        + config["lexical_weight"]
        - 1.0
    ) < 1e-9

    best = tuning["best_configuration"]

    assert config["dense_weight"] == best["dense_weight"]
    assert config["lexical_weight"] == best["lexical_weight"]
    assert config["rrf_k"] == best["rrf_k"]

    retriever = HybridRetriever(
        project_root=PROJECT_ROOT,
        dense_weight=config["dense_weight"],
        rrf_k=config["rrf_k"],
        bm25_k1=config["bm25_k1"],
        bm25_b=config["bm25_b"],
    )

    ranks: list[int] = []

    for case in development_cases:
        results = retriever.search(
            query=case["query"],
            top_k=5,
        )

        rank = first_relevant_rank(
            results,
            set(case["expected_chunk_ids"]),
        )

        assert rank is not None, (
            f"No relevant development result for "
            f"{case['case_id']}."
        )

        ranks.append(rank)

    assert len(ranks) == 10
    assert sum(rank <= 1 for rank in ranks) == 8
    assert sum(rank <= 3 for rank in ranks) == 10
    assert sum(rank <= 5 for rank in ranks) == 10

    impossible = retriever.search(
        query="revenue",
        top_k=5,
        filters={
            "document_id": "DOCUMENT_THAT_DOES_NOT_EXIST"
        },
    )

    assert impossible == []

    formal_global = formal_report["global_summary"]

    assert formal_global["case_count"] == 30
    assert formal_global["mrr"] == 0.8107
    assert formal_global["hit_rate_at_1"] == 0.7667
    assert formal_global["hit_rate_at_5"] == 0.9
    assert formal_global["mean_rank"] == 3.97

    print("=" * 80)
    print("HYBRID RETRIEVER VALIDATION PASSED")
    print("=" * 80)
    print(f"Development ranks: {ranks}")
    print("Development Hit@1: 8/10")
    print("Development Hit@3: 10/10")
    print("Development Hit@5: 10/10")
    print("Formal MRR: 0.8107")
    print("Formal Hit@5: 90.00%")


if __name__ == "__main__":
    main()
