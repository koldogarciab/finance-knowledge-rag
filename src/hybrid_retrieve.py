from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1]),
    )

from src.retrieve import SemanticRetriever, compact_text


TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[._%/-][A-Za-z0-9]+)*"
)


def lexical_tokens(text: str) -> list[str]:
    """
    Tokenize financial text for lexical retrieval.

    The tokenizer preserves useful forms such as dates, percentages,
    identifiers and hyphenated financial terms.
    """
    return [
        token.casefold()
        for token in TOKEN_PATTERN.findall(text)
    ]


class BM25Index:
    """Small in-memory BM25 implementation for the local corpus."""

    def __init__(
        self,
        texts: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not texts:
            raise ValueError(
                "BM25 requires at least one document."
            )

        if k1 <= 0:
            raise ValueError("BM25 k1 must be positive.")

        if not 0 <= b <= 1:
            raise ValueError(
                "BM25 b must be between 0 and 1."
            )

        self.k1 = float(k1)
        self.b = float(b)

        self.document_tokens = [
            lexical_tokens(text)
            for text in texts
        ]

        self.document_lengths = np.asarray(
            [
                len(tokens)
                for tokens in self.document_tokens
            ],
            dtype=np.float32,
        )

        self.average_document_length = float(
            self.document_lengths.mean()
        )

        if self.average_document_length <= 0:
            raise ValueError(
                "BM25 documents cannot all be empty."
            )

        self.term_frequencies = [
            Counter(tokens)
            for tokens in self.document_tokens
        ]

        document_frequencies: Counter[str] = Counter()

        for tokens in self.document_tokens:
            document_frequencies.update(set(tokens))

        document_count = len(self.document_tokens)

        self.inverse_document_frequency = {
            term: math.log(
                1.0
                + (
                    document_count
                    - frequency
                    + 0.5
                )
                / (
                    frequency
                    + 0.5
                )
            )
            for term, frequency
            in document_frequencies.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """Calculate BM25 scores for every corpus row."""
        query_terms = lexical_tokens(query)

        scores = np.zeros(
            len(self.document_tokens),
            dtype=np.float32,
        )

        if not query_terms:
            return scores

        unique_query_terms = set(query_terms)

        for row_index, frequencies in enumerate(
            self.term_frequencies
        ):
            document_length = float(
                self.document_lengths[row_index]
            )

            normalisation = self.k1 * (
                1.0
                - self.b
                + self.b
                * document_length
                / self.average_document_length
            )

            score = 0.0

            for term in unique_query_terms:
                term_frequency = frequencies.get(term, 0)

                if term_frequency == 0:
                    continue

                idf = self.inverse_document_frequency.get(
                    term,
                    0.0,
                )

                numerator = term_frequency * (
                    self.k1 + 1.0
                )
                denominator = (
                    term_frequency + normalisation
                )

                score += idf * numerator / denominator

            scores[row_index] = score

        return scores


def descending_ranks(
    scores: np.ndarray,
) -> np.ndarray:
    """
    Convert scores into one-based ranks.

    Stable sorting makes ties deterministic and preserves corpus order.
    """
    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    ranks = np.empty(
        scores.shape[0],
        dtype=np.int64,
    )

    ranks[order] = np.arange(
        1,
        scores.shape[0] + 1,
        dtype=np.int64,
    )

    return ranks


class HybridRetriever:
    """
    Hybrid dense and lexical retriever.

    Dense semantic retrieval is combined with BM25 using weighted
    reciprocal rank fusion.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        device: str = "cpu",
        dense_weight: float = 0.2,
        rrf_k: int = 10,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        if not 0 <= dense_weight <= 1:
            raise ValueError(
                "dense_weight must be between 0 and 1."
            )

        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        self.semantic = SemanticRetriever(
            project_root=project_root,
            device=device,
        )

        self.project_root = self.semantic.project_root
        self.rows = self.semantic.rows
        self.embeddings = self.semantic.embeddings
        self.model_name = self.semantic.model_name

        self.dense_weight = float(dense_weight)
        self.lexical_weight = 1.0 - self.dense_weight
        self.rrf_k = int(rrf_k)

        lexical_texts = [
            chunk["retrieval_text"]
            for chunk in self.rows
        ]

        self.bm25 = BM25Index(
            texts=lexical_texts,
            k1=bm25_k1,
            b=bm25_b,
        )

    def _candidate_rows(
        self,
        filters: dict[str, Any],
    ) -> np.ndarray:
        return np.asarray(
            [
                row_index
                for row_index, chunk
                in enumerate(self.rows)
                if self.semantic._matches_filters(
                    chunk,
                    filters,
                )
            ],
            dtype=np.int64,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k chunks using dense plus BM25 rank fusion.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("The query cannot be empty.")

        active_filters = filters or {}

        candidate_rows = self._candidate_rows(
            active_filters
        )

        if candidate_rows.size == 0:
            return []

        query_embedding = self.semantic.encode_query(
            clean_query
        )

        dense_scores_all = (
            self.embeddings @ query_embedding
        )

        lexical_scores_all = self.bm25.scores(
            clean_query
        )

        dense_scores = dense_scores_all[
            candidate_rows
        ]
        lexical_scores = lexical_scores_all[
            candidate_rows
        ]

        dense_ranks = descending_ranks(
            dense_scores
        )
        lexical_ranks = descending_ranks(
            lexical_scores
        )

        dense_rrf = (
            self.dense_weight
            / (
                self.rrf_k + dense_ranks
            )
        )

        lexical_rrf = np.zeros_like(
            dense_rrf,
            dtype=np.float64,
        )

        lexical_matches = lexical_scores > 0

        lexical_rrf[lexical_matches] = (
            self.lexical_weight
            / (
                self.rrf_k
                + lexical_ranks[lexical_matches]
            )
        )

        hybrid_scores = dense_rrf + lexical_rrf

        sorted_positions = np.argsort(
            -hybrid_scores,
            kind="mergesort",
        )

        selected_positions = sorted_positions[
            : min(top_k, candidate_rows.size)
        ]

        results: list[dict[str, Any]] = []

        for rank, candidate_position in enumerate(
            selected_positions,
            start=1,
        ):
            row_index = int(
                candidate_rows[candidate_position]
            )
            chunk = self.rows[row_index]
            metadata = chunk.get("metadata", {})

            results.append(
                {
                    "rank": rank,
                    "score": float(
                        hybrid_scores[
                            candidate_position
                        ]
                    ),
                    "hybrid_score": float(
                        hybrid_scores[
                            candidate_position
                        ]
                    ),
                    "dense_score": float(
                        dense_scores[
                            candidate_position
                        ]
                    ),
                    "lexical_score": float(
                        lexical_scores[
                            candidate_position
                        ]
                    ),
                    "dense_rank": int(
                        dense_ranks[
                            candidate_position
                        ]
                    ),
                    "lexical_rank": int(
                        lexical_ranks[
                            candidate_position
                        ]
                    ),
                    "row_index": row_index,
                    "chunk_id": chunk["chunk_id"],
                    "record_id": chunk["record_id"],
                    "document_id": chunk["document_id"],
                    "document_title": chunk[
                        "document_title"
                    ],
                    "document_name": chunk[
                        "document_name"
                    ],
                    "file_type": chunk["file_type"],
                    "source_path": chunk["source_path"],
                    "citation": chunk["citation"],
                    "citation_locator": metadata.get(
                        "citation_locator"
                    ),
                    "content": chunk["content"],
                    "retrieval_text": chunk[
                        "retrieval_text"
                    ],
                    "metadata": metadata,
                }
            )

        return results


def print_results(
    query: str,
    results: list[dict[str, Any]],
    filters: dict[str, Any],
    dense_weight: float,
    rrf_k: int,
) -> None:
    print("=" * 88)
    print(f"QUERY: {query}")
    print(f"FILTERS: {filters or 'None'}")
    print(
        "CONFIGURATION: "
        f"dense_weight={dense_weight:.2f}, "
        f"lexical_weight={1.0 - dense_weight:.2f}, "
        f"rrf_k={rrf_k}"
    )
    print(f"RESULTS: {len(results)}")
    print("=" * 88)

    if not results:
        print(
            "No chunks matched the query and filters."
        )
        return

    for result in results:
        print()
        print(
            f"[{result['rank']}] "
            f"hybrid={result['hybrid_score']:.6f} | "
            f"dense={result['dense_score']:.4f} "
            f"(rank {result['dense_rank']}) | "
            f"bm25={result['lexical_score']:.4f} "
            f"(rank {result['lexical_rank']})"
        )
        print(f"Chunk: {result['chunk_id']}")
        print(f"Title: {result['document_title']}")
        print(f"Type: {result['file_type']}")
        print(f"Citation: {result['citation']}")
        print(f"Source: {result['source_path']}")
        print(
            f"Text: {compact_text(result['content'])}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid dense and BM25 search over the "
            "finance knowledge corpus."
        )
    )

    parser.add_argument(
        "query",
        help="Natural-language search query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return. Default: 5.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.2,
        help=(
            "Dense contribution to weighted RRF. "
            "Default: 0.2."
        ),
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=10,
        help="RRF rank constant. Default: 10.",
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
        "--document-id",
        help="Optional document_id filter.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Return results as JSON.",
    )

    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()

    filters: dict[str, Any] = {}

    if args.file_type:
        filters["file_type"] = args.file_type

    if args.granularity:
        filters["granularity"] = (
            args.granularity
        )

    if args.document_id:
        filters["document_id"] = (
            args.document_id
        )

    retriever = HybridRetriever(
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
    )

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        filters=filters,
    )

    if args.json:
        print(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_results(
            query=args.query,
            results=results,
            filters=filters,
            dense_weight=args.dense_weight,
            rrf_k=args.rrf_k,
        )


if __name__ == "__main__":
    main()
