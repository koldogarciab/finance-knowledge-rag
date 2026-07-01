from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSON Lines file preserving row order."""
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from exc

    return rows


class SemanticRetriever:
    """Exact semantic retriever using normalized sentence embeddings."""

    def __init__(
        self,
        project_root: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.project_root = (
            project_root
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )

        self.processed_dir = self.project_root / "data" / "processed"

        self.embeddings_path = (
            self.processed_dir / "chunk_embeddings.npy"
        )
        self.index_path = (
            self.processed_dir / "embedding_index.jsonl"
        )
        self.chunks_path = (
            self.processed_dir / "corpus_chunks.jsonl"
        )
        self.manifest_path = (
            self.processed_dir / "embedding_manifest.json"
        )

        self.manifest = load_json(self.manifest_path)
        self.index_rows = load_jsonl(self.index_path)
        self.chunks = load_jsonl(self.chunks_path)
        self.embeddings = np.load(self.embeddings_path)

        self.chunk_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
        }

        if len(self.chunk_by_id) != len(self.chunks):
            raise ValueError(
                "Duplicate chunk_id values found in corpus_chunks.jsonl."
            )

        self.rows: list[dict[str, Any]] = []

        for expected_row, index_record in enumerate(self.index_rows):
            actual_row = index_record.get("row_index")

            if actual_row != expected_row:
                raise ValueError(
                    "Embedding index row order is invalid: "
                    f"expected {expected_row}, found {actual_row}."
                )

            chunk_id = index_record["chunk_id"]
            chunk = self.chunk_by_id.get(chunk_id)

            if chunk is None:
                raise ValueError(
                    f"Chunk {chunk_id!r} from the embedding index "
                    "does not exist in corpus_chunks.jsonl."
                )

            self.rows.append(chunk)

        self._validate_artifacts()

        self.model_name = self.manifest["model"]["name"]
        self.model = SentenceTransformer(
            self.model_name,
            device=device,
        )

    def _validate_artifacts(self) -> None:
        """Validate dimensions, row counts and vector normalisation."""
        if self.embeddings.ndim != 2:
            raise ValueError(
                "The embedding matrix must have two dimensions."
            )

        embedding_rows, embedding_dimensions = self.embeddings.shape

        if embedding_rows != len(self.index_rows):
            raise ValueError(
                "Embedding row count does not match embedding_index.jsonl."
            )

        if embedding_rows != len(self.chunks):
            raise ValueError(
                "Embedding row count does not match corpus_chunks.jsonl."
            )

        model_info = self.manifest.get("model", {})
        expected_dimensions = (
            model_info.get("embedding_dimensions")
            or model_info.get("embedding_dimension")
            or model_info.get("embedding_dim")
        )

        if expected_dimensions is None:
            shape = self.manifest.get("encoding", {}).get("shape", [])
            if len(shape) == 2:
                expected_dimensions = shape[1]

        if expected_dimensions is None:
            raise ValueError(
                "Embedding dimension is missing from the manifest."
            )

        expected_dimensions = int(expected_dimensions)

        if embedding_dimensions != expected_dimensions:
            raise ValueError(
                "Embedding dimension does not match the manifest: "
                f"{embedding_dimensions} != {expected_dimensions}."
            )

        if self.embeddings.dtype != np.float32:
            raise ValueError(
                f"Expected float32 embeddings, found {self.embeddings.dtype}."
            )

        norms = np.linalg.norm(self.embeddings, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError(
                "The embedding matrix contains non-normalised vectors."
            )

    def encode_query(self, query: str) -> np.ndarray:
        """Generate a normalized embedding for a user query."""
        clean_query = query.strip()

        if not clean_query:
            raise ValueError("The query cannot be empty.")

        query_embedding = self.model.encode(
            clean_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(-1)

        if query_embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension does not match the index."
            )

        return query_embedding

    @staticmethod
    def _normalise_filter_value(value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().casefold()

        return value

    @classmethod
    def _matches_filters(
        cls,
        chunk: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        metadata = chunk.get("metadata", {})

        for key, expected in filters.items():
            actual = (
                chunk[key]
                if key in chunk
                else metadata.get(key)
            )

            if isinstance(expected, (list, tuple, set)):
                expected_values = {
                    cls._normalise_filter_value(value)
                    for value in expected
                }

                if (
                    cls._normalise_filter_value(actual)
                    not in expected_values
                ):
                    return False
            else:
                if (
                    cls._normalise_filter_value(actual)
                    != cls._normalise_filter_value(expected)
                ):
                    return False

        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k chunks using exact cosine similarity.

        Since document and query embeddings are normalized, the matrix
        product is equivalent to cosine similarity.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        active_filters = filters or {}
        query_embedding = self.encode_query(query)

        scores = self.embeddings @ query_embedding

        candidate_rows = np.array(
            [
                row_index
                for row_index, chunk in enumerate(self.rows)
                if self._matches_filters(chunk, active_filters)
            ],
            dtype=np.int64,
        )

        if candidate_rows.size == 0:
            return []

        candidate_scores = scores[candidate_rows]

        sorted_positions = np.argsort(
            -candidate_scores,
            kind="mergesort",
        )

        selected_rows = candidate_rows[
            sorted_positions[: min(top_k, candidate_rows.size)]
        ]

        results: list[dict[str, Any]] = []

        for rank, row_index in enumerate(selected_rows, start=1):
            chunk = self.rows[int(row_index)]
            metadata = chunk.get("metadata", {})

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[row_index]),
                    "row_index": int(row_index),
                    "chunk_id": chunk["chunk_id"],
                    "record_id": chunk["record_id"],
                    "document_id": chunk["document_id"],
                    "document_title": chunk["document_title"],
                    "document_name": chunk["document_name"],
                    "file_type": chunk["file_type"],
                    "source_path": chunk["source_path"],
                    "citation": chunk["citation"],
                    "citation_locator": metadata.get(
                        "citation_locator"
                    ),
                    "content": chunk["content"],
                    "retrieval_text": chunk["retrieval_text"],
                    "metadata": metadata,
                }
            )

        return results


def compact_text(text: str, max_chars: int = 900) -> str:
    """Make retrieved content easier to inspect in the terminal."""
    compact = " ".join(text.split())

    if len(compact) <= max_chars:
        return compact

    return compact[: max_chars - 3].rstrip() + "..."


def print_results(
    query: str,
    results: list[dict[str, Any]],
    filters: dict[str, Any],
) -> None:
    print("=" * 80)
    print(f"QUERY: {query}")
    print(f"FILTERS: {filters or 'None'}")
    print(f"RESULTS: {len(results)}")
    print("=" * 80)

    if not results:
        print("No chunks matched the query and filters.")
        return

    for result in results:
        print()
        print(
            f"[{result['rank']}] "
            f"score={result['score']:.4f} | "
            f"{result['chunk_id']}"
        )
        print(f"Title: {result['document_title']}")
        print(f"Type: {result['file_type']}")
        print(f"Citation: {result['citation']}")
        print(f"Source: {result['source_path']}")
        print(f"Text: {compact_text(result['content'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exact semantic search over the finance knowledge corpus."
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
        "--file-type",
        help="Optional file type filter, for example pdf or csv.",
    )
    parser.add_argument(
        "--granularity",
        help=(
            "Optional granularity filter, for example page, section, "
            "kpi, account_category_row or monthly_department_summary."
        ),
    )
    parser.add_argument(
        "--document-id",
        help="Optional document_id filter.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Return results as JSON instead of terminal cards.",
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
        filters["granularity"] = args.granularity

    if args.document_id:
        filters["document_id"] = args.document_id

    retriever = SemanticRetriever()

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
        )


if __name__ == "__main__":
    main()
