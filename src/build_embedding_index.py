from collections import Counter
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/processed/corpus_chunks.jsonl")
EMBEDDINGS_PATH = Path(
    "data/processed/chunk_embeddings.npy"
)
INDEX_PATH = Path(
    "data/processed/embedding_index.jsonl"
)
MANIFEST_PATH = Path(
    "data/processed/embedding_manifest.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
DEVICE = "cpu"
EMBEDDING_FIELD = "retrieval_text"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def load_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: "
                    f"{error}"
                ) from error

    return chunks


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_PATH}"
        )

    chunks = load_chunks(CHUNKS_PATH)

    if len(chunks) != 316:
        raise ValueError(
            f"Expected 316 chunks, found {len(chunks)}"
        )

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    duplicate_ids = [
        chunk_id
        for chunk_id, count in Counter(chunk_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        raise ValueError(
            f"Duplicate chunk IDs: {duplicate_ids}"
        )

    texts = [
        chunk[EMBEDDING_FIELD]
        for chunk in chunks
    ]

    if any(not str(text).strip() for text in texts):
        raise ValueError(
            "One or more chunks have empty retrieval text"
        )

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )

    embedding_dimension = (
        model.get_embedding_dimension()
    )

    model_max_sequence_length = model.max_seq_length

    token_lengths = []

    for text in texts:
        encoded = model.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )

        token_lengths.append(
            len(encoded["input_ids"])
        )

    texts_over_limit = sum(
        length > model_max_sequence_length
        for length in token_lengths
    )

    print(f"Chunks to encode: {len(texts)}")
    print(
        "Model maximum sequence length: "
        f"{model_max_sequence_length} tokens"
    )
    print(
        "Retrieval texts above model limit: "
        f"{texts_over_limit}"
    )
    print("Creating normalized embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    expected_shape = (
        len(chunks),
        embedding_dimension
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Expected embedding shape {expected_shape}, "
            f"found {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values"
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
        allow_pickle=False
    )

    with INDEX_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as file:
        for row_index, chunk in enumerate(chunks):
            index_record = {
                "row_index": row_index,
                "chunk_id": chunk["chunk_id"],
                "record_id": chunk["record_id"],
                "document_id": chunk["document_id"],
                "document_title": chunk[
                    "document_title"
                ],
                "file_type": chunk["file_type"],
                "citation": chunk["citation"],
                "source_path": chunk["source_path"]
            }

            file.write(
                json.dumps(
                    index_record,
                    ensure_ascii=False
                )
            )
            file.write("\n")

    manifest = {
        "input_file": CHUNKS_PATH.as_posix(),
        "embeddings_file": EMBEDDINGS_PATH.as_posix(),
        "index_file": INDEX_PATH.as_posix(),
        "source_chunk_sha256": sha256_file(
            CHUNKS_PATH
        ),
        "model": {
            "name": MODEL_NAME,
            "device": DEVICE,
            "embedding_field": EMBEDDING_FIELD,
            "embedding_dimension": (
                embedding_dimension
            ),
            "maximum_sequence_length": (
                model_max_sequence_length
            ),
            "normalised": True
        },
        "encoding": {
            "chunk_count": len(chunks),
            "batch_size": BATCH_SIZE,
            "dtype": str(embeddings.dtype),
            "shape": list(embeddings.shape)
        },
        "token_statistics": {
            "minimum": min(token_lengths),
            "maximum": max(token_lengths),
            "average": round(
                sum(token_lengths) / len(token_lengths),
                2
            ),
            "texts_over_model_limit": (
                texts_over_limit
            )
        },
        "norm_statistics": {
            "minimum": round(float(norms.min()), 6),
            "maximum": round(float(norms.max()), 6),
            "average": round(float(norms.mean()), 6)
        },
        "library_versions": {
            "python_runtime": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": (
                sentence_transformers.__version__
            ),
            "numpy": np.__version__
        }
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False
        )
        file.write("\n")

    print("Embedding index created successfully")
    print(f"Chunks encoded: {len(chunks)}")
    print(
        f"Embedding shape: {embeddings.shape}"
    )
    print(f"Data type: {embeddings.dtype}")
    print(
        "Norm range: "
        f"{norms.min():.6f}-{norms.max():.6f}"
    )
    print(f"Created: {EMBEDDINGS_PATH}")
    print(f"Created: {INDEX_PATH}")
    print(f"Created: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
