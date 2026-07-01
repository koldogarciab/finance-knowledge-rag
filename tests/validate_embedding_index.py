from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/processed/corpus_chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/chunk_embeddings.npy")
INDEX_PATH = Path("data/processed/embedding_index.jsonl")
MANIFEST_PATH = Path("data/processed/embedding_manifest.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_CHUNKS = 316
EXPECTED_DIMENSION = 384
EXPECTED_DTYPE = np.float32


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                fail(
                    f"{path}: invalid JSON on line "
                    f"{line_number}: {error}"
                )

    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


for path in [
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    INDEX_PATH,
    MANIFEST_PATH
]:
    if not path.exists():
        fail(f"Missing file: {path}")

    if path.stat().st_size == 0:
        fail(f"File is empty: {path}")


chunks = load_jsonl(CHUNKS_PATH)
index_records = load_jsonl(INDEX_PATH)

with MANIFEST_PATH.open("r", encoding="utf-8") as file:
    manifest = json.load(file)

try:
    embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False
    )
except Exception as error:
    fail(f"Embedding matrix cannot be loaded: {error}")


# -------------------------------------------------------------------
# Matrix structure
# -------------------------------------------------------------------

if len(chunks) != EXPECTED_CHUNKS:
    fail(
        f"Expected {EXPECTED_CHUNKS} chunks, "
        f"found {len(chunks)}"
    )

if embeddings.shape != (
    EXPECTED_CHUNKS,
    EXPECTED_DIMENSION
):
    fail(
        "Unexpected embedding shape. "
        f"Expected {(EXPECTED_CHUNKS, EXPECTED_DIMENSION)}, "
        f"found {embeddings.shape}"
    )

if embeddings.dtype != EXPECTED_DTYPE:
    fail(
        f"Expected float32 embeddings, "
        f"found {embeddings.dtype}"
    )

if not np.isfinite(embeddings).all():
    fail("Embedding matrix contains NaN or infinite values")

norms = np.linalg.norm(embeddings, axis=1)

if not np.allclose(
    norms,
    np.ones(EXPECTED_CHUNKS),
    atol=1e-5
):
    fail(
        "Embeddings are not correctly normalised. "
        f"Norm range: {norms.min()}-{norms.max()}"
    )


# -------------------------------------------------------------------
# Index order and chunk mapping
# -------------------------------------------------------------------

if len(index_records) != EXPECTED_CHUNKS:
    fail(
        f"Expected {EXPECTED_CHUNKS} index records, "
        f"found {len(index_records)}"
    )

expected_row_indices = list(range(EXPECTED_CHUNKS))
actual_row_indices = [
    record["row_index"]
    for record in index_records
]

if actual_row_indices != expected_row_indices:
    fail("Embedding index row numbers are not sequential")

chunk_ids = [
    chunk["chunk_id"]
    for chunk in chunks
]

index_chunk_ids = [
    record["chunk_id"]
    for record in index_records
]

if index_chunk_ids != chunk_ids:
    fail(
        "Embedding index chunk order does not match "
        "corpus_chunks.jsonl"
    )

if len(chunk_ids) != len(set(chunk_ids)):
    fail("Duplicate chunk IDs detected")

required_index_fields = {
    "row_index",
    "chunk_id",
    "record_id",
    "document_id",
    "document_title",
    "file_type",
    "citation",
    "source_path"
}

for row_index, (chunk, index_record) in enumerate(
    zip(chunks, index_records)
):
    missing_fields = (
        required_index_fields - set(index_record)
    )

    if missing_fields:
        fail(
            f"Index row {row_index} is missing fields: "
            f"{sorted(missing_fields)}"
        )

    expected_values = {
        "row_index": row_index,
        "chunk_id": chunk["chunk_id"],
        "record_id": chunk["record_id"],
        "document_id": chunk["document_id"],
        "document_title": chunk["document_title"],
        "file_type": chunk["file_type"],
        "citation": chunk["citation"],
        "source_path": chunk["source_path"]
    }

    for field, expected_value in expected_values.items():
        if index_record[field] != expected_value:
            fail(
                f"Index row {row_index}: field '{field}' "
                "does not match the source chunk"
            )


# -------------------------------------------------------------------
# Manifest and source hash
# -------------------------------------------------------------------

current_hash = sha256_file(CHUNKS_PATH)

if manifest.get("source_chunk_sha256") != current_hash:
    fail(
        "Manifest hash does not match the current "
        "corpus_chunks.jsonl"
    )

expected_paths = {
    "input_file": CHUNKS_PATH.as_posix(),
    "embeddings_file": EMBEDDINGS_PATH.as_posix(),
    "index_file": INDEX_PATH.as_posix()
}

for field, expected_value in expected_paths.items():
    if manifest.get(field) != expected_value:
        fail(
            f"Manifest field '{field}' should be "
            f"'{expected_value}'"
        )

model_config = manifest.get("model", {})
encoding_config = manifest.get("encoding", {})

expected_model_values = {
    "name": MODEL_NAME,
    "device": "cpu",
    "embedding_field": "retrieval_text",
    "embedding_dimension": EXPECTED_DIMENSION,
    "maximum_sequence_length": 256,
    "normalised": True
}

for field, expected_value in expected_model_values.items():
    if model_config.get(field) != expected_value:
        fail(
            f"Manifest model field '{field}' should be "
            f"{expected_value}, found {model_config.get(field)}"
        )

if encoding_config.get("chunk_count") != EXPECTED_CHUNKS:
    fail("Manifest chunk_count is incorrect")

if encoding_config.get("batch_size") != 32:
    fail("Manifest batch_size is incorrect")

if encoding_config.get("dtype") != "float32":
    fail("Manifest dtype is incorrect")

if encoding_config.get("shape") != [
    EXPECTED_CHUNKS,
    EXPECTED_DIMENSION
]:
    fail("Manifest embedding shape is incorrect")


# -------------------------------------------------------------------
# Token and norm statistics
# -------------------------------------------------------------------

token_counts = [
    chunk["metadata"]["chunk_token_count"]
    for chunk in chunks
]

token_statistics = manifest.get("token_statistics", {})
norm_statistics = manifest.get("norm_statistics", {})

expected_token_average = round(
    sum(token_counts) / len(token_counts),
    2
)

if token_statistics.get("minimum") != min(token_counts):
    fail("Manifest minimum token count is incorrect")

if token_statistics.get("maximum") != max(token_counts):
    fail("Manifest maximum token count is incorrect")

if token_statistics.get("average") != expected_token_average:
    fail("Manifest average token count is incorrect")

if token_statistics.get("texts_over_model_limit") != 0:
    fail(
        "Manifest should report zero texts above "
        "the model limit"
    )

expected_norm_values = {
    "minimum": round(float(norms.min()), 6),
    "maximum": round(float(norms.max()), 6),
    "average": round(float(norms.mean()), 6)
}

for field, expected_value in expected_norm_values.items():
    if norm_statistics.get(field) != expected_value:
        fail(
            f"Manifest norm statistic '{field}' "
            f"should be {expected_value}"
        )


# -------------------------------------------------------------------
# Installed-library versions
# -------------------------------------------------------------------

library_versions = manifest.get("library_versions", {})

expected_versions = {
    "python_runtime": platform.python_version(),
    "torch": torch.__version__,
    "sentence_transformers": (
        sentence_transformers.__version__
    ),
    "numpy": np.__version__
}

for library, expected_version in expected_versions.items():
    if library_versions.get(library) != expected_version:
        fail(
            f"Manifest version for {library} should be "
            f"{expected_version}, found "
            f"{library_versions.get(library)}"
        )


# -------------------------------------------------------------------
# Re-encode selected chunks to verify row alignment
# -------------------------------------------------------------------

probe_indices = {
    0,
    len(chunks) // 2,
    len(chunks) - 1
}

for index, chunk in enumerate(chunks):
    metadata = chunk["metadata"]

    if (
        metadata.get("granularity")
        == "monthly_department_summary"
        and metadata.get("month") == "2026-03"
        and metadata.get("department") == "Marketing"
    ):
        probe_indices.add(index)

probe_indices = sorted(probe_indices)
probe_texts = [
    chunks[index]["retrieval_text"]
    for index in probe_indices
]

print(f"Loading model for {len(probe_indices)} alignment probes")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

probe_embeddings = model.encode(
    probe_texts,
    batch_size=len(probe_texts),
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)

alignment_scores = []

for probe_position, matrix_index in enumerate(probe_indices):
    score = float(
        np.dot(
            probe_embeddings[probe_position],
            embeddings[matrix_index]
        )
    )

    alignment_scores.append(score)

    if score < 0.9999:
        fail(
            f"Embedding row {matrix_index} does not match "
            f"its source text. Cosine similarity: {score}"
        )


print("[PASS] Embedding files exist and are non-empty")
print(f"[PASS] Chunks and index rows: {EXPECTED_CHUNKS}")
print(
    f"[PASS] Embedding matrix shape: "
    f"{embeddings.shape}"
)
print(f"[PASS] Embedding dtype: {embeddings.dtype}")
print("[PASS] All embedding values are finite")
print(
    f"[PASS] Embedding norm range: "
    f"{norms.min():.6f}-{norms.max():.6f}"
)
print("[PASS] Index row order matches the chunk order")
print("[PASS] Index metadata matches every source chunk")
print("[PASS] Source chunk SHA-256 matches the manifest")
print("[PASS] Model and encoding configuration are correct")
print("[PASS] No retrieval text exceeds the model limit")
print("[PASS] Library versions match the active environment")
print(
    "[PASS] Re-encoded alignment probes: "
    + ", ".join(
        f"{score:.6f}"
        for score in alignment_scores
    )
)
print("[PASS] Embedding index is fully consistent")
