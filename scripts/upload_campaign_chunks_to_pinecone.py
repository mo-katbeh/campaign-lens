from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, cast

from pinecone import Pinecone, ServerlessSpec

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class EmbeddingModel(Protocol):
    def encode(
        self,
        texts: list[str],
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any: ...

    def get_embedding_dimension(self) -> int: ...

    def get_sentence_embedding_dimension(self) -> int: ...


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "campaign_search_chunks.jsonl"

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "campaigns-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
# Re-indexing is required whenever this embedding model changes because existing
# Pinecone vectors become incompatible with newly generated embeddings.
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))
NAMESPACE = os.getenv("PINECONE_NAMESPACE")
# Keep vectors normalized so Pinecone cosine similarity compares compatible embeddings.
NORMALIZE_EMBEDDINGS = True


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_chunk_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc

            if not record.get("chunk_id") or not record.get("text"):
                raise ValueError(f"Chunk record on line {line_number} is missing chunk_id or text")

            records.append(record)

    if not records:
        raise ValueError(f"No chunk records found in {path}")

    return records


@lru_cache(maxsize=1)
def load_local_model() -> EmbeddingModel:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Run `python -m pip install -r requirements.txt` first."
        )

    try:
        print(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}")
        return cast(EmbeddingModel, SentenceTransformer(LOCAL_EMBEDDING_MODEL))
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the local embedding model. "
            "If this is the first run, make sure you have internet access so the model can be downloaded."
        ) from exc


def get_index_dimension(pc: Pinecone, index_name: str) -> int | None:
    try:
        description = pc.describe_index(index_name)
    except Exception:
        return None

    if hasattr(description, "dimension"):
        return description.dimension
    if isinstance(description, dict):
        return description.get("dimension")
    return None


def ensure_index(pc: Pinecone, index_name: str, expected_dimension: int) -> None:
    existing_indexes = set(pc.list_indexes().names())
    if index_name in existing_indexes:
        current_dimension = get_index_dimension(pc, index_name)
        if current_dimension is not None and current_dimension != expected_dimension:
            raise RuntimeError(
                f"Pinecone index '{index_name}' already exists with dimension {current_dimension}, "
                f"but the local model '{LOCAL_EMBEDDING_MODEL}' outputs dimension {expected_dimension}. "
                "Use a different PINECONE_INDEX_NAME or recreate the index."
            )
        return

    pc.create_index(
        name=index_name,
        dimension=expected_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )


def embed_text_batch(model: EmbeddingModel, texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def sanitize_metadata_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, str)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, list):
        cleaned_items = [item for item in value if isinstance(item, str) and item]
        return cleaned_items if cleaned_items else None

    return str(value)


def sanitize_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "campaign_id": int(record["campaign_id"]),
        "title": record["title"],
        "year_label": record["year_label"],
        "campaign_theme": record["campaign_theme"],
        "beneficiary_group": record["beneficiary_group"],
        "funding_status": record["funding_status"],
        "funding_ratio": float(record["funding_ratio"]) if record.get("funding_ratio") is not None else None,
        "donations_count": int(record["donations_count"]),
        "text": record["text"][:1000],
        "location_mentions": record.get("location_mentions", []),
    }
    cleaned_metadata = {
        key: sanitize_metadata_value(value)
        for key, value in metadata.items()
    }
    return {key: value for key, value in cleaned_metadata.items() if value is not None}


def build_pinecone_vectors(model: EmbeddingModel, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = []
    total_batches = (len(records) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for start in range(0, len(records), EMBED_BATCH_SIZE):
        batch_records = records[start : start + EMBED_BATCH_SIZE]
        batch_texts = [record["text"] for record in batch_records]
        batch_number = (start // EMBED_BATCH_SIZE) + 1
        print(f"Embedding batch {batch_number}/{total_batches} ({len(batch_records)} chunks)")
        embeddings = embed_text_batch(model, batch_texts)

        for record, embedding in zip(batch_records, embeddings):
            vectors.append(
                {
                    "id": str(record["chunk_id"]),
                    "values": embedding,
                    "metadata": sanitize_metadata(record),
                }
            )

    return vectors


def upsert_vectors(index: Any, vectors: list[dict[str, Any]]) -> None:
    total_batches = (len(vectors) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

    for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[start : start + UPSERT_BATCH_SIZE]
        batch_number = (start // UPSERT_BATCH_SIZE) + 1
        print(f"Upserting batch {batch_number}/{total_batches} ({len(batch)} vectors)")
        if NAMESPACE:
            index.upsert(vectors=batch, namespace=NAMESPACE)
        else:
            index.upsert(vectors=batch)


def main() -> None:
    pinecone_api_key = require_env("PINECONE_API_KEY")
    chunk_records = load_chunk_records(CHUNKS_PATH)
    print(f"Loaded {len(chunk_records)} chunk records from {CHUNKS_PATH}")

    local_model = load_local_model()
    if hasattr(local_model, "get_embedding_dimension"):
        embedding_dimension = local_model.get_embedding_dimension()
    else:
        embedding_dimension = local_model.get_sentence_embedding_dimension()
    print(f"Local model dimension: {embedding_dimension}")

    pc = Pinecone(api_key=pinecone_api_key)
    ensure_index(pc, INDEX_NAME, embedding_dimension)
    index = pc.Index(INDEX_NAME)

    vectors = build_pinecone_vectors(local_model, chunk_records)
    print(f"Built {len(vectors)} vectors using local model '{LOCAL_EMBEDDING_MODEL}'")

    upsert_vectors(index, vectors)
    print(f"Upserted {len(vectors)} vectors into Pinecone index '{INDEX_NAME}'")


if __name__ == "__main__":
    main()
