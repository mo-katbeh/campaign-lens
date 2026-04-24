from __future__ import annotations

import os
from pathlib import Path

from scripts.upload_campaign_chunks_to_pinecone import (
    INDEX_NAME,
    LOCAL_EMBEDDING_MODEL,
    NAMESPACE,
    NORMALIZE_EMBEDDINGS,
    PINECONE_CLOUD,
    PINECONE_REGION,
    ROOT,
    require_env,
)


PROJECT_ROOT = ROOT
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "campaign_search_chunks.jsonl"
ENRICHED_DATA_PATH = DATA_DIR / "campaigns_enriched.csv"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "5000"))
RECOMMENDATION_QUERY_MULTIPLIER = int(os.getenv("RAG_RECOMMENDATION_QUERY_MULTIPLIER", "4"))
FETCH_BATCH_SIZE = int(os.getenv("PINECONE_FETCH_BATCH_SIZE", "100"))
UPDATE_BATCH_SIZE = int(os.getenv("PINECONE_UPDATE_BATCH_SIZE", "100"))


def require_gemini_api_key() -> str:
    return require_env("GEMINI_API_KEY")


def optional_namespace() -> str | None:
    return NAMESPACE or None


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CHUNKS_PATH",
    "ENRICHED_DATA_PATH",
    "INDEX_NAME",
    "LOCAL_EMBEDDING_MODEL",
    "NORMALIZE_EMBEDDINGS",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
    "GEMINI_MODEL",
    "DEFAULT_TOP_K",
    "MAX_CONTEXT_CHARS",
    "RECOMMENDATION_QUERY_MULTIPLIER",
    "FETCH_BATCH_SIZE",
    "UPDATE_BATCH_SIZE",
    "optional_namespace",
    "require_gemini_api_key",
]
