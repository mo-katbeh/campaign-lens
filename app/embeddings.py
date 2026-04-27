from __future__ import annotations

from typing import Sequence

from scripts.upload_campaign_chunks_to_pinecone import embed_text_batch, load_local_model

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

def get_embedding_model():
    return load_local_model()


def get_embedding_dimension() -> int:
    model = get_embedding_model()
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    cleaned_texts = [str(text).strip() for text in texts if str(text).strip()]
    if not cleaned_texts:
        return []
    model = get_embedding_model()
    return embed_text_batch(model, cleaned_texts)


def embed_query(query: str) -> list[float]:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query must not be empty.")
    # BGE recommends an instruction on queries only so the query embedding is aligned
    # with passage embeddings for cosine retrieval, while document/chunk text stays raw.
    embeddings = embed_texts([f"{BGE_QUERY_INSTRUCTION}{cleaned_query}"])
    if not embeddings:
        raise RuntimeError("Failed to create an embedding for the query.")
    return embeddings[0]
