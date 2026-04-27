from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from .schemas import RetrievedChunk

SentencePair = tuple[str, str]
SentencePairsInput = list[SentencePair] | SentencePair

RERANKER_MODEL = "BAAI/bge-reranker-base"
MAX_RERANK_CANDIDATES = 50
_NORMALIZATION_EPSILON = 1e-8


def _is_fp16_available() -> bool:
    try:
        import torch
    except ImportError:
        return False

    return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())


class _CompatibleFlagReranker:
    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def compute_score(self, sentence_pairs: SentencePairsInput) -> Any:
        if hasattr(self._backend.tokenizer, "prepare_for_model"):
            try:
                return self._backend.compute_score(sentence_pairs)
            except AttributeError as exc:
                if "prepare_for_model" not in str(exc):
                    raise
        return _compute_scores_with_transformers_fallback(self._backend, sentence_pairs)


def _normalize_sentence_pairs(sentence_pairs: SentencePairsInput) -> list[SentencePair]:
    if isinstance(sentence_pairs, tuple):
        return [sentence_pairs]
    return sentence_pairs


def _compute_scores_with_transformers_fallback(
    backend: Any,
    sentence_pairs: SentencePairsInput,
) -> list[float]:
    import torch

    normalized_sentence_pairs = _normalize_sentence_pairs(sentence_pairs)
    detailed_pairs = backend.get_detailed_inputs(normalized_sentence_pairs)
    batch_size = max(1, int(getattr(backend, "batch_size", 128) or 128))
    max_length = int(getattr(backend, "max_length", 512) or 512)
    normalize = bool(getattr(backend, "normalize", False))
    device = getattr(backend, "target_devices", ["cpu"])[0]
    tokenizer = backend.tokenizer
    model = backend.model

    model.to(device)
    model.eval()

    all_scores: list[float] = []
    with torch.no_grad():
        for start_index in range(0, len(detailed_pairs), batch_size):
            batch = detailed_pairs[start_index : start_index + batch_size]
            queries = [pair[0] for pair in batch]
            passages = [pair[1] for pair in batch]
            inputs = tokenizer(
                queries,
                passages,
                padding=True,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            logits = model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
            all_scores.extend(float(score) for score in logits)

    if normalize:
        all_scores = [1 / (1 + math.exp(-score)) for score in all_scores]

    return all_scores


@lru_cache(maxsize=1)
def get_reranker() -> Any:
    from FlagEmbedding import FlagReranker

    backend = FlagReranker(RERANKER_MODEL, use_fp16=_is_fp16_available())
    return _CompatibleFlagReranker(backend)


def _prepare_rerank_candidates(chunks: list[RetrievedChunk]) -> list[tuple[RetrievedChunk, str]]:
    seen_keys: set[tuple[int | None, str]] = set()
    candidates: list[tuple[RetrievedChunk, str]] = []

    for chunk in chunks[:MAX_RERANK_CANDIDATES]:
        text = chunk.text
        if text is None:
            continue

        cleaned_text = text.strip()
        if not cleaned_text:
            continue

        dedupe_key = (chunk.campaign_id, cleaned_text[:50])
        if dedupe_key in seen_keys:
            continue

        seen_keys.add(dedupe_key)
        candidates.append((chunk, cleaned_text))

    return candidates


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    denominator = max_score - min_score + _NORMALIZATION_EPSILON
    return [(score - min_score) / denominator for score in scores]


def build_rerank_debug_scores(chunks: list[RetrievedChunk], limit: int = 10) -> list[dict[str, float | str | None]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "vector_score": chunk.score,
            "rerank_score": chunk.rerank_score,
            "final_score": chunk.final_score,
        }
        for chunk in chunks[:limit]
    ]


def rerank_chunks(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if top_k <= 0:
        return []

    candidates = _prepare_rerank_candidates(chunks)
    if not candidates:
        return []

    reranker = get_reranker()
    scores = reranker.compute_score([(query, cleaned_text) for _, cleaned_text in candidates])

    if isinstance(scores, (int, float)):
        raw_scores = [float(scores)]
    else:
        raw_scores = [float(score) for score in scores]

    reranked_chunks = [chunk for chunk, _ in candidates]
    normalized_scores = _normalize_scores(raw_scores)

    for chunk, raw_score, normalized_rerank in zip(reranked_chunks, raw_scores, normalized_scores):
        quality_score = (chunk.quality or 0.0) / 100.0
        vector_score = chunk.score or 0.0
        chunk.rerank_score = raw_score
        chunk.final_score = 0.85 * normalized_rerank + 0.1 * quality_score + 0.05 * vector_score

    reranked_chunks.sort(key=lambda chunk: chunk.final_score if chunk.final_score is not None else float("-inf"), reverse=True)
    return reranked_chunks[:top_k]
