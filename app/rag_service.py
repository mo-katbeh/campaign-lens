from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from typing import Any, Iterable

try:
    from google import genai
    
except ImportError:
    genai = None

from .config import (
    DEFAULT_TOP_K,
    ENRICHED_DATA_PATH,
    GEMINI_MODEL,
    MAX_CONTEXT_CHARS,
    RECOMMENDATION_QUERY_MULTIPLIER,
    require_gemini_api_key,
)
from .embeddings import embed_query
from .pinecone_store import extract_matches, extract_metadata, get_seed_vectors_for_campaign, query_index
from .schemas import AnswerResult, CampaignRecommendation, RetrievedChunk


def _normalize_retrieved_chunk(match: Any) -> RetrievedChunk:
    chunk_id = str(getattr(match, "id", None) or getattr(match, "get", lambda *_: None)("id") or "")
    score = getattr(match, "score", None)
    if score is None and isinstance(match, dict):
        score = match.get("score")
    metadata = extract_metadata(match)
    campaign_id = metadata.get("campaign_id")
    if campaign_id is not None:
        campaign_id = int(campaign_id)
    funding_ratio = metadata.get("funding_ratio")
    if funding_ratio is not None:
        funding_ratio = float(funding_ratio)
    quality = metadata.get("record_quality_score")
    if quality is not None:
        quality = float(quality)

    return RetrievedChunk(
        chunk_id=chunk_id,
        score=float(score) if score is not None else None,
        campaign_id=campaign_id,
        title=metadata.get("title"),
        theme=metadata.get("campaign_theme"),
        beneficiary=metadata.get("beneficiary_group"),
        year=str(metadata.get("year_label")) if metadata.get("year_label") is not None else None,
        funding_ratio=funding_ratio,
        quality=quality,
        text=str(metadata.get("text") or ""),
        metadata=metadata,
    )


def retrieve_chunks(query: str, filters: dict | None = None, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    query_vector = embed_query(query)
    response = query_index(vector=query_vector, top_k=top_k, filters=filters)
    return [_normalize_retrieved_chunk(match) for match in extract_matches(response)]


def build_context(chunks: Iterable[RetrievedChunk], max_context_chars: int = MAX_CONTEXT_CHARS) -> str:
    context_sections: list[str] = []
    total_length = 0

    for chunk in chunks:
        cleaned_text = chunk.text.strip()
        if not cleaned_text:
            continue

        section = (
            f"[Chunk {chunk.chunk_id}]\n"
            f"Campaign ID: {chunk.campaign_id}\n"
            f"Title: {chunk.title or 'Unknown'}\n"
            f"Theme: {chunk.theme or 'Unknown'}\n"
            f"Year: {chunk.year or 'Unknown'}\n"
            f"Text: {cleaned_text}"
        )
        projected_length = total_length + len(section) + (2 if context_sections else 0)
        if projected_length > max_context_chars:
            break
        context_sections.append(section)
        total_length = projected_length

    return "\n\n".join(context_sections)


def build_grounded_prompt(question: str, context: str) -> str:
    return (
        "You are answering questions about fundraising campaigns.\n"
        "Use only the provided context.\n"
        "If the context is insufficient, say that the answer cannot be determined from the retrieved campaign data.\n"
        "Do not invent facts, external history, statistics, or campaign details.\n"
        "Prefer concise factual answers.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def resolve_gemini_model_name(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name
    return f"models/{model_name}"


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _detect_funding_ratio_ranking(question: str) -> str | None:
    normalized_question = _normalize_text(question)
    if not normalized_question:
        return None

    has_campaign_intent = any(
        token in normalized_question
        for token in ("campaign", "campaigns", "campign", "campigns", "campain", "campains")
    )
    has_funding_intent = any(
        phrase in normalized_question
        for phrase in ("funding", "funded", "fundraising", "funding ratio")
    )
    if not has_campaign_intent or not has_funding_intent:
        return None

    if any(phrase in normalized_question for phrase in ("lowest", "least", "underfunded", "lowest funded")):
        return "asc"
    if any(phrase in normalized_question for phrase in ("highest", "most", "best funded", "top funded")):
        return "desc"
    return None


def _parse_location_mentions(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value if str(item).strip()]
    if not raw_value:
        return []
    try:
        parsed = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


@lru_cache(maxsize=1)
def load_campaign_analytics_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ENRICHED_DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            campaign_id = row.get("campaign_id")
            funding_ratio = row.get("funding_ratio")
            if not campaign_id or funding_ratio in {None, ""}:
                continue
            try:
                campaign_id_value = int(campaign_id)
                funding_ratio_value = float(funding_ratio)
            except ValueError:
                continue

            rows.append(
                {
                    "campaign_id": campaign_id_value,
                    "title": str(row.get("title") or "").strip() or None,
                    "year": str(row.get("year_label") or "").strip() or None,
                    "theme": str(row.get("campaign_theme") or "").strip() or None,
                    "beneficiary": str(row.get("beneficiary_group") or "").strip() or None,
                    "funding_ratio": funding_ratio_value,
                    "quality": float(row["record_quality_score"]) if row.get("record_quality_score") else None,
                    "funding_status": str(row.get("funding_status") or "").strip() or None,
                    "location_mentions": _parse_location_mentions(row.get("location_mentions")),
                    "search_chunks": row.get("search_chunks") or "[]",
                }
            )
    return rows


@lru_cache(maxsize=1)
def get_known_locations() -> dict[str, str]:
    known_locations: dict[str, str] = {}
    for row in load_campaign_analytics_rows():
        for location in row["location_mentions"]:
            normalized_location = _normalize_text(location)
            if normalized_location:
                known_locations.setdefault(normalized_location, location)
    return known_locations


def _extract_locations_from_question(question: str) -> list[str]:
    normalized_question = f" {_normalize_text(question)} "
    matched_locations: list[str] = []
    for normalized_location, display_location in get_known_locations().items():
        if f" {normalized_location} " in normalized_question:
            matched_locations.append(display_location)
    return sorted(set(matched_locations))


def _build_analytics_chunk(row: dict[str, Any]) -> RetrievedChunk:
    search_chunks = row.get("search_chunks") or "[]"
    try:
        parsed_chunks = json.loads(search_chunks)
    except json.JSONDecodeError:
        parsed_chunks = []

    first_chunk = parsed_chunks[0] if parsed_chunks else {}
    metadata = dict(first_chunk) if isinstance(first_chunk, dict) else {}
    metadata.setdefault("campaign_id", row["campaign_id"])
    metadata.setdefault("title", row["title"])
    metadata.setdefault("campaign_theme", row["theme"])
    metadata.setdefault("beneficiary_group", row["beneficiary"])
    metadata.setdefault("year_label", row["year"])
    metadata.setdefault("funding_ratio", row["funding_ratio"])
    metadata.setdefault("record_quality_score", row["quality"])
    metadata.setdefault("location_mentions", row["location_mentions"])

    return RetrievedChunk(
        chunk_id=str(metadata.get("chunk_id") or f"{row['campaign_id']}-analytics"),
        campaign_id=row["campaign_id"],
        title=row["title"],
        theme=row["theme"],
        beneficiary=row["beneficiary"],
        year=row["year"],
        funding_ratio=row["funding_ratio"],
        quality=row["quality"],
        text=str(metadata.get("text") or ""),
        metadata=metadata,
    )


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def answer_funding_ratio_ranking_question(question: str, top_k: int = DEFAULT_TOP_K) -> AnswerResult | None:
    sort_direction = _detect_funding_ratio_ranking(question)
    if sort_direction is None:
        return None

    candidate_rows = load_campaign_analytics_rows()
    requested_locations = _extract_locations_from_question(question)
    if requested_locations:
        requested_location_set = set(requested_locations)
        candidate_rows = [
            row
            for row in candidate_rows
            if requested_location_set.intersection(set(row["location_mentions"]))
        ]

    if not candidate_rows:
        return None

    reverse = sort_direction == "desc"
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (
            row["funding_ratio"],
            -(row["quality"] or 0.0),
            row["campaign_id"],
        ),
        reverse=reverse,
    )
    selected_rows = candidate_rows[: max(top_k, 1)]
    retrieved_chunks = [_build_analytics_chunk(row) for row in selected_rows]

    ranking_label = "Highest-funded" if reverse else "Lowest-funded"
    location_label = ""
    if requested_locations:
        joined_locations = ", ".join(requested_locations)
        location_label = f" in {joined_locations}"

    answer_lines = [
        f"{ranking_label} campaigns{location_label} by funding ratio:"
    ]
    for index, row in enumerate(selected_rows, start=1):
        funding_status = f", status: {row['funding_status']}" if row.get("funding_status") else ""
        answer_lines.append(
            f"{index}. {row['title']} (Campaign {row['campaign_id']}): {_format_percentage(row['funding_ratio'])} funded{funding_status}."
        )

    return AnswerResult(
        question=question,
        answer="\n".join(answer_lines),
        retrieved_chunks=retrieved_chunks,
        source_campaign_ids=[row["campaign_id"] for row in selected_rows],
        source_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
    )


@lru_cache(maxsize=1)
def get_gemini_client():
    if genai is None:
        raise RuntimeError(
            "google-genai is not installed. Run `pip install google-genai`."
        )
    return genai.Client(api_key=require_gemini_api_key())

def answer_question(question: str, filters: dict | None = None, top_k: int = DEFAULT_TOP_K) -> AnswerResult:
    analytics_answer = answer_funding_ratio_ranking_question(question, top_k=top_k)
    if analytics_answer is not None:
        return analytics_answer

    chunks = retrieve_chunks(question, filters=filters, top_k=top_k)
    source_campaign_ids = sorted({chunk.campaign_id for chunk in chunks if chunk.campaign_id is not None})
    source_chunk_ids = [chunk.chunk_id for chunk in chunks]

    context = build_context(chunks)
    if not context:
        return AnswerResult(
            question=question,
            answer="The answer cannot be determined from the retrieved campaign data.",
            retrieved_chunks=chunks,
            source_campaign_ids=source_campaign_ids,
            source_chunk_ids=source_chunk_ids,
        )

    prompt = build_grounded_prompt(question, context)
    try:
        client = get_gemini_client()

        response = client.models.generate_content(
            model=resolve_gemini_model_name(GEMINI_MODEL),
            contents=prompt,
            config={
                "temperature": 0.2,
                # "maxOutputTokens": 1000,
                },
)
    except Exception as exc:
        raise RuntimeError(f"Gemini grounded answer generation failed: {exc}") from exc
    answer_text = getattr(response, "text", None)

    if not answer_text and hasattr(response, "candidates"):
        try:
            answer_text = response.candidates[0].content.parts[0].text
        except Exception:
            answer_text = None

    if not answer_text:
        answer_text = "The answer cannot be determined from the retrieved campaign data."
    return AnswerResult(
        question=question,
        answer=answer_text.strip(),
        retrieved_chunks=chunks,
        source_campaign_ids=source_campaign_ids,
        source_chunk_ids=source_chunk_ids,
    )


def recommend_similar_campaigns(
    campaign_id: int,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    seed_chunk_limit: int = 3,
) -> list[CampaignRecommendation]:
    seed_vectors = get_seed_vectors_for_campaign(campaign_id, limit=seed_chunk_limit)
    if not seed_vectors:
        return []

    aggregated: dict[int, dict[str, Any]] = {}
    per_seed_top_k = max(top_k * RECOMMENDATION_QUERY_MULTIPLIER, top_k + 5)

    for seed in seed_vectors:
        response = query_index(vector=seed["values"], top_k=per_seed_top_k, filters=filters)
        for match in extract_matches(response):
            chunk = _normalize_retrieved_chunk(match)
            if chunk.campaign_id is None or chunk.campaign_id == int(campaign_id):
                continue

            entry = aggregated.setdefault(
                chunk.campaign_id,
                {
                    "campaign_id": chunk.campaign_id,
                    "title": chunk.title,
                    "score": float("-inf"),
                    "source_chunk_ids": set(),
                    "representative_chunk_id": None,
                    "representative_text": None,
                },
            )
            entry["source_chunk_ids"].add(chunk.chunk_id)
            if chunk.score is not None and chunk.score > entry["score"]:
                entry["score"] = chunk.score
                entry["title"] = chunk.title
                entry["representative_chunk_id"] = chunk.chunk_id
                entry["representative_text"] = chunk.text

    recommendations = [
        CampaignRecommendation(
            campaign_id=entry["campaign_id"],
            title=entry["title"],
            score=float(entry["score"]),
            supporting_chunk_count=len(entry["source_chunk_ids"]),
            representative_chunk_id=entry["representative_chunk_id"],
            representative_text=entry["representative_text"],
            source_chunk_ids=sorted(entry["source_chunk_ids"]),
        )
        for entry in aggregated.values()
        if entry["score"] != float("-inf")
    ]
    recommendations.sort(key=lambda item: (-item.score, -item.supporting_chunk_count, item.campaign_id))
    return recommendations[:top_k]
