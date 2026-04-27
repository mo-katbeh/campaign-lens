from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import logging
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Iterable, Mapping

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
from .reranker import build_rerank_debug_scores, rerank_chunks
from .schemas import AnswerResult, CampaignRecommendation, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YearConstraint:
    operator: str
    year: int


@dataclass
class QuestionPlan:
    answer_mode: str = "rag_generation"
    funding_intent: str | None = None
    sort_direction: str | None = None
    year_constraint: YearConstraint | None = None
    requested_count: int | None = None
    location_filters: tuple[str, ...] = ()
    theme_filters: tuple[str, ...] = ()
    beneficiary_filters: tuple[str, ...] = ()
    merged_filters: dict[str, Any] = field(default_factory=dict)

    @property
    def is_confident_structured(self) -> bool:
        return self.answer_mode in {"structured_ranking", "structured_filter"}


QUERY_FILTER_RULES = (
    (
        re.compile(r"\blow funding\b", re.IGNORECASE),
        {"funding_ratio": {"lt": 1}},
    ),
    (
        re.compile(r"\bunder funded\b", re.IGNORECASE),
        {"funding_ratio": {"lt": 1}},
    ),
    (
        re.compile(r"\bunderfunded\b", re.IGNORECASE),
        {"funding_ratio": {"lt": 1}},
    ),
    (
        re.compile(r"\bhigh quality\b", re.IGNORECASE),
        {"quality": {"gte": 80}},
    ),
    (
        re.compile(r"\bmedical\b", re.IGNORECASE),
        {"theme": "medical"},
    ),
    (
        re.compile(r"\beducation\b", re.IGNORECASE),
        {"theme": "education"},
    ),
    (
        re.compile(r"\brecent\b", re.IGNORECASE),
        {"year": {"gte": "__RECENT_YEAR__"}},
    ),
)

LOW_FUNDING_PHRASES = (
    "low funding",
    "low funded",
    "lowest funding",
    "lowest funded",
    "least funded",
    "underfunded",
    "under funded",
)

HIGH_FUNDING_PHRASES = (
    "highest funded",
    "high funded",
    "most funded",
    "best funded",
    "top funded",
)

FAILED_TO_REACH_GOAL_PHRASES = (
    "failed to reach their goal",
    "failed to reach the goal",
    "failed to reach goal",
    "did not reach their goal",
    "did not reach the goal",
    "did not reach goal",
    "didn't reach their goal",
    "didn't reach the goal",
    "didn't reach goal",
)

YEAR_CONSTRAINT_PATTERNS = (
    (re.compile(r"\bon\s+or\s+before\s+(\d{4})\b", re.IGNORECASE), "lte"),
    (re.compile(r"\bon\s+or\s+after\s+(\d{4})\b", re.IGNORECASE), "gte"),
    (re.compile(r"\bbefore\s+(\d{4})\b", re.IGNORECASE), "lt"),
    (re.compile(r"\bafter\s+(\d{4})\b", re.IGNORECASE), "gt"),
    (re.compile(r"\bduring\s+(\d{4})\b", re.IGNORECASE), "eq"),
    (re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE), "eq"),
)

COUNT_PATTERNS = (
    re.compile(r"\btop\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\blist\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\bgive\s+me\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+campaigns?\b", re.IGNORECASE),
)


def _is_rag_debug_enabled() -> bool:
    return os.getenv("RAG_DEBUG", "").strip().lower() == "true"


def _resolve_rule_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    if resolved.get("year") == {"gte": "__RECENT_YEAR__"}:
        resolved["year"] = {"gte": str(datetime.now().year - 2)}
    return resolved


def _merge_filter_dicts(base_filters: Mapping[str, Any] | None, extra_filters: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base_filters or {})
    for key, extra_value in (extra_filters or {}).items():
        if extra_value is None:
            continue

        current_value = merged.get(key)
        if current_value is None:
            merged[key] = extra_value
            continue

        if isinstance(current_value, Mapping) and isinstance(extra_value, Mapping):
            combined = dict(current_value)
            for operator_key, operator_value in extra_value.items():
                combined.setdefault(operator_key, operator_value)
            merged[key] = combined

    return merged


def _analyze_query(query: str) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    extracted_filters: dict[str, Any] = {}
    matched_spans: list[tuple[int, int]] = []

    for pattern, payload in QUERY_FILTER_RULES:
        for match in pattern.finditer(query):
            matched_spans.append(match.span())
            extracted_filters = _merge_filter_dicts(extracted_filters, _resolve_rule_payload(payload))

    return extracted_filters, matched_spans


def parse_query(query: str) -> dict[str, Any]:
    extracted_filters, _ = _analyze_query(query)
    return extracted_filters


def clean_query(query: str) -> str:
    _, matched_spans = _analyze_query(query)
    if not matched_spans:
        return re.sub(r"\s+", " ", query).strip() or "campaign"

    cleaned_parts: list[str] = []
    current_index = 0
    for start, end in sorted(matched_spans):
        if start < current_index:
            continue
        cleaned_parts.append(query[current_index:start])
        current_index = end
    cleaned_parts.append(query[current_index:])

    cleaned_query = "".join(cleaned_parts)
    cleaned_query = re.sub(r"\s+([,.;:!?])", r"\1", cleaned_query)
    cleaned_query = re.sub(r"([(\[])\s+", r"\1", cleaned_query)
    cleaned_query = re.sub(r"\s+([)\]])", r"\1", cleaned_query)
    cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip(" ,.;:!?-_")
    return cleaned_query or "campaign"


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


def _determine_initial_retrieval_k(top_k: int) -> int:
    if top_k <= 5:
        return 20
    if top_k <= 10:
        return 30
    return 50


def retrieve_chunks(query: str, filters: dict | None = None, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    extracted_filters = parse_query(query)
    cleaned_query = clean_query(query)
    merged_filters = _merge_filter_dicts(filters, extracted_filters) or None
    initial_k = _determine_initial_retrieval_k(top_k)
    debug_payload = {
        "raw_query": query,
        "cleaned_query": cleaned_query,
        "initial_k": initial_k,
        "top_k": top_k,
        "extracted_filters": extracted_filters,
        "final_filters": merged_filters,
    }

    query_vector = embed_query(cleaned_query)
    response = query_index(vector=query_vector, top_k=initial_k, filters=merged_filters)
    chunks = [_normalize_retrieved_chunk(match) for match in extract_matches(response)]

    try:
        result = rerank_chunks(cleaned_query, chunks, top_k=top_k)
    except Exception:
        logger.exception("Reranking failed; falling back to vector-ranked results.")
        result = chunks[:top_k]

    if _is_rag_debug_enabled():
        logger.info(
            {
                **debug_payload,
                "scores": build_rerank_debug_scores(result),
            }
        )

    return result


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


def _has_campaign_intent(question: str) -> bool:
    normalized_question = _normalize_text(question)
    return any(
        token in normalized_question
        for token in ("campaign", "campaigns", "campign", "campigns", "campain", "campains")
    )


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


def _parse_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def load_campaign_analytics_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ENRICHED_DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            campaign_id = row.get("campaign_id")
            funding_ratio_value = _parse_float(row.get("funding_ratio"))
            if not campaign_id or funding_ratio_value is None:
                continue
            try:
                campaign_id_value = int(campaign_id)
            except ValueError:
                continue

            quality_value = _parse_float(row.get("record_quality_score"))

            rows.append(
                {
                    "campaign_id": campaign_id_value,
                    "title": str(row.get("title") or "").strip() or None,
                    "year": str(row.get("year_label") or "").strip() or None,
                    "theme": str(row.get("campaign_theme") or "").strip() or None,
                    "beneficiary": str(row.get("beneficiary_group") or "").strip() or None,
                    "funding_ratio": funding_ratio_value,
                    "quality": quality_value,
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


@lru_cache(maxsize=1)
def get_known_themes() -> dict[str, str]:
    known_themes: dict[str, str] = {}
    for row in load_campaign_analytics_rows():
        theme = row.get("theme")
        if not theme:
            continue
        normalized_theme = _normalize_text(theme)
        if normalized_theme:
            known_themes.setdefault(normalized_theme, str(theme))
    return known_themes


@lru_cache(maxsize=1)
def get_known_beneficiaries() -> dict[str, str]:
    known_beneficiaries: dict[str, str] = {}
    for row in load_campaign_analytics_rows():
        beneficiary = row.get("beneficiary")
        if not beneficiary:
            continue
        normalized_beneficiary = _normalize_text(beneficiary)
        if normalized_beneficiary:
            known_beneficiaries.setdefault(normalized_beneficiary, str(beneficiary))
    return known_beneficiaries


def _extract_known_values_from_question(question: str, known_values: Mapping[str, str]) -> list[str]:
    normalized_question = f" {_normalize_text(question)} "
    matched_values: list[str] = []
    for normalized_value, display_value in known_values.items():
        if f" {normalized_value} " in normalized_question:
            matched_values.append(display_value)
    return sorted(set(matched_values))


def _extract_locations_from_question(question: str) -> list[str]:
    return _extract_known_values_from_question(question, get_known_locations())


def _extract_year_constraint(question: str) -> YearConstraint | None:
    for pattern, operator in YEAR_CONSTRAINT_PATTERNS:
        match = pattern.search(question)
        if match is not None:
            return YearConstraint(operator=operator, year=int(match.group(1)))
    return None


def _extract_requested_count(question: str) -> int | None:
    for pattern in COUNT_PATTERNS:
        match = pattern.search(question)
        if match is None:
            continue
        count = int(match.group(1))
        if count <= 0:
            return None
        return min(count, 50)
    return None


def _extract_funding_intent(question: str) -> tuple[str | None, str | None, dict[str, Any]]:
    normalized_question = _normalize_text(question)
    if any(phrase in normalized_question for phrase in FAILED_TO_REACH_GOAL_PHRASES):
        return "failed_to_reach_goal", "asc", {"funding_ratio": {"lt": 1}}
    if any(phrase in normalized_question for phrase in LOW_FUNDING_PHRASES):
        return "underfunded", "asc", {"funding_ratio": {"lt": 1}}
    if any(phrase in normalized_question for phrase in HIGH_FUNDING_PHRASES):
        return "high_funding", "desc", {}
    return None, None, {}


def build_question_plan(question: str, filters: dict | None = None, top_k: int = DEFAULT_TOP_K) -> QuestionPlan:
    merged_filters = _merge_filter_dicts(filters, parse_query(question)) or {}
    location_filters = tuple(_extract_locations_from_question(question))
    theme_filters = tuple(_extract_known_values_from_question(question, get_known_themes()))
    beneficiary_filters = tuple(_extract_known_values_from_question(question, get_known_beneficiaries()))
    year_constraint = _extract_year_constraint(question)
    requested_count = _extract_requested_count(question)
    funding_intent, sort_direction, funding_filters = _extract_funding_intent(question)

    merged_filters = _merge_filter_dicts(merged_filters, funding_filters) or merged_filters
    if theme_filters and "theme" not in merged_filters:
        merged_filters["theme"] = theme_filters[0]
    if beneficiary_filters and "beneficiary" not in merged_filters:
        merged_filters["beneficiary"] = beneficiary_filters[0]

    answer_mode = "rag_generation"
    has_structured_signal = any(
        (
            funding_intent is not None,
            year_constraint is not None,
            requested_count is not None,
            bool(location_filters),
            bool(theme_filters),
            bool(beneficiary_filters),
        )
    )
    if _has_campaign_intent(question) and has_structured_signal:
        answer_mode = "structured_ranking" if sort_direction is not None else "structured_filter"

    return QuestionPlan(
        answer_mode=answer_mode,
        funding_intent=funding_intent,
        sort_direction=sort_direction,
        year_constraint=year_constraint,
        requested_count=requested_count,
        location_filters=location_filters,
        theme_filters=theme_filters,
        beneficiary_filters=beneficiary_filters,
        merged_filters=merged_filters,
    )


def _extract_gemini_text(response: Any) -> str | None:
    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        return None

    content = getattr(candidates[0], "content", None)
    if content is None:
        return None

    parts = getattr(content, "parts", None)
    if not isinstance(parts, list) or not parts:
        return None

    part_text = getattr(parts[0], "text", None)
    if isinstance(part_text, str) and part_text.strip():
        return part_text
    return None


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


def _parse_year(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_filter_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key == "year":
        return _parse_year(value)
    if key in {"funding_ratio", "quality"}:
        return _parse_float(value)
    if key == "campaign_id":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return str(value).strip()


def _get_row_value(row: Mapping[str, Any], key: str) -> Any:
    if key == "year":
        return _parse_year(row.get("year"))
    return row.get(key)


def _matches_filter_value(actual_value: Any, operator: str, expected_value: Any) -> bool:
    if actual_value is None:
        return False
    if operator == "eq":
        return actual_value == expected_value
    if operator == "ne":
        return actual_value != expected_value
    if operator == "lt":
        return actual_value < expected_value
    if operator == "lte":
        return actual_value <= expected_value
    if operator == "gt":
        return actual_value > expected_value
    if operator == "gte":
        return actual_value >= expected_value
    if operator == "in" and isinstance(expected_value, list):
        return actual_value in expected_value
    return False


def _row_matches_filters(row: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    if not filters:
        return True

    for key, raw_value in filters.items():
        actual_value = _get_row_value(row, key)
        if isinstance(raw_value, Mapping):
            for operator, operator_value in raw_value.items():
                if operator == "in" and isinstance(operator_value, list):
                    expected_value = [_coerce_filter_value(key, item) for item in operator_value]
                else:
                    expected_value = _coerce_filter_value(key, operator_value)
                if expected_value is None or not _matches_filter_value(actual_value, operator, expected_value):
                    return False
            continue

        expected_value = _coerce_filter_value(key, raw_value)
        if expected_value is None or actual_value != expected_value:
            return False

    return True


def _matches_year_constraint(row: Mapping[str, Any], year_constraint: YearConstraint | None) -> bool:
    if year_constraint is None:
        return True
    actual_year = _parse_year(row.get("year"))
    if actual_year is None:
        return False
    return _matches_filter_value(actual_year, year_constraint.operator, year_constraint.year)


def _build_question_interpretation(plan: QuestionPlan) -> str:
    lines = [f"Answer mode: {plan.answer_mode}"]
    if plan.funding_intent:
        lines.append(f"Funding signal: {plan.funding_intent}")
    if plan.sort_direction:
        lines.append(f"Sort direction: {plan.sort_direction}")
    if plan.year_constraint is not None:
        lines.append(f"Year constraint: {plan.year_constraint.operator} {plan.year_constraint.year}")
    if plan.requested_count is not None:
        lines.append(f"Requested count: {plan.requested_count}")
    if plan.location_filters:
        lines.append(f"Locations: {', '.join(plan.location_filters)}")
    if plan.theme_filters:
        lines.append(f"Themes: {', '.join(plan.theme_filters)}")
    if plan.beneficiary_filters:
        lines.append(f"Beneficiaries: {', '.join(plan.beneficiary_filters)}")
    if plan.merged_filters:
        lines.append(f"Structured filters: {plan.merged_filters}")
    return "\n".join(lines)


def _build_structured_answer_heading(plan: QuestionPlan) -> str:
    if plan.funding_intent == "failed_to_reach_goal":
        heading = "Campaigns that failed to reach their goal"
    elif plan.sort_direction == "desc":
        heading = "Highest-funded campaigns"
    elif plan.sort_direction == "asc":
        heading = "Lowest-funded campaigns"
    else:
        heading = "Matching campaigns"

    if plan.location_filters:
        heading = f"{heading} in {', '.join(plan.location_filters)}"

    if plan.year_constraint is not None:
        if plan.year_constraint.operator == "lt":
            heading = f"{heading} before {plan.year_constraint.year}"
        elif plan.year_constraint.operator == "lte":
            heading = f"{heading} on or before {plan.year_constraint.year}"
        elif plan.year_constraint.operator == "gt":
            heading = f"{heading} after {plan.year_constraint.year}"
        elif plan.year_constraint.operator == "gte":
            heading = f"{heading} on or after {plan.year_constraint.year}"
        elif plan.year_constraint.operator == "eq":
            heading = f"{heading} in {plan.year_constraint.year}"

    if plan.answer_mode == "structured_ranking":
        heading = f"{heading} by funding ratio"
    return f"{heading}:"


def answer_structured_campaign_question(
    question: str,
    filters: dict | None = None,
    top_k: int = DEFAULT_TOP_K,
    plan: QuestionPlan | None = None,
) -> AnswerResult | None:
    resolved_plan = plan or build_question_plan(question, filters=filters, top_k=top_k)
    if not resolved_plan.is_confident_structured:
        return None

    candidate_rows = [
        row
        for row in load_campaign_analytics_rows()
        if _row_matches_filters(row, resolved_plan.merged_filters)
        and _matches_year_constraint(row, resolved_plan.year_constraint)
    ]
    if resolved_plan.location_filters:
        requested_location_set = set(resolved_plan.location_filters)
        candidate_rows = [
            row
            for row in candidate_rows
            if requested_location_set.intersection(set(row["location_mentions"]))
        ]

    if not candidate_rows:
        return AnswerResult(
            question=question,
            answer="No matching campaigns were found in the campaign data for the requested criteria.",
            retrieved_chunks=[],
            source_campaign_ids=[],
            source_chunk_ids=[],
        )

    limit = resolved_plan.requested_count if resolved_plan.requested_count is not None else max(top_k, 1)
    if resolved_plan.answer_mode == "structured_ranking":
        candidate_rows = sorted(
            candidate_rows,
            key=lambda row: (
                row["funding_ratio"],
                -(row["quality"] or 0.0),
                row["campaign_id"],
            ),
            reverse=resolved_plan.sort_direction == "desc",
        )
    else:
        candidate_rows = sorted(
            candidate_rows,
            key=lambda row: (
                -(_parse_year(row.get("year")) or 0),
                -(row["quality"] or 0.0),
                row["campaign_id"],
            ),
        )

    selected_rows = candidate_rows[: max(limit, 1)]
    retrieved_chunks = [_build_analytics_chunk(row) for row in selected_rows]
    answer_lines = [_build_structured_answer_heading(resolved_plan)]
    for index, row in enumerate(selected_rows, start=1):
        year_segment = f", year: {row['year']}" if row.get("year") else ""
        funding_status = f", status: {row['funding_status']}" if row.get("funding_status") else ""
        answer_lines.append(
            f"{index}. {row['title']} (Campaign {row['campaign_id']}): {_format_percentage(row['funding_ratio'])} funded{year_segment}{funding_status}."
        )

    return AnswerResult(
        question=question,
        answer="\n".join(answer_lines),
        retrieved_chunks=retrieved_chunks,
        source_campaign_ids=[row["campaign_id"] for row in selected_rows],
        source_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
    )


def answer_funding_ratio_ranking_question(question: str, top_k: int = DEFAULT_TOP_K) -> AnswerResult | None:
    return answer_structured_campaign_question(question, top_k=top_k)


@lru_cache(maxsize=1)
def get_gemini_client():
    if genai is None:
        raise RuntimeError(
            "google-genai is not installed. Run `pip install google-genai`."
        )
    return genai.Client(api_key=require_gemini_api_key())


def answer_question(question: str, filters: dict | None = None, top_k: int = DEFAULT_TOP_K) -> AnswerResult:
    question_plan = build_question_plan(question, filters=filters, top_k=top_k)
    structured_answer = answer_structured_campaign_question(
        question,
        filters=filters,
        top_k=top_k,
        plan=question_plan,
    )
    if structured_answer is not None:
        return structured_answer

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

    prompt = (
        "Question interpretation:\n"
        f"{_build_question_interpretation(question_plan)}\n\n"
        f"{build_grounded_prompt(question, context)}"
    )
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
    answer_text = _extract_gemini_text(response)

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
