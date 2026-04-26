from __future__ import annotations

import csv
import json
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

from pinecone import Pinecone

from .config import CHUNKS_PATH, ENRICHED_DATA_PATH, FETCH_BATCH_SIZE, INDEX_NAME, optional_namespace
from scripts.upload_campaign_chunks_to_pinecone import require_env


FIELD_ALIASES = {
    "theme": "campaign_theme",
    "beneficiary": "beneficiary_group",
    "year": "year_label",
    "quality": "record_quality_score",
    "campaign_id": "campaign_id",
    "funding_ratio": "funding_ratio",
    "campaign_theme": "campaign_theme",
    "beneficiary_group": "beneficiary_group",
    "year_label": "year_label",
    "record_quality_score": "record_quality_score",
}

FILTER_OPERATORS = {
    "eq": "$eq",
    "ne": "$ne",
    "lt": "$lt",
    "lte": "$lte",
    "gt": "$gt",
    "gte": "$gte",
    "in": "$in",
}


def _parse_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _batched(items: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _coerce_year(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _coerce_filter_value(field_name: str, value: Any) -> Any:
    if value is None:
        return None
    if field_name == "year_label":
        return _coerce_year(value)
    if field_name == "campaign_id":
        return int(value)
    if field_name in {"funding_ratio", "record_quality_score"}:
        return float(value)
    return value


def translate_filters(filters: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None

    translated: dict[str, Any] = {}
    for raw_key, raw_value in filters.items():
        field_name = FIELD_ALIASES.get(raw_key)
        if field_name is None or raw_value is None:
            continue

        if isinstance(raw_value, Mapping):
            operator_payload: dict[str, Any] = {}
            for operator_key, operator_value in raw_value.items():
                pinecone_operator = FILTER_OPERATORS.get(operator_key)
                if pinecone_operator is None or operator_value is None:
                    continue
                if operator_key == "in" and isinstance(operator_value, Sequence) and not isinstance(
                    operator_value, (str, bytes)
                ):
                    coerced_values = [
                        _coerce_filter_value(field_name, item) for item in operator_value if item is not None
                    ]
                    if coerced_values:
                        operator_payload[pinecone_operator] = coerced_values
                    continue
                operator_payload[pinecone_operator] = _coerce_filter_value(field_name, operator_value)

            if operator_payload:
                translated[field_name] = operator_payload
            continue

        translated[field_name] = _coerce_filter_value(field_name, raw_value)

    return translated or None


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=require_env("PINECONE_API_KEY"))


@lru_cache(maxsize=1)
def get_index():
    return get_pinecone_client().Index(INDEX_NAME)


@lru_cache(maxsize=1)
def load_chunk_catalog() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@lru_cache(maxsize=1)
def get_campaign_chunk_ids() -> dict[int, list[str]]:
    mapping: dict[int, list[str]] = {}
    for record in load_chunk_catalog():
        campaign_id = int(record["campaign_id"])
        mapping.setdefault(campaign_id, []).append(str(record["chunk_id"]))
    return mapping


@lru_cache(maxsize=1)
def get_chunk_record_map() -> dict[str, dict[str, Any]]:
    return {str(record["chunk_id"]): record for record in load_chunk_catalog()}


@lru_cache(maxsize=1)
def get_campaign_quality_map() -> dict[int, float]:
    quality_map: dict[int, float] = {}
    with ENRICHED_DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            campaign_id = row.get("campaign_id")
            quality_value = _parse_float(row.get("record_quality_score"))
            if not campaign_id or quality_value is None:
                continue
            quality_map[int(campaign_id)] = quality_value
    return quality_map


def _extract_attr(value: Any, name: str, default: Any = None) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, Mapping):
        return value.get(name, default)
    return default


def _response_namespace_kwargs() -> dict[str, Any]:
    namespace = optional_namespace()
    if not namespace:
        return {}
    return {"namespace": namespace}


def query_index(
    *,
    vector: list[float],
    top_k: int,
    filters: Mapping[str, Any] | None = None,
    include_values: bool = False,
):
    query_kwargs: dict[str, Any] = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True,
        "include_values": include_values,
    }
    translated_filters = translate_filters(filters)
    if translated_filters:
        query_kwargs["filter"] = translated_filters
    query_kwargs.update(_response_namespace_kwargs())
    return get_index().query(**query_kwargs)


def extract_matches(response: Any) -> list[Any]:
    matches = _extract_attr(response, "matches", [])
    return list(matches or [])


def extract_metadata(match: Any) -> dict[str, Any]:
    metadata = _extract_attr(match, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = dict(metadata)
    chunk_id = str(_extract_attr(match, "id", ""))
    if chunk_id:
        fallback_record = get_chunk_record_map().get(chunk_id, {})
        quality = metadata.get("record_quality_score")
        if quality is None:
            campaign_id = metadata.get("campaign_id", fallback_record.get("campaign_id"))
            if campaign_id is not None:
                metadata["record_quality_score"] = get_campaign_quality_map().get(int(campaign_id))
        for key, value in fallback_record.items():
            metadata.setdefault(key, value)
    return metadata


def fetch_vectors_by_ids(chunk_ids: Sequence[str]) -> dict[str, Any]:
    if not chunk_ids:
        return {}

    vectors: dict[str, Any] = {}
    for batch in _batched([str(chunk_id) for chunk_id in chunk_ids], FETCH_BATCH_SIZE):
        response = get_index().fetch(ids=batch, **_response_namespace_kwargs())
        raw_vectors = _extract_attr(response, "vectors", {}) or {}
        if isinstance(raw_vectors, Mapping):
            vectors.update(dict(raw_vectors))
    return vectors


def get_seed_vectors_for_campaign(campaign_id: int, limit: int = 3) -> list[dict[str, Any]]:
    chunk_ids = get_campaign_chunk_ids().get(int(campaign_id), [])
    if not chunk_ids:
        return []

    fetched = fetch_vectors_by_ids(chunk_ids[: max(limit, 1)])
    seeds: list[dict[str, Any]] = []
    for chunk_id in chunk_ids[: max(limit, 1)]:
        vector_record = fetched.get(str(chunk_id))
        if vector_record is None:
            continue
        values = _extract_attr(vector_record, "values")
        if not values:
            continue
        metadata = _extract_attr(vector_record, "metadata", {}) or {}
        if not metadata:
            metadata = get_chunk_record_map().get(str(chunk_id), {})
        seeds.append(
            {
                "chunk_id": str(chunk_id),
                "values": list(values),
                "metadata": dict(metadata),
            }
        )
    return seeds
