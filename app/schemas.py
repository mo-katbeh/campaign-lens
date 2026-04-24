from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float | None = None
    campaign_id: int | None = None
    title: str | None = None
    theme: str | None = None
    beneficiary: str | None = None
    year: str | None = None
    funding_ratio: float | None = None
    quality: float | None = None
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerResult(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    source_campaign_ids: list[int] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class CampaignRecommendation(BaseModel):
    campaign_id: int
    title: str | None = None
    score: float
    supporting_chunk_count: int = 0
    representative_chunk_id: str | None = None
    representative_text: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    query: str
    filters: dict[str, Any] | None = None
    top_k: int = 5


class RetrieveResponse(BaseModel):
    query: str
    filters: dict[str, Any] | None = None
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class AnswerRequest(BaseModel):
    question: str
    filters: dict[str, Any] | None = None
    top_k: int = 5


class RecommendationRequest(BaseModel):
    campaign_id: int
    filters: dict[str, Any] | None = None
    top_k: int = 5
    seed_chunk_limit: int = 3


class RecommendationResponse(BaseModel):
    campaign_id: int
    filters: dict[str, Any] | None = None
    recommendations: list[CampaignRecommendation] = Field(default_factory=list)
