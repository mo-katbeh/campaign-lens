from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .rag_service import answer_question, recommend_similar_campaigns, retrieve_chunks
from .schemas import (
    AnswerRequest,
    AnswerResult,
    RecommendationRequest,
    RecommendationResponse,
    RetrieveRequest,
    RetrieveResponse,
)


app = FastAPI(title="Campaign RAG Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Campaign RAG Backend is running 🚀",
        "endpoints": ["/retrieve", "/answer", "/recommendations"],
        "docs": "/docs"
    }

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(request: RetrieveRequest) -> RetrieveResponse:
    chunks = retrieve_chunks(query=request.query, filters=request.filters, top_k=request.top_k)
    return RetrieveResponse(query=request.query, filters=request.filters, chunks=chunks)


@app.post("/answer", response_model=AnswerResult)
def answer_endpoint(request: AnswerRequest) -> AnswerResult:
    try:
        return answer_question(question=request.question, filters=request.filters, top_k=request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendation_endpoint(request: RecommendationRequest) -> RecommendationResponse:
    recommendations = recommend_similar_campaigns(
        campaign_id=request.campaign_id,
        filters=request.filters,
        top_k=request.top_k,
        seed_chunk_limit=request.seed_chunk_limit,
    )
    return RecommendationResponse(
        campaign_id=request.campaign_id,
        filters=request.filters,
        recommendations=recommendations,
    )
