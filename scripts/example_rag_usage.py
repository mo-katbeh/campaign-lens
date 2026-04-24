from __future__ import annotations

from pprint import pprint
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import answer_question, recommend_similar_campaigns, retrieve_chunks


def main() -> None:
    chunks = retrieve_chunks(
        query="medical campaigns for patients in 2025 that are still underfunded",
        filters={"theme": "medical", "funding_ratio": {"lt": 1}, "year": "2025"},
        top_k=3,
    )
    print("Retrieved chunks:")
    pprint([chunk.model_dump() for chunk in chunks])

    answer = answer_question(
        question="What do the retrieved medical campaigns say about the needs they are addressing?",
        filters={"theme": "medical", "funding_ratio": {"lt": 1}},
        top_k=3,
    )
    print("\nAnswer:")
    pprint(answer.model_dump())

    recommendations = recommend_similar_campaigns(campaign_id=1, top_k=3, seed_chunk_limit=2)
    print("\nRecommendations:")
    pprint([recommendation.model_dump() for recommendation in recommendations])


if __name__ == "__main__":
    main()
