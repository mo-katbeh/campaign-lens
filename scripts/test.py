from __future__ import annotations

import argparse
from typing import Any

from pinecone import Pinecone

from upload_campaign_chunks_to_pinecone import (
    INDEX_NAME,
    LOCAL_EMBEDDING_MODEL,
    NAMESPACE,
    embed_text_batch,
    get_index_dimension,
    load_local_model,
    require_env,
)


def build_filter(args: argparse.Namespace) -> dict[str, Any] | None:
    filter_dict: dict[str, Any] = {}

    if args.theme:
        filter_dict["campaign_theme"] = args.theme
    if args.beneficiary:
        filter_dict["beneficiary_group"] = args.beneficiary
    if args.status:
        filter_dict["funding_status"] = args.status
    if args.year:
        filter_dict["year_label"] = args.year

    return filter_dict or None


def query_index(
    query: str,
    top_k: int,
    filter_dict: dict[str, Any] | None,
) -> Any:
    pinecone_api_key = require_env("PINECONE_API_KEY")
    model = load_local_model()

    if hasattr(model, "get_embedding_dimension"):
        query_dimension = model.get_embedding_dimension()
    else:
        query_dimension = model.get_sentence_embedding_dimension()

    pc = Pinecone(api_key=pinecone_api_key)
    index_dimension = get_index_dimension(pc, INDEX_NAME)
    if index_dimension is not None and index_dimension != query_dimension:
        raise RuntimeError(
            f"Index '{INDEX_NAME}' has dimension {index_dimension}, "
            f"but the query model '{LOCAL_EMBEDDING_MODEL}' produces {query_dimension}."
        )

    index = pc.Index(INDEX_NAME)
    query_vector = embed_text_batch(model, [query])[0]

    query_kwargs: dict[str, Any] = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
    }
    if filter_dict:
        query_kwargs["filter"] = filter_dict
    if NAMESPACE:
        query_kwargs["namespace"] = NAMESPACE

    return index.query(**query_kwargs)


def print_matches(results: Any) -> None:
    matches = getattr(results, "matches", None)
    if matches is None and isinstance(results, dict):
        matches = results.get("matches", [])
    matches = matches or []

    if not matches:
        print("No matches found.")
        return

    for position, match in enumerate(matches, start=1):
        match_id = getattr(match, "id", None) or match.get("id")
        score = getattr(match, "score", None)
        if score is None and isinstance(match, dict):
            score = match.get("score")
        metadata = getattr(match, "metadata", None)
        if metadata is None and isinstance(match, dict):
            metadata = match.get("metadata", {})
        metadata = metadata or {}

        print(f"\nResult {position}")
        print(f"id: {match_id}")
        print(f"score: {score:.4f}" if isinstance(score, (int, float)) else f"score: {score}")
        print(f"title: {metadata.get('title', '-')}")
        print(f"theme: {metadata.get('campaign_theme', '-')}")
        print(f"beneficiary: {metadata.get('beneficiary_group', '-')}")
        print(f"status: {metadata.get('funding_status', '-')}")
        print(f"year: {metadata.get('year_label', '-')}")
        print(f"campaign_id: {metadata.get('campaign_id', '-')}")
        print(f"locations: {', '.join(metadata.get('location_mentions', [])) or '-'}")
        print(f"text: {metadata.get('text', '-')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Pinecone using the same local embedding model as the uploader.")
    parser.add_argument(
        "--query",
        default="gave me top 5 lowest funded campaigns",
        help="Natural-language query to search for.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to return.")
    parser.add_argument("--theme", help="Optional exact-match filter on campaign_theme.")
    parser.add_argument("--beneficiary", help="Optional exact-match filter on beneficiary_group.")
    parser.add_argument("--status", help="Optional exact-match filter on funding_status.")
    parser.add_argument("--year", help="Optional exact-match filter on year_label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_dict = build_filter(args)

    print(f"Query: {args.query}")
    if filter_dict:
        print(f"Filter: {filter_dict}")
    print(f"Index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE or '(default)'}")
    print(f"Model: {LOCAL_EMBEDDING_MODEL}")

    results = query_index(args.query, args.top_k, filter_dict)
    print_matches(results)


if __name__ == "__main__":
    main()