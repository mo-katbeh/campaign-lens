from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pinecone_store import get_campaign_chunk_ids, get_campaign_quality_map, get_index
from scripts.upload_campaign_chunks_to_pinecone import NAMESPACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill record_quality_score into existing Pinecone metadata without regenerating embeddings."
    )
    parser.add_argument("--campaign-id", type=int, help="Optional single campaign_id to backfill.")
    parser.add_argument("--limit", type=int, help="Optional limit on the number of chunk updates.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned updates without writing to Pinecone.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_chunk_ids = get_campaign_chunk_ids()
    quality_map = get_campaign_quality_map()

    updated = 0
    index = get_index() if not args.dry_run else None

    for current_campaign_id, chunk_ids in campaign_chunk_ids.items():
        if args.campaign_id is not None and current_campaign_id != args.campaign_id:
            continue

        quality_score = quality_map.get(current_campaign_id)
        if quality_score is None:
            continue

        for chunk_id in chunk_ids:
            if args.limit is not None and updated >= args.limit:
                print(f"Reached limit ({args.limit}).")
                print(f"Total chunk metadata updates prepared: {updated}")
                return

            if args.dry_run:
                print(f"[dry-run] chunk_id={chunk_id} campaign_id={current_campaign_id} quality={quality_score}")
            else:
                if index is None:
                    raise RuntimeError("Pinecone index is unavailable while dry-run is disabled.")
                update_kwargs = {
                    "id": str(chunk_id),
                    "set_metadata": {"record_quality_score": float(quality_score)},
                }
                if NAMESPACE:
                    update_kwargs["namespace"] = NAMESPACE
                index.update(**update_kwargs)
                print(f"Updated chunk_id={chunk_id} with quality={quality_score}")
            updated += 1

    print(f"Total chunk metadata updates prepared: {updated}")


if __name__ == "__main__":
    main()
