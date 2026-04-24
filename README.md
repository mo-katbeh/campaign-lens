# Campaign Lens

Campaign Lens is a Python RAG backend for exploring fundraising campaign data. It prepares campaign records for search, indexes campaign chunks in Pinecone with local embeddings, and exposes FastAPI endpoints for retrieval, grounded question answering, and similar-campaign recommendations.

## What It Does

- Cleans and enriches campaign data from CSV files
- Generates search-friendly text chunks for retrieval
- Creates embeddings with a local `sentence-transformers` model
- Stores and queries vectors in Pinecone
- Uses Gemini for grounded answers when retrieval context is needed
- Supports recommendation flows based on campaign similarity

## Project Structure

```text
app/
  config.py                  Runtime config and shared paths
  embeddings.py              Embedding helpers
  main.py                    FastAPI app
  pinecone_store.py          Pinecone query and metadata helpers
  rag_service.py             Retrieval, QA, and recommendations
  schemas.py                 Pydantic request/response models
data/
  campaigns.csv              Raw campaign dataset
  campaigns_enriched.csv     Enriched dataset
  campaign_search_chunks.jsonl Search chunks used for indexing
scripts/
  prepare_campaigns_dataset.py          Data cleaning and chunk generation
  upload_campaign_chunks_to_pinecone.py Pinecone indexing script
  backfill_quality_metadata.py          Metadata update utility
  example_rag_usage.py                  Local usage examples
  test.py                               Manual Pinecone query helper
tests/
  test_rag_backend.py       Unit tests
```

## Requirements

- Python 3.11+
- A Pinecone account and API key
- A Gemini API key for answer generation

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Environment Setup

Create a local `.env` file based on `.env.example`.

Example values:

```env
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=campaigns-index
PINECONE_NAMESPACE=campaigns
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
NORMALIZE_EMBEDDINGS=true

GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-2.5-flash

EMBED_BATCH_SIZE=64
UPSERT_BATCH_SIZE=100
```

## Data Pipeline

### 1. Prepare the dataset

This script reads `data/campaigns.csv`, normalizes campaign content, enriches metadata such as theme and beneficiary group, and writes:

- `data/campaigns_enriched.csv`
- `data/campaign_search_chunks.jsonl`

Run:

```bash
python scripts/prepare_campaigns_dataset.py
```

### 2. Upload chunks to Pinecone

This script loads the prepared JSONL chunks, generates embeddings locally, ensures the Pinecone index exists, and upserts vectors.

Run:

```bash
python scripts/upload_campaign_chunks_to_pinecone.py
```

### 3. Optional metadata backfill

If you want to add `record_quality_score` metadata to already indexed vectors without rebuilding embeddings:

```bash
python scripts/backfill_quality_metadata.py --dry-run
python scripts/backfill_quality_metadata.py
```

## Running the API

Start the FastAPI app with Uvicorn:

```bash
uvicorn app.main:app --reload
```

Once running:

- API root: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

## API Endpoints

### `POST /retrieve`

Retrieves the most relevant campaign chunks for a query.

Example request:

```json
{
  "query": "medical campaigns for patients in 2025",
  "filters": {
    "theme": "medical",
    "year": "2025"
  },
  "top_k": 3
}
```

### `POST /answer`

Retrieves chunks and returns a grounded answer with cited source IDs.

Example request:

```json
{
  "question": "What are the lowest funded campaigns in Syria?",
  "top_k": 5
}
```

### `POST /recommendations`

Returns campaigns similar to a seed campaign based on indexed chunk vectors.

Example request:

```json
{
  "campaign_id": 1,
  "top_k": 3,
  "seed_chunk_limit": 2
}
```

## Helper Scripts

- `python scripts/example_rag_usage.py` runs sample retrieval, answer, and recommendation flows.
- `python scripts/test.py --query "lowest funded campaigns"` sends a manual query to Pinecone using the local embedding model.

## Testing

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

## Notes

- The repo currently includes prepared data files in `data/` for local experimentation.
- Answer generation depends on Gemini, but some ranking-style questions are answered directly from the enriched analytics data.
- The first embedding run may download the local transformer model, so internet access may be required once.
