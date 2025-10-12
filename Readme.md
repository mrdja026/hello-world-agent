## What this repo does

Small local search experiment:

- Embed a query via a local Ollama model (`/api/embeddings`).
- Vector search in Qdrant collection `fuel-vendors`.
- Optionally join top hits with Postgres (metadata) and print results.

Run the demo:

```powershell
npx tsx src\agent\backend\test-agent.ts
```

## Install and run a local Qdrant

1. Start Qdrant (Docker):

```powershell
docker run --rm -p 6333:6333 qdrant/qdrant:latest
```

2. Create a collection (set vectors size to your embedding dim):

```bash
curl -X PUT "http://localhost:6333/collections/fuel-vendors" \
  -H "Content-Type: application/json" \
  -d '{
        "vectors": { "size": 3584, "distance": "Cosine" }
      }'
```

Tip: 3584 is an example. Use the exact dimension your embedding model returns.

3. Insert points (ids + vectors, optional payload) using the Qdrant upsert API.

## Airweave: use and purpose

[Airweave](https://github.com/airweave-ai/airweave) lets agents search any app by syncing data from SaaS, DBs, or files into a searchable knowledge base exposed via REST or MCP. It handles extraction, embedding, and serving so agents can query across sources with minimal glue code. If you outgrow hand-rolled ingestion, Airweave can manage connectors, auth, and scalable vector stores for you, while still letting you use Qdrant locally.

## Why reranking

Vector similarity is fast but can surface semantically close yet not-the-best results. A lightweight reranker (cross-encoder or LLM scoring) on the top-k:

- Improves precision for ambiguous queries.
- Compensates for ANN recall trade-offs.
- Aligns results with task-specific relevance beyond pure cosine distance.

Keep k small (e.g., 50→5) to stay cheap and snappy.

- Todo
- [ ] - make the getter for fuel_qdrant uses sql queries that are in the db that return more semantic lvl for everything, like for vendors every vendor should have a This is a X vendor with owners named Y, Z and other fk keys in text so its more semantic than JSON, then check data

## Dual ingestion and cosine-only search

This repo now supports dual ingestion and search using cosine-only similarity:

- Raw vendors go to collection "fuel-me-raw" via [src/ingest_pogress.py](src/ingest_pogress.py:1)
- Enriched vendor profiles (from [plpgsql.fn_vendor_profiles()](src/data-prep/fuel.me.sql:1)) go to collection "fuel-me-enriched" via [src/ingest_enriched.py](src/ingest_enriched.py:1)
- Dual search performs a cosine search over both collections and de-duplicates by vendor_id via [src/search_dual.py](src/search_dual.py:1)
- Simple evaluation harness to compare raw vs enriched vs fused side-by-side is in [src/eval_dual.py](src/eval_dual.py:1)

Prerequisites

- Postgres running with your "fuel-me-db" and the function in [src/data-prep/fuel.me.sql](src/data-prep/fuel.me.sql:1) applied (CREATE OR REPLACE FUNCTION public.fn_vendor_profiles()).
- Qdrant running locally (Docker recommended, see section below).
- Ollama running locally with an embedding model:
  - Pull once:
    - Windows PowerShell (example): `ollama pull bge-base-en-v1.5`
  - Default endpoints used:
    - Qdrant: http://localhost:6333
    - Ollama: http://localhost:11434

Run raw ingest (vendors → fuel-me-raw)

- Purpose: store a compact JSON-based representation of each vendor with cosine embeddings.
- Command:
  - `python -m src.ingest_pogress`
- What it does:
  - Reads vendors from Postgres (public.vendors).
  - Embeds a compact JSON string of each row using Ollama embeddings.
  - Upserts to Qdrant collection fuel-me-raw with payload:
    - source: "raw"
    - table: "vendors"
    - vendor_id: number
    - semantic_text: compact JSON string
    - data: full vendor row (for debugging)
  - Skips re-embedding unchanged rows using a SHA256 content hash cache.

Run enriched ingest (fn_vendor_profiles() → fuel-me-enriched)

- Purpose: store a concise, task-oriented profile text per vendor for better semantic matching under cosine.
- Command:
  - `python -m src.ingest_enriched`
- What it does:
  - Calls [plpgsql.fn_vendor_profiles()](src/data-prep/fuel.me.sql:1) returning fields like vendor_status, total/completed/pending/cancelled, avg_amount, last_order, profile_summary.
  - Builds a compact "profile_text" (e.g., "Acme. Status: active. Orders total 42...").
  - Embeds with Ollama and upserts to Qdrant collection fuel-me-enriched with payload:
    - source: "enriched"
    - vendor_id, vendor_name, vendor_email, vendor_status
    - total_orders, completed_orders, pending_orders, cancelled_orders
    - avg_amount, last_order, profile_summary, profile_text, hash
  - Skips re-embedding unchanged rows using a SHA256 content hash cache.

Dual search (cosine-only; de-dup by vendor_id)

- Command:
  - `python -m src.search_dual "reliable vendor with many completed orders"`
  - Optional: set environment variable K to control fused top-k (default 5), e.g. `K=5 python -m src.search_dual "active vendor"`
- What it does:
  - Embeds the query once via Ollama.
  - Searches both fuel-me-raw and fuel-me-enriched with cosine.
  - Fuses results by vendor_id, keeping the max cosine score per vendor.
  - Prints RAW top-k, ENRICHED top-k, and FUSED top-k with scores and basic fields.

Evaluation harness (challenge the assumption)

- Command examples:
  - `python -m src.eval_dual`
    - Runs with default sample queries.
  - `python -m src.eval_dual "reliable vendor;active vendor with recent activity;vendors with high average order amount"`
    - Use semicolon to separate multiple queries.
  - Environment variables:
    - `K` (displayed top-k each section, default 5)
    - `K_SEARCH_EACH` (how many to fetch per collection before fusing, default max(10, K))
- Output:
  - Side-by-side RAW, ENRICHED, and FUSED sections for each query, with vendor_id, scores, and summary fields.
  - This is cosine-only, no reranker involved.

Notes and rationale

- Cosine everywhere:
  - Collections are configured with "Cosine" distance in both raw and enriched paths.
- Why two representations:
  - Raw payloads capture exact fields and precise attributes.
  - Enriched profiles encode intent-aligned features (status, volumes, recency) that often improve match quality for natural-language queries.
- De-duplication:
  - Both ingest paths align on the same integer point id (vendor_id) to simplify entity fusion.
  - Dual search keeps the best cosine score per vendor across sources.
- Freshness:
  - Both ingest scripts compute content hashes and skip unchanged rows to reduce embedding cost.

Troubleshooting

- Ensure Qdrant is running:
  - `docker run --rm -p 6333:6333 qdrant/qdrant:latest`
- Ensure Ollama is running and the model is available:
  - `ollama serve` (if needed)
  - `ollama pull bge-base-en-v1.5`
- Verify Postgres credentials in:
  - [src/ingest_pogress.py](src/ingest_pogress.py:1)
  - [src/ingest_enriched.py](src/ingest_enriched.py:1)
- Confirm the SQL function exists:
  - [src/data-prep/fuel.me.sql](src/data-prep/fuel.me.sql:1)
