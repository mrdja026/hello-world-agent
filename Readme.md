# Airweave local Postgres RAG fast lookup

### Uses Airweave that is gitignored
 - Clone the Airweave from (github)(https://github.com/airweave-ai/airweave)
 - Follow instructions to install

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

### TODO
- [ ] The structured vs unstructured data is a pain
- [ ] Structured data (json, rows) should have some semantic meaning in them, so combining the data with like Fuel Vendor with {Name} that is a company type {CompanyType} ect can be better with reranking or even plain cosine similarity
- [ ] Test thhat
