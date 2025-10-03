import fetch from "node-fetch";
import pg from "pg";

const QDRANT_URL = "http://localhost:6333";
const COLLECTION = "fuel-vendors";
const OLLAMA_URL = "http://localhost:11434";
const MODEL = "gemma-fc-test:latest";

type EmbeddingResponse = {
  embedding: number[];
};

type QdrantSearchResponse = {
  result: Array<{
    id: number;
    version?: number;
    score: number;
  }>;
  status: string;
  time?: number;
};

// Postgres config
const pool = new pg.Pool({
  user: "postgres", // change if needed
  host: "localhost",
  database: "fuel-me-db",
  password: "smederevo026", // change this
  port: 54321,
});

// --- Ollama embeddings ---
async function embed(text: string): Promise<number[]> {
  const res = await fetch(`${OLLAMA_URL}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      prompt: text,
    }),
  });

  if (!res.ok) {
    throw new Error(`Ollama error ${res.status}`);
  }
  const data = (await res.json()) as EmbeddingResponse;
  return data.embedding;
}

// --- Qdrant search ---
async function search(
  vector: number[],
  limit = 3
): Promise<QdrantSearchResponse> {
  const res = await fetch(
    `${QDRANT_URL}/collections/${COLLECTION}/points/search`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vector, limit }),
    }
  );

  if (!res.ok) {
    throw new Error(`Qdrant error ${res.status}`);
  }
  const data = (await res.json()) as QdrantSearchResponse;
  return data;
}

// --- Postgres fetch ---
async function fetchVendorById(id: number) {
  const client = await pool.connect();
  try {
    const result = await client.query(
      "SELECT id, name, email, description FROM vendors WHERE id=$1",
      [id]
    );
    return result.rows[0];
  } finally {
    client.release();
  }
}

// --- Main ---
async function main() {
  const query = "eco-friendly fuel delivery service for fleets";
  console.log(`🔎 Query: "${query}" using model ${MODEL}`);

  // 1. Embed
  const vector = await embed(query);
  console.log(`✅ Got embedding (dim=${vector.length})`);

  // 2. Search Qdrant
  const searchResult = await search(vector, 5);

  if (!searchResult.result || searchResult.result.length === 0) {
    console.log("No results.");
    return;
  }

  // 3. Join with Postgres
  console.log("\n📊 Top results:");
  for (const hit of searchResult.result) {
    const vendor = await fetchVendorById(hit.id);
    if (vendor) {
      console.log(
        `\n⭐ Score: ${hit.score.toFixed(4)}\n` +
          `   ID: ${vendor.id}\n` +
          `   Name: ${vendor.name}\n` +
          `   Email: ${vendor.email}\n` +
          `   Description: ${vendor.description}`
      );
    } else {
      console.log(`⚠️ No vendor found in Postgres for ID=${hit.id}`);
    }
  }

  await pool.end();
}

main().catch((err) => {
  console.error("❌ Error:", err);
  process.exit(1);
});
