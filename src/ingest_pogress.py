import psycopg2
import requests
from sentence_transformers import SentenceTransformer
# Using HTTP (requests) for Qdrant operations with schema fallbacks
import json
import os
import hashlib
from typing import Dict, Any, List
from decimal import Decimal
from datetime import datetime
from local_vector_store import save_points_jsonl

# ---- CONFIG ----
PG_CONN = dict(
    dbname="fuel-me-db",
    user="postgres",
    password="smederevo026",
    host="localhost",
    port="54321",
)

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
MODEL = "bge-base-en-v1.5"   # embedding model via Ollama

# Single test collection (cosine only) for RAW pass
COLLECTION = "fuel-me"

# Basic change-detection cache (skip re-embedding unchanged rows)
CACHE_DIR = ".cache"
RAW_HASH_FILE = os.path.join(CACHE_DIR, "raw_hashes.json")

# Local vector store JSONL (single collection for testing)
LOCAL_STORE_FILE = os.path.join(CACHE_DIR, "fuel-me.jsonl")


# ---- HELPERS ----
# Use local SentenceTransformer for embeddings during testing (bypasses Ollama)
_EMBEDDER = SentenceTransformer("BAAI/bge-base-en-v1.5")
def embed(text: str) -> List[float]:
    return _EMBEDDER.encode(text).tolist()


# Qdrant disabled for local testing
def qdrant_create_collection(size: int):
    print("[local-store] Skipping Qdrant collection creation for testing.")


def qdrant_upsert(points):
    print("[local-store] Skipping Qdrant upsert for testing.")


def load_hashes(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_hashes(path: str, h: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, indent=2)

def to_json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-safe types (Decimal, datetime, etc.) to JSON-safe."""
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return None
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (str, int, float)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    # Fallback to string
    return str(obj)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---- MAIN INGEST (RAW: vendors) ----
def main():
    # Connect to Postgres (pylance stub doesn't accept kwargs; ignore for typing only)
    conn = psycopg2.connect(**PG_CONN)  # type: ignore[arg-type]
    cur = conn.cursor()

    # For now, ingest only vendors as "raw" (JSON-based) representation
    cur.execute("SELECT row_to_json(v) FROM public.vendors v;")
    rows = cur.fetchall()

    all_points = []
    cache = load_hashes(RAW_HASH_FILE)

    for row in rows:
        data: Dict[str, Any] = row[0]
        # Expect vendor id present
        vid = data.get("id")
        if vid is None:
            continue

        # Semantic text for raw representation: compact JSON string
        # You can tailor this to fewer fields if desired.
        semantic_text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

        # Change detection (skip if unchanged)
        h = sha256(semantic_text)
        cache_key = str(vid)
        if cache.get(cache_key) == h:
            continue

        vector = embed(semantic_text)

        all_points.append({
            "id": int(vid),  # align ids with vendor_id for entity-level fusion
            "vector": vector,
            "payload": {
                "source": "raw",
                "table": "vendors",
                "vendor_id": int(vid),
                "semantic_text": semantic_text,
                "data": data
            }
        })
        cache[cache_key] = h

    if all_points:
        # Write to local JSONL vector store for testing (bypass Qdrant)
        save_points_jsonl(LOCAL_STORE_FILE, all_points)
        save_hashes(RAW_HASH_FILE, cache)
        print(f"[done] Wrote {len(all_points)} raw vendor rows to local store {LOCAL_STORE_FILE}")
    else:
        print("[done] No changes detected; nothing to write")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
