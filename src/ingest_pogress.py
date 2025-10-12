import psycopg2
import requests
import json
import os
import hashlib
from typing import Dict, Any, List

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

# Raw collection (cosine only)
COLLECTION = "fuel-me-raw"

# Basic change-detection cache (skip re-embedding unchanged rows)
CACHE_DIR = ".cache"
RAW_HASH_FILE = os.path.join(CACHE_DIR, "raw_hashes.json")


# ---- HELPERS ----
def embed(text: str) -> List[float]:
    res = requests.post(f"{OLLAMA_URL}/api/embeddings",
                        json={"model": MODEL, "input": text})
    res.raise_for_status()
    data = res.json()
    # Ollama embeddings API returns {"embedding": [...]}
    return data["embedding"]


def qdrant_create_collection(size: int):
    payload = {
        "vectors": {
            "size": size,
            "distance": "Cosine"
        }
    }
    r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}",
                     json=payload)
    try:
        print("[qdrant] ensure collection:", r.json())
    except Exception:
        print("[qdrant] ensure collection status:", r.status_code)


def qdrant_upsert(points):
    r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
                     json={"points": points})
    try:
        print("[qdrant] upsert:", r.json())
    except Exception:
        print("[qdrant] upsert status:", r.status_code)


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
        dim = len(all_points[0]["vector"])
        qdrant_create_collection(dim)
        qdrant_upsert(all_points)
        save_hashes(RAW_HASH_FILE, cache)
        print(f"[done] Upserted {len(all_points)} raw vendor rows into {COLLECTION}")
    else:
        print("[done] No changes detected; nothing to upsert")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
