import psycopg2
import requests
import json
import os
import hashlib
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
from sentence_transformers import SentenceTransformer
from local_vector_store import save_points_jsonl

# ---- CONFIG ----
PG_CONN = dict(
    dbname="fuel-me-db",
    user="postgres",
    password="smederevo026",
    host="localhost",
    port="54321"
)

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
MODEL = "bge-base-en-v1.5"

# Single test collection for ENRICHED pass
COLLECTION = "fuel-me"
CACHE_DIR = ".cache"
HASH_FILE = os.path.join(CACHE_DIR, "enriched_hashes.json")
# Local vector store JSONL (single collection for testing)
LOCAL_STORE_FILE = os.path.join(CACHE_DIR, "fuel-me.jsonl")

# ---- HELPERS ----
# Use local SentenceTransformer for embeddings during testing (bypasses Ollama)
_EMBEDDER = SentenceTransformer("BAAI/bge-base-en-v1.5")
def embed(text: str):
    return _EMBEDDER.encode(text).tolist()

def qdrant_create_collection(size: int):
    # Disabled for local testing (using JSONL store)
    print("[local-store] Skipping Qdrant collection creation for testing.")

def qdrant_upsert(points):
    # Disabled for local testing (using JSONL store)
    print("[local-store] Skipping Qdrant upsert for testing.")

def load_hashes() -> Dict[str, str]:
    if not os.path.exists(HASH_FILE):
        return {}
    try:
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_hashes(h: Dict[str, str]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, indent=2)

def to_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        try:
            return int(float(val))
        except Exception:
            return None

def to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None

def to_json_safe(obj: Any) -> Any:
    """Recursively convert Decimals, datetimes, and other non-JSON types to JSON-safe."""
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return None
    if isinstance(obj, (str, int, float)) or obj is None:
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    # Fallback: string representation
    return str(obj)

def make_profile_text(row: Dict[str, Any]) -> str:
    name = row.get("vendor_name") or ""
    status = row.get("vendor_status") or ""
    total = row.get("total_orders") or 0
    completed = row.get("completed_orders") or 0
    pending = row.get("pending_orders") or 0
    cancelled = row.get("cancelled_orders") or 0
    avg_amount = row.get("avg_amount") or 0
    last_order = row.get("last_order")
    last_s = str(last_order) if last_order is not None else "N/A"
    summary = row.get("profile_summary") or ""
    text = (
        f"{name}. Status: {status}. "
        f"Orders total {total}, completed {completed}, pending {pending}, cancelled {cancelled}. "
        f"Average amount {avg_amount}. Last order {last_s}. "
        f"{summary}"
    )
    return text.strip()

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---- MAIN INGEST ----
def main():
    # Connect Postgres
    conn = psycopg2.connect(**PG_CONN)  # type: ignore[arg-type]
    cur = conn.cursor()

    # Fetch enriched vendor profiles
    cur.execute("SELECT "
                "vendor_id, vendor_name, vendor_email, vendor_status, "
                "total_orders, completed_orders, pending_orders, cancelled_orders, "
                "avg_amount, last_order, profile_summary "
                "FROM public.fn_vendor_profiles();")
    rows = cur.fetchall()

    # Normalize rows into dicts with column names
    cols = [desc[0] for desc in cur.description]
    dict_rows = [dict(zip(cols, r)) for r in rows]

    cache = load_hashes()
    to_upsert = []
    for r in dict_rows:
        vid = r.get("vendor_id")
        if vid is None:
            continue
        text = make_profile_text(r)
        h = sha256(text)
        prev = cache.get(str(vid))
        if prev == h:
            # unchanged, skip re-embedding
            continue
        vec = embed(text)
        point = {
            "id": int(vid),
            "vector": vec,
            "payload": {
                "source": "enriched",
                "vendor_id": int(vid),
                "vendor_name": r.get("vendor_name"),
                "vendor_email": r.get("vendor_email"),
                "vendor_status": r.get("vendor_status"),
                "total_orders": to_int(r.get("total_orders")),
                "completed_orders": to_int(r.get("completed_orders")),
                "pending_orders": to_int(r.get("pending_orders")),
                "cancelled_orders": to_int(r.get("cancelled_orders")),
                "avg_amount": to_float(r.get("avg_amount")),
                "last_order": str(r.get("last_order")) if r.get("last_order") else None,
                "profile_summary": r.get("profile_summary"),
                "profile_text": text,
                "hash": h,
            }
        }
        to_upsert.append(point)
        cache[str(vid)] = h

    if to_upsert:
        # Write to local JSONL vector store for testing (bypass Qdrant)
        save_points_jsonl(LOCAL_STORE_FILE, to_upsert)
        save_hashes(cache)
        print(f"[done] Wrote {len(to_upsert)} enriched vendor profiles to local store {LOCAL_STORE_FILE}")
    else:
        print("[done] No changes detected; nothing to write")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()