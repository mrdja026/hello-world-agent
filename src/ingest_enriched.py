import psycopg2
import requests
import json
import os
import hashlib
from typing import Dict, Any

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

COLLECTION = "fuel-me-enriched"
CACHE_DIR = ".cache"
HASH_FILE = os.path.join(CACHE_DIR, "enriched_hashes.json")

# ---- HELPERS ----
def embed(text: str):
    res = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": MODEL, "input": text})
    res.raise_for_status()
    return res.json()["embedding"]

def qdrant_create_collection(size: int):
    payload = {"vectors": {"size": size, "distance": "Cosine"}}
    r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}", json=payload)
    try:
        print("[qdrant] ensure collection:", r.json())
    except Exception:
        print("[qdrant] ensure collection status:", r.status_code)

def qdrant_upsert(points):
    r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true", json={"points": points})
    try:
        print("[qdrant] upsert:", r.json())
    except Exception:
        print("[qdrant] upsert status:", r.status_code)

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
                "total_orders": r.get("total_orders"),
                "completed_orders": r.get("completed_orders"),
                "pending_orders": r.get("pending_orders"),
                "cancelled_orders": r.get("cancelled_orders"),
                "avg_amount": r.get("avg_amount"),
                "last_order": str(r.get("last_order")) if r.get("last_order") else None,
                "profile_summary": r.get("profile_summary"),
                "profile_text": text,
                "hash": h,
            }
        }
        to_upsert.append(point)
        cache[str(vid)] = h

    if to_upsert:
        dim = len(to_upsert[0]["vector"])
        qdrant_create_collection(dim)
        qdrant_upsert(to_upsert)
        save_hashes(cache)
        print(f"[done] Upserted {len(to_upsert)} enriched vendor profiles")
    else:
        print("[done] No changes detected; nothing to upsert")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()