import sys
import os
import json
from typing import List, Dict, Any, Tuple
import requests

# ---- CONFIG ----
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
MODEL = "bge-base-en-v1.5"

COLLECTION_RAW = "fuel-me-raw"
COLLECTION_ENRICHED = "fuel-me-enriched"


# ---- EMBEDDINGS ----
def embed(text: str) -> List[float]:
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": MODEL, "input": text})
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


# ---- QDRANT SEARCH ----
def collection_exists(name: str) -> bool:
    try:
        r = requests.get(f"{QDRANT_URL}/collections/{name}")
        if r.status_code != 200:
            return False
        # Qdrant returns a JSON body; if it parses and 200, the collection exists
        return True
    except Exception:
        return False

def qdrant_search(collection: str, vector: List[float], limit: int) -> List[Dict[str, Any]]:
    # Preflight: if collection is missing, return [] with a hint instead of raising
    if not collection_exists(collection):
        print(f"[warn] Qdrant collection '{collection}' is missing. Run the ingesters to create it.")
        return []

    payload = {
        "vector": vector,
        "limit": limit,
        "with_payload": True
    }
    resp = requests.post(f"{QDRANT_URL}/collections/{collection}/points/search", json=payload)
    if resp.status_code == 404:
        print(f"[warn] Qdrant collection '{collection}' not found (404). Run the ingesters to create it.")
        return []
    resp.raise_for_status()
    data = resp.json()
    return data.get("result", [])


# ---- NORMALIZATION ----
def hit_vendor_info(hit: Dict[str, Any]) -> Tuple[int, str, str, str]:
    """Extract vendor_id, name, email, status from payload for printing."""
    p = hit.get("payload") or {}
    source = p.get("source", "")
    vendor_id = p.get("vendor_id")
    name = ""
    email = ""
    status = ""

    if source == "enriched":
        name = p.get("vendor_name") or ""
        email = p.get("vendor_email") or ""
        status = p.get("vendor_status") or ""
    else:
        # raw: payload has data with vendor fields
        data = p.get("data") or {}
        name = data.get("name") or ""
        email = data.get("email") or ""
        status = data.get("status") or ""

    return int(vendor_id) if vendor_id is not None else -1, name, email, status


# ---- FUSION ----
def fuse_by_vendor_id(raw_hits: List[Dict[str, Any]],
                      enriched_hits: List[Dict[str, Any]],
                      top_k: int) -> List[Dict[str, Any]]:
    """
    Fuse two hit lists by vendor_id. Keep the best (max) score per vendor across sources.
    Attach raw_score/enriched_score for inspection.
    """
    fused: Dict[int, Dict[str, Any]] = {}

    def consider(hit: Dict[str, Any]):
        p = hit.get("payload") or {}
        vendor_id = p.get("vendor_id")
        if vendor_id is None:
            return
        vid = int(vendor_id)
        score = hit.get("score", 0.0)
        src = p.get("source", "raw")

        if vid not in fused:
            fused[vid] = {
                "vendor_id": vid,
                "best_score": score,
                "best_source": src,
                "raw_score": score if src == "raw" else None,
                "enriched_score": score if src == "enriched" else None,
                "payload": p
            }
        else:
            # track best
            if score > fused[vid]["best_score"]:
                fused[vid]["best_score"] = score
                fused[vid]["best_source"] = src
                fused[vid]["payload"] = p
            # track per-source scores
            key = "raw_score" if src == "raw" else "enriched_score"
            fused[vid][key] = max(fused[vid].get(key) or 0.0, score)

    for h in raw_hits:
        consider(h)
    for h in enriched_hits:
        consider(h)

    fused_list = list(fused.values())
    fused_list.sort(key=lambda x: x["best_score"], reverse=True)
    return fused_list[:top_k]


# ---- PRINTING ----
def print_section(title: str):
    print("\n" + "=" * 12 + f" {title} " + "=" * 12)


def print_hits(collection: str, hits: List[Dict[str, Any]], max_rows: int):
    for i, h in enumerate(hits[:max_rows], start=1):
        vendor_id, name, email, status = hit_vendor_info(h)
        score = h.get("score")
        payload = h.get("payload") or {}
        source = payload.get("source", collection)
        print(f"{i:2d}. vid={vendor_id} score={score:.4f} src={source} | {name} | {email} | {status}")


def print_fused(hits: List[Dict[str, Any]]):
    for i, h in enumerate(hits, start=1):
        vid = h["vendor_id"]
        best = h["best_score"]
        src = h["best_source"]
        p = h["payload"]
        name = p.get("vendor_name") or (p.get("data") or {}).get("name") or ""
        email = p.get("vendor_email") or (p.get("data") or {}).get("email") or ""
        status = p.get("vendor_status") or (p.get("data") or {}).get("status") or ""
        raw_s = h.get("raw_score")
        enr_s = h.get("enriched_score")
        print(f"{i:2d}. vid={vid} best={best:.4f} src={src} raw={raw_s} enr={enr_s} | {name} | {email} | {status}")


# ---- CLI ----
def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.search_dual <query text> [k=5]")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    try:
        k = int(os.getenv("K", "5"))
    except ValueError:
        k = 5
    # search deeper than k for fusion
    k_raw = max(k, 10)
    k_enr = max(k, 10)

    q_vec = embed(query)

    raw_hits = qdrant_search(COLLECTION_RAW, q_vec, k_raw)
    enr_hits = qdrant_search(COLLECTION_ENRICHED, q_vec, k_enr)

    print_section(f"RAW top-{k_raw} ({COLLECTION_RAW})")
    print_hits(COLLECTION_RAW, raw_hits, k_raw)

    print_section(f"ENRICHED top-{k_enr} ({COLLECTION_ENRICHED})")
    print_hits(COLLECTION_ENRICHED, enr_hits, k_enr)

    fused = fuse_by_vendor_id(raw_hits, enr_hits, k)
    print_section(f"FUSED top-{k} (max cosine score per vendor)")
    print_fused(fused)


if __name__ == "__main__":
    main()