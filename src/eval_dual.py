import sys
import os
from typing import List, Dict, Any
from datetime import datetime, timezone

# Reuse the dual-search helpers and config
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

try:
    from search_dual import (
        embed,
        qdrant_search,
        fuse_by_vendor_id,
        COLLECTION_RAW,
        COLLECTION_ENRICHED,
    )
except Exception as e:
    print(f"Failed to import dual search helpers: {e}. Try running: python -m src.eval_dual (from project root) or python src/eval_dual.py")
    sys.exit(1)


def print_section(title: str):
    print("\n" + "=" * 12 + f" {title} " + "=" * 12)


def print_hits(collection: str, hits: List[Dict[str, Any]], max_rows: int):
    for i, h in enumerate(hits[:max_rows], start=1):
        p = h.get("payload") or {}
        src = p.get("source", collection)
        vendor_id = p.get("vendor_id")
        score = h.get("score", 0.0)

        # Try to extract human-friendly info
        name = p.get("vendor_name") or (p.get("data") or {}).get("name") or ""
        email = p.get("vendor_email") or (p.get("data") or {}).get("email") or ""
        status = p.get("vendor_status") or (p.get("data") or {}).get("status") or ""

        print(f"{i:2d}. vid={vendor_id} score={score:.4f} src={src} | {name} | {email} | {status}")


def evaluate_queries(queries: List[str], k_print: int = 5, k_search_each: int = 10):
    ts = datetime.now(datetime.UTC).isoformat()
    print_section(f"Dual evaluation run @ {ts} (k_print={k_print}, search_each={k_search_each})")

    for q in queries:
        q = q.strip()
        if not q:
            continue

        print_section(f"Query: {q}")
        q_vec = embed(q)

        raw_hits = qdrant_search(COLLECTION_RAW, q_vec, k_search_each)
        enr_hits = qdrant_search(COLLECTION_ENRICHED, q_vec, k_search_each)

        if not raw_hits:
            print("[hint] No raw hits. If collection is missing, run: python -m src.ingest_pogress")
        if not enr_hits:
            print("[hint] No enriched hits. If collection is missing, run: python -m src.ingest_enriched")

        print_section(f"RAW top-{k_print} ({COLLECTION_RAW})")
        print_hits(COLLECTION_RAW, raw_hits, k_print)

        print_section(f"ENRICHED top-{k_print} ({COLLECTION_ENRICHED})")
        print_hits(COLLECTION_ENRICHED, enr_hits, k_print)

        fused = fuse_by_vendor_id(raw_hits, enr_hits, k_print)
        print_section(f"FUSED top-{k_print} (max cosine score per vendor)")
        for i, h in enumerate(fused, start=1):
            vid = h.get("vendor_id")
            best = h.get("best_score", 0.0)
            src = h.get("best_source", "?")
            p = h.get("payload") or {}
            name = p.get("vendor_name") or (p.get("data") or {}).get("name") or ""
            email = p.get("vendor_email") or (p.get("data") or {}).get("email") or ""
            status = p.get("vendor_status") or (p.get("data") or {}).get("status") or ""
            raw_s = h.get("raw_score")
            enr_s = h.get("enriched_score")
            print(f"{i:2d}. vid={vid} best={best:.4f} src={src} raw={raw_s} enr={enr_s} | {name} | {email} | {status}")


def main():
    # Usage:
    #   python -m src.eval_dual "reliable vendor;active vendor with high completion"
    # or set environment:
    #   K=5 python -m src.eval_dual "query one;query two;query three"
    #
    # k_print is how many rows to display for each section.
    # k_search_each controls how many results we retrieve from each collection before fusing.
    try:
        k_print = int(os.getenv("K", "5"))
    except ValueError:
        k_print = 5
    try:
        k_search_each = int(os.getenv("K_SEARCH_EACH", str(max(10, k_print))))
    except ValueError:
        k_search_each = max(10, k_print)

    if len(sys.argv) < 2:
        # Default sample queries if none provided
        queries = [
            "reliable vendor with many completed orders",
            "active vendor with recent activity",
            "vendors with high average order amount",
            "vendors with many pending orders",
            "vendors with cancelled orders issues",
        ]
    else:
        # Join args and split by ';' to allow multiple queries in one string
        joined = " ".join(sys.argv[1:])
        queries = [q.strip() for q in joined.split(";") if q.strip()]

    evaluate_queries(queries, k_print=k_print, k_search_each=k_search_each)


if __name__ == "__main__":
    main()