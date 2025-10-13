import streamlit as st
import psycopg2
from sentence_transformers import SentenceTransformer
import torch
import sys, os

# Ensure we can import modules from the project src/ directory when running via Streamlit
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from local_vector_store import load_points_jsonl, cosine_search

# ---- CONFIG ----
PG_CONN = dict(
    dbname="fuel-me-db",
    user="postgres",
    password="smederevo026",
    host="localhost",
    port="54321",
)

# Local JSONL vector store (single test collection)
# Use absolute path so Streamlit CWD does not matter
LOCAL_STORE_FILE = os.path.join(BASE_DIR, "src", ".cache", "fuel-me.jsonl")

# ---- MODELS ----
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

# ---- DB (optional, not used in retrieval path for local testing) ----
conn = psycopg2.connect(**PG_CONN)  # type: ignore[arg-type]

# ---- UI ----
st.set_page_config(page_title="Fuel-Me Agent", page_icon="⛽")
st.title("🧑‍💻 Fuel-Me Agent (DB-wide Semantic Search)")

st.markdown("""
Ask natural language questions across **users, roles, orders, vendors**.
Pipeline: **Embedding → Local Cosine Retrieval (JSONL) → (optional) CrossEncoder Rerank → Details**.
""")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
min_score = st.sidebar.slider(
    "Minimum score threshold",
    min_value=0.0, max_value=1.0, value=0.40, step=0.05,
    help="When reranker is ON, threshold applies to CrossEncoder scores. When OFF, threshold applies to local cosine scores."
)
debug_mode = st.sidebar.checkbox("Debug: show raw cosine top-10 (ignore threshold)", value=False)

# Example queries
with st.expander("💡 Example questions"):
    st.markdown("### ✅ Good (should work well)")
    st.write("- *Eco-friendly fuel delivery options* (vendors)")
    st.write("- *Which vendors support commercial fleets?* (vendors)")
    st.write("- *Show me pending fuel orders* (orders)")
    st.write("- *Who are the registered users with role user?* (users/roles)")

    st.markdown("### ❌ Bad (not in DB)")
    st.write("- *Who won the Champions League in 2024?* (irrelevant domain)")
    st.write("- *What is the weather tomorrow?* (no weather data)")

    st.markdown("### ⚠️ Ambiguous (may return mixed results)")
    st.write("- *fuel suppliers* (could match orders or vendors)")
    st.write("- *admin users* (depends on role definition in DB)")
    st.write("- *delivery service* (orders vs vendors overlap)")

# Query input
query = st.text_input("🔎 Ask me something:", "eco-friendly fuel delivery")

if st.button("Search"):
    try:
        # Step 1: Embed query
        qvec = embedder.encode(query).tolist()

        # Preflight diagnostics for local store path
        cwd = os.getcwd()
        abs_path = LOCAL_STORE_FILE  # already absolute via BASE_DIR
        exists = os.path.exists(abs_path)
        size_bytes = os.path.getsize(abs_path) if exists else 0
        line_count = 0
        if exists:
            try:
                with open(abs_path, "r", encoding="utf-8") as _f:
                    for _ in _f:
                        line_count += 1
            except Exception:
                pass
        st.caption(f"CWD: {cwd}")
        st.caption(f"Local store path: {abs_path} | exists={exists} | size={size_bytes} bytes | lines={line_count}")

        # Step 2: Load local store and retrieve via cosine
        points = load_points_jsonl(LOCAL_STORE_FILE)
        st.caption(f"Loaded {len(points)} points from local store: {LOCAL_STORE_FILE}")
        # Fetch more than UI needs so reranker (if on) has headroom
        hits = cosine_search(points, qvec, limit=60)
        # Debug: show top-10 raw cosine results irrespective of threshold
        if debug_mode:
            st.markdown("#### Debug: Raw cosine top-10 (pre-rerank)")
            for h in hits[:10]:
                payload = h.get("payload") or {}
                score = float(h.get("score", 0.0))
                name = payload.get("vendor_name") or (payload.get("data") or {}).get("name") or ""
                table = payload.get("table", "vendors")
                st.write(f"- score={score:.4f} | table={table} | name={name}")

        # Build unified candidate texts derived from payloads
        candidates = []
        for h in hits:
            payload = h.get("payload") or {}
            # Prefer enriched-style text if available
            text = payload.get("profile_text") or payload.get("profile_summary") or ""
            if not text:
                # Fallback to raw-style text
                data = payload.get("data", {}) or {}
                table = payload.get("table", "vendors")
                # compact text from key: value fields
                raw_text = " | ".join(f"{k}: {v}" for k, v in data.items() if v is not None)
                text = f"[{table}] {raw_text}" if raw_text else "[vendors]"
            candidates.append((text, payload, float(h.get("score", 0.0))))

        # Step 3: Scoring — cosine-only (no reranker)
        rescored = sorted(candidates, key=lambda x: x[2], reverse=True)

        # Step 4: Apply threshold (CrossEncoder score if ON, else cosine score)
        filtered = [r for r in rescored if r[2] >= min_score]

        # Step 5: Render a single column (local store fuel-me)
        st.markdown("### 🔎 Results (fuel-me, local)")
        to_render = filtered
        # If nothing passed threshold, fall back to top-10 by score so user sees something
        if not to_render:
            st.info("No results above threshold; showing top-10 by score for inspection.")
            to_render = rescored[:10]

        if not to_render:
            st.warning("No candidates returned from local cosine search.")
        else:
            for text, payload, score in to_render[:10]:
                st.markdown("---")
                # Prefer enriched presentation if fields exist; fallback to raw-style
                name = payload.get('vendor_name') or ''
                status = payload.get('vendor_status') or ''
                if name or status:
                    # Enriched-like display
                    st.subheader(f"👤 {name} — 🔥 Score {score:.3f}")
                    st.caption(f"Profile text: {text}")
                    st.json({
                        "vendor_id": payload.get("vendor_id"),
                        "vendor_email": payload.get("vendor_email"),
                        "vendor_status": status,
                        "total_orders": payload.get("total_orders"),
                        "completed_orders": payload.get("completed_orders"),
                        "pending_orders": payload.get("pending_orders"),
                        "cancelled_orders": payload.get("cancelled_orders"),
                        "avg_amount": payload.get("avg_amount"),
                        "last_order": payload.get("last_order"),
                        "profile_summary": payload.get("profile_summary"),
                    })
                else:
                    # Raw-like display
                    table = payload.get('table', 'vendors')
                    st.subheader(f"📊 Table: {table} — 🔥 Score {score:.3f}")
                    st.caption(f"Semantic text: {text}")
                    st.json(payload.get("data", {}))

    except Exception as e:
        st.error(f"Error: {e}")
