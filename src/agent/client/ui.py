import streamlit as st
import psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder
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
LOCAL_STORE_FILE = "src/.cache/fuel-me.jsonl"

# ---- MODELS ----
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)

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
    help="When reranker is ON, threshold applies to CrossEncoder scores. When OFF, threshold applies to Qdrant cosine scores."
)
use_reranker = st.sidebar.checkbox("Use reranker (CrossEncoder)", value=True)

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

        # Step 2: Load local store and retrieve via cosine
        points = load_points_jsonl(LOCAL_STORE_FILE)
        # Fetch more than UI needs so reranker (if on) has headroom
        hits = cosine_search(points, qvec, limit=60)

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

        # Step 3: Scoring — either rerank with CrossEncoder or use cosine scores directly
        if use_reranker:
            pairs = [(query, t) for t, _, _ in candidates]
            scores = reranker.predict(pairs) if pairs else []
            rescored = sorted(
                [(t, p, float(s)) for (t, p, _), s in zip(candidates, scores)],
                key=lambda x: x[2], reverse=True
            )
        else:
            # Cosine-only: trust local cosine scores
            rescored = sorted(candidates, key=lambda x: x[2], reverse=True)

        # Step 4: Apply threshold (CrossEncoder score if ON, else cosine score)
        filtered = [r for r in rescored if r[2] >= min_score]

        # Step 5: Render a single column (local store fuel-me)
        st.markdown("### 🔎 Results (fuel-me, local)")
        if not filtered:
            st.info("No results above threshold.")
        else:
            for text, payload, score in filtered[:10]:
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
