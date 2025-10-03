import streamlit as st
import psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
import torch

# ---- CONFIG ----
PG_CONN = dict(
    dbname="fuel-me-db",
    user="postgres",
    password="smederevo026",
    host="localhost",
    port=54321,
)

QDRANT_URL = "http://localhost:6333"
COLLECTION = "fuel-me-all"

# ---- MODELS ----
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)

# ---- DB + Qdrant ----
conn = psycopg2.connect(**PG_CONN)
qdrant = QdrantClient(url=QDRANT_URL)

# ---- UI ----
st.set_page_config(page_title="Fuel-Me Agent", page_icon="⛽")
st.title("🧑‍💻 Fuel-Me Agent (DB-wide Semantic Search)")

st.markdown("""
Ask natural language questions across **users, roles, orders, vendors**.  
Pipeline: **Embedding → Qdrant Retrieval → CrossEncoder Rerank → Postgres Details**.
""")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
min_score = st.sidebar.slider(
    "Minimum rerank score",
    min_value=0.0, max_value=1.0, value=0.40, step=0.05,
    help="Filter out weak semantic matches (CrossEncoder scores)."
)

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

        # Step 2: Retrieve candidates from Qdrant
        hits = qdrant.search(collection_name=COLLECTION, query_vector=qvec, limit=30)

        if not hits:
            st.warning("No results found in Qdrant.")
        else:
            # Step 3: Build candidate texts
            candidates = []
            for h in hits:
                payload = h.payload
                table = payload["table"]
                data = payload["data"]

                # readable string
                text = " | ".join(f"{k}: {v}" for k, v in data.items() if v is not None)
                candidates.append((f"[{table}] {text}", payload))

            # Step 4: Rerank with cross-encoder
            pairs = [(query, text) for text, _ in candidates]
            scores = reranker.predict(pairs)

            rescored = sorted(
                [(text, payload, float(s)) for (text, payload), s in zip(candidates, scores)],
                key=lambda x: x[2], reverse=True
            )

            # Step 5: Apply threshold
            filtered = [r for r in rescored if r[2] >= min_score]

            if not filtered:
                st.warning(f"No results above threshold {min_score:.2f}. Try lowering it.")
            else:
                for text, payload, score in filtered[:10]:
                    st.markdown("---")
                    st.subheader(f"📊 Table: {payload['table']} — 🔥 Score {score:.3f}")
                    st.caption(f"Semantic text: {text}")
                    st.json(payload["data"])

    except Exception as e:
        st.error(f"Error: {e}")
