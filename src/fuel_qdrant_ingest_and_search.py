import psycopg2
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

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

embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
qdrant = QdrantClient(url=QDRANT_URL)


def build_semantic_text(table: str, row: dict) -> str:
    """Select only semantic fields to embed, depending on table."""
    if table == "users":
        return f"User with email {row.get('email','')} and role_id {row.get('role_id','')}"
    elif table == "roles":
        perms = ", ".join(row.get("permissions", []))
        return f"Role: {row.get('name','')} with permissions: {perms}"
    elif table == "orders":
        return f"Order: {row.get('title','')} | {row.get('description','')} | Status: {row.get('status','')} amount {row.get('amount','')}"
    elif table == "vendors":
        return f"Vendor: {row.get('name','')} | {row.get('description','')} | Contact: {row.get('email','')}"
    else:
        # fallback: embed everything
        return " ".join(f"{k}: {v}" for k, v in row.items() if v is not None)


def main():
    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()

    tables = ["users", "roles", "orders", "vendors"]
    points = []
    point_id = 1

    for table in tables:
        cur.execute(f"SELECT row_to_json(t) FROM {table} t;")
        rows = cur.fetchall()

        for row in rows:
            data = row[0]  # dict
            text = build_semantic_text(table, data)
            vec = embedder.encode(text).tolist()

            points.append({
                "id": point_id,
                "vector": vec,
                "payload": {
                    "table": table,
                    "semantic_text": text,   # store for debugging
                    "data": data             # keep raw row for display
                }
            })
            point_id += 1

    # recreate collection with proper vector size
    dim = len(points[0]["vector"])
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={"size": dim, "distance": "Cosine"},
    )

    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"[done] Inserted {len(points)} rows from {tables}")


if __name__ == "__main__":
    main()
