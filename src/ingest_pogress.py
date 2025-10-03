import psycopg2
import requests
import json

# ---- CONFIG ----
PG_CONN = dict(
    dbname="fuel-me-db",
    user="postgres",
    password="smederevo026",
    host="localhost",
    port=54321
)

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
MODEL = "bge-base-en-v1.5"   # <-- switched to embedding model

COLLECTION = "fuel-me-all"

# ---- HELPERS ----
def embed(text: str):
    res = requests.post(f"{OLLAMA_URL}/api/embeddings",
                        json={"model": MODEL, "input": text})
    res.raise_for_status()
    return res.json()["embedding"]

def qdrant_create_collection(size: int):
    payload = {
        "vectors": {
            "size": size,
            "distance": "Cosine"
        }
    }
    r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}",
                     json=payload)
    print("[qdrant] ensure collection:", r.json())

def qdrant_upsert(points):
    r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
                     json={"points": points})
    print("[qdrant] upsert:", r.json())

# ---- MAIN INGEST ----
def main():
    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()

    tables = ["users", "roles", "orders", "vendors"]

    all_points = []
    point_id = 1

    for table in tables:
        cur.execute(f"SELECT row_to_json(t) FROM {table} t;")
        rows = cur.fetchall()

        for row in rows:
            data = row[0]
            text = json.dumps(data, ensure_ascii=False)
            vector = embed(text)

            all_points.append({
                "id": point_id,
                "vector": vector,
                "payload": {
                    "table": table,
                    "data": data
                }
            })
            point_id += 1

    dim = len(all_points[0]["vector"])
    qdrant_create_collection(dim)
    qdrant_upsert(all_points)

    print(f"[done] Inserted {len(all_points)} rows from {tables}")

if __name__ == "__main__":
    main()
