import psycopg2
import requests
import numpy as np

# --- Postgres connection ---
pg_conn = psycopg2.connect(
    dbname="fuel-me-db",
    user="postgres",         # replace with your username
    password="smederevo026", # replace with your password
    host="localhost",
    port="54321"
)
pg_cur = pg_conn.cursor()

# Fetch vendors
pg_cur.execute("SELECT id, name, email, description FROM vendors;")
vendors = pg_cur.fetchall()

print("Vendors from Postgres:")
for v in vendors:
    print(v)

# --- Create collection in Qdrant ---
QDRANT_URL = "http://localhost:6333"
COLLECTION = "fuel-vendors"
VECTOR_DIM = 768   # change to your embedding model dim (e.g. 768, 1024, 1536)

requests.put(f"{QDRANT_URL}/collections/{COLLECTION}", json={
    "vectors": { "size": VECTOR_DIM, "distance": "Cosine" }
})

# --- Insert vendors into Qdrant ---
points = []
for v in vendors:
    vid, name, email, desc = v
    text = f"{name} - {desc}"

    # TEMP: random embeddings, replace with your model
    vector = np.random.rand(VECTOR_DIM).tolist()

    points.append({
        "id": vid,
        "vector": vector,
        "payload": {
            "name": name,
            "email": email,
            "description": desc
        }
    })

r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true", json={"points": points})
print("Inserted into Qdrant:", r.json())

# --- Search in Qdrant ---
query_vec = np.random.rand(VECTOR_DIM).tolist()  # TEMP: random query embedding

search = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/search", json={
    "vector": query_vec,
    "limit": 3
}).json()

print("Search results from Qdrant:", search)

# --- Lookup metadata in Postgres ---
for hit in search.get("result", []):
    vid = hit["id"]
    pg_cur.execute("SELECT id, name, email, description FROM vendors WHERE id = %s;", (vid,))
    print("From Postgres:", pg_cur.fetchone())

pg_cur.close()
pg_conn.close()
