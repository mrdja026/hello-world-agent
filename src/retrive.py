import psycopg2

# Connection details — adjust if needed
conn = psycopg2.connect(
    dbname="fuel-me-db",
    user="postgres",
    password="smederevo026",  # replace with your password
    host="localhost",
    port="54321"
)

cur = conn.cursor()

# Example: get vendors
cur.execute("SELECT id, name, email, description FROM vendors LIMIT 5;") # can we adjust this to be a function that returns arrow of semantinc normal data?

rows = cur.fetchall()

for row in rows:
    print(row)

cur.close()
conn.close()
