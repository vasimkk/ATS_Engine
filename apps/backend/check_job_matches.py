import psycopg2

db_url = "postgresql://postgres:123456@localhost:5432/auto_apply_jobs"

try:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'job_matches'")
    columns = cur.fetchall()
    print("Columns in job_matches:")
    for col in columns:
        print("-", col[0])
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
