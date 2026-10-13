import psycopg2

db_url = "postgresql://postgres:123456@localhost:5432/auto_apply_jobs"

try:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    cur.execute("""
        ALTER TABLE jobs 
        ADD COLUMN IF NOT EXISTS metadata_json JSONB,
        ADD COLUMN IF NOT EXISTS required_skills_embedding JSONB,
        ADD COLUMN IF NOT EXISTS preferred_skills_embedding JSONB;
    """)
    conn.commit()
    print("Successfully added backend columns to PostgreSQL!")
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
