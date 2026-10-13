import psycopg2

db_url = "postgresql://postgres:123456@localhost:5432/auto_apply_jobs"

try:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    cur.execute("""
        ALTER TABLE job_matches 
        ADD COLUMN IF NOT EXISTS skill_score FLOAT,
        ADD COLUMN IF NOT EXISTS semantic_score FLOAT,
        ADD COLUMN IF NOT EXISTS experience_score FLOAT;
    """)
    conn.commit()
    print("Successfully added sub-score columns to job_matches!")
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
