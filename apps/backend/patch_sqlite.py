import sqlite3
import os

db_path = r"d:\clone_repo_ats-1\apps\backend\data\resume_matcher.db"

try:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("ALTER TABLE jobs ADD COLUMN required_skills_embedding JSON")
    c.execute("ALTER TABLE jobs ADD COLUMN preferred_skills_embedding JSON")
    conn.commit()
    conn.close()
    print("Successfully added columns to SQLite!")
except Exception as e:
    print(f"Error: {e}")
