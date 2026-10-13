import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.database import db
from sqlalchemy import text

with db._sync_write_session() as session:
    res = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='esco_skills';")).fetchall()
    print("Postgres columns:", res)
    try:
        res2 = session.execute(text("PRAGMA table_info(esco_skills)")).fetchall()
        print("SQLite columns:", res2)
    except Exception as e:
        print(e)
