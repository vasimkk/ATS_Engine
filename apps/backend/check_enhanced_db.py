import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

load_dotenv()

async def check_db():
    db_url = os.getenv('DATABASE_URL')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql+asyncpg://', 1)
    elif db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT id, job_id, resume_id, ats_score, enhanced_ats_score FROM job_matches WHERE enhanced_ats_score IS NOT NULL LIMIT 5;"))
        rows = result.fetchall()
        print(f"Found {len(rows)} existing enhanced scores in DB:")
        for r in rows:
            print(r)
    await engine.dispose()

asyncio.run(check_db())
