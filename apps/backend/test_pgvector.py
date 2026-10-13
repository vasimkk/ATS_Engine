import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

load_dotenv()

async def test_endpoint():
    db_url = os.getenv('DATABASE_URL')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql+asyncpg://', 1)
    elif db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        
    engine = create_async_engine(db_url)
    try:
        async with engine.begin() as conn:
            from pgvector.sqlalchemy import Vector
            import json
            # Let's get the resume vector first
            res = await conn.execute(text("SELECT res_embedding FROM resumes LIMIT 1;"))
            resume_vector = res.scalar()
            
            # Now let's try the query that Get_N_moreJDs_for_Res runs
            # Job.job_embedding.cosine_distance(resume_vector)
            query = text("SELECT id, job_embedding <=> :vec AS distance FROM jobs WHERE job_embedding IS NOT NULL ORDER BY job_embedding <=> :vec LIMIT 10")
            
            # resume_vector is returned as a string from asyncpg if we don't configure types, 
            # but SQLAlchemy should handle it if using the ORM. Let's use the ORM directly to reproduce.
    except Exception as e:
        print(f"Failed: {e}")
    finally:
        await engine.dispose()

asyncio.run(test_endpoint())
