import asyncio
from app.db_engine import make_async_engine
from app.models import Base
from sqlalchemy import text
from app.config import settings

async def reset():
    print("Cleaning all tables EXCEPT the 'jobs' table...")
    engine = make_async_engine(settings.db_path)
    async with engine.begin() as conn:
        # Drop everything except jobs
        await conn.execute(text("DROP TABLE IF EXISTS resumes CASCADE;"))
        await conn.execute(text("DROP TABLE IF EXISTS job_matches CASCADE;"))
        await conn.execute(text("DROP TABLE IF EXISTS gap_analysis CASCADE;"))
        await conn.execute(text("DROP TABLE IF EXISTS improvements CASCADE;"))
        await conn.execute(text("DROP TABLE IF EXISTS tailoring_previews CASCADE;"))
        await conn.execute(text("DROP TABLE IF EXISTS applications CASCADE;"))
        
        # Recreate the dropped tables with the latest schema
        await conn.run_sync(Base.metadata.create_all)
        
    print("Successfully wiped all user data, cache, and resumes! Your JDs are perfectly safe.")

if __name__ == "__main__":
    asyncio.run(reset())
