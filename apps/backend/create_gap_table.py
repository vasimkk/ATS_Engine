import asyncio
from app.db_engine import make_async_engine
from app.models import Base
from sqlalchemy import text
from app.config import settings

async def create_new_tables():
    print("Creating any new tables...")
    engine = make_async_engine(settings.db_path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("GapAnalysis table successfully created (if it didn't exist)!")

asyncio.run(create_new_tables())
