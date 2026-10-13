import asyncio
from app.db_engine import make_async_engine
from app.config import settings

async def patch_pg():
    print("Connecting to Postgres...")
    engine = make_async_engine(settings.data_dir / "resume_matcher.db")
    async with engine.begin() as conn:
        try:
            await conn.execute(text("ALTER TABLE jobs ADD COLUMN required_skills_embedding JSON"))
            print("Added required")
        except Exception as e:
            print("Required:", e)
            
        try:
            await conn.execute(text("ALTER TABLE jobs ADD COLUMN preferred_skills_embedding JSON"))
            print("Added preferred")
        except Exception as e:
            print("Preferred:", e)
    await engine.dispose()
    print("Done!")

import sys
if 'sqlalchemy' in sys.modules:
    from sqlalchemy import text
    asyncio.run(patch_pg())
