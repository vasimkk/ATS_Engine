import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

load_dotenv()

async def check_nulls():
    db_url = os.getenv('DATABASE_URL')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql+asyncpg://', 1)
    elif db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        
    engine = create_async_engine(db_url)
    try:
        async with engine.begin() as conn:
            # Let's fix the database by setting empty arrays to NULL where metadata_json is empty
            await conn.execute(text("UPDATE jobs SET required_skills = NULL, preferred_skills = NULL WHERE metadata_json = '{}'::jsonb;"))
            print("Fixed empty arrays to NULL.")
    except Exception as e:
        print(f"Failed: {e}")
    finally:
        await engine.dispose()

asyncio.run(check_nulls())
