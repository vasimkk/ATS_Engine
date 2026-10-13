import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv
import os

load_dotenv()

async def drop_columns():
    db_url = os.getenv('DATABASE_URL')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql+asyncpg://', 1)
    elif db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        try:
            await conn.execute(text('ALTER TABLE job_matches DROP COLUMN IF EXISTS skill_score;'))
            await conn.execute(text('ALTER TABLE job_matches DROP COLUMN IF EXISTS semantic_score;'))
            await conn.execute(text('ALTER TABLE job_matches DROP COLUMN IF EXISTS experience_score;'))
            print('Successfully dropped columns from the database.')
        except Exception as e:
            print(f'Error dropping columns: {e}')
    await engine.dispose()

asyncio.run(drop_columns())
