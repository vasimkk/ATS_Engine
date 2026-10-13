import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

load_dotenv()

async def migrate_db():
    db_url = os.getenv('DATABASE_URL')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql+asyncpg://', 1)
    elif db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        
    engine = create_async_engine(db_url)
    try:
        async with engine.begin() as conn:
            # For jobs
            await conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS job_embedding_v vector(768);"))
            await conn.execute(text("""
                UPDATE jobs 
                SET job_embedding_v = (
                    SELECT array_agg(value::text::real)
                    FROM jsonb_array_elements(job_embedding::jsonb)
                )::vector
                WHERE job_embedding IS NOT NULL;
            """))
            await conn.execute(text("ALTER TABLE jobs DROP COLUMN job_embedding;"))
            await conn.execute(text("ALTER TABLE jobs RENAME COLUMN job_embedding_v TO job_embedding;"))
            
            # For resumes
            await conn.execute(text("ALTER TABLE resumes ADD COLUMN IF NOT EXISTS res_embedding_v vector(768);"))
            await conn.execute(text("""
                UPDATE resumes 
                SET res_embedding_v = (
                    SELECT array_agg(value::text::real)
                    FROM jsonb_array_elements(res_embedding::jsonb)
                )::vector
                WHERE res_embedding IS NOT NULL;
            """))
            await conn.execute(text("ALTER TABLE resumes DROP COLUMN res_embedding;"))
            await conn.execute(text("ALTER TABLE resumes RENAME COLUMN res_embedding_v TO res_embedding;"))

            print("Successfully migrated columns to pgvector.")
    except Exception as e:
        print(f"Failed to migrate: {e}")
    finally:
        await engine.dispose()

asyncio.run(migrate_db())
