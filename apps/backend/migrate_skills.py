import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

load_dotenv()

async def migrate_skills():
    db_url = os.getenv('DATABASE_URL')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql+asyncpg://', 1)
    elif db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        
    engine = create_async_engine(db_url)
    try:
        async with engine.begin() as conn:
            # Migrate data from metadata_json by converting JSON arrays to Postgres TEXT[] arrays
            await conn.execute(text("""
                UPDATE jobs 
                SET required_skills = ARRAY(
                    SELECT json_array_elements_text((metadata_json->>'required_skills')::json)
                    WHERE metadata_json ? 'required_skills'
                ),
                preferred_skills = ARRAY(
                    SELECT json_array_elements_text((metadata_json->>'preferred_skills')::json)
                    WHERE metadata_json ? 'preferred_skills'
                )
                WHERE metadata_json IS NOT NULL;
            """))
            print("Successfully migrated data to ARRAY columns.")
    except Exception as e:
        print(f"Failed: {e}")
    finally:
        await engine.dispose()

asyncio.run(migrate_skills())
