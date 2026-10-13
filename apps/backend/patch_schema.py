import asyncio
from app.database import db
from sqlalchemy import text

async def patch():
    queries = [
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS content_type VARCHAR DEFAULT 'md';",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS filename VARCHAR;",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS is_master BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS parent_id VARCHAR;",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS processed_data JSON;",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS processing_status VARCHAR DEFAULT 'pending';",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS processing_token VARCHAR;",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS cover_letter TEXT;",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS outreach_message TEXT;",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS interview_prep TEXT;",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS title VARCHAR;",
        "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS original_markdown TEXT;"
    ]
    
    async with db._write_session() as session:
        for q in queries:
            try:
                await session.execute(text(q))
            except Exception as e:
                print(f"Error executing {q}: {e}")
        await session.commit()
    print("All columns patched successfully!")

asyncio.run(patch())
