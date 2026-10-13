import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.database import db
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def add_columns():
    db._ensure_initialized()
    async with db._async_engine.begin() as conn:
        try:
            logger.info("Adding enhanced_resume_content and enhanced_ats_score to job_matches table...")
            await conn.execute(text("ALTER TABLE job_matches ADD COLUMN IF NOT EXISTS enhanced_resume_content TEXT;"))
            await conn.execute(text("ALTER TABLE job_matches ADD COLUMN IF NOT EXISTS enhanced_ats_score FLOAT;"))
            logger.info("Columns added successfully.")
        except Exception as e:
            logger.error(f"Error adding columns: {e}")

if __name__ == "__main__":
    asyncio.run(add_columns())
