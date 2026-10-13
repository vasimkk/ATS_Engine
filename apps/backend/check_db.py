import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.database import db
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify():
    db._ensure_initialized()
    async with db._async_engine.begin() as conn:
        result = await conn.execute(text("SELECT job_id, ats_score, enhanced_ats_score, enhanced_resume_content FROM job_matches LIMIT 1;"))
        row = result.fetchone()
        if row:
            logger.info(f"Row found! Job ID: {row[0]}")
            logger.info(f"Original ATS: {row[1]}")
            logger.info(f"Enhanced ATS: {row[2]}")
            logger.info(f"Enhanced Resume (first 100 chars): {str(row[3])[:100] if row[3] else None}")
        else:
            logger.info("No rows in job_matches.")

if __name__ == "__main__":
    asyncio.run(verify())
