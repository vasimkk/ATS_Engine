import asyncio
import logging
from sqlalchemy import delete
from app.database import db
from app.models import Resume, JobMatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def clear_cache():
    logger.info("Connecting to database to clear Resumes and Scores...")
    async with db._session() as session:
        # Delete all scores
        await session.execute(delete(JobMatch))
        logger.info("Cleared all JobMatches (Cached Scores).")
        
        # Delete all resumes
        await session.execute(delete(Resume))
        logger.info("Cleared all Resumes.")
        
        await session.commit()
    logger.info("Successfully wiped resumes and scores! Job Descriptions are safe.")

if __name__ == "__main__":
    asyncio.run(clear_cache())
