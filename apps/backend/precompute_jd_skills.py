import asyncio
import logging
from sqlalchemy import select

from app.database import db
from app.models import Job
from app.services.extractor import extract_job_keywords_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_job(job_id: str, job_content: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            # 1. Call the LLM to extract the skills
            extracted_skills = await extract_job_keywords_llm(job_content)
            req_skills = extracted_skills.get("required_skills", [])
            pref_skills = extracted_skills.get("preferred_skills", [])
            
            # Load model locally to compute embeddings
            import asyncio
            from sentence_transformers import SentenceTransformer
            loop = asyncio.get_running_loop()
            
            def embed(text):
                if not text: return None
                model = SentenceTransformer(r"D:\bge-base-en-v1.5")
                return model.encode(text).tolist()
                
            req_emb = await loop.run_in_executor(None, embed, ", ".join(req_skills)) if req_skills else None
            pref_emb = await loop.run_in_executor(None, embed, ", ".join(pref_skills)) if pref_skills else None
            
            # 2. Open a dedicated database session for this task
            async with db._session() as session:
                # Fetch the job fresh in this session
                stmt = select(Job).where(Job.job_id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                
                if not job:
                    return
                    
                meta = job.metadata_json or {}
                
                # Update the metadata dictionary
                meta["required_skills"] = req_skills
                meta["preferred_skills"] = pref_skills
                
                # Save directly back to the database
                job.metadata_json = meta
                job.required_skills_embedding = req_emb
                job.preferred_skills_embedding = pref_emb
                session.add(job)
                await session.commit()
                
            logger.info(f"Successfully saved skills and embeddings for {job_id}!")
        except Exception as e:
            logger.error(f"Failed to extract/save job {job_id}: {e}")

async def main():
    logger.info("Starting JD Skills Pre-computation script...")
    
    semaphore = asyncio.Semaphore(20) 
    
    # Fetch job metadata first
    jobs_to_process = []
    async with db._session() as session:
        stmt = select(Job.job_id, Job.content, Job.metadata_json)
        result = await session.execute(stmt)
        for job_id, content, meta in result.all():
            meta = meta or {}
            if "required_skills" not in meta or "preferred_skills" not in meta:
                jobs_to_process.append((job_id, content))
        
    logger.info(f"Found {len(jobs_to_process)} jobs needing skill extraction. Processing...")
    
    # Fire off all jobs asynchronously with their own sessions
    tasks = [process_job(job_id, content, semaphore) for job_id, content in jobs_to_process]
    await asyncio.gather(*tasks)
        
    logger.info("Finished pre-computing all JD skills!")

if __name__ == "__main__":
    asyncio.run(main())
