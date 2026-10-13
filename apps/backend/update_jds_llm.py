import asyncio
import json
from sqlalchemy import select, text
from app.database import db
from app.models import Job
from app.services.llm import ask_llm

LLM_PROMPT = """You are an expert tech recruiter. Analyze the following job description and extract the domain, required skills, and preferred skills. 
Return ONLY a strictly valid JSON object with the following format, nothing else:
{
    "domain": "Software Engineering, Marketing, etc",
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill3", "skill4"]
}

Job Description:
"""

async def process_jobs():
    print("Starting LLM extraction for all jobs...")
    
    async with db._write_session() as session:
        result = await session.execute(select(Job))
        jobs = result.scalars().all()
        
        for idx, job in enumerate(jobs):
            if job.content:
                print(f"Processing Job {idx + 1}/{len(jobs)}: {job.title}")
                try:
                    prompt = LLM_PROMPT + job.content
                    response = await ask_llm(prompt)
                    extracted_data = json.loads(response)
                    
                    # Update job with the extracted skills
                    job.processed_data = extracted_data
                    await session.commit()
                except Exception as e:
                    print(f"Failed to process job {job.job_id}: {e}")
                    
    print("Finished processing all jobs!")

if __name__ == "__main__":
    asyncio.run(process_jobs())
