import asyncio
import uuid
import time
from app.database import db
from app.models import JobMatch
from sqlalchemy import select, delete
from app.services.ats import compute_ats_score
from app.services.extractor import extract_job_keywords

async def run_batch_scoring():
    print("Fetching the latest resume...")
    resumes = await db.list_resumes()
    if not resumes:
        print("No resumes found in the database. Please upload one first.")
        return
        
    # Get the latest resume
    latest_resume = resumes[-1]
    resume_id = latest_resume["resume_id"]
    resume_text = latest_resume.get("content", "")
    resume_vector = latest_resume.get("resume_embedding")
    resume_skills = latest_resume.get("metadata_json", {}).get("skills", [])
    
    print(f"Resume found: {resume_id}")
    
    print("Fetching all jobs...")
    jobs = await db.list_jobs()
    print(f"Found {len(jobs)} jobs to score.")
    
    print("Calculating scores and saving to database...")
    start_time = time.time()
    
    async with db._write_session() as session:
        # Clear old matches for this resume
        await session.execute(delete(JobMatch).where(JobMatch.resume_id == resume_id))
        
        matches = []
        for i, job in enumerate(jobs):
            job_id = job["job_id"]
            jd_text = job.get("content", "")
            jd_vector = job.get("job_embedding")
            
            # Extract keywords if not already in job metadata
            job_keywords = job.get("metadata_json", {}).get("job_keywords")
            if not job_keywords:
                job_keywords = extract_job_keywords(jd_text)
                
            # Compute score
            try:
                refined_resume = {
                    "id": resume_id,
                    "additional": {"technicalSkills": resume_skills},
                    "workExperience": [{"description": resume_text}],
                    "summary": ""
                }
                score_data = await compute_ats_score(
                    refined_resume=refined_resume,
                    job_keywords=job_keywords,
                    injectable_keywords=[],
                    jd_content=jd_text,
                    job_id=job_id,
                    precomputed_jd_vector=jd_vector,
                    precomputed_domain_score=None
                )
                overall_score = score_data["overall_score"]
                
                # Create JobMatch record
                matches.append(
                    JobMatch(
                        match_id=str(uuid.uuid4()),
                        resume_id=resume_id,
                        job_id=job_id,
                        overall_score=overall_score,
                        sub_scores_json=score_data
                    )
                )
            except Exception as e:
                print(f"Error scoring job {job_id}: {e}")
                
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(jobs)}...")
                session.add_all(matches)
                await session.flush()
                matches = []
                
        if matches:
            session.add_all(matches)
            
        await session.commit()
        
    elapsed = time.time() - start_time
    print(f"\nDone! Successfully scored {len(jobs)} jobs and saved them to the JobMatch database table in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(run_batch_scoring())
