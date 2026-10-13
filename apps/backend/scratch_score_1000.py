import asyncio
import os
import sys

# add backend dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import db
from app.models import Job, Resume
from app.services.ats import compute_ats_score
from sqlalchemy import select
import pandas as pd
from datetime import datetime

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def main():
    print("Loading AI Models into memory...")
    from sentence_transformers import SentenceTransformer
    import app.main
    # Force load the model into the app's global state so ats.py can see it
    try:
        app.main._SENTENCE_MODEL = SentenceTransformer(r"D:\bge-base-en-v1.5")
    except Exception:
        app.main._SENTENCE_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print("Models loaded successfully!")
    
    async with db._session() as session:
        # Get the specific resume
        resume_id = "d8ec7558e0ae3802339a87c693e628d11dbe25d85236a716612c6660c008f1db"
        resume = (await session.execute(select(Resume).where(Resume.resume_id == resume_id))).scalar_one_or_none()
        if not resume or resume.res_embedding is None:
            print(f"Resume {resume_id} not found or has no embedding.")
            return
            
        print("Searching full database using pgvector HNSW...")
        stmt = (
            select(Job)
            .where(Job.job_embedding.is_not(None))
            .order_by(Job.job_embedding.cosine_distance(resume.res_embedding))
            .limit(100)
        )
        jobs = (await session.execute(stmt)).scalars().all()
        print(f"Found {len(jobs)} jobs. Calculating scores...")
        
        results = []
        for i, job in enumerate(jobs):
            job_keywords = {
                "required_skills": job.required_skills or [],
                "preferred_skills": job.preferred_skills or []
            }
            import json
            resume_data = resume.content
            if isinstance(resume_data, str):
                try:
                    resume_data = json.loads(resume_data)
                except:
                    resume_data = {"content": resume_data}
            
            score_data = await compute_ats_score(
                refined_resume=resume_data,
                job_keywords=job_keywords,
                injectable_keywords=[],
                jd_content=job.content,
                job_id=job.job_id
            )
            
            # Print to screen!
            print(f"[{i+1}/100] Job {job.job_id} | Overall: {score_data['overall_score']}% | Domain: {score_data['sub_scores']['domain_score']}% | Skills: {score_data['sub_scores']['skills_coverage']}%")
            
            results.append({
                "Job ID": job.job_id,
                "Resume ID": resume_id,
                "Resume Text": resume.content,
                "Job Description": job.content,
                "Overall ATS Score": score_data["overall_score"],
                "Cosine Score": score_data["sub_scores"]["domain_score"],
                "Skill Score": score_data["sub_scores"]["skills_coverage"],
                "Keyword Score": score_data["sub_scores"]["skills_coverage"],
                "Experience Fit": score_data["sub_scores"]["experience_fit"],
                "Section Completeness": score_data["sub_scores"]["section_completeness"]
            })
            
        df = pd.DataFrame(results)
        
        # Sort by Overall Score descending so highest matches are at the top!
        df = df.sort_values(by="Overall ATS Score", ascending=False)
        
        # Save directly to the project root folder
        artifacts_dir = r"D:\clone_repo_ats-1 (2)\clone_repo_ats-1\clone_repo_ats-1"
        filename = os.path.join(artifacts_dir, "resume_scores_100.xlsx")
        
        try:
            df.to_excel(filename, index=False)
            print(f"Excel saved to {filename}")
        except Exception as e:
            print(f"Excel save failed: {e}")
            csv_filename = filename.replace('.xlsx', '.csv')
            df.to_csv(csv_filename, index=False)
            print(f"CSV saved to {csv_filename} instead.")

if __name__ == "__main__":
    asyncio.run(main())
