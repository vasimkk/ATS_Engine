import sys
import os
import asyncio
import pandas as pd
from litellm import aembedding
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Add backend to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import db
from app.models import JobMatch, Resume, Job
from sqlalchemy import select
from app.services.extractor import extract_job_keywords_llm, extract_resume_entities_llm
from app.services.ats import compute_ats_score
import json

load_dotenv()
if "LLM_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]

async def main():
    print("Connecting to database...")
    async with db._session() as session:
        stmt = select(JobMatch)
        result = await session.execute(stmt)
        matches = result.scalars().all()
        
        results_data = []
        
        print(f"Processing {len(matches)} matches...")
        for match in matches:
            resume_id = match.resume_id
            job_id = match.job_id
            y_db_score = float(match.match_score) # OLD DB ATS SCORE
            
            res = (await session.execute(select(Resume).where(Resume.resume_id == resume_id))).scalar_one_or_none()
            job = None
            try:
                job = (await session.execute(select(Job).where(Job.job_id == int(job_id)))).scalar_one_or_none()
            except:
                pass
            
            if not res or not job:
                continue
                
            resume_text = res.content
            jd_text = job.content
            
            try:
                # 1. Generate OpenAI Embeddings & Parse Skills
                res_skills_task = extract_resume_entities_llm(resume_text)
                jd_skills_task = extract_job_keywords_llm(jd_text)
                res_emb_task = aembedding(model="text-embedding-3-small", input=resume_text)
                jd_emb_task = aembedding(model="text-embedding-3-small", input=jd_text)
                
                res_skills, jd_keywords, res_resp, jd_resp = await asyncio.gather(
                    res_skills_task, jd_skills_task, res_emb_task, jd_emb_task
                )
                
                resume_emb = res_resp.data[0]["embedding"]
                jd_emb = jd_resp.data[0]["embedding"]
                
                # 2. Raw JD Score (x) = Raw OpenAI Cosine Similarity
                x_raw_openai_score = float(cosine_similarity([resume_emb], [jd_emb])[0][0] * 100)
                
                refined_resume = {"content": resume_text, "additional": {"technicalSkills": res_skills}}
                if not jd_keywords:
                    jd_keywords = {"required_skills": [], "preferred_skills": []}
                    
                # 3. Database JD Score (y) = New ATS Engine Score using OpenAI embeddings
                final_ats_result = await compute_ats_score(
                    refined_resume=refined_resume,
                    job_keywords=jd_keywords,
                    injectable_keywords=[],
                    jd_content=jd_text,
                    precomputed_domain_score=x_raw_openai_score
                )
                
                y_new_ats_score = float(final_ats_result.get("overall_score", 0.0))
                
                # 4. Apply Formula (100 / x) * y
                formula_score = (100 / x_raw_openai_score) * y_new_ats_score if x_raw_openai_score > 0 else 0.0
                
                results_data.append({
                    "Resume ID": resume_id,
                    "Job ID": job_id,
                    "Resume Text": resume_text,
                    "JD Text": jd_text,
                    "Resume OpenAI Embedding": json.dumps(resume_emb),
                    "JD OpenAI Embedding": json.dumps(jd_emb),
                    "Raw JD Score (x)": x_raw_openai_score,
                    "Database JD Score (y)": y_new_ats_score,
                    "Formula Score (100/x)*y": formula_score
                })
                print(f"Processed Job {job_id} | Raw JD (x): {x_raw_openai_score:.2f} | Database JD (y): {y_new_ats_score:.2f} | Formula: {formula_score:.2f}")
                
            except Exception as e:
                print(f"Error for Resume {resume_id[:8]}: {e}")
                
        if results_data:
            df = pd.DataFrame(results_data)
            excel_path = "openai_formula_scores.xlsx"
            df.to_excel(excel_path, index=False)
            print(f"\nSUCCESS! Saved {len(results_data)} results and raw embeddings to {excel_path}!")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
