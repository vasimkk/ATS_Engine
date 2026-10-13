import sys
import os
import io
import asyncio
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
import pypdf
from litellm import aembedding
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import db
from app.models import Job
from sqlalchemy import select
from app.services.extractor import extract_job_keywords_llm, extract_resume_entities_llm
from app.services.ats import compute_ats_score

load_dotenv()
if "LLM_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]

app = FastAPI(
    title="OpenAI Dual-JD ATS Engine Test",
    description="Upload a Resume, provide a Raw JD and a Database Job ID, and calculate the normalized (100/x)*y formula, outputting to Excel.",
    version="1.0"
)

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

@app.post("/api/v1/test_dual_jd_formula", summary="Test Resume vs 2 JDs (Raw & DB)")
async def test_dual_jd_formula(
    resume_pdf: UploadFile = File(...),
    raw_jd_text: str = Form(..., description="The 'Raw JD' text you are pasting"),
    database_job_id: int = Form(..., description="The ID of the JD in the database (e.g., 8393)")
):
    if not resume_pdf.filename.endswith(".pdf"):
        return {"error": "Valid PDF required."}
        
    if not raw_jd_text:
        return {"error": "raw_jd_text is required."}
        
    try:
        # 1. Fetch DB JD
        async with db._session() as session:
            job = (await session.execute(select(Job).where(Job.job_id == database_job_id))).scalar_one_or_none()
            
        if not job:
            return {"error": f"Job {database_job_id} not found in database."}
        database_jd_text = job.content
        
        # 2. Extract PDF Text
        pdf_bytes = await resume_pdf.read()
        resume_text = extract_pdf_text(pdf_bytes)
        
        # 3. OpenAI Embeddings and LLM Skill Extraction
        res_skills_task = extract_resume_entities_llm(resume_text)
        raw_jd_skills_task = extract_job_keywords_llm(raw_jd_text)
        db_jd_skills_task = extract_job_keywords_llm(database_jd_text)
        
        res_emb_task = aembedding(model="text-embedding-3-small", input=resume_text)
        raw_jd_emb_task = aembedding(model="text-embedding-3-small", input=raw_jd_text)
        db_jd_emb_task = aembedding(model="text-embedding-3-small", input=database_jd_text)
        
        (res_skills, raw_jd_keywords, db_jd_keywords,
         res_resp, raw_jd_resp, db_jd_resp) = await asyncio.gather(
            res_skills_task, raw_jd_skills_task, db_jd_skills_task,
            res_emb_task, raw_jd_emb_task, db_jd_emb_task
        )
        
        resume_emb = res_resp.data[0]["embedding"]
        raw_jd_emb = raw_jd_resp.data[0]["embedding"]
        db_jd_emb = db_jd_resp.data[0]["embedding"]
        
        # 4. Calculate X: Resume vs RAW JD
        x_openai_score = float(cosine_similarity([resume_emb], [raw_jd_emb])[0][0] * 100)
        refined_resume = {"content": resume_text, "additional": {"technicalSkills": res_skills}}
        raw_jd_keywords = raw_jd_keywords or {"required_skills": [], "preferred_skills": []}
        
        x_ats_result = await compute_ats_score(
            refined_resume=refined_resume,
            job_keywords=raw_jd_keywords,
            injectable_keywords=[],
            jd_content=raw_jd_text,
            precomputed_domain_score=x_openai_score
        )
        x_raw_jd_score = float(x_ats_result.get("overall_score", 0.0))
        
        # 5. Calculate Y: Resume vs DATABASE JD
        y_openai_score = float(cosine_similarity([resume_emb], [db_jd_emb])[0][0] * 100)
        db_jd_keywords = db_jd_keywords or {"required_skills": [], "preferred_skills": []}
        
        y_ats_result = await compute_ats_score(
            refined_resume=refined_resume,
            job_keywords=db_jd_keywords,
            injectable_keywords=[],
            jd_content=database_jd_text,
            precomputed_domain_score=y_openai_score
        )
        y_database_jd_score = float(y_ats_result.get("overall_score", 0.0))
        
        # 6. Calculate Formula (100 / x) * y
        formula_score = (100 / x_raw_jd_score) * y_database_jd_score if x_raw_jd_score > 0 else 0.0
        
        import json
        
        # 7. Save to Excel
        results_data = [{
            "Database Job ID": database_job_id,
            "Resume Text": resume_text,
            "Raw JD Text": raw_jd_text,
            "Database JD Text": database_jd_text,
            "Resume OpenAI Embedding": json.dumps(resume_emb),
            "Raw JD OpenAI Embedding": json.dumps(raw_jd_emb),
            "Database JD OpenAI Embedding": json.dumps(db_jd_emb),
            "Raw JD Score (x)": x_raw_jd_score,
            "Raw JD Cosine Score": x_ats_result.get("sub_scores", {}).get("domain_score", 0),
            "Raw JD Skill Score": x_ats_result.get("sub_scores", {}).get("skills_coverage", 0),
            "Database JD Score (y)": y_database_jd_score,
            "Database JD Cosine Score": y_ats_result.get("sub_scores", {}).get("domain_score", 0),
            "Database JD Skill Score": y_ats_result.get("sub_scores", {}).get("skills_coverage", 0),
            "Formula Score (100/x)*y": formula_score
        }]
        
        df = pd.DataFrame(results_data)
        
        # Save to the root of the project instead of the backend folder
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        excel_path = os.path.join(project_root, "single_test_formula_score.xlsx")
        df.to_excel(excel_path, index=False)
        
        return {
            "status": "success",
            "message": f"Saved results to {excel_path}",
            "raw_jd_score_x": x_raw_jd_score,
            "raw_jd_cosine_score": x_ats_result.get("sub_scores", {}).get("domain_score", 0),
            "raw_jd_skill_score": x_ats_result.get("sub_scores", {}).get("skills_coverage", 0),
            "database_jd_score_y": y_database_jd_score,
            "database_jd_cosine_score": y_ats_result.get("sub_scores", {}).get("domain_score", 0),
            "database_jd_skill_score": y_ats_result.get("sub_scores", {}).get("skills_coverage", 0),
            "formula_score": formula_score,
            "raw_jd_full_breakdown": x_ats_result,
            "database_jd_full_breakdown": y_ats_result
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    print("======================================================")
    print("🚀 Starting OpenAI Dual-JD Swagger Server on PORT 8001")
    print("👉 GO TO: http://127.0.0.1:8001/docs")
    print("======================================================")
    uvicorn.run(app, host="127.0.0.1", port=8001)
