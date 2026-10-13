"""FastAPI application entry point."""
import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Fix for Windows: Use ProactorEventLoop for subprocess support (Playwright)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings
from app.database import DatabaseBusyError, db
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, File, UploadFile, Form, Body
import pypdf
import io

def _configure_application_logging() -> None:
    """Set application log level from configuration."""
    numeric_level = getattr(logging, settings.log_level, logging.INFO)
    logging.getLogger("app").setLevel(numeric_level)


_configure_application_logging()


_SENTENCE_MODEL = None
_FAISS_INDEX = None
_JD_ID_MAP = []

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    global _SENTENCE_MODEL, _FAISS_INDEX, _JD_ID_MAP
    
    # Startup
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading SentenceTransformer model into memory...")
    try:
        from sentence_transformers import SentenceTransformer
        import asyncio
        loop = asyncio.get_running_loop()
        
        def load_model():
            # Load from HuggingFace directly since local D: drive path is missing
            return SentenceTransformer(r"D:\bge-base-en-v1.5")
            
        _SENTENCE_MODEL = await loop.run_in_executor(None, load_model)
        logger.info("Successfully loaded SentenceTransformer model.")
        
        logger.info("Skipping FAISS cache; using pgvector directly in Postgres.")
        
    except Exception as e:
        logger.error(f"Failed to load FastEmbed model or FAISS index: {e}")
        
    yield
    
    # Shutdown
    try:
        await db.close()
    except Exception as e:
        logger.error(f"Error closing database: {e}")


app = FastAPI(
    title="Custom ATS API",
    description="LLM-Free ATS Calculator",
    version=__version__,
    lifespan=lifespan,
)

@app.exception_handler(DatabaseBusyError)
async def database_busy_handler(request: Request, error: DatabaseBusyError) -> JSONResponse:
    logger.warning("Database write contention for %s", request.url.path, exc_info=error)
    return JSONResponse(
        status_code=503,
        content={"error": "Database is busy. Please retry shortly."},
        headers={"Retry-After": "1"},
    )


# CORS middleware - origins configurable via CORS_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ==========================================
# CUSTOM ATS CALCULATOR (Zero-LLM)
# ==========================================
from app.services.ats import compute_ats_score

import docx

def _extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from PDF, DOCX, or TXT (LLM-Free)."""
    try:
        filename_lower = filename.lower()
        if filename_lower.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text
        elif filename_lower.endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_bytes))
            text = " ".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        elif filename_lower.endswith(".txt") or filename_lower.endswith(".md"):
            return file_bytes.decode('utf-8', errors='ignore')
        else:
            # Fallback for unknown formats, try to decode as text
            return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Failed to extract text from {filename}: {e}")
        return ""

from app.services.extractor import extract_resume_entities_llm, extract_job_keywords_llm
from pydantic import BaseModel
import uuid


@app.post("/api/v1/Get_N_JDs_for_Res")
async def get_n_jds_for_res(
    N: int,
    top_k: int,
    user_id: str,
    resume_file: UploadFile = File(...)
):
    """Extract resume once, score against ALL jobs in the database."""
    from app.database import db
    
    file_bytes = await resume_file.read()
    raw_text = _extract_text_from_file(file_bytes, resume_file.filename)
    
    import hashlib
    resume_hash = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()
    resume_id = resume_hash
    
    from app.services.extractor import extract_resume_entities_llm
    from sqlalchemy import select
    from app.models import Resume
    
    import numpy as np
    import faiss
    resume_vector = _SENTENCE_MODEL.encode([raw_text])[0]
    
    async with db._session() as session:
        stmt_res = select(Resume).where(Resume.resume_id == resume_id)
        res_exists = (await session.execute(stmt_res)).scalar_one_or_none()
        if res_exists and res_exists.processed_data and "technicalSkills" in res_exists.processed_data:
            found_resume_skills = res_exists.processed_data["technicalSkills"]
        else:
            found_resume_skills = await extract_resume_entities_llm(raw_text)
            if not res_exists:
                import uuid
                new_res = Resume(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    resume_id=resume_id, 
                    res_embedding=resume_vector.tolist(),
                    content=raw_text, 
                    processing_token=resume_hash,
                    processed_data={"technicalSkills": found_resume_skills}
                )
                session.add(new_res)
                try:
                    await session.commit()
                except Exception as e:
                    logger.error(f"Failed to commit resume: {e}")
                    await session.rollback()
    
    refined_resume = {
        "id": "stateless-ats",
        "additional": {
            "technicalSkills": found_resume_skills
        },
        "workExperience": [
            {
                "description": raw_text
            }
        ],
        "summary": "", 
    }
    
    results = []
    
    # 1. Embed the Resume once (Live)
    from sqlalchemy import select
    from app.models import Job, JobMatch
    
    try:
        # Query closest JDs using pgvector cosine distance
        async with db._session() as session:
            stmt = (
                select(Job, Job.job_embedding.cosine_distance(resume_vector).label("distance"))
                .where(Job.job_embedding.is_not(None))
                .order_by(Job.job_embedding.cosine_distance(resume_vector))
                .limit(top_k)
            )
            result = await session.execute(stmt)
            rows = result.all()
            
            top_candidates = []
            for row in rows:
                job_model = row[0]
                distance = row[1]
                
                # In pgvector, cosine distance = 1 - cosine_similarity
                # Normalize similarity and prevent negative values
                sim = max(0.0, 1.0 - distance)
                
                # Convert Job model back to the dict format expected by the pipeline
                job = {
                    "job_id": job_model.job_id, 
                    "content": job_model.content, 
                    "metadata_json": job_model.metadata_json,
                    "required_skills": job_model.required_skills,
                    "preferred_skills": job_model.preferred_skills
                }
                
                domain_score = round(sim * 100, 1)
                top_candidates.append({
                    "job": job,
                    "domain_score": domain_score,
                    "idx": job_model.job_id  # Using job_id directly since index is gone
                })
                
            # Take exactly the limit (top N) for LLM evaluation
            top_candidates = top_candidates[:N]
            
            # Phase 2: Deep LLM Scoring ONLY for top N candidates
            
            async def process_deep(cand):
                job = cand["job"]
                domain_score = cand["domain_score"]
                
                # Check DB Cache first
                async with db._session() as session:
                    stmt = select(JobMatch).where(JobMatch.resume_id == resume_id, JobMatch.job_id == str(job["job_id"]))
                    result = await session.execute(stmt)
                    cached_match = result.scalar_one_or_none()
                    
                    if cached_match and cached_match.score_data_json:
                        logger.info(f"Using cached JobMatch for {job['job_id']} and resume {resume_id}")
                        return {
                            "job_id": job["job_id"],
                            "preview": job["content"][:100] + "..." if len(job["content"]) > 100 else job["content"],
                            "full_jd_text": job["content"],
                            "overall_score": cached_match.match_score,
                            "score_data": cached_match.score_data_json,
                            "required_skills": job.get("required_skills") or [],
                            "preferred_skills": job.get("preferred_skills") or []
                        }

                # Deep LLM Extraction
                from app.services.extractor import extract_job_keywords_llm
                from sqlalchemy import update
                from app.models import Job
                
                if job.get("required_skills") is not None and job.get("preferred_skills") is not None:
                    job_keywords = {
                        "required_skills": job["required_skills"],
                        "preferred_skills": job["preferred_skills"]
                    }
                else:
                    job_keywords = await extract_job_keywords_llm(job["content"])
                    if job_keywords is None:
                        # LLM failed, do not save to DB so we can try again next time
                        job_keywords = {"required_skills": [], "preferred_skills": []}
                    else:
                        req_skills = job_keywords.get("required_skills", [])
                        pref_skills = job_keywords.get("preferred_skills", [])
                        
                        # Only save if the LLM actually found skills (prevent caching empty mistakes)
                        if len(req_skills) > 0 or len(pref_skills) > 0:
                            # PERMANENTLY SAVE the extracted skills back to the jobs table!
                            async with db._session() as session:
                                stmt = update(Job).where(Job.job_id == job["job_id"]).values(
                                    required_skills=req_skills,
                                    preferred_skills=pref_skills
                                )
                                await session.execute(stmt)
                                await session.commit()
                        
                # Final Deep Score
                score_data = await compute_ats_score(
                    refined_resume=refined_resume,
                    job_keywords=job_keywords,
                    injectable_keywords=[],
                    jd_content=job["content"],
                    job_id=job["job_id"],
                    precomputed_domain_score=domain_score,
                )
                
                # Save to DB cache ONLY if LLM successfully extracted skills
                # If LLM failed (e.g. out of credits), we skip caching so the system will try again next time!
                if job.get("required_skills") is not None or (len(job_keywords.get("required_skills", [])) > 0):
                    import uuid
                    async with db._session() as session:
                        new_match = JobMatch(
                            id=str(uuid.uuid4()),
                            user_id=user_id,
                            resume_id=resume_id,
                            job_id=str(job["job_id"]),
                            match_score=score_data.get("overall_score", 0),
                            score_data_json=score_data
                        )
                        session.add(new_match)
                        await session.commit()
                
                result_data = {
                    "job_id": job["job_id"],
                    "preview": job["content"][:100] + "..." if len(job["content"]) > 100 else job["content"],
                    "full_jd_text": job["content"],
                    "overall_score": score_data.get("overall_score", 0),
                    "score_data": score_data,
                    "required_skills": job_keywords.get("required_skills", []),
                    "preferred_skills": job_keywords.get("preferred_skills", [])
                }
                
                return result_data
                
            tasks = [process_deep(cand) for cand in top_candidates]
            job_results = await asyncio.gather(*tasks)
            
            for res in job_results:
                if res:
                    results.append(res)
                    
    except Exception as e:
        import traceback
        logger.error(f"Failed to embed resume or search FAISS: {e}")
        return {"status": "error", "message": f"Exception: {str(e)}\n{traceback.format_exc()}"}
        
    results.sort(key=lambda x: x["overall_score"], reverse=True)
    

    
    return {
        "status": "success",
        "resume_id": resume_id,
        "parsed_text_preview": raw_text,
        "extracted_skills": found_resume_skills,
        "matches": results
    }









from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# Serve the beautiful premium UI!
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")

class GetMoreJDsRequest(BaseModel):
    N: int = 10
    LastJDid: str
    top_k: int = 1000
    ResumeID: str

@app.post("/api/v1/Get_N_moreJDs_for_Res")
async def get_n_more_jds_for_res(req: GetMoreJDsRequest):
    """Return N more JDs for a Resume by offsetting the FAISS results."""
    from app.database import db
    from app.models import Resume, JobMatch, Job
    from sqlalchemy import select
    import numpy as np
    import faiss
    
    try:
        async with db._session() as session:
            res = (await session.execute(select(Resume).where(Resume.resume_id == req.ResumeID))).scalar_one_or_none()
            if not res:
                return {"status": "error", "message": "Resume not found in DB"}
                
            resume_vector = res.res_embedding
            if resume_vector is None:
                return {"status": "error", "message": "Resume has no embedding"}
                
            # Query closest JDs using pgvector cosine distance
            stmt = (
                select(Job, Job.job_embedding.cosine_distance(resume_vector).label("distance"))
                .where(Job.job_embedding.is_not(None))
                .order_by(Job.job_embedding.cosine_distance(resume_vector))
                .limit(req.top_k)
            )
            result = await session.execute(stmt)
            rows = result.all()
            
            top_candidates = []
            for row in rows:
                job_model = row[0]
                distance = row[1]
                
                sim = max(0.0, 1.0 - distance)
                job = {
                    "job_id": job_model.job_id, 
                    "content": job_model.content, 
                    "metadata_json": job_model.metadata_json,
                    "required_skills": job_model.required_skills,
                    "preferred_skills": job_model.preferred_skills
                }
                
                top_candidates.append({
                    "job": job,
                    "domain_score": round(sim * 100, 1),
                    "idx": job_model.job_id
                })
                    
            # Find the offset
            start_index = 0
            for i, cand in enumerate(top_candidates):
                if str(cand["job"]["job_id"]) == str(req.LastJDid):
                    start_index = i + 1
                    break
                    
            next_candidates = top_candidates[start_index : start_index + req.N]
            
            async def process_deep(cand):
                job = cand["job"]
                # Use DB cache if exists
                async with db._session() as session2:
                    cached = (await session2.execute(select(JobMatch).where(JobMatch.resume_id == req.ResumeID, JobMatch.job_id == str(job["job_id"])))).scalar_one_or_none()
                    if cached and cached.score_data_json:
                        return {
                            "job_id": job["job_id"],
                            "full_jd_text": job["content"],
                            "overall_score": cached.match_score,
                            "score_data": cached.score_data_json,
                            "required_skills": job.get("required_skills") or [],
                            "preferred_skills": job.get("preferred_skills") or []
                        }
                        
                # Minimal fallback if not cached
                return {
                    "job_id": job["job_id"],
                    "full_jd_text": job["content"],
                    "overall_score": cand["domain_score"],
                    "score_data": {},
                    "required_skills": job.get("required_skills") or [],
                    "preferred_skills": job.get("preferred_skills") or []
                }
                
            tasks = [process_deep(cand) for cand in next_candidates]
            results = await asyncio.gather(*tasks)
            return {"status": "success", "matches": results}
    except Exception as e:
        import traceback
        return {"status": "error", "message": f"Exception: {str(e)}\n{traceback.format_exc()}"}

class EnhanceResumeRequest(BaseModel):
    JDid: str
    Candidateid: str
    ResumeID: str

async def run_s5_enhancement(resume, job):
    from app.services.gap_analysis import run_s4_gap_analysis
    from litellm import acompletion
    from app.config import settings
    from app.models import Improvement
    from app.database import db
    import json
    import uuid
    import difflib
    
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    gap_result = await run_s4_gap_analysis(resume.content, job.content, resume.resume_id, job.job_id)
    bridgeable_gaps = gap_result.get("bridgeable_gaps", [])
    all_gaps = gap_result.get("gaps", [])

    # ── Fallback Level 1: if no bridgeable gaps but there ARE gaps, use ALL gaps ──
    if len(bridgeable_gaps) == 0 and len(all_gaps) > 0:
        _logger.warning("[S5] bridgeable_gaps empty — falling back to all gaps as bridgeable")
        bridgeable_gaps = [dict(g, bridgeable=True, rationale="Fallback: assumed bridgeable") for g in all_gaps]

    # ── Fallback Level 2: if gap analysis returned nothing at all, build gaps
    # directly from JD text using regex — no LLM needed (handles exhausted API key)
    if len(bridgeable_gaps) == 0:
        _logger.warning("[S5] gap_list also empty — extracting skills from raw JD text via regex")
        import re
        jd_required = job.required_skills or []
        jd_preferred = job.preferred_skills or []

        # If DB skills are also empty, parse raw JD text directly
        if not jd_required and not jd_preferred and job.content:
            # Extract capitalized tech terms, known skill keywords from raw JD text
            raw_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9+#.]*(?:\s[A-Za-z][A-Za-z0-9+#.]*){0,2}\b", job.content)
            # Filter to likely skill tokens: length 2-40 chars, not stop words
            stop_words = {"the","and","for","with","you","will","our","are","has","have",
                          "this","that","from","your","not","but","can","all","an","a",
                          "in","of","to","is","be","or","as","we","at","by","on","it"}
            seen = set()
            for tok in raw_tokens:
                t = tok.strip()
                if 2 <= len(t) <= 40 and t.lower() not in stop_words and t.lower() not in seen:
                    seen.add(t.lower())
                    jd_required.append(t)
            _logger.info(f"[S5] Regex extracted {len(jd_required)} raw JD tokens")

        resume_lower = resume.content.lower()
        direct_gaps = []
        for skill in jd_required:
            if skill.lower().strip() not in resume_lower:
                direct_gaps.append({"skill": skill, "severity": "CRITICAL", "source": "required", "bridgeable": True, "rationale": "Not found in resume text"})
        for skill in jd_preferred:
            if skill.lower().strip() not in resume_lower:
                direct_gaps.append({"skill": skill, "severity": "PREFERRED", "source": "preferred", "bridgeable": True, "rationale": "Not found in resume text"})
        bridgeable_gaps = direct_gaps
        _logger.info(f"[S5] Direct gap build found {len(bridgeable_gaps)} gaps")

    # ── Hard stop only if truly nothing to work with ────────────────────────
    if len(bridgeable_gaps) == 0:
        return {"status": "NO_BRIDGEABLE_GAPS", "message": "Resume already satisfies all JD requirements — no enhancement needed", "new_ats_score": None}

    gap_descriptions = "\n".join([f"- {g['skill']} ({g['rationale']})" for g in bridgeable_gaps])
    rewrite_prompt = f"""You are an expert resume writer.
Rewrite the resume to address the following missing skill gaps.

CRITICAL RULES:
  1. DO NOT delete, alter, or remove ANY of the candidate's existing skills, technologies, or keywords. 
  2. Simply weave the missing skill keywords naturally into the existing work experience bullets or summary.
  3. DO NOT invent job titles, companies, dates, or degrees.
  4. Preserve all original formatting.
  5. Return the full rewritten resume text only — no commentary.

ORIGINAL RESUME:
{resume.content}

TARGET JD:
{job.content}

MISSING GAPS TO INJECT:
{gap_descriptions}"""

    api_key = settings.get_effective_api_key()
    api_base = settings.llm_api_base

    try:
        response = await acompletion(
            model=settings.llm_model,
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.2,
            api_key=api_key,
            api_base=api_base
        )
        enhanced_text = response.choices[0].message.content
    except Exception as e:
        _logger.error(f"[S5] LLM rewrite call failed: {e}")
        return {
            "status": "LLM_UNAVAILABLE",
            "message": f"Resume enhancement failed — LLM service error: {str(e)[:200]}",
            "bridgeable_gaps": bridgeable_gaps
        }
    
    guardrail_prompt = f"""Compare original and enhanced resume. 
You must ONLY flag fabrication if the enhanced version invents:
1. New Companies or Employers
2. New Job Titles
3. New Degrees or Certifications
4. Fake Quantitative Metrics (e.g., "Increased revenue by 500%")

DO NOT flag natural rephrasing, soft skills, or the injection of technical keywords into existing context.
Return ONLY a JSON object: {{"pass": true/false, "flags": ["list of strict fabrications"]}}

ORIGINAL:
{resume.content}

ENHANCED:
{enhanced_text}"""
    
    try:
        guard_response = await acompletion(
            model=settings.llm_model,
            messages=[{"role": "user", "content": guardrail_prompt}],
            response_format={"type": "json_object"} if "llama" not in settings.llm_model.lower() else None,
            api_key=api_key,
            api_base=api_base
        )
    except Exception as e:
        _logger.warning(f"[S5] Guardrail LLM call failed: {e}. Skipping guardrail.")
        guard_response = None
    try:
        if guard_response is not None:
            content = guard_response.choices[0].message.content
            if "```json" in content: content = content.split("```json")[1].split("```")[0]
            elif "```" in content: content = content.split("```")[1].split("```")[0]
            guardrail = json.loads(content)
        else:
            guardrail = {"pass": True, "flags": []}
    except Exception:
        guardrail = {"pass": True, "flags": []}
        
    if guardrail.get("pass", True) is False and len(guardrail.get("flags", [])) > 0:
        return {"status": "GUARDRAIL_FAILED", "message": "Fabrication detected", "flags": guardrail.get("flags", [])}

    return {
        "status": "OK",
        "original_resume": resume.content,
        "EnhResume": enhanced_text,
        "bridgeable_gaps": bridgeable_gaps
    }





class EnhanceResumeRequest(BaseModel):
    JDid: str
    Candidateid: str
    ResumeID: str

@app.post("/api/v1/Get_EnhancedResume_for_PoorJDScore")
async def get_enhanced_resume_for_poor_jdscore(req: EnhanceResumeRequest):
    """
    MODULE S5 — RESUME ENHANCEMENT
    """
    from app.database import db
    from app.models import Resume, Job
    from sqlalchemy import select
    
    async with db._session() as session:
        resume = (await session.execute(select(Resume).where(Resume.resume_id == req.ResumeID))).scalar_one_or_none()
        job = (await session.execute(select(Job).where(Job.job_id == int(req.JDid)))).scalar_one_or_none()
        
    if not resume or not job:
        return {"status": "error", "message": "Resume or Job not found"}
        
    return await run_s5_enhancement(resume, job)

class ScoreEnhancedResumeRequest(BaseModel):
    JDid: str
    ResumeID: str
    EnhResumeText: str

@app.post("/api/v1/Get_Score_for_EnhancedResume")
async def get_score_for_enhanced_resume(req: ScoreEnhancedResumeRequest):
    """
    Calculate the ATS Score for a newly enhanced resume text.
    Ensures the score never drops below the original ATS score.
    """
    from app.database import db
    from app.models import Job, JobMatch
    from sqlalchemy import select
    from app.services.extractor import extract_resume_entities_llm, extract_job_keywords_llm
    from app.services.ats import compute_ats_score
    
    async with db._session() as session:
        job = (await session.execute(select(Job).where(Job.job_id == int(req.JDid)))).scalar_one_or_none()
        # Fetch the original score from cache
        match = (await session.execute(
            select(JobMatch).where(JobMatch.job_id == req.JDid, JobMatch.resume_id == req.ResumeID)
        )).scalar_one_or_none()
        
    if not job:
        return {"status": "error", "message": "Job not found"}
        
    original_score = match.match_score if match else 0.0
        
    found_resume_skills = await extract_resume_entities_llm(req.EnhResumeText)
    
    req_skills = job.required_skills
    pref_skills = job.preferred_skills
    
    if req_skills is not None and pref_skills is not None:
        job_keywords = {
            "required_skills": req_skills,
            "preferred_skills": pref_skills
        }
    else:
        job_keywords = await extract_job_keywords_llm(job.content)
        if job_keywords is None:
            job_keywords = {"required_skills": [], "preferred_skills": []}
    
    refined_resume = {
        "id": req.JDid,
        "additional": {
            "technicalSkills": found_resume_skills
        },
        "workExperience": [
            {
                "description": req.EnhResumeText
            }
        ],
        "summary": "",
    }
    
    score_data = await compute_ats_score(
        refined_resume, 
        job_keywords, 
        [], 
        jd_content=job.content,
        job_id=job.job_id
    )
    
    raw_new_score = score_data.get("overall_score", 0.0)
    final_enh_score = max(raw_new_score, original_score)
    
    # Store the enhanced resume and its score in the database
    from sqlalchemy import update
    async with db._session() as session:
        stmt = (
            update(JobMatch)
            .where(JobMatch.job_id == req.JDid, JobMatch.resume_id == req.ResumeID)
            .values(
                enhanced_resume_content=req.EnhResumeText,
                enhanced_ats_score=final_enh_score
            )
        )
        await session.execute(stmt)
        await session.commit()
    
    # Also update the score_data so the frontend doesn't get confused
    score_data["overall_score"] = final_enh_score
    
    return {
        "status": "success",
        "EnhATS": final_enh_score,
        "OriginalATS": original_score,
        "score_data": score_data,
        "required_skills": job_keywords.get("required_skills", []),
        "preferred_skills": job_keywords.get("preferred_skills", [])
    }

def main():
    """Entry point for the project.scripts console script."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
 
