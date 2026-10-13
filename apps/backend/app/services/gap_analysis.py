import json
import uuid
import logging
from typing import Any
from app.database import db
from sqlalchemy import select
from app.models import GapAnalysis
from app.services.extractor import extract_job_keywords_llm, extract_resume_entities_llm

logger = logging.getLogger(__name__)

async def run_s4_gap_analysis(resume_text: str, jd_text: str, resume_id: str, job_id: str) -> dict[str, Any]:
    """
    MODULE S4 — GAP ANALYSIS
    Extracts required/preferred skills from JD, compares against
    resume skills, builds a ranked gap list with severity and
    bridgeable flag. Writes to gap store.
    """
    
    job_record = await db.get_job(job_id)
    jd_required = []
    jd_preferred = []
    
    if job_record and job_record.get("required_skills") is not None and job_record.get("preferred_skills") is not None:
        jd_required = job_record.get("required_skills", [])
        jd_preferred = job_record.get("preferred_skills", [])
        logger.info(f"[S4] JD skills loaded from DB — required: {len(jd_required)}, preferred: {len(jd_preferred)}")
    else:
        logger.warning(f"[S4] JD skills not in DB for job_id={job_id}, calling LLM extractor...")
        jd_skills = await extract_job_keywords_llm(jd_text)
        logger.info(f"[S4] LLM JD extraction result: {jd_skills}")
        if jd_skills:
            jd_required = jd_skills.get("required_skills", [])
            jd_preferred = jd_skills.get("preferred_skills", [])
        
    jd_required_set = {s.lower().strip() for s in jd_required}
    jd_preferred_set = {s.lower().strip() for s in jd_preferred}
    logger.info(f"[S4] jd_required_set({len(jd_required_set)}): {list(jd_required_set)[:10]}")
    logger.info(f"[S4] jd_preferred_set({len(jd_preferred_set)}): {list(jd_preferred_set)[:10]}")

    # ── Step 2: extract skills from resume ────────────────────
    # In a real system, we might query the Resume table. Here we extract on the fly if needed.
    # We will use the existing extraction, plus an implied skills prompt
    base_resume_skills = await extract_resume_entities_llm(resume_text)
    logger.info(f"[S4] base_resume_skills({len(base_resume_skills)}): {base_resume_skills[:10]}")
    
    # Extract IMPLIED skills
    from litellm import acompletion
    from app.config import settings
    model = settings.llm_model
    
    implied_prompt = f"""List implied skills evidenced by the roles and projects in this resume. 
Return ONLY a valid JSON array of strings. No other text.
RESUME:
{resume_text}
"""
    try:
        api_key = settings.get_effective_api_key()
        api_base = settings.llm_api_base
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": implied_prompt}],
            response_format={"type": "json_object"} if "llama3.2" not in model else None,
            api_key=api_key,
            api_base=api_base
        )
        content = response.choices[0].message.content
        # Basic JSON extraction
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        implied_skills = json.loads(content)
        if isinstance(implied_skills, dict):
            # Sometimes it wraps it in an object like {"skills": [...]}
            for k, v in implied_skills.items():
                if isinstance(v, list):
                    implied_skills = v
                    break
    except Exception as e:
        logger.error(f"Failed to extract implied skills: {e}")
        implied_skills = []
        
    if not isinstance(implied_skills, list):
        implied_skills = []
        
    all_resume_skills = set(s.lower().strip() for s in base_resume_skills) | set(s.lower().strip() for s in implied_skills)
    logger.info(f"[S4] all_resume_skills({len(all_resume_skills)}): {list(all_resume_skills)[:10]}")
    
    # ── Step 3: compute gap set ────────────────────────────────
    missing_required = jd_required_set - all_resume_skills
    missing_preferred = jd_preferred_set - all_resume_skills
    logger.info(f"[S4] missing_required({len(missing_required)}): {list(missing_required)}")
    logger.info(f"[S4] missing_preferred({len(missing_preferred)}): {list(missing_preferred)}")
    
    # ── Step 4: severity rank each gap ────────────────────────
    gap_list = []
    for skill in missing_required:
        # Get original casing if possible
        orig = next((s for s in jd_required if s.lower().strip() == skill), skill)
        gap_list.append({"skill": orig, "severity": "CRITICAL", "source": "required"})
        
    for skill in missing_preferred:
        orig = next((s for s in jd_preferred if s.lower().strip() == skill), skill)
        gap_list.append({"skill": orig, "severity": "PREFERRED", "source": "preferred"})
        
    if not gap_list:
        logger.warning(f"[S4] gap_list is EMPTY — jd_required_set size={len(jd_required_set)}, jd_preferred_set size={len(jd_preferred_set)}, all_resume_skills size={len(all_resume_skills)}. Resume may already cover all JD skills, or JD extraction failed.")
        return {"gap_id": None, "gaps": [], "bridgeable_gaps": []}
        
    # ── Step 5: tag each gap as bridgeable ────────────────────
    bridge_prompt = f"""For each skill in the GAPS list, decide if it is "bridgeable" given the candidate's RESUME.
A gap is bridgeable if the candidate has evidence in their experience that could support adding this skill truthfully (e.g. they know a very similar framework, or they did a project that implies they learned it).
Return ONLY a JSON array of objects with keys: "skill" (string), "bridgeable" (boolean), "rationale" (short string).

RESUME:
{resume_text}

GAPS:
{json.dumps([g["skill"] for g in gap_list])}
"""
    
    bridge_result = []
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": bridge_prompt}],
            response_format={"type": "json_object"} if "llama3.2" not in model else None,
            api_key=api_key,
            api_base=api_base
        )
        content = response.choices[0].message.content
        logger.info(f"Bridge LLM raw response: {content[:500]}")
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        bridge_result = json.loads(content)
        if isinstance(bridge_result, dict):
            for k, v in bridge_result.items():
                if isinstance(v, list):
                    bridge_result = v
                    break
    except Exception as e:
        logger.error(f"Failed to calculate bridgeable gaps: {e}. Defaulting all gaps to bridgeable=True.")
        bridge_result = []
        
    if not isinstance(bridge_result, list):
        logger.warning(f"bridge_result was not a list (type={type(bridge_result)}). Defaulting all gaps to bridgeable=True.")
        bridge_result = []
        
    # Merge bridgeable results into gap_list.
    # If bridge_result is empty (LLM failed), default ALL gaps to bridgeable=True
    # so the enhancement pipeline is never silently blocked.
    fallback_to_all_bridgeable = len(bridge_result) == 0
    for gap in gap_list:
        match = next((b for b in bridge_result if isinstance(b, dict) and b.get("skill", "").lower().strip() == gap["skill"].lower().strip()), None)
        if match:
            gap["bridgeable"] = match.get("bridgeable", True)
            gap["rationale"] = match.get("rationale", "")
        else:
            gap["bridgeable"] = True if fallback_to_all_bridgeable else False
            gap["rationale"] = "Assumed bridgeable (LLM fallback)." if fallback_to_all_bridgeable else "No evidence found."
            
    # ── Step 6: write to gap store ────────────────────────────
    gap_id = str(uuid.uuid4())
    async with db._session() as session:
        # Check if one exists by job_id and user_id (unique constraint)
        stmt = select(GapAnalysis).where(GapAnalysis.job_id == str(job_id), GapAnalysis.user_id == "default_user")
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.resume_id = resume_id
            existing.missing_skills_json = gap_list
            gap_id = existing.id
        else:
            new_gap = GapAnalysis(
                id=gap_id,
                user_id="default_user",
                resume_id=resume_id,
                job_id=str(job_id),
                missing_skills_json=gap_list
            )
            session.add(new_gap)
            
        await session.commit()
        
    bridgeable_gaps = [g for g in gap_list if g.get("bridgeable") == True]
    
    logger.info(f"S4 complete — job: {job_id}, total gaps: {len(gap_list)}, bridgeable: {len(bridgeable_gaps)}")
    
    return {
        "gap_id": gap_id,
        "gaps": gap_list,
        "bridgeable_gaps": bridgeable_gaps
    }
