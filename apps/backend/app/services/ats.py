"""ATS score computation utilities (Stateless).

Calculates an ATS-style breakdown score from already-processed resume and job data:
  - keyword_match: final keyword match % 
  - skills_coverage: overlap between resume technical skills and JD required skills
  - section_completeness: presence of essential resume sections
  - domain_score: whole-text semantic similarity

The overall_score is a weighted composite.
"""

import logging
import re
from typing import Any
from thefuzz import fuzz

logger = logging.getLogger(__name__)

# Patterns to detect resume section headings
_SECTION_PATTERNS = {
    "summary": ["summary", "objective", "profile", "about"],
    "experience": ["experience", "work history", "employment"],
    "education": ["education", "academic", "degree"],
    "skills": ["skills", "technologies", "competencies", "technical"],
}

def _extract_all_text(data: dict[str, Any]) -> str:
    """Flatten all string values from a resume dict into a single text block."""
    parts: list[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)

    _walk(data)
    return " ".join(parts)


def _normalize_skill(skill: str) -> str:
    """Removes spaces and non-alphanumeric chars for matching."""
    return re.sub(r'[^a-z0-9]', '', skill.lower())

def _keyword_in_text(keyword: str, text_lower: str) -> bool:
    """Match against text handling punctuation and word boundaries properly."""
    keyword_norm = _normalize_skill(keyword)
    if not keyword_norm:
        return False
        
    # Check for single word fuzzy match
    if len(keyword.strip().split()) == 1:
        words = re.findall(r'[a-z0-9]+', text_lower)
        for w in words:
            if fuzz.ratio(keyword_norm, w) > 85:
                return True
            
    # Use regex word boundaries instead of spaces to handle punctuation correctly
    # Escape the keyword, but allow flexible whitespace internally
    escaped_keyword = re.escape(keyword.lower())
    # Replace escaped spaces with a pattern that allows any whitespace
    escaped_keyword = escaped_keyword.replace(r'\ ', r'\s+')
    
    if re.search(rf'\b{escaped_keyword}\b', text_lower):
        return True
        
    return False


ACRONYM_DICT = {
    "ml": "machine learning",
    "gcp": "google cloud",
    "aws": "amazon web services",
    "k8s": "kubernetes",
    "ci/cd": "continuous integration",
    "nlp": "natural language processing",
    "llm": "large language model",
    "sql": "structured query language",
    "ai": "artificial intelligence",
    "ui": "user interface",
    "ux": "user experience",
    "qa": "quality assurance",
    "api": "application programming interface",
    "db": "database",
}

async def embedding_skill_match_batch(jd_skills: list[str], resume_skills: list[str]) -> dict[str, str]:
    """Batch encode skills and return a mapping of JD skill -> Resume skill for high confidence semantic matches."""
    if not jd_skills or not resume_skills:
        return {}
        
    try:
        from app.main import _SENTENCE_MODEL
        if _SENTENCE_MODEL is None:
            return {}
            
        from sklearn.metrics.pairwise import cosine_similarity
        import asyncio
        loop = asyncio.get_running_loop()
        
        # Batch encode asynchronously
        jd_vectors = await loop.run_in_executor(None, _SENTENCE_MODEL.encode, jd_skills)
        resume_vectors = await loop.run_in_executor(None, _SENTENCE_MODEL.encode, resume_skills)
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(jd_vectors, resume_vectors)
        
        semantic_matches = {}
        for i, jd_skill in enumerate(jd_skills):
            # Find the best matching resume skill for this JD skill
            best_idx = sim_matrix[i].argmax()
            best_score = sim_matrix[i][best_idx]
            
            # Threshold based on calibration results (ReactJS/React = 0.89)
            # Frontend/Backend = 0.815. So 0.85 safely separates them.
            if best_score >= 0.85:
                semantic_matches[jd_skill] = resume_skills[best_idx]
                
        return semantic_matches
    except Exception as e:
        logger.error(f"Semantic match batch failed: {e}")
        return {}


async def _compute_skills_coverage(
    resume: dict[str, Any],
    job_keywords: dict[str, Any]
) -> tuple[float | None, list[dict[str, str]]]:
    """Return skills coverage score (0-100) and detailed skill matching breakdown."""
    required = job_keywords.get("required_skills", [])
    preferred = job_keywords.get("preferred_skills", [])
    jd_skills: list[str] = required + preferred

    if not jd_skills:
        return 0.0, []

    resume_skills: list[str] = (
        resume.get("additional", {}).get("technicalSkills", []) or []
    )
    resume_text = _extract_all_text(resume).lower()
    
    resume_skills_norm = [_normalize_skill(s) for s in resume_skills if isinstance(s, str)]
    
    # Pre-compute semantic matrix for unmatched skills to avoid per-pair encoding
    semantic_matches = await embedding_skill_match_batch(jd_skills, resume_skills)

    total_weight = 0.0
    matched_weight = 0.0
    
    # Store detailed matching results
    match_details = []
    
    for skill in jd_skills:
        if not isinstance(skill, str):
            continue
            
        skill_norm = _normalize_skill(skill)
        skill_lower = skill.lower()
        
        weight = 3.0 if skill in required else 1.0
        req_level = "required" if skill in required else "preferred"
        total_weight += weight
        
        matched = False
        match_type = "missing"
        
        # 1. Exact/Fuzzy Match (TheFuzz)
        if len(skill_norm) >= 4:
            for r_skill in resume_skills_norm:
                if fuzz.ratio(skill_norm, r_skill) > 85:
                    matched = True
                    match_type = "exact"
                    break
        else:
            if skill_norm in resume_skills_norm:
                matched = True
                match_type = "exact"
                
        # 2. Free-text Keyword Match
        if not matched and _keyword_in_text(skill, resume_text):
            matched = True
            match_type = "exact"
            
        # 3. Acronym Dictionary Match
        if not matched:
            # Check if JD skill is an acronym, and its full phrase is in the resume
            mapped_full = ACRONYM_DICT.get(skill_lower)
            if mapped_full and _keyword_in_text(mapped_full, resume_text):
                matched = True
                match_type = "acronym"
                
            # Check if JD skill is a full phrase, and its acronym is in the resume
            if not matched:
                for acronym, full_phrase in ACRONYM_DICT.items():
                    # Handle plurals/substrings (e.g. 'large language models' vs 'large language model')
                    if full_phrase in skill_lower or skill_lower in full_phrase:
                        if _keyword_in_text(acronym, resume_text):
                            matched = True
                            match_type = "acronym"
                            break
                    
        # 4. Semantic Similarity Match (High Threshold)
        if not matched:
            if skill in semantic_matches:
                matched = True
                match_type = "semantic"
                
        if matched:
            matched_weight += weight
            match_details.append({"skill": skill, "status": "matched", "type": match_type, "requirement_level": req_level})
        else:
            match_details.append({"skill": skill, "status": "missing", "type": "missing", "requirement_level": req_level})

    score = min(100.0, (matched_weight / total_weight) * 100.0) if total_weight > 0 else 0.0
    return score, match_details


def _compute_section_completeness(resume: dict[str, Any]) -> float:
    """Return section completeness score (0–100)."""
    found = 0

    if resume.get("summary"):
        found += 1
    if resume.get("workExperience"):
        found += 1
    if resume.get("education"):
        found += 1
    skills = resume.get("additional", {}).get("technicalSkills", [])
    if skills:
        found += 1

    if found == 0:
        import re
        text = _extract_all_text(resume).lower()
        for patterns in _SECTION_PATTERNS.values():
            if any(re.search(rf'\b{re.escape(p)}\b', text) for p in patterns):
                found += 1

    total = len(_SECTION_PATTERNS)
    return (found / total) * 100


async def _generate_deep_insights_llm(
    resume_text: str,
    jd_text: str,
    missing_keywords: list[str]
) -> tuple[float, list[str]]:
    """Phase 4: F5 Deep LLM Analysis to evaluate the candidate and generate insights."""
    prompt = f"""You are an expert ATS (Applicant Tracking System) reviewer. 
Analyze the candidate's resume against the Job Description.

First, assign an 'LLM ATS Score' from 0 to 100 based on how well their actual experience and skills match the job domain.
Second, provide exactly 3 concise, highly actionable recommendations on how they can improve their resume for this role.

Missing Key Skills Extracted: {", ".join(missing_keywords[:5]) if missing_keywords else "None!"}

Job Description: {jd_text[:1000]}...
Resume: {resume_text[:1000]}...

Return ONLY a valid JSON object with two keys: "score" (a number 0-100) and "recommendations" (an array of 3 strings). No markdown, no text.
Example: {{"score": 85.0, "recommendations": ["Add 'AWS' and 'Docker' to your skills section to improve your keyword match.", "Quantify your backend experience with specific metrics to improve domain relevance.", "Your experience perfectly matches the requirement."]}}
"""
    try:
        from litellm import acompletion
        import json
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        api_key = os.environ.get("LLM_API_KEY", None)
        api_base = os.environ.get("LLM_ENDPOINT", "https://api.openai.com/v1")
        
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            api_base=api_base
        )
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        data = json.loads(content.strip())
        score = float(data.get("score", 70.0))
        tips = data.get("recommendations", [])
        if isinstance(tips, list):
            tips = [str(t) for t in tips[:3]]
        else:
            tips = ["Improve keyword matching by reviewing the JD.", "Ensure your experience aligns with the role.", "Add missing technical skills."]
            
        return score, tips
    except Exception as e:
        logger.error(f"Deep LLM Phase failed: {e}")
        return 70.0, ["Focus on missing keywords.", "Review domain alignment.", "Check experience requirements."]

async def _compute_domain_relevance(
    resume_text: str, 
    jd_text: str, 
    precomputed_jd_vector: list[float] | None = None
) -> float | None:
    """Calculate the cosine similarity between the full Resume text and JD text using the global embedding model."""
    if not resume_text or not jd_text:
        return 0.0
        
    try:
        from app.main import _SENTENCE_MODEL
        if _SENTENCE_MODEL is None:
            logger.warning("SentenceTransformer model is not loaded. Skipping domain relevance.")
            return None
            
        from sklearn.metrics.pairwise import cosine_similarity
        import asyncio
        loop = asyncio.get_running_loop()
        
        res = await loop.run_in_executor(None, _SENTENCE_MODEL.encode, resume_text)
        resume_vector = res.tolist()
        
        if precomputed_jd_vector is not None:
            import numpy as np
            jd_vector = np.array(precomputed_jd_vector)
        else:
            res_jd = await loop.run_in_executor(None, _SENTENCE_MODEL.encode, jd_text)
            jd_vector = res_jd.tolist()
        
        score = cosine_similarity([resume_vector], [jd_vector])[0][0]
        return max(0.0, float(score) * 100)
    except Exception as e:
        logger.error(f"Domain Relevance Math Error: {e}")
        return None

from datetime import datetime
from dateutil import parser as date_parser

def _extract_experience_years(resume_text: str) -> float:
    """Extract deterministic YOE using interval merging on the Experience section."""
    import re
    text_lower = resume_text.lower()
    
    # 1. Isolate the Experience section
    lines = resume_text.split('\n')
    exp_start_line = -1
    for i, line in enumerate(lines):
        line_clean = line.strip().lower()
        if any(re.search(rf'\b{re.escape(p)}\b', line_clean) for p in _SECTION_PATTERNS.get("experience", [])) and len(line_clean) < 40:
            exp_start_line = i
            break
            
    exp_section = resume_text
    if exp_start_line != -1:
        next_idx = len(lines)
        for i in range(exp_start_line + 1, len(lines)):
            line_clean = lines[i].strip().lower()
            is_header = False
            for section, patterns in _SECTION_PATTERNS.items():
                if section == "experience":
                    continue
                if any(re.search(rf'\b{re.escape(p)}\b', line_clean) for p in patterns) and len(line_clean) < 40:
                    is_header = True
                    break
            if is_header:
                next_idx = i
                break
        exp_section = '\n'.join(lines[exp_start_line:next_idx])

    # 2. Extract date ranges
    date_regex = r'([a-zA-Z]+\s+\d{4}|\d{4})\s*[-–—to]+\s*([a-zA-Z]+\s+\d{4}|\d{4}|present|current|now)'
    date_ranges = re.findall(date_regex, exp_section, re.IGNORECASE)
    
    if not date_ranges:
        return 0.0
        
    intervals = []
    for start_str, end_str in date_ranges:
        try:
            start_date = date_parser.parse(start_str, default=datetime(2020, 1, 1))
            if end_str.lower() in ["present", "current", "now"]:
                end_date = datetime.now()
            else:
                end_date = date_parser.parse(end_str, default=datetime(2020, 1, 1))
            
            if start_date <= end_date:
                intervals.append([start_date, end_date])
        except Exception:
            continue
            
    if not intervals:
        return 0.0
        
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
            
    total_months = 0
    for start, end in merged:
        months = (end.year - start.year) * 12 + (end.month - start.month)
        total_months += max(0, months)
        
    return round(total_months / 12.0, 1)

def _extract_jd_min_years(jd_text: str) -> float:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', jd_text)
    exp_sentences = [s for s in sentences if 'experience' in s.lower()]
    pattern = r'(\d+(?:\.\d+)?)(?:\s*[-–—]\s*\d+(?:\.\d+)?)?\+?\s*years?'
    for s in exp_sentences:
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    match = re.search(pattern, jd_text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0

def _compute_yoe_penalty(resume_yoe: float, jd_yoe: float, job_id: str = "Unknown", resume_id: str = "Unknown") -> float:
    if jd_yoe == 0.0:
        return 0.0
    if resume_yoe == 0.0:
        logger.warning(f"[UNVERIFIED EXPERIENCE] Applied 40-point flat penalty for Job {job_id} / Resume {resume_id} (Parsed 0.0 YOE but JD requires {jd_yoe} yrs).")
        return 40.0
    if resume_yoe >= jd_yoe:
        return 0.0
    gap = jd_yoe - resume_yoe
    percent_miss = gap / jd_yoe
    penalty = (percent_miss / 0.5) * 100.0
    return min(100.0, penalty)

async def compute_ats_score(
    refined_resume: dict[str, Any],
    job_keywords: dict[str, Any],
    injectable_keywords: list[str],
    jd_content: str = "",
    job_id: str = None,
    precomputed_jd_vector: list[float] | None = None,
    precomputed_domain_score: float | None = None,
) -> dict[str, Any]:
    """Compute the ATS score breakdown dict."""
    resume_text = _extract_all_text(refined_resume)
    jd_text = jd_content if jd_content else " ".join(job_keywords.get("required_skills", []))
    
    resume_yoe = _extract_experience_years(resume_text)
    jd_yoe = _extract_jd_min_years(jd_text)
    yoe_penalty = _compute_yoe_penalty(resume_yoe, jd_yoe, job_id=job_id, resume_id=refined_resume.get("id", "Unknown"))
    exp_fit_score = max(0.0, 100.0 - yoe_penalty)
    
    sec_score = _compute_section_completeness(refined_resume)
    
    # Calculate Domain Relevance unconditionally (No Short Circuit)
    if precomputed_domain_score is not None:
        domain_score = precomputed_domain_score
    else:
        domain_score = await _compute_domain_relevance(resume_text, jd_text, precomputed_jd_vector)
    
    # Calculate Skill Match unconditionally
    sk_score, match_details = await _compute_skills_coverage(refined_resume, job_keywords)
    actual_missing = [m["skill"] for m in match_details if m["status"] == "missing"]
    missing_required = [m["skill"] for m in match_details if m["status"] == "missing" and m.get("requirement_level") == "required"]
    missing_preferred = [m["skill"] for m in match_details if m["status"] == "missing" and m.get("requirement_level") == "preferred"]

    domain_computed = domain_score is not None
    if not domain_computed:
        domain_score = 0.0

    # 1. F1: Semantic Vector Similarity (50% Weight) - Linear pass-through
    f1 = domain_score

    # 2. F2: Skill Overlap
    f2 = sk_score if sk_score is not None else 0.0

    # 3. F3: Experience Fit (0% Weight - Removed)
    f3 = exp_fit_score

    # 4. F4: Deep LLM Phase (Disabled for testing)
    f4 = 0.0
    recommendations = []

    # Cosine > Skills > Experience
    overall = (f1 * 0.60) + (f2 * 0.30) + (f3 * 0.10)
    overall_score = round(min(100.0, max(0.0, overall)), 1)

    return {
        "overall_score": overall_score,
        "missing_keywords": actual_missing,
        "missing_required_skills": missing_required,
        "missing_preferred_skills": missing_preferred,
        "sub_scores": {
            "skills_coverage": round(sk_score if sk_score is not None else 0.0, 1),
            "domain_score": round(domain_score, 1) if domain_computed else 0.0,
            "experience_fit": round(exp_fit_score, 1),
            "section_completeness": round(sec_score, 1)
        },
        "match_details": match_details,
        "injectable_keywords": injectable_keywords,
        "recommendations": recommendations,
    }
