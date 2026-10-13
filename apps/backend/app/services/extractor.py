import logging

logger = logging.getLogger(__name__)

async def extract_resume_entities_llm(resume_text: str) -> list[str]:
    """Extracts technical and soft skills from a resume using an LLM."""
    if not resume_text:
        return []
        
    prompt = f"""You are an expert technical recruiter. 
First, identify the core technical domain of the candidate (e.g., Frontend Engineering, Data Science, AI/ML, DevSecOps).
Then, strictly based on that specific domain, extract all relevant technical and soft skills from the following resume.
Ignore generic filler words or skills entirely outside this domain.
Return ONLY a JSON array of strings containing the skills. No markdown, no explanations, just the JSON array. Example: ["Python", "Machine Learning", "Communication"]

Resume:
{resume_text}
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
            
        skills = json.loads(content.strip())
        if isinstance(skills, list):
            return [str(s) for s in skills]
        return []
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return []

async def extract_job_keywords_llm(jd_text: str) -> dict[str, list[str]]:
    """Extracts required and preferred technical/soft skills from a JD using an LLM."""
    if not jd_text:
        return {"required_skills": [], "preferred_skills": []}
        
    prompt = f"""You are an expert technical recruiter. 
First, identify the core technical domain of this job (e.g., Frontend Engineering, Data Science, AI/ML, DevSecOps).
Then, strictly based on that specific domain, extract the relevant required and preferred skills.
Ignore generic filler words or skills entirely outside this domain.
Return ONLY a valid JSON object with two keys: "required_skills" and "preferred_skills". Both should contain arrays of strings. No markdown, no explanations.
Example: {{"required_skills": ["Python", "AWS"], "preferred_skills": ["Docker", "Communication"]}}

Job Description:
{jd_text}
"""
    try:
        from litellm import acompletion
        import json
        from app.config import settings
        
        model = settings.llm_model
        api_key = settings.get_effective_api_key()
        api_base = settings.llm_api_base
        
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
            
        result = json.loads(content.strip())
        return {
            "required_skills": result.get("required_skills", []),
            "preferred_skills": result.get("preferred_skills", [])
        }
    except Exception as e:
        logger.error(f"LLM JD extraction failed: {e}")
        return None
