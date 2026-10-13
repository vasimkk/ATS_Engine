import asyncio
import sys
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from litellm import aembedding
from dotenv import load_dotenv
import numpy as np

# Add backend to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import db
from app.models import Job, Resume
from sqlalchemy import select

# Load environment variables
load_dotenv()
if "LLM_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]

async def get_openai_embedding(text: str) -> list[float]:
    """Get embeddings using OpenAI's text-embedding-3-small model via litellm"""
    response = await aembedding(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0]["embedding"]

async def main():
    print("Connecting to database...")
    async with db._session() as session:
        # Use the resume you previously tested
        resume_id = "c94b738a1558c9db558d2037fb455ec691f6f4de19d66bbc9ea1135b0ae893d3"
        resume = (await session.execute(select(Resume).where(Resume.resume_id == resume_id))).scalar_one_or_none()
        
        # Grab Job 8393 (The UX Designer role that scored well earlier)
        job_id = 8393
        job = (await session.execute(select(Job).where(Job.job_id == job_id))).scalar_one_or_none()
        
    if not resume or not job:
        print("Error: Could not find the test Resume or Job in the DB.")
        return
        
    print(f"Successfully loaded Resume ({resume_id[:8]}) and Job ({job_id})")
    
    # Extract raw text from Resume JSON
    resume_text = resume.content
    try:
        data = json.loads(resume_text)
        parts = []
        def _walk(obj):
            if isinstance(obj, str): parts.append(obj)
            elif isinstance(obj, list):
                for item in obj: _walk(item)
            elif isinstance(obj, dict):
                for v in obj.values(): _walk(v)
        _walk(data)
        resume_text = " ".join(parts)
    except:
        pass # It's already just a raw string
        
    jd_text = job.content
    
    print("\nCalling OpenAI API to generate embeddings (text-embedding-3-small)...")
    try:
        # Call OpenAI concurrently for speed
        resume_task = get_openai_embedding(resume_text)
        jd_task = get_openai_embedding(jd_text)
        
        resume_emb, jd_emb = await asyncio.gather(resume_task, jd_task)
        
        print(f"Success! Generated vectors of length: {len(resume_emb)}")
        
        # Compare them using Cosine Similarity
        score = cosine_similarity([resume_emb], [jd_emb])[0][0]
        
        print("-" * 50)
        print(f"OPENAI SEMANTIC MATCH SCORE: {score * 100:.2f}%")
        print("-" * 50)
        
    except Exception as e:
        print(f"\n[ERROR] OpenAI API failed: {e}")
        print("Please make sure you have a valid OPENAI_API_KEY set in your .env file!")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
