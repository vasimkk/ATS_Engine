import asyncio
import os
import sys

# Ensure backend is in path so absolute imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database import db
from fastembed import TextEmbedding

async def backfill_embeddings():
    print("Loading SentenceTransformer model (this may take a second)...")
    model = TextEmbedding('BAAI/bge-base-en-v1.5')
    
    print("Fetching jobs from database...")
    jobs = await db.list_jobs()
    
    total_jobs = len(jobs)
    print(f"Found {total_jobs} total jobs.")
    
    processed = 0
    skipped = 0
    
    for job in jobs:
        job_id = job.get("job_id")
        content = job.get("content")
        embedding = job.get("job_embedding")
        
        if not content:
            print(f"Skipping job {job_id}: No content.")
            skipped += 1
            continue
            
        print(f"Precomputing 768d embedding for job {job_id}...")
        vector = list(model.embed([content]))[0]
        
        await db.update_job(job_id, {"job_embedding": vector.tolist()})
        processed += 1
        
    print(f"\n--- Backfill Complete ---")
    print(f"Processed & Saved: {processed}")
    print(f"Skipped: {skipped}")

if __name__ == "__main__":
    asyncio.run(backfill_embeddings())
