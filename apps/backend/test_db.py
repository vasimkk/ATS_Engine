import asyncio
from app.database import db

async def test():
    jobs = await db.list_jobs()
    if jobs:
        print("Type of job_embedding:", type(jobs[0].get("job_embedding")))
        print("Length of job_embedding:", len(jobs[0].get("job_embedding")) if jobs[0].get("job_embedding") else "None")
        print("First 5 elements:", jobs[0].get("job_embedding")[:5])

asyncio.run(test())
