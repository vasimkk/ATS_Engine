import asyncio
from app.database import db
from app.models import Job, Resume
from sqlalchemy import select

async def test_db():
    try:
        async with db._session() as session:
            res = await session.execute(select(Resume).limit(1))
            resume = res.scalar_one_or_none()
            if not resume:
                print("No resume found")
                return
            
            vec = resume.res_embedding
            print(f"Type of vec: {type(vec)}")
            
            # Now try the query
            stmt = select(Job, Job.job_embedding.cosine_distance(vec).label("distance")).where(Job.job_embedding.is_not(None)).order_by(Job.job_embedding.cosine_distance(vec)).limit(10)
            result = await session.execute(stmt)
            rows = result.all()
            print(f"Found {len(rows)} rows")
            
    except Exception as e:
        import traceback
        print(f"Exception: {e}")
        traceback.print_exc()

asyncio.run(test_db())
