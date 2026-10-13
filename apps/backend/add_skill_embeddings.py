import asyncio
from app.database import db

async def add_columns():
    print("Connecting to database...")
    await db.connect()
    
    print("Adding required_skills_embedding and preferred_skills_embedding columns...")
    async with db._pool.acquire() as conn:
        try:
            await conn.execute("ALTER TABLE jobs ADD COLUMN required_skills_embedding FLOAT[]")
            print("Added required_skills_embedding")
        except Exception as e:
            print(e)
            
        try:
            await conn.execute("ALTER TABLE jobs ADD COLUMN preferred_skills_embedding FLOAT[]")
            print("Added preferred_skills_embedding")
        except Exception as e:
            print(e)
            
    print("Done!")
    
asyncio.run(add_columns())
