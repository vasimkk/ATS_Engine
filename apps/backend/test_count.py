from app.database import db
from sqlalchemy import text

with db._sync() as s: 
    print(f'Total skills: {s.execute(text("SELECT COUNT(id) FROM esco_skills")).scalar()}')
    print(f'Skills with alt labels: {s.execute(text("SELECT COUNT(id) FROM esco_skills WHERE alt_labels != \'[]\'")).scalar()}')
