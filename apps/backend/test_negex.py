import sys
from app.services.extractor import _get_skills_from_text

test_text = "I have no experience with React, but 5 years of Python."
skills = _get_skills_from_text(test_text)

print(f"Extracted skills: {skills}")

assert "React" not in skills, "Negex failed: React was extracted but it should have been negated!"
assert "Python" in skills, "Negex failed: Python was NOT extracted!"

print("Negex test passed successfully!")
