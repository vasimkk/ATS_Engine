import json
from app.services.extractor import extract_job_keywords, extract_resume_entities
from app.services.ats import compute_ats_score, compute_resume_centroid

resume_text = """
MANOJ BODDULA
Python Developer | AI & Software Engineering
+91-8179210860 • boddulamanoj525@gmail.com • LinkedIn • GitHub • Hyderabad, Telangana
CAREER OBJECTIVE
Python Developer with hands-on experience building scalable software solutions, REST APIs, and AI-powered applications. Adept at 
writing clean, maintainable code, working in agile environments, and collaborating across cross-functional teams to deliver high performing software tailored to complex business requirements. Eager to contribute strong problem-solving and software engineering 
capabilities to a fast-paced, innovation-driven engineering team.
EDUCATION
B.Tech in Computer Science (AI & ML) | Vaagdevi Engineering College, Warangal Sep 2021 – Apr 2025
CGPA: 6.73
TECHNICAL SKILLS
Languages: Python (Primary), SQL
Frameworks: FastAPI, LangChain, LangGraph, NumPy, Pandas, BeautifulSoup4
Databases: PostgreSQL (pgvector), MySQL, ChromaDB, FAISS
Dev Tools: Git, GitHub, GitHub Actions (CI/CD), Docker, Streamlit, REST APIs
AI / ML: LLMs, RAG Pipelines, Semantic Search & Vector Embeddings, NLP, OpenAI & HuggingFace APIs, Scikit-learn, 
TensorFlow, Keras, CNN, LangSmith (Observability), Agent Tool Calling
INTERNSHIP EXPERIENCE
AI & ML Intern | Galactix Solutions Pvt Ltd — Hyderabad May 4, 2026 – Present
Claim Radar — AI-Powered Legal-Tech Matching Platform
Python · FastAPI · React · PostgreSQL (pgvector) · ChromaDB · OpenAI API · Playwright · Sentence-Transformers
• Designed a semantic search matching engine using sentence-transformers and ChromaDB/pgvector, running cosine-similarity 
search over dense vector embeddings for real-time claim-to-user matching.
• Built an LLM-based information extraction pipeline (OpenAI API) with structured JSON schema parsing to convert unstructured 
legal text into entities — settlement payouts, filing deadlines, eligibility criteria.
• Developed an automated ETL pipeline using Playwright to crawl JavaScript-rendered legal directories, with text pre-processing 
and tokenization feeding the vector index.
• Modeled and optimized PostgreSQL schemas (SQLAlchemy, Alembic) for vector-search performance, and served ML inference 
via FastAPI endpoints integrated with a React dashboard.
Project Intern | DLRL, DRDO — Hyderabad Jun 2025 – Jul 2025
• Developed a production-grade RAG pipeline in Python using LangChain and ChromaDB, enabling efficient retrieval over large 
document corpora for defense research applications.
"""

bad_jd = """
AI Automation Engineer
Experience : 4.0- 6.0 Years
Job Location - Hyderabad
We are looking for an AI Engineer with a strong foundation in Agentic AI and Natural Language Processing (NLP). The role will involve designing, developing, and deploying intelligent systems using LLMs, multi-agent frameworks, and autonomous agents to solve real-world problems.

Roles & Responsibilities
Agentic AI & Multi-Agent Systems
Design and implement agent-based architectures using CrewAI, LangGraph, AutoGen, LangChain, or similar frameworks
Build multi-agent workflows for task orchestration, autonomous decision-making, and execution
Integrate LLMs with tools, APIs, and memory systems
Required Qualifications
Bachelor’s or Master’s degree in Computer Science, AI, Data Science, or related fields
Strong programming skills in Python
Experience with PyTorch / TensorFlow
Hands-on experience with LLMs, prompt engineering, and fine-tuning
Exposure to Agentic AI / multi-agent systems
Experience with distributed systems and large-scale data processing
Strong problem-solving and collaboration skills
Preferred Skills
Experience with WatsonX, LangChain, AutoGen, LangGraph
Cloud exposure: IBM Cloud, AWS, GCP, Azure
Open-source contributions
Experience with RAG and vector databases
"""

good_jd = """
Python Backend Developer (AI/ML focused)
Experience: 0-2 years
We are looking for a Python Developer to build robust AI-powered applications.

Required Qualifications:
- Strong programming in Python and SQL.
- Experience building REST APIs with FastAPI.
- Hands-on experience with LangChain and RAG pipelines.
- Experience with Vector Databases (pgvector, ChromaDB, FAISS).
- Familiarity with OpenAI APIs and HuggingFace models.

Nice to have:
- Experience with React and Docker.
- Experience with web scraping (Playwright, BeautifulSoup).
- Familiarity with GitHub Actions for CI/CD.
"""

def test_case(jd_content, label):
    print(f"\\n{'='*50}\\nTESTING: {label}\\n{'='*50}")
    found_resume_skills = extract_resume_entities(resume_text)
    job_keywords = extract_job_keywords(jd_content)
    
    found_jd_skills = job_keywords["required_skills"] + job_keywords["preferred_skills"]
    resume_centroid, resume_cov = compute_resume_centroid(found_resume_skills)
    jd_centroid, _ = compute_resume_centroid(found_jd_skills)
    
    refined_resume = {
        "id": "custom-llm-free",
        "additional": {"technicalSkills": found_resume_skills},
        "workExperience": [{"description": resume_text}],
        "summary": "", 
    }
    
    result = compute_ats_score(
        refined_resume=refined_resume,
        job_keywords=job_keywords,
        injectable_keywords=[],
        jd_content=jd_content,
        job_id="test-job",
        precomputed_resume_centroid=resume_centroid,
        precomputed_resume_coverage=resume_cov,
        precomputed_jd_centroid=jd_centroid,
    )
    
    print(json.dumps(result, indent=2))
    print(f"\\nRequired Skills Extracted: {job_keywords['required_skills']}")
    print(f"Preferred Skills Extracted: {job_keywords['preferred_skills']}")

test_case(bad_jd, "KNOWN-BAD MATCH (AI Automation Engineer)")
test_case(good_jd, "KNOWN-GOOD MATCH (Python Backend Developer)")
