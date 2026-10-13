import sys
import os

# Add backend to path so we can import sentence_transformers if needed, or just load it directly.
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def run_calibration():
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # "Should match" pairs (Synonyms or near-equivalents)
    positive_pairs = [
        ("GCP", "Google Cloud"),
        ("ML", "Machine Learning"),
        ("ReactJS", "React"),
        ("ChromaDB", "Vector Database"),
        ("NextJS", "Next.js"),
        ("Postgres", "PostgreSQL"),
        ("AWS", "Amazon Web Services"),
        ("K8s", "Kubernetes"),
        ("Vue.js", "Vue"),
        ("Node", "Node.js")
    ]
    
    # "Should NOT match" pairs (Related but distinct)
    negative_pairs = [
        ("AWS", "GCP"),
        ("Python", "Java"),
        ("React", "Angular"),
        ("Docker", "Kubernetes"),
        ("SQL", "NoSQL"),
        ("MongoDB", "PostgreSQL"),
        ("HTML", "CSS"),
        ("Frontend", "Backend"),
        ("Data Science", "Software Engineering"),
        ("C++", "C#")
    ]
    
    print("\n--- POSITIVE PAIRS (Should Match) ---")
    pos_scores = []
    for s1, s2 in positive_pairs:
        v1 = model.encode([s1])[0]
        v2 = model.encode([s2])[0]
        score = cosine_similarity([v1], [v2])[0][0]
        pos_scores.append(score)
        print(f"{s1} <-> {s2}: {score:.3f}")
        
    print("\n--- NEGATIVE PAIRS (Should NOT Match) ---")
    neg_scores = []
    for s1, s2 in negative_pairs:
        v1 = model.encode([s1])[0]
        v2 = model.encode([s2])[0]
        score = cosine_similarity([v1], [v2])[0][0]
        neg_scores.append(score)
        print(f"{s1} <-> {s2}: {score:.3f}")
        
    print("\n--- SUMMARY ---")
    print(f"Min Positive Score: {min(pos_scores):.3f}")
    print(f"Max Negative Score: {max(neg_scores):.3f}")
    
    thresholds = [0.65, 0.70, 0.75, 0.80]
    for t in thresholds:
        false_negatives = sum(1 for s in pos_scores if s < t)
        false_positives = sum(1 for s in neg_scores if s >= t)
        print(f"Threshold {t:.2f} -> False Negatives: {false_negatives}/{len(positive_pairs)}, False Positives: {false_positives}/{len(negative_pairs)}")

if __name__ == "__main__":
    run_calibration()
