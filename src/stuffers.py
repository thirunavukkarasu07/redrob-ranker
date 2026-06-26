"""Phase 1.5: characterize keyword-stuffer traps (AI skills + non-AI title/career)."""
from collections import Counter
import pandas as pd
from explore import load_raw

# AI/ML/IR skills that look impressive for THIS jd
AI_CORE = {s.lower() for s in [
    "LLMs","RAG","Embeddings","FAISS","Pinecone","Milvus","Weaviate","Qdrant",
    "Sentence Transformers","BM25","Semantic Search","Information Retrieval","Learning to Rank",
    "NLP","PyTorch","TensorFlow","Hugging Face Transformers","Transformers","Fine-tuning LLMs",
    "LoRA","QLoRA","PEFT","MLOps","Kubeflow","MLflow","Deep Learning","Machine Learning",
    "Prompt Engineering","OpenSearch","Elasticsearch","LangChain","Haystack","Recommendation Systems",
    "Neural Networks","BERT","Vector Search","Reinforcement Learning","Feature Engineering","scikit-learn",
]}

# title categories
def title_cat(t):
    t = (t or "").lower()
    if any(k in t for k in ["machine learning","ml engineer","ai engineer","data scientist",
                            "nlp","search engineer","applied scientist","research engineer","deep learning"]):
        return "AI/ML"
    if any(k in t for k in ["software engineer","backend","frontend","full stack","devops",
                            "cloud engineer","data engineer","analytics engineer","mobile",
                            "java developer",".net","qa engineer"]):
        return "SWE/Data"
    return "Non-tech"

raw = load_raw()
freq = Counter()
rows = []
for r in raw:
    sk = [s["name"] for s in r.get("skills", [])]
    freq.update(s.lower() for s in sk)
    n_ai = sum(1 for s in sk if s.lower() in AI_CORE)
    rows.append({"candidate_id": r["candidate_id"],
                 "title": r["profile"]["current_title"],
                 "cat": title_cat(r["profile"]["current_title"]),
                 "yoe": r["profile"]["years_of_experience"],
                 "n_ai": n_ai, "n_skills": len(sk)})
d = pd.DataFrame(rows)

print("=== TOP 40 SKILLS IN POOL (to calibrate the AI set) ===")
for name, c in freq.most_common(40):
    print(f"  {name:30} {c}")

print("\n=== mean AI-skill count by title category ===")
print(d.groupby("cat")["n_ai"].agg(["mean","max","count"]))

print("\n=== KEYWORD STUFFERS: Non-tech title with >=5 AI core skills ===")
stuffers = d[(d.cat == "Non-tech") & (d.n_ai >= 5)]
print(f"count (>=5 AI skills): {len(stuffers)}")
print(f"count (>=7 AI skills): {len(d[(d.cat=='Non-tech') & (d.n_ai>=7)])}")
print("\ntop stuffer titles:")
print(stuffers["title"].value_counts().head(12))
print("\nexamples:")
print(stuffers.sort_values("n_ai", ascending=False)
      [["candidate_id","title","yoe","n_ai","n_skills"]].head(10).to_string(index=False))