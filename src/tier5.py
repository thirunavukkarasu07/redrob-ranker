"""Phase 1.6: find plain-language Tier-5s via career-history prose relevance."""
import re
import pandas as pd
from explore import load_raw

def title_cat(t):
    t = (t or "").lower()
    if any(k in t for k in ["machine learning","ml engineer","ai engineer","data scientist","nlp",
                            "search engineer","applied scientist","research engineer","deep learning",
                            "recommendation","recommender","ranking"]):
        return "AI/ML"
    if any(k in t for k in ["software engineer","backend","frontend","full stack","devops","cloud engineer",
                            "data engineer","analytics engineer","mobile","java developer",".net","qa engineer"]):
        return "SWE/Data"
    return "Non-tech"

# JD-core concepts (retrieval / ranking / search / recsys / eval) in plain language
PHRASES = ["ranking","rank ","retrieval","retriev","recommend","recommender","search system","search engine",
           "embedding","vector","relevance","personaliz","semantic","information retrieval","learning to rank",
           "matching","nearest neighbor","recsys","ndcg","a/b test","experimentation","evaluation framework",
           "recommendation system","feature store","candidate generation"]

SERVICES = {"tcs","infosys","wipro","accenture","cognizant","capgemini","mindtree","hcl","tech mahindra","ltimindtree"}

raw = load_raw()
rows = []
for r in raw:
    p = r["profile"]
    text = (p.get("summary","") + " " + " ".join(c.get("description","") for c in r.get("career_history",[]))).lower()
    rel = sum(1 for ph in PHRASES if ph in text)
    companies = {c.get("company","").lower() for c in r.get("career_history",[])}
    in_services = any(any(s in comp for s in SERVICES) for comp in companies)
    rows.append({"candidate_id": r["candidate_id"], "title": p["current_title"],
                 "cat": title_cat(p["current_title"]), "yoe": p["years_of_experience"],
                 "industry": p.get("current_industry",""), "rel": rel, "services": in_services})
d = pd.DataFrame(rows)

print("=== mean career-prose relevance by title category ===")
print(d.groupby("cat")["rel"].agg(["mean","max","count"]))

print("\n=== HIDDEN GEMS: SWE/Data or Non-tech title but HIGH career relevance (>=5) ===")
gems = d[(d.cat != "AI/ML") & (d.rel >= 5)].sort_values("rel", ascending=False)
print(f"count: {len(gems)}  (these are candidates pure keyword/title matching would miss)")
print(gems[["candidate_id","title","cat","yoe","industry","rel","services"]].head(12).to_string(index=False))

print("\n=== for contrast: AI/ML-titled relevance distribution ===")
print(d[d.cat=='AI/ML']["rel"].describe()[["mean","50%","max"]])