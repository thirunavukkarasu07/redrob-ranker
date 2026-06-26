"""Phase 1.7: build a stratified gold-set labeling sheet."""
import random, re
from datetime import date
from pathlib import Path
import pandas as pd
from explore import load_raw

random.seed(42)
REF = date(2026, 6, 1)
ROOT = Path(__file__).resolve().parent.parent
HONEYPOTS = set((ROOT / "gold" / "honeypots.txt").read_text().split())

AI_CORE = {s.lower() for s in ["LLMs","RAG","Embeddings","FAISS","Pinecone","Milvus","Weaviate","Qdrant",
    "Sentence Transformers","BM25","Semantic Search","Information Retrieval","Learning to Rank","NLP",
    "PyTorch","TensorFlow","Hugging Face Transformers","Fine-tuning LLMs","LoRA","QLoRA","PEFT","MLOps",
    "Kubeflow","MLflow","Deep Learning","Machine Learning","Prompt Engineering","OpenSearch","LangChain","Haystack"]}
PHRASES = ["ranking","retriev","recommend","search system","search engine","embedding","vector","relevance",
           "personaliz","semantic","information retrieval","learning to rank","matching","recsys","ndcg",
           "a/b test","experimentation","recommendation system","candidate generation"]
SERVICES = {"tcs","infosys","wipro","accenture","cognizant","capgemini","mindtree","hcl","tech mahindra"}

def title_cat(t):
    t=(t or "").lower()
    if any(k in t for k in ["machine learning","ml engineer","ai engineer","data scientist","nlp","search engineer",
        "applied scientist","research engineer","deep learning","recommendation","recommender","ranking"]): return "AI/ML"
    if any(k in t for k in ["software engineer","backend","frontend","full stack","devops","cloud engineer",
        "data engineer","analytics engineer","mobile","java developer",".net","qa engineer"]): return "SWE/Data"
    return "Non-tech"

def days_since(s):
    try: y,m,d=map(int,s.split("-")); return (REF-date(y,m,d)).days
    except: return 9999

raw = load_raw()
rows=[]
for r in raw:
    p,s=r["profile"],r["redrob_signals"]
    text=(p.get("summary","")+" "+" ".join(c.get("description","") for c in r.get("career_history",[]))).lower()
    cos={c.get("company","").lower() for c in r.get("career_history",[])}
    rows.append({"candidate_id":r["candidate_id"],"title":p["current_title"],"cat":title_cat(p["current_title"]),
        "yoe":p["years_of_experience"],"industry":p.get("current_industry",""),"country":p.get("country",""),
        "n_ai":sum(1 for sk in r.get("skills",[]) if sk["name"].lower() in AI_CORE),
        "rel":sum(1 for ph in PHRASES if ph in text),
        "services":any(any(x in c for x in SERVICES) for c in cos),
        "resp":s["recruiter_response_rate"],"days_inactive":days_since(s["last_active_date"]),
        "open":s["open_to_work_flag"],"honeypot":r["candidate_id"] in HONEYPOTS,
        "cv_speech":bool(re.search(r"computer vision|speech|robotics|vision engineer",p["current_title"].lower())),
        "summary":(p.get("summary","")[:200]).replace("\n"," ")})
d=pd.DataFrame(rows)

def auto_tier(r):
    if r["honeypot"]: return 0
    if r["cat"]=="Non-tech": return 1 if r["n_ai"]>=5 else 0
    if r["cv_speech"]: return 2
    if r["cat"]=="AI/ML":
        b=4
        if r["services"]: b-=1
        if not (5<=r["yoe"]<=9): b-=1
        if r["resp"]<0.15 or r["days_inactive"]>180: b-=1
        if r["rel"]>=5: b+=1
        return max(1,min(5,b))
    if r["cat"]=="SWE/Data": return 3 if r["rel"]>=3 else (2 if r["n_ai"]>=2 else 1)
    return 1
d["auto_tier"]=d.apply(auto_tier,axis=1)

# strata
def samp(mask,n): 
    sub=d[mask]; return sub.sample(min(n,len(sub)),random_state=42)
parts=[
 samp(d.honeypot,10),
 samp((d.cat=="Non-tech")&(d.n_ai>=7),15),
 samp((d.cat=="AI/ML")&(~d.services)&(d.yoe.between(5,9))&(d.resp>=0.3)&(d.days_inactive<=90),20),
 samp((d.cat=="AI/ML")&(d.services),10),
 samp((d.cat=="AI/ML")&((d.resp<0.15)|(d.days_inactive>180)),10),
 samp(d.cv_speech,10),
 samp((d.cat=="SWE/Data")&(d.rel>=3),15),
 samp((d.cat=="Non-tech")&(d.n_ai<2),10),
 samp((d.cat=="SWE/Data")&(d.rel==0),10),
 samp((d.cat=="AI/ML")&(~d.yoe.between(5,9)),10),
]
gold=pd.concat(parts).drop_duplicates("candidate_id").reset_index(drop=True)
gold["final_tier"]=gold["auto_tier"]   # <-- you will review/edit this column
cols=["candidate_id","title","cat","yoe","industry","country","n_ai","rel","services",
      "resp","days_inactive","open","honeypot","cv_speech","auto_tier","final_tier","summary"]
out=ROOT/"gold"/"to_label.csv"
gold[cols].to_csv(out,index=False)
print(f"Wrote {len(gold)} candidates to {out}")
print("\nauto_tier distribution:")
print(gold["auto_tier"].value_counts().sort_index())
print("\nby category:"); print(gold["cat"].value_counts())