"""
Redrob Hybrid Ranker — sandbox demo (HuggingFace Spaces).

Upload a small candidates JSONL/JSON sample (<=100) and the app runs the SAME ranking
logic as the full submission: structured features + contrastive sentence-embeddings
(rescue) + internal-consistency honeypot zeroing + multiplicative behavioural modifier.
Embeddings are computed on the fly for the small sample (the full pipeline precomputes
and caches them). CPU only, no network calls beyond the one-time model download.
"""
import json
import os
import tempfile
from datetime import date
import numpy as np
import pandas as pd
import gradio as gr

REF = date(2026, 6, 1)
W_CONTENT, W_CAREER, W_EXP = 0.50, 0.25, 0.25
SEM_DIV = 0.15  # fixed scaling (full-pool p99.5 of the contrastive score) for sample comparability

SERVICES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
            "mindtree", "hcl", "tech mahindra", "ltimindtree"}
PHRASES = ["ranking", "retriev", "recommend", "search system", "search engine", "embedding",
           "vector", "relevance", "personaliz", "semantic", "information retrieval",
           "learning to rank", "matching", "recsys", "ndcg", "a/b test", "experimentation",
           "recommendation system", "candidate generation", "nearest neighbor"]
PFX = "Represent this sentence for searching relevant passages: "
POS = [
    "Senior engineer who built and deployed embedding-based retrieval and ranking systems for search or "
    "recommendations at a product company, owning relevance quality, offline NDCG and MRR evaluation, A/B testing.",
    "Applied machine learning engineer with production experience in hybrid search, vector databases, "
    "learning-to-rank, and candidate-to-job or product-to-user matching, shipping end-to-end systems to real users.",
    "Machine learning engineer who rebuilt a ranking pipeline from keyword search to dense retrieval and "
    "LLM-based re-ranking, improving engagement, with strong Python and rigorous ranking evaluation.",
]
NEG = [
    "Computer vision engineer working on image classification, object detection, segmentation, or medical imaging.",
    "Speech recognition, text-to-speech, or audio signal processing engineer.",
    "Marketing manager, sales executive, operations manager, HR manager, or accountant managing teams and business KPIs.",
    "IT services consultant at an outsourcing firm delivering generic web and backend CRUD applications in Java or .NET.",
    "Academic researcher publishing machine learning papers without production deployment to real users.",
]

try:
    from sentence_transformers import SentenceTransformer
    print("loading bge-small-en-v1.5 ...")
    MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5"); MODEL.max_seq_length = 256
    POS_EMB = MODEL.encode([PFX + t for t in POS], normalize_embeddings=True)
    NEG_EMB = MODEL.encode([PFX + t for t in NEG], normalize_embeddings=True)
    EMBED_OK = True
except Exception as e:
    print(f"[warn] embeddings unavailable ({e}); running rule-only (sem=0).")
    EMBED_OK = False


# ---------- features (identical to src/features.py) ----------
def title_relevance(t):
    t = (t or "").lower()
    if any(k in t for k in ["computer vision", "speech", "robotics", "vision engineer"]): return 0.30
    if any(k in t for k in ["search engineer", "recommendation", "ranking", "nlp engineer", "applied scientist"]): return 1.00
    if any(k in t for k in ["machine learning", "ml engineer", "ai engineer"]): return 0.95
    if "data scientist" in t: return 0.85
    if "research engineer" in t or "research scientist" in t: return 0.70
    if "(ml)" in t or "ml)" in t: return 0.80
    if any(k in t for k in ["data engineer", "analytics engineer", "backend engineer"]): return 0.50
    if any(k in t for k in ["software engineer", "full stack", "frontend", "devops", "cloud engineer",
                            "java developer", ".net", "mobile", "qa engineer"]): return 0.35
    return 0.0

def doc_text(rec):
    p = rec["profile"]
    parts = [p.get("headline", ""), p.get("summary", "")]
    parts += [c.get("description", "") for c in rec.get("career_history", [])]
    return ". ".join(x for x in parts if x)

def career_relevance(rec):
    text = doc_text(rec).lower()
    return min(sum(1 for ph in PHRASES if ph in text) / 6.0, 1.0)

def experience_fit(yoe):
    if 5 <= yoe <= 9: return 1.0
    if 4 <= yoe < 5 or 9 < yoe <= 10.5: return 0.8
    if 3 <= yoe < 4 or 10.5 < yoe <= 12: return 0.6
    if yoe < 3: return 0.3
    return 0.45

def services_factor(rec):
    cos = {c.get("company", "").lower() for c in rec.get("career_history", [])}
    return 0.65 if any(any(s in c for s in SERVICES) for c in cos) else 1.0

def geography(rec):
    c = rec["profile"].get("country", "")
    if "india" in c.lower(): return 1.0
    return 0.7 if rec["redrob_signals"].get("willing_to_relocate") else 0.5

def pdate(s):
    if not s: return None
    try:
        y, m, d = map(int, s.split("-")); return date(y, m, d)
    except Exception:
        return None

def mbetween(a, b): return (b.year - a.year) * 12 + (b.month - a.month)

def days_inactive(s):
    d = pdate(s.get("last_active_date"))
    return (REF - d).days if d else 9999

def behavioral_modifier(rec):
    s = rec["redrob_signals"]; di = days_inactive(s)
    recency = 1.0 if di <= 30 else 0.9 if di <= 90 else 0.7 if di <= 180 else 0.45
    resp_f = 0.5 + 0.5 * min(s.get("recruiter_response_rate", 0.0) / 0.6, 1.0)
    icr_f = 0.85 + 0.15 * s.get("interview_completion_rate", 0.6)
    np_ = s.get("notice_period_days", 90)
    np_f = 1.0 if np_ <= 30 else 0.95 if np_ <= 60 else 0.9 if np_ <= 90 else 0.85
    otw = 1.05 if s.get("open_to_work_flag") else 1.0
    return max(0.25, min(1.15, recency * resp_f * icr_f * np_f * otw))

def is_honeypot(rec):
    yoe = rec["profile"].get("years_of_experience") or 0; ym = yoe * 12
    ch = rec.get("career_history", []) or []; sk = rec.get("skills", []) or []; ed = rec.get("education", []) or []
    durs = [c.get("duration_months", 0) or 0 for c in ch]
    sum_dur = sum(durs) > ym + 18
    single = any(d > ym + 6 for d in durs)
    starts = [pdate(c.get("start_date")) for c in ch if pdate(c.get("start_date"))]
    ends = [pdate(c.get("end_date")) or REF for c in ch]
    span = (mbetween(min(starts), max(ends)) > ym + 24) if starts else False
    cur = sae = False
    for c in ch:
        s0 = pdate(c.get("start_date"))
        if c.get("is_current") and s0 and abs(mbetween(s0, REF) - (c.get("duration_months") or 0)) > 3: cur = True
        e0 = pdate(c.get("end_date"))
        if s0 and e0 and s0 > e0: sae = True
    skill_ext = sum(1 for s in sk if (s.get("duration_months") or 0) > ym + 24) >= 2
    exp_zero = sum(1 for s in sk if s.get("proficiency") in ("expert", "advanced") and (s.get("duration_months") or 0) == 0) >= 2
    edu_bad = any((e.get("end_year") or 0) < (e.get("start_year") or 0) for e in ed)
    n_thresh = sum([sum_dur, single, span, skill_ext, exp_zero]); n_logic = sum([cur, sae, edu_bad])
    return n_thresh >= 2 or n_logic >= 1


def reasoning(rec, sem, career_rel, svc):
    p, s = rec["profile"], rec["redrob_signals"]; di = days_inactive(s); yoe = p["years_of_experience"]
    parts = [f"{p['current_title']} with {yoe:.1f}y experience at {p['current_company']} ({p['current_industry']})"]
    if sem >= 0.75 or career_rel >= 0.5:
        parts.append("career history shows hands-on search / ranking / recommendation work")
    elif title_relevance(p["current_title"]) >= 0.85:
        parts.append("applied ML/AI background relevant to the retrieval-and-ranking mandate")
    avail = (["open to work"] if s.get("open_to_work_flag") else []) + \
            [f"responds to {s['recruiter_response_rate']:.0%} of recruiters", f"last active {di}d ago"]
    parts.append(", ".join(avail))
    concerns = []
    if yoe < 5: concerns.append(f"under the 5-9y band ({yoe:.1f}y)")
    elif yoe > 9: concerns.append(f"above the 5-9y band ({yoe:.1f}y)")
    if svc < 1.0: concerns.append("services-firm background")
    if "india" not in p["country"].lower(): concerns.append(f"based in {p['country']} (relocation/visa)")
    if s.get("notice_period_days", 0) > 90: concerns.append(f"{s['notice_period_days']}d notice")
    if di > 120: concerns.append(f"limited recent activity ({di}d)")
    txt = "; ".join(parts)
    return (txt + (". Watch: " + ", ".join(concerns) if concerns else "") + ".")


def parse(content):
    content = content.strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(ln) for ln in content.splitlines() if ln.strip()]


def rank(file):
    if file is None:
        return pd.DataFrame(), None
    with open(file.name, "r", encoding="utf-8") as f:
        recs = parse(f.read())
    recs = recs[:100]
    texts = [doc_text(r) for r in recs]
    if EMBED_OK:
        doc_emb = MODEL.encode(texts, normalize_embeddings=True)
        contrast = (doc_emb @ POS_EMB.T).max(1) - (doc_emb @ NEG_EMB.T).max(1)
        sem_all = np.clip(contrast / SEM_DIV, 0, 1)
    else:
        sem_all = np.zeros(len(recs))

    rows = []
    for r, sem in zip(recs, sem_all):
        if is_honeypot(r):
            score = 0.0
        else:
            cr = career_relevance(r); svc = services_factor(r)
            content = max(title_relevance(r["profile"]["current_title"]), float(sem))
            base = W_CONTENT * content + W_CAREER * cr + W_EXP * experience_fit(r["profile"]["years_of_experience"])
            score = base * svc * geography(r) * behavioral_modifier(r)
        cr = career_relevance(r); svc = services_factor(r)
        rows.append({"candidate_id": r["candidate_id"], "title": r["profile"]["current_title"],
                     "score": round(float(score), 4),
                     "reasoning": reasoning(r, float(sem), cr, svc)})
    df = pd.DataFrame(rows).sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    out = os.path.join(tempfile.gettempdir(), "ranked_sample.csv")
    df[["candidate_id", "rank", "score", "reasoning"]].to_csv(out, index=False)
    return df, out


with gr.Blocks(title="Redrob Hybrid Ranker") as demo:
    gr.Markdown(
        "# Redrob Hybrid Candidate Ranker — sandbox\n"
        "Upload a small candidates sample (`.jsonl` or `.json`, ≤100) for the *Senior AI Engineer* JD. "
        "Runs the **full hybrid** live: structured features (title / career-prose / experience / "
        "product-vs-services / geography) + **contrastive sentence-embeddings** (retrieval/ranking "
        "anchors minus CV/speech/non-tech/services/research anchors, used as a rescue signal) + "
        "**internal-consistency honeypot zeroing** + a **behavioural availability modifier**, with "
        "fact-grounded reasoning. Honeypots score 0; keyword-stuffers and inactive 'ghosts' are "
        "down-weighted; plain-language builders are rescued by embeddings. (Embeddings are computed "
        "on the fly for the small sample; the full 100K pipeline precomputes and caches them.)"
    )
    inp = gr.File(label="candidates sample (.jsonl / .json)", file_types=[".jsonl", ".json"])
    btn = gr.Button("Rank", variant="primary")
    out_df = gr.Dataframe(label="Ranked candidates", wrap=True)
    out_file = gr.File(label="Download ranked CSV")
    btn.click(rank, inputs=inp, outputs=[out_df, out_file])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
