"""Phase 2.1: per-candidate feature extraction encoding the JD intent."""
import re
from datetime import date
import pandas as pd
from explore import load_raw
from honeypots import check as honeypot_check

REF = date(2026, 6, 1)
SERVICES = {"tcs","infosys","wipro","accenture","cognizant","capgemini","mindtree","hcl","tech mahindra","ltimindtree"}
PHRASES = ["ranking","retriev","recommend","search system","search engine","embedding","vector","relevance",
           "personaliz","semantic","information retrieval","learning to rank","matching","recsys","ndcg",
           "a/b test","experimentation","recommendation system","candidate generation","nearest neighbor"]

def title_relevance(t):
    t = (t or "").lower()
    if any(k in t for k in ["computer vision","speech","robotics","vision engineer"]): return 0.30  # JD-negative
    if any(k in t for k in ["search engineer","recommendation","ranking","nlp engineer","applied scientist"]): return 1.00
    if any(k in t for k in ["machine learning","ml engineer","ai engineer"]): return 0.95
    if "data scientist" in t: return 0.85
    if "research engineer" in t or "research scientist" in t: return 0.70  # research tilt, JD wary
    if "(ml)" in t or "ml)" in t: return 0.80                              # SWE (ML)
    if any(k in t for k in ["data engineer","analytics engineer","backend engineer"]): return 0.50
    if any(k in t for k in ["software engineer","full stack","frontend","devops","cloud engineer",
                            "java developer",".net","mobile","qa engineer"]): return 0.35
    return 0.0  # non-tech

def career_text(rec):
    p = rec["profile"]
    return (p.get("summary","") + " " + " ".join(c.get("description","") for c in rec.get("career_history",[]))).lower()

def career_relevance(text):
    return min(sum(1 for ph in PHRASES if ph in text) / 6.0, 1.0)

def experience_fit(yoe):
    if 5 <= yoe <= 9: return 1.0
    if 4 <= yoe < 5 or 9 < yoe <= 10.5: return 0.8
    if 3 <= yoe < 4 or 10.5 < yoe <= 12: return 0.6
    if yoe < 3: return 0.3
    return 0.45  # >12

def services_factor(rec):
    cos = {c.get("company","").lower() for c in rec.get("career_history",[])}
    return 0.65 if any(any(s in c for s in SERVICES) for c in cos) else 1.0

def geography(rec):
    c = rec["profile"].get("country","")
    if "india" in c.lower(): return 1.0
    return 0.7 if rec["redrob_signals"].get("willing_to_relocate") else 0.5

def days_inactive(s):
    try: y,m,d = map(int, s["last_active_date"].split("-")); return (REF - date(y,m,d)).days
    except: return 9999

def behavioral_modifier(rec):
    s = rec["redrob_signals"]
    di = days_inactive(s)
    recency = 1.0 if di<=30 else 0.9 if di<=90 else 0.7 if di<=180 else 0.45
    resp = s.get("recruiter_response_rate", 0.0)
    resp_f = 0.5 + 0.5*min(resp/0.6, 1.0)            # 0.5 .. 1.0
    icr = s.get("interview_completion_rate", 0.6)
    icr_f = 0.85 + 0.15*icr
    np_ = s.get("notice_period_days", 90)
    np_f = 1.0 if np_<=30 else 0.95 if np_<=60 else 0.9 if np_<=90 else 0.85
    otw = 1.05 if s.get("open_to_work_flag") else 1.0
    m = recency * resp_f * icr_f * np_f * otw
    return max(0.25, min(1.15, m))

THRESH = ["sum_dur_gt_yoe","single_gt_yoe","span_gt_yoe","skill_dur_extreme","expert_zero_months"]
LOGIC  = ["current_dur_mismatch","start_after_end","edu_bad"]
def is_honeypot(rec):
    f,_,_ = honeypot_check(rec)
    return sum(bool(f[c]) for c in THRESH) >= 2 or any(bool(f[c]) for c in LOGIC)

def compute_features(rec):
    text = career_text(rec)
    return {
        "candidate_id": rec["candidate_id"],
        "title_rel": title_relevance(rec["profile"]["current_title"]),
        "career_rel": career_relevance(text),
        "exp_fit": experience_fit(rec["profile"].get("years_of_experience",0)),
        "services": services_factor(rec),
        "geo": geography(rec),
        "behav": behavioral_modifier(rec),
        "honeypot": is_honeypot(rec),
    }

if __name__ == "__main__":
    gold = pd.read_csv(__import__("pathlib").Path(__file__).resolve().parent.parent / "gold" / "gold_labels.csv")
    gold_ids = set(gold["candidate_id"])
    feats = [compute_features(r) for r in load_raw() if r["candidate_id"] in gold_ids]
    fdf = pd.DataFrame(feats).merge(gold[["candidate_id","final_tier"]], on="candidate_id")

    print("=== mean feature value by gold tier (should rise with tier) ===")
    print(fdf.groupby("final_tier")[["title_rel","career_rel","exp_fit","services","geo","behav"]].mean().round(2))
    print("\n=== honeypots caught in gold set (should all be tier 0) ===")
    print(fdf[fdf.honeypot][["candidate_id","final_tier"]].to_string(index=False))