"""Full-pool A/B: rule-baseline vs hybrid (sem-rescue). WHO does the embedding rescue?"""
import pandas as pd
from explore import load_raw
from features import compute_features
from semantic import load_semantic

W_CONTENT, W_CAREER, W_EXP = 0.50, 0.25, 0.25
def score(f, content):
    if f["honeypot"]: return 0.0
    return (W_CONTENT*content + W_CAREER*f["career_rel"] + W_EXP*f["exp_fit"]) \
           * f["services"] * f["geo"] * f["behav"]

sem = load_semantic()
rows = []
for r in load_raw():
    f = compute_features(r); s = sem.get(r["candidate_id"], 0.0); p = r["profile"]
    rows.append({"candidate_id": r["candidate_id"], "title": p["current_title"],
                 "title_rel": f["title_rel"], "sem": s, "career_rel": f["career_rel"],
                 "yoe": p["years_of_experience"], "industry": p.get("current_industry", ""),
                 "base": score(f, f["title_rel"]), "hyb": score(f, max(f["title_rel"], s)),
                 "summary": p.get("summary", "")[:170].replace("\n", " ")})
df = pd.DataFrame(rows)

def top100(col): return set(df.sort_values([col, "candidate_id"], ascending=[False, True]).head(100).candidate_id)
b, h = top100("base"), top100("hyb")
print(f"overlap: {len(b & h)} / 100   rescued by sem: {len(h - b)}   dropped: {len(b - h)}")

print("\n===== RESCUED by embeddings — are these GENUINE builders? =====")
for _, x in df[df.candidate_id.isin(h - b)].sort_values("hyb", ascending=False).iterrows():
    print(f"\n{x['candidate_id']} | {x['title']} | title_rel={x['title_rel']:.2f} "
          f"sem={x['sem']:.2f} career_rel={x['career_rel']:.2f} | {x['yoe']}y {x['industry']}")
    print(f"   {x['summary']}")

print("\n===== titles DROPPED from baseline top-100 =====")
print(df[df.candidate_id.isin(b - h)].title.value_counts().head(10))