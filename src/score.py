"""Phase 3.3: scorer with semantic blend; evaluate against gold set."""
import numpy as np, pandas as pd
from pathlib import Path
from explore import load_raw
from features import compute_features
from semantic import load_semantic

# weights: content (title-or-semantic rescue) + keyword-career + experience
W_CONTENT, W_CAREER, W_EXP = 0.50, 0.25, 0.25

def score_features(f):
    if f["honeypot"]:
        return 0.0
    content = max(f["title_rel"], f.get("sem", 0.0))   # sem can only rescue, never drag down
    base = W_CONTENT*content + W_CAREER*f["career_rel"] + W_EXP*f["exp_fit"]
    return base * f["services"] * f["geo"] * f["behav"]

def dcg(rels, k):
    return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels[:k]))
def ndcg(tiers, k):
    idcg = dcg(sorted(tiers, reverse=True), k)
    return dcg(tiers, k) / idcg if idcg > 0 else 0.0
def average_precision(tiers, thr=3):
    rel = [1 if t >= thr else 0 for t in tiers]; tot = sum(rel)
    if not tot: return 0.0
    hits = s = 0
    for i, r in enumerate(rel):
        if r: hits += 1; s += hits/(i+1)
    return s/tot

if __name__ == "__main__":
    gold = pd.read_csv(Path(__file__).resolve().parent.parent / "gold" / "gold_labels.csv")
    gmap = dict(zip(gold.candidate_id, gold.final_tier)); gold_ids = set(gold.candidate_id)
    sem_map = load_semantic()

    rows = []
    for r in load_raw():
        if r["candidate_id"] in gold_ids:
            f = compute_features(r); f["sem"] = sem_map.get(r["candidate_id"], 0.0)
            rows.append({"candidate_id": r["candidate_id"], "score": score_features(f),
                         "tier": gmap[r["candidate_id"]], "sem": f["sem"],
                         "title": r["profile"]["current_title"]})
    df = pd.DataFrame(rows).sort_values(["score","candidate_id"], ascending=[False,True]).reset_index(drop=True)
    tiers = df["tier"].tolist()
    n10, n50, ap = ndcg(tiers,10), ndcg(tiers,50), average_precision(tiers)
    p10 = sum(1 for t in tiers[:10] if t>=3)/10
    comp = 0.50*n10 + 0.30*n50 + 0.15*ap + 0.05*p10

    print(f"NDCG@10={n10:.4f}  NDCG@50={n50:.4f}  MAP={ap:.4f}  P@10={p10:.4f}")
    print(f"--- COMPOSITE = {comp:.4f}   (baseline was 0.9370) ---")
    print("\ntop 15:")
    print(df.head(15)[["candidate_id","title","tier","sem","score"]].round(3).to_string(index=False))
    print("\ntier-5 ranks:")
    print(df[df.tier==5].assign(rank=lambda x:x.index+1)[["rank","candidate_id","title","sem"]].to_string(index=False))