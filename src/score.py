"""Phase 2.2/2.3: score candidates and evaluate NDCG/MAP/P@10 against the gold set."""
import numpy as np, pandas as pd
from pathlib import Path
from explore import load_raw
from features import compute_features

# --- scoring weights (tune these) ---
W_TITLE, W_CAREER, W_EXP = 0.45, 0.30, 0.25

def score_features(f):
    if f["honeypot"]:
        return 0.0
    base = W_TITLE*f["title_rel"] + W_CAREER*f["career_rel"] + W_EXP*f["exp_fit"]
    fit = base * f["services"] * f["geo"]
    return fit * f["behav"]

# --- metrics ---
def dcg(rels, k):
    rels = rels[:k]
    return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))

def ndcg(ranked_tiers, k):
    ideal = sorted(ranked_tiers, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(ranked_tiers, k) / idcg if idcg > 0 else 0.0

def average_precision(ranked_tiers, rel_threshold=3):
    rel = [1 if t >= rel_threshold else 0 for t in ranked_tiers]
    total = sum(rel)
    if total == 0: return 0.0
    hits, s = 0, 0.0
    for i, r in enumerate(rel):
        if r:
            hits += 1; s += hits / (i + 1)
    return s / total

if __name__ == "__main__":
    gold = pd.read_csv(Path(__file__).resolve().parent.parent / "gold" / "gold_labels.csv")
    gmap = dict(zip(gold.candidate_id, gold.final_tier))
    gold_ids = set(gold.candidate_id)

    rows = []
    for r in load_raw():
        if r["candidate_id"] in gold_ids:
            f = compute_features(r)
            rows.append({"candidate_id": r["candidate_id"], "score": score_features(f),
                         "tier": gmap[r["candidate_id"]], "title": r["profile"]["current_title"]})
    df = pd.DataFrame(rows)
    # rank: score desc, tie-break candidate_id asc (per spec)
    df = df.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    tiers = df["tier"].tolist()

    n10, n50 = ndcg(tiers, 10), ndcg(tiers, 50)
    ap = average_precision(tiers)
    p10 = sum(1 for t in tiers[:10] if t >= 3) / 10
    composite = 0.50*n10 + 0.30*n50 + 0.15*ap + 0.05*p10

    print(f"NDCG@10 = {n10:.4f}")
    print(f"NDCG@50 = {n50:.4f}")
    print(f"MAP     = {ap:.4f}")
    print(f"P@10    = {p10:.4f}")
    print(f"--- COMPOSITE = {composite:.4f} ---")

    print("\n=== our top 15 (want tier 5s on top, no honeypots) ===")
    print(df.head(15)[["candidate_id","title","tier","score"]].to_string(index=False))
    print("\n=== where the 8 ideal (tier 5) candidates ranked ===")
    print(df[df.tier == 5].assign(rank=lambda x: x.index+1)[["rank","candidate_id","title","score"]].to_string(index=False))