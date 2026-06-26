"""Contrastive semantic fit: positive-anchor similarity minus negative-anchor similarity."""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent; ART = ROOT / "artifacts"

def _raw():
    emb = np.load(ART / "emb.npy")
    pos = np.load(ART / "anchor_pos.npy"); neg = np.load(ART / "anchor_neg.npy")
    ids = (ART / "emb_ids.txt").read_text().splitlines()
    return ids, (emb @ pos.T).max(1) - (emb @ neg.T).max(1)

def load_semantic():
    ids, c = _raw()
    hi = np.percentile(c, 99.5)
    return dict(zip(ids, np.clip(c / (hi + 1e-9), 0, 1)))

if __name__ == "__main__":
    import pandas as pd
    ids, c = _raw()
    print("contrastive sem percentiles:")
    for p in [1, 5, 25, 50, 75, 90, 95, 99]: print(f"  p{p:>2}: {np.percentile(c, p):.3f}")
    gold = pd.read_csv(ROOT / "gold" / "gold_labels.csv"); gold["sem"] = gold.candidate_id.map(dict(zip(ids, c)))
    print("\nmean contrastive sem by tier (want clear rise, esp 3->4->5):")
    print(gold.groupby("final_tier")["sem"].agg(["mean", "min", "max"]).round(3))
    print("\nCV-engineer sem (want LOW / negative now):")
    print(sorted(gold[gold.cv_speech]["sem"].round(3).tolist()))
    print("\nSWE-ML gem sem (want HIGH):")
    print(sorted(gold[gold.title.str.contains(r"\(ML\)", na=False)]["sem"].round(3).tolist()))