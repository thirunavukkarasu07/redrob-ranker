"""Phase 2.4: rank the full 100K pool, emit top-100 submission, audit for traps."""
import time, numpy as np, pandas as pd
from pathlib import Path
from explore import load_raw
from features import compute_features, days_inactive
from score import score_features

ROOT = Path(__file__).resolve().parent.parent

def reasoning(rec, f):
    p, s = rec["profile"], rec["redrob_signals"]
    di = days_inactive(s)
    bits = [f"{p['current_title']} with {p['years_of_experience']:.1f} yrs at {p['current_company']} ({p['current_industry']})"]
    if f["career_rel"] >= 0.5: bits.append("career shows search/ranking/recsys work")
    bits.append(f"recruiter response {s['recruiter_response_rate']:.2f}, last active {di}d ago")
    if f["services"] < 1.0: bits.append("services-firm background")
    return "; ".join(bits) + "."

if __name__ == "__main__":
    t0 = time.time()
    raw = load_raw()
    print(f"loaded {len(raw):,} in {time.time()-t0:.1f}s")

    t1 = time.time()
    rows = []
    for r in raw:
        f = compute_features(r)
        rows.append({"candidate_id": r["candidate_id"], "score": score_features(f),
                     "title": r["profile"]["current_title"], "yoe": r["profile"]["years_of_experience"],
                     "country": r["profile"]["country"], "honeypot": f["honeypot"],
                     "services": f["services"] < 1.0, "career_rel": f["career_rel"],
                     "resp": r["redrob_signals"]["recruiter_response_rate"],
                     "inactive": days_inactive(r["redrob_signals"]), "_rec": r, "_f": f})
    df = pd.DataFrame(rows)
    print(f"scored {len(df):,} in {time.time()-t1:.1f}s")

    df = df.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    top = df.head(100).copy()
    top["rank"] = range(1, 101)

    # --- write submission ---
    sub = pd.DataFrame({"candidate_id": top.candidate_id, "rank": top["rank"],
                        "score": top.score.round(6),
                        "reasoning": [reasoning(r["_rec"], r["_f"]) for _, r in top.iterrows()]})
    out = ROOT / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # --- trap audit on the top 100 ---
    print("\n===== TOP-100 AUDIT =====")
    print(f"honeypots in top 100 : {top.honeypot.sum()}   (DQ if >10)")
    print(f"services-firm in top : {top.services.sum()}")
    print(f"non-India in top     : {(~top.country.str.contains('India')).sum()}")
    print(f"mean response rate   : {top.resp.mean():.2f}")
    print(f"mean days inactive   : {top.inactive.mean():.0f}")
    print(f"score range          : {top.score.min():.3f} .. {top.score.max():.3f}")
    print("\ntop-100 title distribution:")
    print(top.title.value_counts().head(15))
    print("\n===== TOP 20 PREVIEW =====")
    print(top.head(20)[["rank","candidate_id","title","yoe","country","resp","inactive","score"]].to_string(index=False))