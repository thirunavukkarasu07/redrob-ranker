"""Phase 2.4: rank the full 100K pool, emit top-100 submission, audit for traps."""
import time, numpy as np, pandas as pd
from pathlib import Path
from explore import load_raw, DATA
from features import compute_features, days_inactive
from score import score_features
from semantic import load_semantic
import argparse 

ROOT = Path(__file__).resolve().parent.parent

def reasoning(rec, f):
    p, s = rec["profile"], rec["redrob_signals"]
    di = days_inactive(s); yoe = p["years_of_experience"]
    parts = [f"{p['current_title']} with {yoe:.1f}y experience at {p['current_company']} ({p['current_industry']})"]

    # JD-connected strength
    if f.get("sem", 0) >= 0.75 or f["career_rel"] >= 0.5:
        parts.append("career history shows hands-on search / ranking / recommendation work")
    elif f["title_rel"] >= 0.85:
        parts.append("applied ML/AI background relevant to the retrieval-and-ranking mandate")

    # availability (the JD's explicit 'can we actually hire them' signal)
    avail = []
    if s.get("open_to_work_flag"): avail.append("open to work")
    avail.append(f"responds to {s['recruiter_response_rate']:.0%} of recruiters")
    avail.append(f"last active {di}d ago")
    parts.append(", ".join(avail))

    # honest concerns
    concerns = []
    if yoe < 5: concerns.append(f"under the 5-9y band ({yoe:.1f}y)")
    elif yoe > 9: concerns.append(f"above the 5-9y band ({yoe:.1f}y)")
    if f["services"] < 1.0: concerns.append("services-firm background")
    if "india" not in p["country"].lower():
        concerns.append(f"based in {p['country']} (relocation/visa)")
    if s.get("notice_period_days", 0) > 90: concerns.append(f"{s['notice_period_days']}d notice")
    if di > 120: concerns.append(f"limited recent activity ({di}d)")

    txt = "; ".join(parts)
    if concerns:
        txt += ". Watch: " + ", ".join(concerns)
    return txt + "."

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rank candidates for the Redrob JD; emit top-100 CSV.")
    ap.add_argument("--candidates", default=str(DATA), help="path to candidates.jsonl")
    ap.add_argument("--out", default=str(ROOT / "submission.csv"), help="output CSV path")
    args = ap.parse_args()
    t0 = time.time()
    raw = load_raw(args.candidates)
    sem_map = load_semantic()
    print(f"loaded {len(raw):,} in {time.time()-t0:.1f}s")

    t1 = time.time()
    rows = []
    for r in raw:
        f = compute_features(r)
        f["sem"] = sem_map.get(r["candidate_id"], 0.0)
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
    out = args.out
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
    print("\n===== SAMPLE REASONINGS (ranks 1, 25, 50, 100) =====")
    for rk in [1, 25, 50, 100]:
        row = top[top["rank"] == rk].iloc[0]
        print(f"\n#{rk} {row['candidate_id']} ({row['title']}):\n   {reasoning(row['_rec'], row['_f'])}")