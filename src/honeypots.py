"""Phase 1.4 (final): detect internally-impossible (honeypot) profiles."""
from datetime import date
import pandas as pd
from explore import load_raw

REF = date(2026, 6, 1)  # "today" for current-role math

def pdate(s):
    if not s: return None
    try:
        y, m, d = map(int, s.split("-")); return date(y, m, d)
    except Exception:
        return None

def mbetween(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month)

def check(rec):
    yoe = rec.get("profile", {}).get("years_of_experience") or 0
    yoe_m = yoe * 12
    ch = rec.get("career_history", []) or []
    sk = rec.get("skills", []) or []
    ed = rec.get("education", []) or []
    f = {}

    durs = [c.get("duration_months", 0) or 0 for c in ch]
    f["sum_dur_gt_yoe"] = sum(durs) > yoe_m + 18
    f["single_gt_yoe"]  = any(d > yoe_m + 6 for d in durs)

    # career span (earliest start -> latest end / now) vs claimed experience
    starts = [pdate(c.get("start_date")) for c in ch if pdate(c.get("start_date"))]
    ends   = [pdate(c.get("end_date")) or REF for c in ch]
    span = mbetween(min(starts), max(ends)) if starts else 0
    f["span_gt_yoe"] = span > yoe_m + 24   # 2y slack for gaps/breaks

    # current-role duration must fit between its start and today
    cur_mismatch = 0; cur_diff = 0
    for c in ch:
        s = pdate(c.get("start_date"))
        if c.get("is_current") and s:
            diff = abs(mbetween(s, REF) - (c.get("duration_months") or 0))
            if diff > 3: cur_mismatch += 1; cur_diff = max(cur_diff, diff)
        e = pdate(c.get("end_date"))
        if s and e and s > e: f["start_after_end"] = True
    f["current_dur_mismatch"] = cur_mismatch > 0
    f["cur_diff"] = cur_diff
    f.setdefault("start_after_end", False)

    # skill durations far exceeding total career
    f["skill_dur_extreme"] = sum(1 for s in sk if (s.get("duration_months") or 0) > yoe_m + 24) >= 2
    # "expert/advanced in many skills, 0 months used"
    f["expert_zero_months"] = sum(
        1 for s in sk if s.get("proficiency") in ("expert", "advanced")
        and (s.get("duration_months") or 0) == 0) >= 2
    f["edu_bad"] = any((e.get("end_year") or 0) < (e.get("start_year") or 0) for e in ed)

    checks = ["sum_dur_gt_yoe","single_gt_yoe","span_gt_yoe","current_dur_mismatch",
              "start_after_end","skill_dur_extreme","expert_zero_months","edu_bad"]
    hard = sum(bool(f[c]) for c in checks)
    return f, hard, checks

if __name__ == "__main__":
    raw = load_raw()
    rows = []
    for rec in raw:
        f, hard, checks = check(rec)
        f["candidate_id"] = rec["candidate_id"]
        f["title"] = rec.get("profile", {}).get("current_title", "")
        f["yoe"] = rec.get("profile", {}).get("years_of_experience")
        rows.append(f)
    d = pd.DataFrame(rows)

    threshold = ["sum_dur_gt_yoe","single_gt_yoe","span_gt_yoe","skill_dur_extreme","expert_zero_months"]
    logic = ["current_dur_mismatch","start_after_end","edu_bad"]
    d["n_thresh"] = d[threshold].sum(axis=1)
    d["n_logic"]  = d[logic].sum(axis=1)
    d["honeypot"] = (d.n_thresh >= 2) | (d.n_logic >= 1)

    print(f"DEFINITE honeypots: {int(d.honeypot.sum())}")
    print("\ntitle breakdown of honeypots:")
    print(d[d.honeypot]["title"].value_counts().head(15))

    print("\n--- TRUST CHECK: current_dur_mismatch cases (diff = months impossible) ---")
    cm = d[d.current_dur_mismatch].sort_values("cur_diff", ascending=False)
    print(cm[["candidate_id","title","yoe","cur_diff"]].head(20).to_string(index=False))

    print("\n--- FALSE-ALARM CHECK: skill_dur_extreme-only (NOT killed) ---")
    so = d[(d.skill_dur_extreme) & (d.n_thresh == 1) & (d.n_logic == 0)]
    print(f"count = {len(so)}; sample:")
    print(so[["candidate_id","title","yoe"]].head(8).to_string(index=False))

    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "gold" / "honeypots.txt"
    out.write_text("\n".join(d[d.honeypot]["candidate_id"]))
    print(f"\nSaved {int(d.honeypot.sum())} honeypot ids -> {out}")