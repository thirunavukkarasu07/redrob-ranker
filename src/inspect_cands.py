"""Print full profiles for specific candidates to eyeball honeypot construction."""
import sys, json
from explore import load_raw

import sys
IDS = sys.argv[1:] if len(sys.argv) > 1 else ["CAND_0093547", "CAND_0000001"]  # last one = normal control

raw = {r["candidate_id"]: r for r in load_raw()}

for cid in IDS:
    r = raw.get(cid)
    if not r:
        print(f"\n{cid}: NOT FOUND"); continue
    p, s = r["profile"], r["redrob_signals"]
    print("\n" + "=" * 70)
    print(f"{cid} | {p['current_title']} | yoe={p['years_of_experience']} | {p['current_company']} ({p['current_industry']})")
    print(f"  headline: {p['headline']}")
    print("  CAREER HISTORY:")
    for c in r["career_history"]:
        print(f"    - {c['title']} @ {c['company']} | {c['start_date']}->{c['end_date']} "
              f"| dur={c['duration_months']}mo | current={c['is_current']}")
    tot = sum(c["duration_months"] for c in r["career_history"])
    print(f"    [sum of tenures = {tot}mo = {tot/12:.1f}y  vs  yoe = {p['years_of_experience']}y]")
    print("  SKILLS:")
    for sk in r["skills"]:
        print(f"    - {sk['name']:24} {sk['proficiency']:12} "
              f"endorse={sk['endorsements']:>3} dur={sk.get('duration_months','?')}mo")
    print(f"  SIGNALS: response_rate={s['recruiter_response_rate']} "
          f"last_active={s['last_active_date']} open_to_work={s['open_to_work_flag']} "
          f"github={s['github_activity_score']}")