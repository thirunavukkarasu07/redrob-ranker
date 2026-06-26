"""Phase 1.3: profile the candidate pool from the cached flat table."""
import pandas as pd
from pathlib import Path

FLAT = Path(__file__).resolve().parent.parent / "artifacts" / "flat.pkl"
df = pd.read_pickle(FLAT)
pd.set_option("display.max_rows", 60)

def section(t): print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)

section("EXPERIENCE (years_of_experience)")
print(df["yoe"].describe())
print("\nIn JD's 5-9y band:", ((df.yoe >= 5) & (df.yoe <= 9)).sum())

section("TOP 25 CURRENT TITLES")
print(df["current_title"].value_counts().head(25))

section("TOP 15 INDUSTRIES")
print(df["current_industry"].value_counts().head(15))

section("CURRENT COMPANY SIZE")
print(df["current_company_size"].value_counts())

section("COUNTRY (top 10)")
print(df["country"].value_counts().head(10))
print("\nIndia vs non-India:")
print((df.country.str.contains("India", case=False, na=False)).value_counts())

section("PREFERRED WORK MODE")
print(df["preferred_work_mode"].value_counts())

section("BEHAVIORAL SIGNALS — summary stats")
for c in ["profile_completeness", "recruiter_response_rate", "interview_completion_rate",
          "offer_acceptance_rate", "github_activity", "saved_by_recruiters_30d",
          "notice_period_days", "n_skills", "n_jobs"]:
    print(f"\n{c}:"); print(df[c].describe()[["mean","min","25%","50%","75%","max"]])

section("SENTINEL / MISSING-VALUE FLAGS (-1 conventions)")
print("github_activity == -1 (no GitHub):", (df.github_activity == -1).sum())
print("offer_acceptance_rate == -1 (no offers):", (df.offer_acceptance_rate == -1).sum())
print("open_to_work == True:", (df.open_to_work == True).sum())

section("LAST-ACTIVE RECENCY (relative to most recent date in pool)")
d = pd.to_datetime(df["last_active_date"], errors="coerce")
ref = d.max()
days = (ref - d).dt.days
print("Reference 'now' (max last_active):", ref.date())
print(pd.cut(days, [-1, 7, 30, 90, 180, 9999],
      labels=["<=7d","8-30d","31-90d","91-180d",">180d"]).value_counts().sort_index())