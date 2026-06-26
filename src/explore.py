"""Phase 1 exploration: load candidates.jsonl, flatten to a DataFrame, cache it."""
import time, orjson, pandas as pd
from pathlib import Path

# --- paths ---
DATA = Path(r"C:\Users\Thiru\indiaruns\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl")
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
FLAT = ARTIFACTS / "flat.pkl"


def load_raw(path=DATA, limit=None):
    """Stream-parse the JSONL into a list of dicts."""
    out = []
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            out.append(orjson.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def flatten(rec):
    """Pull the scalar fields we need for profiling into one flat row."""
    p = rec.get("profile", {})
    s = rec.get("redrob_signals", {})
    sal = s.get("expected_salary_range_inr_lpa", {}) or {}
    return {
        "candidate_id": rec["candidate_id"],
        "headline": p.get("headline", ""),
        "summary": p.get("summary", ""),
        "location": p.get("location", ""),
        "country": p.get("country", ""),
        "yoe": p.get("years_of_experience"),
        "current_title": p.get("current_title", ""),
        "current_company": p.get("current_company", ""),
        "current_company_size": p.get("current_company_size", ""),
        "current_industry": p.get("current_industry", ""),
        "n_jobs": len(rec.get("career_history", [])),
        "n_skills": len(rec.get("skills", [])),
        "n_edu": len(rec.get("education", [])),
        "n_certs": len(rec.get("certifications", []) or []),
        # behavioral signals
        "profile_completeness": s.get("profile_completeness_score"),
        "last_active_date": s.get("last_active_date"),
        "open_to_work": s.get("open_to_work_flag"),
        "recruiter_response_rate": s.get("recruiter_response_rate"),
        "avg_response_time_hours": s.get("avg_response_time_hours"),
        "notice_period_days": s.get("notice_period_days"),
        "willing_to_relocate": s.get("willing_to_relocate"),
        "github_activity": s.get("github_activity_score"),
        "saved_by_recruiters_30d": s.get("saved_by_recruiters_30d"),
        "interview_completion_rate": s.get("interview_completion_rate"),
        "offer_acceptance_rate": s.get("offer_acceptance_rate"),
        "salary_min": sal.get("min"),
        "salary_max": sal.get("max"),
        "preferred_work_mode": s.get("preferred_work_mode"),
        "n_skill_assessments": len(s.get("skill_assessment_scores", {}) or {}),
    }


if __name__ == "__main__":
    t0 = time.time()
    raw = load_raw()
    print(f"Loaded {len(raw):,} candidates in {time.time()-t0:.1f}s")

    df = pd.DataFrame(flatten(r) for r in raw)
    df.to_pickle(FLAT)
    print(f"Flat table cached -> {FLAT}")
    print(f"Shape: {df.shape}")
    print("\nColumns:", list(df.columns))
    print("\n--- first 3 rows (key cols) ---")
    print(df[["candidate_id", "current_title", "yoe", "n_skills",
              "recruiter_response_rate", "last_active_date"]].head(3).to_string())