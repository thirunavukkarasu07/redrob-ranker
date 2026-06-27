# Redrob Intelligent Candidate Discovery & Ranking — Hybrid Ranker

Solution for the **INDIA.RUNS** (Redrob AI × Hack2skill) Data & AI Challenge: rank the top-100
candidates from a 100,000-candidate pool against the *Senior AI Engineer — Founding Team* job
description.

## TL;DR

A **hybrid ranker** that combines an interpretable rule/feature engine with a contrastive
sentence-embedding signal and a multiplicative behavioural modifier. It is deliberately *not* a
keyword counter — the dataset's `skills[]` array is randomized noise by design, so the ranker
reasons about **title + career-history prose + behaviour**, exactly as the JD asks ("seeing beyond
keywords to understand semantic fit").

- **Ranking step: ~7 seconds** on CPU for the full 100K, no GPU, no network. (Embeddings are
  precomputed offline and cached.)
- **Trap-aware**: honeypots hard-zeroed, keyword-stuffers neutralized, behavioural "ghosts"
  down-weighted, CV/speech/services/geography negatives penalized, plain-language Tier-5s rescued
  by embeddings.
- **0 honeypots / 0 services-firms / 0 non-India** candidates in the produced top-100.
- Passes the official `validate_submission.py`.

## How it works

The final score for each candidate is:

```
score = ( 0.50 * content_relevance        # max(title_relevance, semantic_fit)  <- embeddings can only RESCUE
        + 0.25 * career_keyword_relevance  # search/ranking/recsys phrases in career prose
        + 0.25 * experience_fit )          # peaks in the JD's 5-9y band
        * services_factor                  # 0.65 if services-only career (TCS/Infosys/...)
        * geography_factor                 # 1.0 India, lower otherwise (no visa sponsorship)
        * behavioural_modifier             # recency, recruiter response, notice period, open-to-work
# honeypots are forced to score 0.
```

### Key design decisions (and why)

| Decision | Rationale |
|---|---|
| **Skills are down-weighted, gated by title/career** | Every skill appears ~12,000x in the pool — the `skills[]` array is randomized noise. A keyword counter walks straight into the JD's trap. |
| **Contrastive embeddings** (`max(cos to POSITIVE anchors) - max(cos to NEGATIVE anchors)`) | Plain cosine to "ideal candidate" text rated computer-vision engineers (a JD-negative) as highly as real retrieval engineers. Subtracting similarity to negative anchors (CV / speech / non-tech / services / research) cleanly separates genuine builders from look-alikes. |
| **Embeddings used as a *rescue* signal** (`max(title_rel, sem)`) | The embedding can only lift a candidate whose *title* understates them (e.g. a "Senior Software Engineer (ML)" who built a recommender), never drag the well-ordered top around or pull in CV/non-tech (whose contrastive sem is already low). |
| **Multiplicative behavioural modifier** | The JD is explicit: a perfect-on-paper candidate who hasn't logged in for months and ignores recruiters is *not actually hireable*. High fit x low availability = demoted. |
| **Honeypot detector = internal consistency only** | The ~80 honeypots are *internally impossible* profiles (e.g. 7.8 years of a skill on a 2.9-year career; a current role lasting longer than time since it started). Detected with pure self-consistency checks — no external knowledge — and hard-zeroed. |

Validated against a hand-labelled gold set of 115 candidates (tiers 0-5) spanning every trap and
fit type — the project's private "leaderboard" since the competition has no live feedback.

## Reproduce

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Pre-compute embeddings (one-time, offline; may exceed the 5-min ranking budget)
This is the only slow step (~45 min on CPU). It caches candidate + anchor embeddings to `artifacts/`.
```bash
python src/embed.py   --candidates ./candidates.jsonl   # caches artifacts/emb.npy + emb_ids.txt
python src/anchors.py                                    # caches artifacts/anchor_pos.npy + anchor_neg.npy
```

### 3. Rank (the scored step — runs in ~7 s on CPU, no network)
```bash
python src/rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
> If the embedding artifacts are absent, `rank.py` falls back to the rule-only ranker (it warns and
> sets `sem = 0`), so it always produces a valid submission.

### 4. Validate the format
```bash
python validate_submission.py submission.csv   # -> "Submission is valid."
```

## Compute compliance

| Constraint | This solution |
|---|---|
| Ranking <= 5 min wall-clock | **~7 s** for 100K |
| <= 16 GB RAM | well under |
| CPU only, no GPU | yes (embeddings precomputed; ranking is numpy only) |
| No network during ranking | yes — `rank.py` loads cached `.npy` files; no model, no API |
| Pre-computation | embeddings, ~45 min offline, cached (documented above) |

## Repository layout

```
src/
  explore.py        # fast JSONL loader + flat-table cache
  features.py       # per-candidate structured features (title/career/exp/services/geo/behaviour/honeypot)
  honeypots.py      # internal-consistency honeypot detector
  semantic.py       # contrastive embedding fit (loads cached vectors)
  embed.py          # OFFLINE: encode candidate career-prose -> artifacts/emb.npy
  anchors.py        # OFFLINE: encode positive/negative JD anchors
  score.py          # scoring formula + NDCG/MAP/P@10 evaluation vs the gold set
  rank.py           # MAIN: rank full pool -> submission.csv (+ top-100 trap audit)
gold/
  gold_labels.csv   # 115 hand-labelled candidates (tiers 0-5) — validation set
  honeypots.txt     # detected honeypot ids
artifacts/          # cached embeddings (generated; not committed)
submission.csv      # the top-100 ranking
```

## AI tools

Claude (Anthropic) was used for architecture discussion, code review, and pair-programming
throughout. All engineering decisions, data analysis, gold-set labelling judgement, and the final
design were directed and verified by the author. No candidate data was sent to any hosted LLM during
ranking (the ranking step makes no network calls).
