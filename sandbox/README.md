---
title: Redrob Hybrid Candidate Ranker
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Redrob Hybrid Candidate Ranker — sandbox

Live demo of the INDIA.RUNS candidate-ranking submission (Docker Space, CPU). Upload a small
candidates sample (`.jsonl` or `.json`, ≤100 records following the challenge schema) and the app
ranks them for the *Senior AI Engineer — Founding Team* JD using the **full hybrid** logic:

- structured features (title relevance, career-prose relevance, experience-band fit, product-vs-services, geography)
- **contrastive sentence-embeddings** (retrieval/ranking anchors minus CV/speech/non-tech/services/research anchors) used as a rescue signal
- **internal-consistency honeypot detection** (impossible profiles scored 0)
- a **multiplicative behavioural modifier** (recruiter response, recency, notice period, open-to-work)
- fact-grounded reasoning for each candidate

Embeddings are computed on the fly for the small sample (the full 100K pipeline precomputes and
caches them). Full code: see the linked GitHub repository.
