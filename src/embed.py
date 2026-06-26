"""Phase 3.1+3.2: precompute candidate + JD-anchor embeddings (offline, cached).
Usage: python embed.py [limit]   (limit = probe only, no save)"""
import sys, time, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from explore import load_raw

ART = Path(__file__).resolve().parent.parent / "artifacts"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Plain-language "ideal candidate" anchors capturing what the JD MEANS (not its keywords)
ANCHORS = [
    "Senior engineer who built and deployed embedding-based retrieval and ranking systems for "
    "search or recommendations at a product company, owning relevance quality, offline NDCG and MRR "
    "evaluation, A/B testing, and index refresh in production.",
    "Applied machine learning engineer with production experience in hybrid search, vector databases, "
    "learning-to-rank, and candidate-to-job or product-to-user matching, who shipped end-to-end systems "
    "to real users rather than research prototypes.",
    "Machine learning engineer who rebuilt a ranking pipeline from keyword search to dense retrieval "
    "and LLM-based re-ranking, improving engagement, with strong Python and rigorous ranking evaluation.",
]

def doc_text(rec):
    p = rec["profile"]
    parts = [p.get("headline", ""), p.get("summary", "")]
    parts += [c.get("description", "") for c in rec.get("career_history", [])]
    return ". ".join(x for x in parts if x)

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
raw = load_raw(limit=limit)
ids = [r["candidate_id"] for r in raw]
texts = [doc_text(r) for r in raw]

print("loading bge-small-en-v1.5 ...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
model.max_seq_length = 256          # speed: cap tokens (text is front-loaded)

t = time.time()
emb = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
dt = time.time() - t
print(f"\nencoded {len(texts):,} docs in {dt:.1f}s ({1000*dt/len(texts):.1f} ms/doc)")

# anchors use the bge query instruction prefix
anc = model.encode([BGE_QUERY_PREFIX + a for a in ANCHORS], normalize_embeddings=True)

if limit is None:
    np.save(ART / "emb.npy", emb.astype("float32"))
    (ART / "emb_ids.txt").write_text("\n".join(ids))
    np.save(ART / "anchor_emb.npy", anc.astype("float32"))
    print(f"saved emb.npy {emb.shape}, anchor_emb.npy {anc.shape}, emb_ids.txt")
else:
    print(f"(probe — projected 100K: {dt/len(texts)*100000/60:.1f} min; not saved)")