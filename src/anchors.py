"""Embed positive + negative JD anchors (fast; candidate cache untouched)."""
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ART = Path(__file__).resolve().parent.parent / "artifacts"
PFX = "Represent this sentence for searching relevant passages: "
POS = [
    "Senior engineer who built and deployed embedding-based retrieval and ranking systems for search or "
    "recommendations at a product company, owning relevance quality, offline NDCG and MRR evaluation, A/B testing.",
    "Applied machine learning engineer with production experience in hybrid search, vector databases, "
    "learning-to-rank, and candidate-to-job or product-to-user matching, shipping end-to-end systems to real users.",
    "Machine learning engineer who rebuilt a ranking pipeline from keyword search to dense retrieval and "
    "LLM-based re-ranking, improving engagement, with strong Python and rigorous ranking evaluation.",
]
NEG = [
    "Computer vision engineer working on image classification, object detection, segmentation, or medical imaging.",
    "Speech recognition, text-to-speech, or audio signal processing engineer.",
    "Marketing manager, sales executive, operations manager, HR manager, or accountant managing teams and business KPIs.",
    "IT services consultant at an outsourcing firm delivering generic web and backend CRUD applications in Java or .NET.",
    "Academic researcher publishing machine learning papers without production deployment to real users.",
]
m = SentenceTransformer("BAAI/bge-small-en-v1.5"); m.max_seq_length = 256
np.save(ART / "anchor_pos.npy", m.encode([PFX+t for t in POS], normalize_embeddings=True).astype("float32"))
np.save(ART / "anchor_neg.npy", m.encode([PFX+t for t in NEG], normalize_embeddings=True).astype("float32"))
print("saved anchor_pos.npy + anchor_neg.npy")