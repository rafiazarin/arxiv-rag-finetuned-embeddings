"""
step11_eval_pubmed_baselines.py

Tests whether the PubMed gain comes from fine-tuning or from choosing BGE
over nomic. All models are evaluated in ONE run, on the same machine, with
the same cross-encoder matcher (step10e.is_match).

Eval set: pubmed_manual_eval90.json (90 human-written queries)
Models:
  - BGE-base-en-v1.5, no fine-tuning
  - BGE-base, PubMed FT, final model (5 epochs)
  - BGE-base, PubMed FT, epoch-1 checkpoint (what the original notebook saved)
  - nomic-embed-text via Ollama, no fine-tuning

Tests: McNemar (Hit@3) + paired sign-flip permutation test (MRR).
Output: experiments/pubmed_step11_scores.json (includes per-query scores)

Usage (Ollama must be running):
    python step11_eval_pubmed_baselines.py
"""
import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import EXPERIMENTS_DIR
from baseline_rag import embed_text as nomic_embed
import step10e_eval_pubmed as pub

np.random.seed(42)  # reproducible bootstrap CIs

BGE_BASE = "BAAI/bge-base-en-v1.5"
OUT_PATH = os.path.join(EXPERIMENTS_DIR, "pubmed_step11_scores.json")
E = lambda *p: os.path.join(EXPERIMENTS_DIR, *p)


def st_embed(model):
    return lambda t: model.encode(t, normalize_embeddings=True).tolist()


def need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return path


def get_or_build_index(model, chunks, path):
    if os.path.exists(path):
        print(f"  Loading cached index: {path}")
        return faiss.read_index(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  Building untuned BGE index for {len(chunks):,} chunks (one-time)...")
    emb = model.encode([c["text"] for c in chunks], batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True,
                       convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, path)
    return index


def paired_permutation_test(a, b, n=10000, seed=42):
    """Two-sided paired test: randomly flips the sign of each per-query difference."""
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return round(float((null >= observed - 1e-12).mean()), 4)


def main():
    print("Checking Ollama...")
    try:
        assert len(nomic_embed("test")) > 0
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable: {e}\nRun in another window: ollama serve")

    pairs = pub.load_eval_pairs("manual")
    with open(need(E("pubmed_baseline", "pubmed_baseline_chunks.pkl")), "rb") as f:
        chunks = pickle.load(f)

    bge = SentenceTransformer(BGE_BASE)
    ft5 = SentenceTransformer(need(E("pubmed_bge", "model")))
    ft1 = SentenceTransformer(need(E("pubmed_bge_ep1", "model")))

    models = {
        "BGE-base (no FT)": (st_embed(bge),
                             get_or_build_index(bge, chunks, E("bge_base_noft", "pubmed.faiss"))),
        "BGE-base (PubMed FT, 5 ep)": (st_embed(ft5),
                                       faiss.read_index(need(E("pubmed_bge", "pubmed_bge.faiss")))),
        "BGE-base (PubMed FT, ep 1)": (st_embed(ft1),
                                       faiss.read_index(need(E("pubmed_bge_ep1", "pubmed_bge.faiss")))),
        "nomic (no FT)": (nomic_embed,
                          faiss.read_index(need(E("pubmed_baseline", "pubmed_baseline.faiss")))),
    }

    scores = {}
    for label, (embed_fn, index) in models.items():
        print(f"\nEvaluating: {label}")
        scores[label] = pub.evaluate(embed_fn, index, chunks, pairs)

    print(f"\n{'='*74}\n  PubMed | n={len(pairs)} | cross-encoder matcher | 95% bootstrap CI\n{'='*74}")
    print(f"  {'Model':<30} {'Hit@3':>7} {'MRR':>7} {'NDCG@10':>8}")
    for label, s in scores.items():
        print(f"  {label:<30} {s['hit_rate']:>7.4f} {s['mrr']:>7.4f} {s['ndcg_10']:>8.4f}")
        print(f"  {'':<30} Hit CI {s['hit_rate_ci']}  MRR CI {s['mrr_ci']}")

    tests = [("BGE-base (PubMed FT, 5 ep)", "BGE-base (no FT)"),
             ("BGE-base (no FT)", "nomic (no FT)"),
             ("BGE-base (PubMed FT, 5 ep)", "nomic (no FT)"),
             ("BGE-base (PubMed FT, 5 ep)", "BGE-base (PubMed FT, ep 1)")]
    sig = []
    print(f"\n  Paired significance (two-sided)")
    print(f"  {'A vs B':<58} {'McNemar':>8} {'PermMRR':>8}")
    for a, b in tests:
        p_hit = pub.mcnemar_test(scores[a]["_hit_scores"], scores[b]["_hit_scores"])
        p_mrr = paired_permutation_test(scores[a]["_rr_scores"], scores[b]["_rr_scores"])
        print(f"  {a + ' vs ' + b:<58} {p_hit:>8.4f} {p_mrr:>8.4f}")
        sig.append({"a": a, "b": b, "mcnemar_hit": p_hit, "paired_perm_mrr": p_mrr})
    print(f"{'='*74}")

    out = {"n_queries": len(pairs), "scores": {}, "significance": sig}
    for label, s in scores.items():
        out["scores"][label] = {k: v for k, v in s.items() if not k.startswith("_")}
        out["scores"][label]["per_query_hit"] = s["_hit_scores"]
        out["scores"][label]["per_query_rr"] = s["_rr_scores"]
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
