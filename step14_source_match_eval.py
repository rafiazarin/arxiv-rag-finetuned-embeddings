"""
step14_source_match_eval.py

Re-scores PubMed retrieval with an objective relevance rule that needs no model:
a retrieved chunk is relevant if it comes from the query's source abstract.

Why: human validation (step13, experiments/matcher_validation.json) found the
cross-encoder matcher agrees poorly with human judgment (kappa 0.2).

PRE-REGISTERED (decided before running):
  PRIMARY: fine-tuned model (5 epochs, published) vs untuned BGE-base,
           paired permutation test on per-query reciprocal rank @3.
  Also reported: Hit@3 (McNemar), NDCG@10, and the other step11 comparisons.
Limitation: a chunk from the source abstract may still not contain the answer.

Usage (Ollama must be running, for nomic):
    python step14_source_match_eval.py
"""
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from baseline_rag import embed_text as nomic_embed
import step10e_eval_pubmed as pub
import step11_eval_pubmed_baselines as s11

np.random.seed(42)  # reproducible bootstrap CIs
K = 3
E = s11.E
OUT_PATH = E("pubmed_source_match_scores.json")


def evaluate(embed_fn, index, chunks, pairs, n_rel):
    hits, rrs, ndcgs = [], [], []
    for qi, p in enumerate(pairs):
        q = np.array([embed_fn(p["query"])], dtype="float32")
        faiss.normalize_L2(q)
        _, idxs = index.search(q, 10)
        rel = [chunks[i]["text"] in p["positive"] for i in idxs[0] if i >= 0]
        hits.append(int(any(rel[:K])))
        rrs.append(next((1.0 / (i + 1) for i, r in enumerate(rel[:K]) if r), 0.0))
        dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(rel[:10]) if r)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_rel[qi], 10)))
        ndcgs.append(dcg / idcg)
    return {"hit_rate": round(float(np.mean(hits)), 4), "hit_rate_ci": pub.bootstrap_ci(hits),
            "mrr": round(float(np.mean(rrs)), 4), "mrr_ci": pub.bootstrap_ci(rrs),
            "ndcg_10": round(float(np.mean(ndcgs)), 4), "ndcg_10_ci": pub.bootstrap_ci(ndcgs),
            "n_queries": len(pairs), "k": K, "_hit": hits, "_rr": rrs}


def main():
    print("Checking Ollama...")
    try:
        assert len(nomic_embed("test")) > 0
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable: {e}\nRun in another window: ollama serve")

    pairs = pub.load_eval_pairs("manual")
    with open(s11.need(E("pubmed_baseline", "pubmed_baseline_chunks.pkl")), "rb") as f:
        chunks = pickle.load(f)
    n_rel = [sum(1 for c in chunks if c["text"] in p["positive"]) for p in pairs]
    assert min(n_rel) > 0, "a source abstract has no chunks in the corpus"
    print(f"Source-abstract chunks per query: min {min(n_rel)}, max {max(n_rel)}")

    bge = SentenceTransformer(s11.BGE_BASE)
    ft5 = SentenceTransformer(s11.need(E("pubmed_bge", "model")))
    ft1 = SentenceTransformer(s11.need(E("pubmed_bge_ep1", "model")))
    models = {
        "BGE-base (no FT)": (s11.st_embed(bge), faiss.read_index(s11.need(E("bge_base_noft", "pubmed.faiss")))),
        "BGE-base (PubMed FT, 5 ep)": (s11.st_embed(ft5), faiss.read_index(s11.need(E("pubmed_bge", "pubmed_bge.faiss")))),
        "BGE-base (PubMed FT, ep 1)": (s11.st_embed(ft1), faiss.read_index(s11.need(E("pubmed_bge_ep1", "pubmed_bge.faiss")))),
        "nomic (no FT)": (nomic_embed, faiss.read_index(s11.need(E("pubmed_baseline", "pubmed_baseline.faiss")))),
    }

    scores = {}
    for label, (embed_fn, index) in models.items():
        print(f"Evaluating: {label}")
        scores[label] = evaluate(embed_fn, index, chunks, pairs, n_rel)

    print(f"\n{'='*74}\n  PubMed | n={len(pairs)} | relevance = chunk from source abstract\n{'='*74}")
    print(f"  {'Model':<30} {'Hit@3':>7} {'MRR':>7} {'NDCG@10':>8}")
    for label, s in scores.items():
        print(f"  {label:<30} {s['hit_rate']:>7.4f} {s['mrr']:>7.4f} {s['ndcg_10']:>8.4f}")
        print(f"  {'':<30} Hit CI {s['hit_rate_ci']}  MRR CI {s['mrr_ci']}")

    tests = [("BGE-base (PubMed FT, 5 ep)", "BGE-base (no FT)"),   # PRIMARY
             ("BGE-base (no FT)", "nomic (no FT)"),
             ("BGE-base (PubMed FT, 5 ep)", "nomic (no FT)"),
             ("BGE-base (PubMed FT, 5 ep)", "BGE-base (PubMed FT, ep 1)")]
    sig = []
    print(f"\n  Paired significance (two-sided); first line is the PRIMARY test")
    print(f"  {'A vs B':<58} {'McNemar':>8} {'PermMRR':>8}")
    for a, b in tests:
        p_hit = pub.mcnemar_test(scores[a]["_hit"], scores[b]["_hit"])
        p_mrr = s11.paired_permutation_test(scores[a]["_rr"], scores[b]["_rr"])
        print(f"  {a + ' vs ' + b:<58} {p_hit:>8.4f} {p_mrr:>8.4f}")
        sig.append({"a": a, "b": b, "mcnemar_hit": p_hit, "paired_perm_mrr": p_mrr})
    print(f"{'='*74}")

    out = {"relevance": "retrieved chunk is a substring of the query's source abstract",
           "preregistered_primary": sig[0], "n_queries": len(pairs), "scores": {}, "significance": sig}
    for label, s in scores.items():
        out["scores"][label] = {k: v for k, v in s.items() if not k.startswith("_")}
        out["scores"][label]["per_query_hit"] = s["_hit"]
        out["scores"][label]["per_query_rr"] = s["_rr"]
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
