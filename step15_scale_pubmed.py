"""
step15_scale_pubmed.py

Ceiling check: repeats the Metric-1 evaluation (retrieved chunk comes from the
query's source abstract) on a 10x larger PubMed corpus, so retrieval is harder.

Corpus: the first 20,000 abstracts of ccdv/pubmed-summarization (train, 'section')
passing the step10a length filter, in dataset order. This is a superset of the
original 2,000-abstract pool (checked below). Same 400/50 chunking.

PRE-REGISTERED (decided before running):
  PRIMARY: fine-tuned (5 epochs, published) vs untuned BGE-base on Metric 1,
           paired permutation test on per-query reciprocal rank @3.
  Also reported: Hit@3 (McNemar), NDCG@10, untuned BGE vs nomic, FT vs nomic.

Outputs: data/pubmed_pool_20k.json (gitignored), indexes in experiments/pubmed_20k/
(gitignored), experiments/pubmed_20k_source_match_scores.json.
Finished indexes are cached, and the nomic embedding step resumes if interrupted.

Usage (Ollama must be running):
    python step15_scale_pubmed.py
"""
import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from config import DATA_DIR
from baseline_rag import embed_text as nomic_embed
import step10b_build_pubmed_index as b
import step10e_eval_pubmed as pub
import step11_eval_pubmed_baselines as s11
import step14_source_match_eval as s14

N = 20000
POOL_20K = os.path.join(DATA_DIR, "pubmed_pool_20k.json")
E = s11.E
OUT_DIR = E("pubmed_20k")
OUT_PATH = E("pubmed_20k_source_match_scores.json")


def build_pool():
    if os.path.exists(POOL_20K):
        print(f"Loading cached pool: {POOL_20K}")
        return json.load(open(POOL_20K))
    print(f"Streaming the first {N:,} filtered abstracts...")
    ds = load_dataset("ccdv/pubmed-summarization", "section", split="train", streaming=True)
    pool = []
    for item in tqdm(ds, desc="Scanning"):
        a = (item.get("abstract") or "").strip()
        if len(a) < 200 or len(a) > 2000:      # same filter as step10a
            continue
        if len(a.split()) < 50:
            continue
        pool.append({"text": a})
        if len(pool) >= N:
            break
    with open(POOL_20K, "w") as f:
        json.dump(pool, f)
    return pool


def nomic_index(chunks, path):
    if os.path.exists(path):
        print(f"  Loading cached index: {path}")
        return faiss.read_index(path)
    partial = path + ".partial.npy"
    embs = list(np.load(partial)) if os.path.exists(partial) else []
    if embs:
        print(f"  Resuming nomic embedding from chunk {len(embs):,}")
    for i in tqdm(range(len(embs), len(chunks)), initial=len(embs), total=len(chunks), desc="nomic"):
        embs.append(nomic_embed(chunks[i]["text"]))
        if (i + 1) % 2000 == 0:
            np.save(partial, np.array(embs, dtype="float32"))
    arr = np.array(embs, dtype="float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, path)
    if os.path.exists(partial):
        os.remove(partial)
    return index


def main():
    print("Checking Ollama...")
    try:
        assert len(nomic_embed("test")) > 0
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable: {e}\nRun in another window: ollama serve")
    os.makedirs(OUT_DIR, exist_ok=True)

    pool = build_pool()
    original = json.load(open(os.path.join(DATA_DIR, "pubmed_pool.json")))
    texts = {p["text"] for p in pool}
    missing = sum(1 for p in original if p["text"] not in texts)
    assert missing == 0, f"{missing} original abstracts missing from the 20k pool"
    print(f"Pool: {len(pool):,} abstracts (contains all {len(original):,} originals)")

    chunks = b.chunk_documents(pool)
    print(f"Chunks: {len(chunks):,}")
    pairs = pub.load_eval_pairs("manual")
    n_rel = [sum(1 for c in chunks if c["text"] in p["positive"]) for p in pairs]
    assert min(n_rel) > 0

    bge = SentenceTransformer(s11.BGE_BASE)
    ft5 = SentenceTransformer(s11.need(E("pubmed_bge", "model")))
    print("Untuned BGE index:")
    bge_idx = s11.get_or_build_index(bge, chunks, os.path.join(OUT_DIR, "bge_noft.faiss"))
    print("Fine-tuned BGE index:")
    ft_idx = s11.get_or_build_index(ft5, chunks, os.path.join(OUT_DIR, "bge_ft5.faiss"))
    print("nomic index (slowest step):")
    nm_idx = nomic_index(chunks, os.path.join(OUT_DIR, "nomic.faiss"))

    models = {"BGE-base (no FT)": (s11.st_embed(bge), bge_idx),
              "BGE-base (PubMed FT, 5 ep)": (s11.st_embed(ft5), ft_idx),
              "nomic (no FT)": (nomic_embed, nm_idx)}
    np.random.seed(42)
    scores = {}
    for label, (fn, idx) in models.items():
        print(f"Evaluating: {label}")
        scores[label] = s14.evaluate(fn, idx, chunks, pairs, n_rel)

    print(f"\n{'='*74}\n  PubMed 20k | {len(chunks):,} chunks | n={len(pairs)} | relevance = chunk from source abstract\n{'='*74}")
    print(f"  {'Model':<30} {'Hit@3':>7} {'MRR':>7} {'NDCG@10':>8}")
    for label, s in scores.items():
        print(f"  {label:<30} {s['hit_rate']:>7.4f} {s['mrr']:>7.4f} {s['ndcg_10']:>8.4f}")
        print(f"  {'':<30} Hit CI {s['hit_rate_ci']}  MRR CI {s['mrr_ci']}")
    tests = [("BGE-base (PubMed FT, 5 ep)", "BGE-base (no FT)"),   # PRIMARY
             ("BGE-base (no FT)", "nomic (no FT)"),
             ("BGE-base (PubMed FT, 5 ep)", "nomic (no FT)")]
    sig = []
    print(f"\n  Paired significance (two-sided); first line is the PRIMARY test")
    print(f"  {'A vs B':<58} {'McNemar':>8} {'PermMRR':>8}")
    for a, c in tests:
        p_hit = pub.mcnemar_test(scores[a]["_hit"], scores[c]["_hit"])
        p_mrr = s11.paired_permutation_test(scores[a]["_rr"], scores[c]["_rr"])
        print(f"  {a + ' vs ' + c:<58} {p_hit:>8.4f} {p_mrr:>8.4f}")
        sig.append({"a": a, "b": c, "mcnemar_hit": p_hit, "paired_perm_mrr": p_mrr})
    print(f"{'='*74}")

    out = {"corpus": f"first {len(pool)} filtered abstracts, {len(chunks)} chunks",
           "relevance": "retrieved chunk is a substring of the query's source abstract",
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
