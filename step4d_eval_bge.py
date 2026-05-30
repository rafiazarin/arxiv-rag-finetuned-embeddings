"""
step4d_eval_bge.py  (updated for step 9)

Evaluates bge-base model variants on manual_eval140.json (140 queries).
The 10 held-out validation queries (manual_val.json) are excluded from eval
to avoid contamination with training-time validation.

Usage:
    python step4d_eval_bge.py --eval manual
    python step4d_eval_bge.py --eval synthetic
    python step4d_eval_bge.py --eval combined
"""
import os
import json
import argparse
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import DATA_DIR, EXPERIMENTS_DIR, TOP_K

# ── Models to evaluate ────────────────────────────────────────────────────────
BGE_RUNS = {
    "BGE-base (arXiv only)": {
        "model_dir":  "bge_arxiv100/model",
        "index_file": "bge_arxiv100/bge_arxiv100.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
    "BGE-base (mixed 50/50)": {
        "model_dir":  "bge_mixed50/model",
        "index_file": "bge_mixed50/bge_mixed50.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
    "BGE-base (hard negatives)": {
        "model_dir":  "bge_hardneg/model",
        "index_file": "bge_hardneg/bge_hardneg.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
    "BGE-base (mistral pairs)": {
    "model_dir":  "bge_mistral/model",
    "index_file": "bge_mistral/bge_mistral.faiss",
    "chunks_src": "baseline/baseline_chunks.pkl",
    },
    "Staged gen->dom (MARCO then arXiv)": {
        "model_dir":  "staged_gen_dom/model",
        "index_file": "staged_gen_dom/staged_gen_dom.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
    "Staged dom->gen (arXiv then MARCO)": {
        "model_dir":  "staged_dom_gen/model",
        "index_file": "staged_dom_gen/staged_dom_gen.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
}

# ── Metrics ───────────────────────────────────────────────────────────────────

def hit_at_k(retrieved, relevant, k):
    return int(any(r in relevant for r in retrieved[:k]))

def reciprocal_rank(retrieved, relevant, k):
    for i, r in enumerate(retrieved[:k], 1):
        if r in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    dcg  = sum(1.0 / np.log2(i + 2) for i, r in enumerate(retrieved[:k]) if r in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def bootstrap_ci(scores, n_resamples=1000, ci=0.95):
    scores = np.array(scores)
    means  = [np.mean(np.random.choice(scores, size=len(scores), replace=True))
              for _ in range(n_resamples)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return round(float(lo), 4), round(float(hi), 4)

# ── Match function ────────────────────────────────────────────────────────────

_match_model = None

def get_match_model():
    global _match_model
    if _match_model is None:
        _match_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _match_model

def is_match(retrieved_text, ground_truth, threshold=0.75):
    try:
        model = get_match_model()
        embs  = model.encode([retrieved_text, ground_truth],
                             normalize_embeddings=True, convert_to_numpy=True)
        return float(np.dot(embs[0], embs[1])) >= threshold
    except Exception:
        a = set(retrieved_text.lower().split())
        b = set(ground_truth.lower().split())
        return len(a & b) / len(a) > 0.5 if a else False

# ── Eval set loader ───────────────────────────────────────────────────────────

def load_eval_pairs(eval_type="manual"):
    pairs = []
    if eval_type in ("synthetic", "combined"):
        with open(os.path.join(DATA_DIR, "training_pairs.json")) as f:
            pairs += json.load(f)[-100:]

    if eval_type in ("manual", "combined"):
        # Use manual_eval140.json if it exists (post step 9 split)
        # Fall back to manual_eval.json for backwards compatibility
        path140 = os.path.join(DATA_DIR, "manual_eval140.json")
        path150 = os.path.join(DATA_DIR, "manual_eval.json")
        if os.path.exists(path140):
            with open(path140) as f:
                pairs += json.load(f)
            print("Using manual_eval140.json (140 queries, 10 held out for val)")
        elif os.path.exists(path150):
            with open(path150) as f:
                pairs += json.load(f)
            print("Using manual_eval.json (150 queries) — run step9_split_manual_eval.py to split")
        else:
            print("Warning: no manual eval file found.")
    return pairs

# ── Core evaluator ────────────────────────────────────────────────────────────

def evaluate(model_dir, index_path, chunks, eval_pairs, k=TOP_K):
    model    = SentenceTransformer(model_dir)
    index    = faiss.read_index(index_path)
    embed_fn = lambda t: model.encode(t, normalize_embeddings=True).tolist()

    hit_scores, rr_scores, ndcg_scores = [], [], []

    for pair in tqdm(eval_pairs, desc=os.path.basename(model_dir)):
        q_emb = np.array([embed_fn(pair["query"])], dtype="float32")
        faiss.normalize_L2(q_emb)
        _, idxs    = index.search(q_emb, 10)
        retrieved  = [chunks[i]["text"] for i in idxs[0] if i >= 0]
        relevant   = {t for t in retrieved if is_match(t, pair["positive"])}
        relevant.add(pair["positive"])

        hit_scores.append(hit_at_k(retrieved, relevant, k))
        rr_scores.append(reciprocal_rank(retrieved, relevant, k))
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, 10))

    return {
        "hit_rate":    round(float(np.mean(hit_scores)),  4),
        "hit_rate_ci": bootstrap_ci(hit_scores),
        "mrr":         round(float(np.mean(rr_scores)),   4),
        "mrr_ci":      bootstrap_ci(rr_scores),
        "ndcg_10":     round(float(np.mean(ndcg_scores)), 4),
        "ndcg_10_ci":  bootstrap_ci(ndcg_scores),
        "n_queries":   len(eval_pairs),
        "k":           k,
    }

# ── Pretty print ──────────────────────────────────────────────────────────────

def print_results(all_scores, eval_type):
    col = 35
    n   = list(all_scores.values())[0]["n_queries"]
    print(f"\n{'='*72}")
    print(f"  BGE-BASE EVAL  |  {eval_type}  |  n={n}  |  95% CI bootstrap")
    print(f"{'='*72}")
    print(f"  {'Model':<{col}} {'Hit@3':>8} {'MRR':>8} {'NDCG@10':>10}")
    print(f"  {'-'*65}")
    for label, s in sorted(all_scores.items(), key=lambda x: -x[1]["mrr"]):
        print(f"  {label:<{col}} {s['hit_rate']:>8.4f} {s['mrr']:>8.4f} {s['ndcg_10']:>10.4f}")
        lo_h, hi_h = s["hit_rate_ci"]
        lo_m, hi_m = s["mrr_ci"]
        lo_n, hi_n = s["ndcg_10_ci"]
        print(f"  {'':>{col}} {f'[{lo_h},{hi_h}]':>8} {f'[{lo_m},{hi_m}]':>8} {f'[{lo_n},{hi_n}]':>10}")
    print(f"{'='*72}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", choices=["synthetic", "manual", "combined"],
                        default="manual")
    args       = parser.parse_args()
    eval_type  = args.eval
    eval_pairs = load_eval_pairs(eval_type)
    print(f"\nEval set: {len(eval_pairs)} queries ({eval_type})\n")

    all_scores = {}

    for label, cfg in BGE_RUNS.items():
        model_dir   = os.path.join(EXPERIMENTS_DIR, cfg["model_dir"])
        index_path  = os.path.join(EXPERIMENTS_DIR, cfg["index_file"])
        chunks_path = os.path.join(EXPERIMENTS_DIR, cfg["chunks_src"])

        for p, name in [(model_dir, "model"), (index_path, "index"), (chunks_path, "chunks")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing {name}: {p}")

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        print(f"\nEvaluating: {label}")
        all_scores[label] = evaluate(model_dir, index_path, chunks, eval_pairs)

    out_path = os.path.join(EXPERIMENTS_DIR, f"bge_scores_{eval_type}.json")
    with open(out_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    print_results(all_scores, eval_type)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
