"""
Full evaluation script with:
- Hit Rate@K
- MRR@K  
- NDCG@10
- Bootstrap 95% confidence intervals (1000 resamples)

Accepts a label and embed_fn so it works for any model.
Run once per model, results saved to experiments/{label}/full_scores.json
"""
import os
import json
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import mlflow
from config import (
    DATA_DIR, BASELINE_DIR, FINETUNED_DIR,
    FINETUNED_MODEL_PATH, TOP_K, EXPERIMENTS_DIR
)
from baseline_rag import load_index, embed_text as baseline_embed

# ── Metrics ───────────────────────────────────────────────────────────────────

def hit_at_k(retrieved, relevant, k):
    return int(any(r in relevant for r in retrieved[:k]))

def reciprocal_rank(retrieved, relevant, k):
    for i, r in enumerate(retrieved[:k], 1):
        if r in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    """NDCG@K — normalized discounted cumulative gain."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, r in enumerate(retrieved[:k])
        if r in relevant
    )
    # Ideal DCG: all relevant docs at top positions
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def bootstrap_ci(scores, n_resamples=1000, ci=0.95):
    """Bootstrap confidence interval for a list of per-query scores."""
    scores = np.array(scores)
    means = [
        np.mean(np.random.choice(scores, size=len(scores), replace=True))
        for _ in range(n_resamples)
    ]
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return round(float(lower), 4), round(float(upper), 4)

# ── Eval set loader ───────────────────────────────────────────────────────────

def load_eval_pairs(eval_type="synthetic"):
    """
    eval_type:
        'synthetic' — last 100 auto-generated pairs
        'manual'    — data/manual_eval.json (human-written queries)
        'combined'  — both merged
    """
    pairs = []

    if eval_type in ("synthetic", "combined"):
        with open(os.path.join(DATA_DIR, "training_pairs.json")) as f:
            all_pairs = json.load(f)
        pairs += all_pairs[-100:]

    if eval_type in ("manual", "combined"):
        manual_path = os.path.join(DATA_DIR, "manual_eval.json")
        if os.path.exists(manual_path):
            with open(manual_path) as f:
                pairs += json.load(f)
        else:
            print("Warning: manual_eval.json not found, skipping manual queries.")

    return pairs

# ── Match function ────────────────────────────────────────────────────────────



# Cache the baseline model for match scoring
_match_model = None

def get_match_model():
    global _match_model
    if _match_model is None:
        from sentence_transformers import SentenceTransformer
        _match_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _match_model

def is_match(retrieved_text, ground_truth, threshold=0.75):
    """
    Semantic similarity match using embedding cosine similarity.
    More robust than word overlap for paraphrases and domain jargon.
    Falls back to word overlap if model unavailable.
    """
    try:
        model = get_match_model()
        embeddings = model.encode(
            [retrieved_text, ground_truth],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return similarity >= threshold
    except Exception:
        # Fallback to word overlap
        a = set(retrieved_text.lower().split())
        b = set(ground_truth.lower().split())
        return len(a & b) / len(a) > 0.5 if a else False

# ── Core evaluator ────────────────────────────────────────────────────────────

def evaluate(index, chunks, embed_fn, eval_pairs, k=TOP_K):
    hit_scores  = []
    rr_scores   = []
    ndcg_scores = []

    for pair in tqdm(eval_pairs):
        query    = pair["query"]
        gt_text  = pair["positive"]

        # Retrieve
        q_emb = np.array([embed_fn(query)], dtype="float32")
        faiss.normalize_L2(q_emb)
        _, idxs = index.search(q_emb, 10)  # retrieve 10 for NDCG@10
        retrieved = [chunks[i]["text"] for i in idxs[0] if i >= 0]

        # Build relevant set (all retrieved texts that match ground truth)
        relevant = {t for t in retrieved if is_match(t, gt_text)}
        # Always include ground truth itself as relevant
        relevant.add(gt_text)

        hit_scores.append(hit_at_k(retrieved, relevant, k))
        rr_scores.append(reciprocal_rank(retrieved, relevant, k))
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, 10))

    mean_hit  = round(float(np.mean(hit_scores)),  4)
    mean_mrr  = round(float(np.mean(rr_scores)),   4)
    mean_ndcg = round(float(np.mean(ndcg_scores)), 4)

    hit_ci  = bootstrap_ci(hit_scores)
    mrr_ci  = bootstrap_ci(rr_scores)
    ndcg_ci = bootstrap_ci(ndcg_scores)

    return {
        "hit_rate":    mean_hit,
        "hit_rate_ci": hit_ci,
        "mrr":         mean_mrr,
        "mrr_ci":      mrr_ci,
        "ndcg_10":     mean_ndcg,
        "ndcg_10_ci":  ndcg_ci,
        "n_queries":   len(eval_pairs),
        "k":           k
    }

# ── BM25 evaluator ────────────────────────────────────────────────────────────

def evaluate_bm25(chunks, eval_pairs, k=TOP_K):
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)

    hit_scores  = []
    rr_scores   = []
    ndcg_scores = []

    for pair in tqdm(eval_pairs):
        query   = pair["query"]
        gt_text = pair["positive"]

        scores   = bm25.get_scores(query.lower().split())
        top_idxs = np.argsort(scores)[::-1][:10]
        retrieved = [chunks[i]["text"] for i in top_idxs]

        relevant = {t for t in retrieved if is_match(t, gt_text)}
        relevant.add(gt_text)

        hit_scores.append(hit_at_k(retrieved, relevant, k))
        rr_scores.append(reciprocal_rank(retrieved, relevant, k))
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, 10))

    mean_hit  = round(float(np.mean(hit_scores)),  4)
    mean_mrr  = round(float(np.mean(rr_scores)),   4)
    mean_ndcg = round(float(np.mean(ndcg_scores)), 4)

    return {
        "hit_rate":    mean_hit,
        "hit_rate_ci": bootstrap_ci(hit_scores),
        "mrr":         mean_mrr,
        "mrr_ci":      bootstrap_ci(rr_scores),
        "ndcg_10":     mean_ndcg,
        "ndcg_10_ci":  bootstrap_ci(ndcg_scores),
        "n_queries":   len(eval_pairs),
        "k":           k
    }

# ── Pretty print ──────────────────────────────────────────────────────────────

def print_results(all_scores):
    col = 32
    print(f"\n{'='*72}")
    print(f"  RAG RETRIEVAL EVALUATION")
    print(f"  n={list(all_scores.values())[0]['n_queries']} queries  |  "
          f"Hit Rate@{TOP_K}, MRR@{TOP_K}, NDCG@10  |  95% CI (bootstrap)")
    print(f"{'='*72}")
    print(f"  {'Method':<{col}} {'Hit@K':>8} {'MRR':>8} {'NDCG@10':>10}")
    print(f"  {'-'*64}")

    for label, s in sorted(all_scores.items(), key=lambda x: x[1]["mrr"]):
        marker = " ◀" if "mixed" in label.lower() or "finetuned" in label.lower() else ""
        print(f"  {label:<{col}} {s['hit_rate']:>8.4f} {s['mrr']:>8.4f} {s['ndcg_10']:>10.4f}{marker}")
        print(f"  {'':>{col}} "
              f"{'['+str(s['hit_rate_ci'][0])+','+str(s['hit_rate_ci'][1])+']':>8} "
              f"{'['+str(s['mrr_ci'][0])+','+str(s['mrr_ci'][1])+']':>8} "
              f"{'['+str(s['ndcg_10_ci'][0])+','+str(s['ndcg_10_ci'][1])+']':>10}")

    print(f"{'='*72}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    eval_pairs = load_eval_pairs(eval_type="synthetic")
    print(f"Eval set: {len(eval_pairs)} queries (synthetic)\n")

    b_index, b_chunks = load_index(name="baseline", models_dir=BASELINE_DIR)
    ft_index = faiss.read_index(os.path.join(FINETUNED_DIR, "finetuned.faiss"))
    with open(os.path.join(FINETUNED_DIR, "finetuned_chunks.pkl"), "rb") as f:
        ft_chunks = pickle.load(f)
    ft_model  = SentenceTransformer(FINETUNED_MODEL_PATH)
    ft_embed  = lambda t: ft_model.encode(t, normalize_embeddings=True).tolist()

    mlflow.set_experiment("rag-full-eval")
    all_scores = {}

    # BM25
    print("Evaluating BM25...")
    with mlflow.start_run(run_name="bm25"):
        s = evaluate_bm25(b_chunks, eval_pairs)
        mlflow.log_metrics({k: v for k, v in s.items()
                           if isinstance(v, float)})
        all_scores["BM25"] = s

    # Baseline embedding
    print("Evaluating baseline embedding...")
    with mlflow.start_run(run_name="baseline-nomic"):
        s = evaluate(b_index, b_chunks, baseline_embed, eval_pairs)
        mlflow.log_metrics({k: v for k, v in s.items()
                           if isinstance(v, float)})
        all_scores["Baseline (nomic-embed-text)"] = s

    # Fine-tuned embedding
    print("Evaluating fine-tuned embedding...")
    with mlflow.start_run(run_name="finetuned-arxiv"):
        s = evaluate(ft_index, ft_chunks, ft_embed, eval_pairs)
        mlflow.log_metrics({k: v for k, v in s.items()
                           if isinstance(v, float)})
        all_scores["Fine-tuned (arXiv only)"] = s

    # Save
    out = os.path.join(FINETUNED_DIR, "full_scores.json")
    with open(out, "w") as f:
        json.dump(all_scores, f, indent=2)

    print_results(all_scores)
    print(f"Saved to {out}")
    print("Run `mlflow ui` to view dashboard.")

if __name__ == "__main__":
    main()