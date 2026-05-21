import os
import json
import pickle
import numpy as np
import faiss
import mlflow
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config import (
    DATA_DIR, BASELINE_DIR, FINETUNED_DIR,
    FINETUNED_MODEL_PATH, TOP_K
)
from baseline_rag import load_index as load_baseline_index, embed_text as baseline_embed

# ── Eval set — always last 100 pairs, never used in training ──────────────────

def load_eval_pairs():
    with open(os.path.join(DATA_DIR, "training_pairs.json")) as f:
        pairs = json.load(f)
    return pairs[-100:]       # same 100 every time

# ── Shared scoring logic ───────────────────────────────────────────────────────

def score_results(retrieved_texts, ground_truth):
    """Return (hit, reciprocal_rank) for one query."""
    for rank, text in enumerate(retrieved_texts, start=1):
        gt_words  = set(ground_truth.lower().split())
        ret_words = set(text.lower().split())
        overlap   = len(gt_words & ret_words) / len(gt_words) if gt_words else 0
        if overlap > 0.5:
            return 1, 1.0 / rank
    return 0, 0.0

def compute_metrics(results_list):
    """results_list: list of (hit, rr) tuples."""
    hits = [h for h, _ in results_list]
    rrs  = [r for _, r in results_list]
    return {
        "hit_rate": round(float(np.mean(hits)), 4),
        "mrr":      round(float(np.mean(rrs)),  4),
    }

# ── BM25 evaluator ────────────────────────────────────────────────────────────

def evaluate_bm25(eval_pairs, chunks):
    print("\nEvaluating BM25...")
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    results = []
    for pair in tqdm(eval_pairs):
        query_tokens    = pair["query"].lower().split()
        scores          = bm25.get_scores(query_tokens)
        top_k_idxs      = np.argsort(scores)[::-1][:TOP_K]
        retrieved_texts = [chunks[i]["text"] for i in top_k_idxs]
        results.append(score_results(retrieved_texts, pair["positive"]))

    return compute_metrics(results)

# ── Dense retrieval evaluator (works for both baseline and finetuned) ─────────

def evaluate_dense(eval_pairs, index, chunks, embed_fn):
    results = []
    for pair in tqdm(eval_pairs):
        q_emb = np.array([embed_fn(pair["query"])], dtype="float32")
        faiss.normalize_L2(q_emb)
        _, idxs = index.search(q_emb, TOP_K)
        retrieved_texts = [
            chunks[i]["text"] for i in idxs[0] if i >= 0
        ]
        results.append(score_results(retrieved_texts, pair["positive"]))
    return compute_metrics(results)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    eval_pairs = load_eval_pairs()
    print(f"Eval set: {len(eval_pairs)} pairs")

    # Load indexes and chunks
    baseline_index, baseline_chunks = load_baseline_index(
        name="baseline",
        models_dir=BASELINE_DIR
    )

    ft_index  = faiss.read_index(os.path.join(FINETUNED_DIR, "finetuned.faiss"))
    with open(os.path.join(FINETUNED_DIR, "finetuned_chunks.pkl"), "rb") as f:
        ft_chunks = pickle.load(f)

    ft_model = SentenceTransformer(FINETUNED_MODEL_PATH)
    ft_embed = lambda text: ft_model.encode(
        text, normalize_embeddings=True
    ).tolist()

    # ── MLflow experiment ─────────────────────────────────────────────────────
    mlflow.set_experiment("rag-embedding-comparison")

    all_scores = {}

    # BM25
    with mlflow.start_run(run_name="bm25"):
        mlflow.log_param("method", "BM25-Okapi")
        mlflow.log_param("top_k", TOP_K)
        scores = evaluate_bm25(eval_pairs, baseline_chunks)
        mlflow.log_metrics(scores)
        all_scores["BM25"] = scores
        print(f"BM25         → Hit Rate: {scores['hit_rate']}  MRR: {scores['mrr']}")

    # Baseline embedding
    with mlflow.start_run(run_name="baseline-embedding"):
        mlflow.log_param("method", "nomic-embed-text (off-the-shelf)")
        mlflow.log_param("top_k", TOP_K)
        print("\nEvaluating baseline embedding...")
        scores = evaluate_dense(eval_pairs, baseline_index, baseline_chunks, baseline_embed)
        mlflow.log_metrics(scores)
        all_scores["Baseline (nomic-embed-text)"] = scores
        print(f"Baseline     → Hit Rate: {scores['hit_rate']}  MRR: {scores['mrr']}")

    # Fine-tuned embedding
    with mlflow.start_run(run_name="finetuned-embedding"):
        mlflow.log_param("method", "all-MiniLM-L6-v2 fine-tuned on arXiv")
        mlflow.log_param("base_model", "all-MiniLM-L6-v2")
        mlflow.log_param("train_pairs", 2700)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("top_k", TOP_K)
        print("\nEvaluating fine-tuned embedding...")
        scores = evaluate_dense(eval_pairs, ft_index, ft_chunks, ft_embed)
        mlflow.log_metrics(scores)
        all_scores["Fine-tuned (ours)"] = scores
        print(f"Fine-tuned   → Hit Rate: {scores['hit_rate']}  MRR: {scores['mrr']}")

    # Save results
    out_path = os.path.join(FINETUNED_DIR, "comparison_scores.json")
    with open(out_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"\nAll scores saved to {out_path}")
    print("Run `mlflow ui` to view experiment tracking dashboard.")

if __name__ == "__main__":
    main()
    
#source /Users/alisha/rag-finetuning-project/.venv/bin/activate