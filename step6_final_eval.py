"""
Final four-way evaluation:
  BM25 | Baseline (nomic) | Fine-tuned (arXiv only) | Fine-tuned (mixed)

Runs on three eval sets:
  synthetic (100q) | manual (30q) | combined (130q)

Produces the definitive results table for README and arXiv preprint.
"""
import os
import json
import faiss
import pickle
import numpy as np
import mlflow
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from config import (
    DATA_DIR, BASELINE_DIR, FINETUNED_DIR,
    FINETUNED_MODEL_PATH, EXPERIMENTS_DIR
)
from baseline_rag import load_index, embed_text as baseline_embed
from step4c_full_eval import (
    evaluate, evaluate_bm25, load_eval_pairs, print_results
)

MIXED_DIR        = os.path.join(EXPERIMENTS_DIR, "mixed_50")
MIXED_MODEL_PATH = os.path.join(MIXED_DIR, "model")

def build_mixed_index(chunks):
    """Build FAISS index for mixed model if not already built."""
    index_path  = os.path.join(MIXED_DIR, "mixed.faiss")
    chunks_path = os.path.join(MIXED_DIR, "mixed_chunks.pkl")

    if os.path.exists(index_path):
        print("Mixed index already exists, loading...")
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            saved_chunks = pickle.load(f)
        return index, saved_chunks

    print("Building mixed model index (~30 sec)...")
    model = SentenceTransformer(MIXED_MODEL_PATH)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved mixed index: {index.ntotal} vectors")
    return index, chunks

def run_eval_set(eval_type, b_index, b_chunks, ft_index, ft_chunks,
                 ft_embed, mx_index, mx_chunks, mx_embed, k=3):
    eval_pairs = load_eval_pairs(eval_type=eval_type)
    if not eval_pairs:
        return None

    print(f"\n{'='*60}")
    print(f"Eval set: {eval_type.upper()} ({len(eval_pairs)} queries)")
    print(f"{'='*60}")

    scores = {}

    print("BM25...")
    scores["BM25"] = evaluate_bm25(b_chunks, eval_pairs)

    print("Baseline (nomic-embed-text)...")
    scores["Baseline (nomic-embed-text)"] = evaluate(
        b_index, b_chunks, baseline_embed, eval_pairs)

    print("Fine-tuned (arXiv only)...")
    scores["Fine-tuned (arXiv only)"] = evaluate(
        ft_index, ft_chunks, ft_embed, eval_pairs)

    print("Fine-tuned (mixed arXiv+MARCO)...")
    scores["Fine-tuned (mixed)"] = evaluate(
        mx_index, mx_chunks, mx_embed, eval_pairs)

    print_results(scores)
    return scores

def main():
    # Load all indexes
    print("Loading indexes...")
    b_index, b_chunks = load_index(name="baseline", models_dir=BASELINE_DIR)

    ft_index = faiss.read_index(os.path.join(FINETUNED_DIR, "finetuned.faiss"))
    with open(os.path.join(FINETUNED_DIR, "finetuned_chunks.pkl"), "rb") as f:
        ft_chunks = pickle.load(f)

    mx_index, mx_chunks = build_mixed_index(b_chunks)

    # Embed functions
    ft_model = SentenceTransformer(FINETUNED_MODEL_PATH)
    ft_embed = lambda t: ft_model.encode(
        t, normalize_embeddings=True).tolist()

    mx_model = SentenceTransformer(MIXED_MODEL_PATH)
    mx_embed = lambda t: mx_model.encode(
        t, normalize_embeddings=True).tolist()

    # Run all three eval sets
    mlflow.set_experiment("rag-final-eval")
    all_results = {}

    for eval_type in ["synthetic", "manual", "combined"]:
        scores = run_eval_set(
            eval_type,
            b_index, b_chunks,
            ft_index, ft_chunks, ft_embed,
            mx_index, mx_chunks, mx_embed
        )
        if scores:
            all_results[eval_type] = scores

            with mlflow.start_run(run_name=f"final-{eval_type}"):
                for method, s in scores.items():
                    # sanitize method name for MLflow
                    key = method.replace("(", "").replace(")", "").replace(" ", "_").replace("-", "_")
                    mlflow.log_metrics({
                        f"{key}_hit":  s["hit_rate"],
                        f"{key}_mrr":  s["mrr"],
                        f"{key}_ndcg": s["ndcg_10"]
                    })

    # Save everything
    out = os.path.join(EXPERIMENTS_DIR, "final_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out}")

    # Print the key comparison — manual eval is the honest one
    print("\n" + "="*60)
    print("KEY FINDING — Manual eval (human-written queries):")
    print("="*60)
    manual = all_results.get("manual", {})
    for method, s in sorted(manual.items(), key=lambda x: x[1]["mrr"]):
        marker = " ◀" if "mixed" in method else ""
        print(f"  {method:<35} MRR={s['mrr']:.4f}  "
              f"NDCG@10={s['ndcg_10']:.4f}{marker}")

if __name__ == "__main__":
    main()