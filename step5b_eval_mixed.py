"""
step5b_eval_mixed.py

Evaluates all mixed-ratio models against baseline and arXiv-only fine-tuned.
Produces a ratio comparison table showing the specialization/generalization tradeoff.


# 1. Train all three ratio models (this will take a few hours total — let it run)
python step5_train_mixed.py

# 2. Or train one at a time if you want to check results as you go
python step5_train_mixed.py --ratio 80
python step5_train_mixed.py --ratio 70
python step5_train_mixed.py --ratio 50

# 3. Evaluate all ratios on manual (honest) eval set
python step5b_eval_mixed.py --eval manual

# 4. Then on synthetic to see the contamination contrast
python step5b_eval_mixed.py --eval synthetic


Usage:
    python step5b_eval_mixed.py
"""
import os
import json
import faiss
import pickle
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from config import (
    DATA_DIR, BASELINE_DIR, FINETUNED_DIR,
    FINETUNED_MODEL_PATH, EXPERIMENTS_DIR, TOP_K
)
from baseline_rag import load_index, embed_text as baseline_embed
from step4c_full_eval import evaluate, evaluate_bm25, load_eval_pairs, print_results

RATIOS = [80, 70, 50]   # must match step5_train_mixed.py

# ── Build FAISS index for a mixed model ───────────────────────────────────────

def load_or_build_index(ratio, base_chunks):
    mixed_dir   = os.path.join(EXPERIMENTS_DIR, f"mixed_{ratio}")
    model_path  = os.path.join(mixed_dir, "model")
    index_path  = os.path.join(mixed_dir, "index.faiss")
    chunks_path = os.path.join(mixed_dir, "chunks.pkl")

    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(model_path) or not os.path.exists(config_file):
        print(f"  Model for ratio {ratio} not trained yet — skipping.")
        return None, None, None

    if os.path.exists(index_path):
        print(f"  Loading cached index for ratio {ratio}...")
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
    else:
        print(f"  Building index for ratio {ratio} (~30 sec)...")
        model = SentenceTransformer(model_path)
        texts = [c["text"] for c in base_chunks]
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
            pickle.dump(base_chunks, f)
        chunks = base_chunks
        print(f"  Saved index: {index.ntotal} vectors")

    model     = SentenceTransformer(model_path)
    embed_fn  = lambda t: model.encode(t, normalize_embeddings=True).tolist()
    return index, chunks, embed_fn

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval", choices=["synthetic", "manual", "combined"],
        default="manual",
        help="Eval set to use (default: manual — the honest one)"
    )
    args = parser.parse_args()

    print(f"Evaluating all mixed-ratio models on '{args.eval}' eval set\n")

    # Load baseline
    b_index, b_chunks = load_index(name="baseline", models_dir=BASELINE_DIR)

    # Load arXiv-only fine-tuned
    ft_index = faiss.read_index(os.path.join(FINETUNED_DIR, "finetuned.faiss"))
    with open(os.path.join(FINETUNED_DIR, "finetuned_chunks.pkl"), "rb") as f:
        ft_chunks = pickle.load(f)
    ft_model  = SentenceTransformer(FINETUNED_MODEL_PATH)
    ft_embed  = lambda t: ft_model.encode(t, normalize_embeddings=True).tolist()

    eval_pairs = load_eval_pairs(eval_type=args.eval)
    print(f"Eval set: {len(eval_pairs)} queries ({args.eval})\n")

    all_scores = {}

    # BM25
    print("Evaluating BM25...")
    all_scores["BM25"] = evaluate_bm25(b_chunks, eval_pairs)

    # Baseline
    print("Evaluating Baseline (nomic-embed-text)...")
    all_scores["Baseline (nomic-embed-text)"] = evaluate(
        b_index, b_chunks, baseline_embed, eval_pairs)

    # arXiv-only fine-tuned
    print("Evaluating Fine-tuned (arXiv only)...")
    all_scores["Fine-tuned (arXiv 100%)"] = evaluate(
        ft_index, ft_chunks, ft_embed, eval_pairs)

    # Each mixed ratio
    for ratio in RATIOS:
        label = f"Mixed ({ratio}% arXiv / {100-ratio}% MARCO)"
        print(f"Evaluating {label}...")
        mx_index, mx_chunks, mx_embed = load_or_build_index(ratio, b_chunks)
        if mx_index is None:
            print(f"  Skipping {label} — model not trained yet.")
            continue
        all_scores[label] = evaluate(mx_index, mx_chunks, mx_embed, eval_pairs)

    # Print table
    print_results(all_scores)

    # Save
    out = os.path.join(EXPERIMENTS_DIR, f"mixed_ratio_scores_{args.eval}.json")
    with open(out, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nSaved to {out}")

    # Print the tradeoff summary
    print("\n── Specialization / Generalization Tradeoff ──────────────────")
    print(f"  {'Method':<40} {'Hit@3':>8} {'MRR':>8}")
    print(f"  {'-'*58}")
    for label, s in all_scores.items():
        print(f"  {label:<40} {s['hit_rate']:>8.4f} {s['mrr']:>8.4f}")

if __name__ == "__main__":
    main()