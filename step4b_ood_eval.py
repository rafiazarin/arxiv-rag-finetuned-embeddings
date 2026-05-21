"""
Out-of-distribution evaluation.
Tests both models on general CS queries NOT from the arXiv physics corpus
to detect catastrophic forgetting from domain fine-tuning.
"""
import faiss
import pickle
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from baseline_rag import embed_text as baseline_embed
from config import BASELINE_DIR, FINETUNED_DIR, FINETUNED_MODEL_PATH, TOP_K

# General CS queries that should be answerable from arXiv CS abstracts
# but are phrased differently from our training distribution
OOD_QUERIES = [
    {
        "query": "How do transformer attention mechanisms work?",
        "relevant_keywords": ["attention", "transformer", "self-attention", "query", "key", "value"]
    },
    {
        "query": "What are methods for training neural language models?",
        "relevant_keywords": ["language model", "neural network", "training", "gradient", "backpropagation"]
    },
    {
        "query": "How does reinforcement learning handle exploration vs exploitation?",
        "relevant_keywords": ["reinforcement learning", "exploration", "exploitation", "reward", "policy"]
    },
    {
        "query": "What is the role of convolutional layers in image recognition?",
        "relevant_keywords": ["convolutional", "image", "feature", "classification", "CNN"]
    },
    {
        "query": "How do graph neural networks propagate information?",
        "relevant_keywords": ["graph", "neural network", "node", "propagation", "message passing"]
    },
    {
        "query": "What clustering algorithms work well for high dimensional data?",
        "relevant_keywords": ["clustering", "high dimensional", "k-means", "dimensionality"]
    },
    {
        "query": "How does dropout prevent overfitting in deep networks?",
        "relevant_keywords": ["dropout", "overfitting", "regularization", "neural network"]
    },
    {
        "query": "What are common approaches to machine translation?",
        "relevant_keywords": ["translation", "sequence", "encoder", "decoder", "language"]
    },
    {
        "query": "How do generative adversarial networks produce realistic images?",
        "relevant_keywords": ["generative", "adversarial", "GAN", "discriminator", "generator"]
    },
    {
        "query": "What optimization algorithms are used in deep learning?",
        "relevant_keywords": ["optimization", "gradient descent", "Adam", "learning rate", "momentum"]
    },
]

def keyword_relevance(text, keywords):
    """Check if retrieved text contains relevant keywords."""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)

def evaluate_ood(index, chunks, embed_fn, label):
    results = []
    for item in OOD_QUERIES:
        q_emb = np.array([embed_fn(item["query"])], dtype="float32")
        faiss.normalize_L2(q_emb)
        _, idxs = index.search(q_emb, TOP_K)

        retrieved = [chunks[i]["text"] for i in idxs[0] if i >= 0]

        # Score: average keyword relevance across top-k results
        scores = [keyword_relevance(t, item["relevant_keywords"]) for t in retrieved]
        avg_score = np.mean(scores) if scores else 0.0
        results.append({
            "query": item["query"],
            "avg_keyword_relevance": round(float(avg_score), 4),
            "top1_preview": retrieved[0][:120] if retrieved else ""
        })

    mean_relevance = round(float(np.mean([r["avg_keyword_relevance"] for r in results])), 4)
    return mean_relevance, results

def main():
    # Load indexes
    b_index = faiss.read_index(f"{BASELINE_DIR}/baseline.faiss")
    with open(f"{BASELINE_DIR}/baseline_chunks.pkl", "rb") as f:
        b_chunks = pickle.load(f)

    ft_index = faiss.read_index(f"{FINETUNED_DIR}/finetuned.faiss")
    with open(f"{FINETUNED_DIR}/finetuned_chunks.pkl", "rb") as f:
        ft_chunks = pickle.load(f)

    ft_model  = SentenceTransformer(FINETUNED_MODEL_PATH)
    ft_embed  = lambda t: ft_model.encode(t, normalize_embeddings=True).tolist()

    print("Running out-of-distribution evaluation...")
    print("(10 general CS queries not from training distribution)\n")

    b_score,  b_details  = evaluate_ood(b_index,  b_chunks,  baseline_embed, "baseline")
    ft_score, ft_details = evaluate_ood(ft_index, ft_chunks, ft_embed,       "finetuned")

    # Print per-query comparison
    col = 48
    print(f"{'Query':<{col}} {'Baseline':>10} {'Fine-tuned':>12}")
    print("-" * (col + 24))
    for b, ft in zip(b_details, ft_details):
        marker = " ⚠" if ft["avg_keyword_relevance"] < b["avg_keyword_relevance"] else " ✓"
        print(f"  {b['query'][:col-2]:<{col}} {b['avg_keyword_relevance']:>10.4f} {ft['avg_keyword_relevance']:>12.4f}{marker}")

    print("-" * (col + 24))
    print(f"  {'Mean keyword relevance':<{col}} {b_score:>10.4f} {ft_score:>12.4f}")

    delta = round(ft_score - b_score, 4)
    direction = "degradation" if delta < 0 else "improvement"
    print(f"\n  OOD {direction}: {delta:+.4f}")

    # Save
    ood_results = {
        "baseline_mean_relevance":  b_score,
        "finetuned_mean_relevance": ft_score,
        "delta": delta,
        "queries": [
            {
                "query":    b["query"],
                "baseline": b["avg_keyword_relevance"],
                "finetuned": ft["avg_keyword_relevance"]
            }
            for b, ft in zip(b_details, ft_details)
        ]
    }
    out = f"{FINETUNED_DIR}/ood_eval.json"
    with open(out, "w") as f:
        json.dump(ood_results, f, indent=2)
    print(f"\n  Saved to {out}")
    print("\n  Note: OOD degradation on fine-tuned model is expected behaviour")
    print("  (catastrophic forgetting). See README limitations section.")

if __name__ == "__main__":
    main()