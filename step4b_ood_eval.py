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
from config import BASELINE_DIR, FINETUNED_DIR, FINETUNED_MODEL_PATH, TOP_K, EXPERIMENTS_DIR
#from config import BASELINE_DIR, FINETUNED_DIR, FINETUNED_MODEL_PATH, TOP_K

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
#from config import BASELINE_DIR, FINETUNED_DIR, FINETUNED_MODEL_PATH, TOP_K, EXPERIMENTS_DIR
def main():
    b_index = faiss.read_index(f"{BASELINE_DIR}/baseline.faiss")
    with open(f"{BASELINE_DIR}/baseline_chunks.pkl", "rb") as f:
        b_chunks = pickle.load(f)

    ft_index = faiss.read_index(f"{FINETUNED_DIR}/finetuned.faiss")
    with open(f"{FINETUNED_DIR}/finetuned_chunks.pkl", "rb") as f:
        ft_chunks = pickle.load(f)

    ft_model = SentenceTransformer(FINETUNED_MODEL_PATH)
    ft_embed = lambda t: ft_model.encode(t, normalize_embeddings=True).tolist()

    # Load mixed models
    mixed_models = {}
    for ratio in [80, 70, 50]:
        model_path  = os.path.join(EXPERIMENTS_DIR, f"mixed_{ratio}", "model")
        index_path  = os.path.join(EXPERIMENTS_DIR, f"mixed_{ratio}", "index.faiss")
        chunks_path = os.path.join(EXPERIMENTS_DIR, f"mixed_{ratio}", "chunks.pkl")
        if os.path.exists(model_path) and os.path.exists(index_path):
            mx_index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                mx_chunks = pickle.load(f)
            mx_model = SentenceTransformer(model_path)
            mx_embed = lambda t, m=mx_model: m.encode(
                t, normalize_embeddings=True).tolist()
            mixed_models[ratio] = (mx_index, mx_chunks, mx_embed)

    print("Running out-of-distribution evaluation...\n")

    results = {}
    results["Baseline"] = evaluate_ood(b_index, b_chunks, baseline_embed, "baseline")
    results["Fine-tuned (arXiv 100%)"] = evaluate_ood(
        ft_index, ft_chunks, ft_embed, "finetuned")
    for ratio, (mx_index, mx_chunks, mx_embed) in mixed_models.items():
        label = f"Mixed {ratio}/{100-ratio}"
        results[label] = evaluate_ood(mx_index, mx_chunks, mx_embed, label)

    # Print table
    col = 30
    print(f"\n{'Method':<{col}} {'OOD Score':>10}")
    print("-" * (col + 12))
    for label, (score, _) in sorted(results.items(), key=lambda x: x[1][0]):
        delta = score - results["Baseline"][0]
        marker = " ⚠" if delta < 0 else " ✓"
        print(f"  {label:<{col}} {score:>10.4f}  ({delta:+.4f}){marker}")

    # Save
    out = os.path.join(EXPERIMENTS_DIR, "ood_results_all.json")
    with open(out, "w") as f:
        json.dump(
            {k: {"mean_relevance": v[0], "queries": v[1]}
             for k, v in results.items()},
            f, indent=2
        )
    print(f"\nSaved to {out}")
if __name__ == "__main__":
    main()