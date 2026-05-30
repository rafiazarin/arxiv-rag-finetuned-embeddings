"""
step3b_mine_negatives.py

Mines hard negatives for the 2700 arXiv training pairs using the existing
baseline FAISS index (nomic-embed-text embeddings).

For each (query, positive) pair:
  1. Embed the query with nomic via Ollama
  2. Retrieve top-K chunks from the baseline index
  3. Skip the chunk that matches the positive passage
  4. Take the highest-scoring non-positive chunk as the hard negative

Output: data/hard_negative_pairs.json
  [{query, positive, hard_negative}, ...]

This file is then uploaded to Colab for retraining with TripletLoss.

Usage:
    python step3b_mine_negatives.py

Requirements:
    - Ollama running locally with nomic-embed-text pulled
    - experiments/baseline/baseline.faiss
    - experiments/baseline/baseline_chunks.pkl
    - data/training_pairs.json
"""

import os
import json
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from baseline_rag import embed_text, load_index
from config import DATA_DIR, BASELINE_DIR, EXPERIMENTS_DIR

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_SIZE       = 2700        # same slice as all previous runs
SEARCH_K         = 10          # retrieve top-10, pick best non-positive
MATCH_THRESHOLD  = 0.5         # word overlap threshold to detect positive match
OUTPUT_PATH      = os.path.join(DATA_DIR, "hard_negative_pairs.json")

# ── Match: detect if a retrieved chunk is the positive passage ────────────────

def is_positive(chunk_text, positive_text, threshold=MATCH_THRESHOLD):
    """
    Lightweight word-overlap check — just used to skip the positive chunk.
    Does NOT need to be perfect; false negatives just mean we pick a
    slightly easier negative, which is fine.
    """
    a = set(chunk_text.lower().split())
    b = set(positive_text.lower().split())
    if not a:
        return False
    return len(a & b) / len(a) > threshold

# ── Main mining loop ──────────────────────────────────────────────────────────

def mine(train_pairs, index, chunks):
    triplets   = []
    skipped    = 0   # pairs where no hard negative was found

    for pair in tqdm(train_pairs, desc="Mining hard negatives"):
        query    = pair["query"]
        positive = pair["positive"]

        # Embed query with nomic via Ollama (same model that built the index)
        q_emb = np.array([embed_text(query)], dtype="float32")
        faiss.normalize_L2(q_emb)

        # Retrieve top-K
        scores, idxs = index.search(q_emb, SEARCH_K)

        hard_negative = None
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            candidate = chunks[idx]["text"]
            # Skip if this chunk IS the positive passage
            if is_positive(candidate, positive):
                continue
            # First non-positive hit = hardest negative
            hard_negative = candidate
            break

        if hard_negative is None:
            # Rare: all top-K were positive matches — skip this pair
            skipped += 1
            continue

        triplets.append({
            "query":         query,
            "positive":      positive,
            "hard_negative": hard_negative
        })

    print(f"\nMined {len(triplets)} triplets  |  skipped {skipped} pairs (no negative found)")
    return triplets

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load training pairs — same split as all previous runs
    pairs_path = os.path.join(DATA_DIR, "training_pairs.json")
    with open(pairs_path) as f:
        all_pairs = json.load(f)

    train_pairs = all_pairs[:TRAIN_SIZE]   # first 2700
    print(f"Loaded {len(train_pairs)} training pairs")

    # Load baseline index
    print("Loading baseline FAISS index...")
    index, chunks = load_index(name="baseline", models_dir=BASELINE_DIR)
    print(f"Index: {index.ntotal} vectors  |  Chunks: {len(chunks)}")

    # Verify Ollama is reachable before starting the long loop
    print("Verifying Ollama connection...")
    try:
        test = embed_text("test")
        assert len(test) > 0
        print(f"Ollama OK — embedding dim: {len(test)}")
    except Exception as e:
        raise RuntimeError(
            f"Ollama not reachable: {e}\n"
            "Make sure Ollama is running: `ollama serve`\n"
            "And nomic-embed-text is pulled: `ollama pull nomic-embed-text`"
        )

    # Mine
    print(f"\nMining hard negatives for {len(train_pairs)} pairs...")
    print("This will take ~30-40 minutes on M2 (one Ollama call per query).\n")
    triplets = mine(train_pairs, index, chunks)

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(triplets, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

    # Quick sanity check — print one example
    if triplets:
        ex = triplets[0]
        print("\nExample triplet:")
        print(f"  Query    : {ex['query'][:100]}")
        print(f"  Positive : {ex['positive'][:100]}")
        print(f"  Hard neg : {ex['hard_negative'][:100]}")

if __name__ == "__main__":
    main()
