"""
step10b_build_pubmed_index.py

Builds the PubMed baseline FAISS index from the cached pubmed_pool.json.
Also splits pubmed_manual_eval.json into val (10) and eval (90) sets.

Run this on M2 before starting pair generation.
Runtime: ~15 min (embedding 2000 abstracts chunked to ~5000 chunks)

Usage:
    python step10b_build_pubmed_index.py
"""

import os
import re
import json
import pickle
import random
import numpy as np
import faiss
import ollama
from tqdm import tqdm
from config import DATA_DIR, EXPERIMENTS_DIR

# ── Config ────────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50
OLLAMA_EMBED  = "nomic-embed-text"
SEED          = 42

POOL_PATH     = os.path.join(DATA_DIR, "pubmed_pool.json")
MANUAL_PATH   = os.path.join(DATA_DIR, "pubmed_manual_eval.json")
OUT_DIR       = os.path.join(EXPERIMENTS_DIR, "pubmed_baseline")
OUT_INDEX     = os.path.join(OUT_DIR, "pubmed_baseline.faiss")
OUT_CHUNKS    = os.path.join(OUT_DIR, "pubmed_baseline_chunks.pkl")
OUT_VAL       = os.path.join(DATA_DIR, "pubmed_manual_val.json")
OUT_EVAL      = os.path.join(DATA_DIR, "pubmed_manual_eval90.json")

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)

# ── Step 1: Split manual eval ─────────────────────────────────────────────────

def split_manual_eval():
    if os.path.exists(OUT_VAL) and os.path.exists(OUT_EVAL):
        print("Manual eval already split — skipping.")
        return

    with open(MANUAL_PATH) as f:
        queries = json.load(f)

    indices  = list(range(len(queries)))
    val_idx  = set(random.sample(indices, 10))
    eval_idx = [i for i in indices if i not in val_idx]

    val_queries  = [queries[i] for i in sorted(val_idx)]
    eval_queries = [queries[i] for i in eval_idx]

    with open(OUT_VAL, "w") as f:
        json.dump(val_queries, f, indent=2)
    with open(OUT_EVAL, "w") as f:
        json.dump(eval_queries, f, indent=2)

    print(f"Split manual eval: {len(val_queries)} val, {len(eval_queries)} eval")

# ── Step 2: Chunk ─────────────────────────────────────────────────────────────

def chunk_documents(docs):
    chunks = []
    for doc in docs:
        text  = doc["text"]
        start = 0
        idx   = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunks.append({
                "chunk_id": f"pubmed_{len(chunks)}",
                "doc_id":   str(idx),
                "text":     text[start:end]
            })
            if end == len(text):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks

# ── Step 3: Embed ─────────────────────────────────────────────────────────────

def embed_chunks(chunks):
    partial_path = os.path.join(OUT_DIR, "pubmed_embeddings_partial.npy")
    start_idx    = 0
    embeddings   = []

    if os.path.exists(partial_path):
        embeddings = list(np.load(partial_path))
        start_idx  = len(embeddings)
        print(f"Resuming from chunk {start_idx:,}")

    print(f"Embedding {len(chunks):,} chunks with {OLLAMA_EMBED}...")
    for i, chunk in enumerate(tqdm(chunks[start_idx:], initial=start_idx,
                                    total=len(chunks), desc="Embedding")):
        response = ollama.embeddings(model=OLLAMA_EMBED, prompt=chunk["text"])
        embeddings.append(response["embedding"])
        if (start_idx + i + 1) % 500 == 0:
            np.save(partial_path, np.array(embeddings, dtype="float32"))

    emb_array = np.array(embeddings, dtype="float32")
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return emb_array

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Verify Ollama
    print("Checking Ollama...")
    try:
        test = ollama.embeddings(model=OLLAMA_EMBED, prompt="test")
        print(f"Ollama OK — dim={len(test['embedding'])}\n")
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable: {e}")

    # Split manual eval first
    split_manual_eval()

    # Load corpus
    with open(POOL_PATH) as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} PubMed abstracts")

    # Skip if index already exists
    if os.path.exists(OUT_INDEX) and os.path.exists(OUT_CHUNKS):
        print(f"Index already exists — skipping. Delete {OUT_DIR} to rebuild.")
        return

    chunks = chunk_documents(docs)
    print(f"Chunked into {len(chunks):,} chunks")

    embeddings = embed_chunks(chunks)

    print("\nBuilding FAISS index...")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, OUT_INDEX)
    with open(OUT_CHUNKS, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nSaved index  : {OUT_INDEX}  ({index.ntotal} vectors)")
    print(f"Saved chunks : {OUT_CHUNKS}")
    print(f"\nNext: run step10c_generate_pubmed_pairs.py")

if __name__ == "__main__":
    main()
