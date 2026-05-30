"""
step8_scale_corpus.py

Scales the retrieval corpus from 5,000 to 20,000 arXiv abstracts and builds
a new baseline FAISS index. Designed to run overnight on M2.

What it does:
    1. Loads 20,000 abstracts from gfissore/arxiv-abstracts-2021
       (identical cleaning logic to step1_load_dataset.py)
    2. Saves to data/corpus_20k.json
    3. Chunks all docs at 400 chars / 50-char overlap (~52,000 chunks)
    4. Embeds every chunk with nomic-embed-text via Ollama
    5. Builds FAISS IndexFlatIP and saves to experiments/baseline_20k/

Scientific purpose:
    Tests whether fine-tuned models trained on the 5k corpus generalise to
    retrieval from a 4x larger corpus. The existing training pairs are a
    subset of the new corpus so retraining is not required for the first
    evaluation — just re-index and re-eval.

Runtime estimate on M2:
    Chunking  : ~2 min
    Embedding : ~25-35 min  (nomic processes ~25 chunks/sec via Ollama)
    Indexing  : ~1 min
    Total     : ~30-40 min

Usage:
    python step8_scale_corpus.py

Safe to run overnight — progress is printed every 1000 chunks.
"""

import os
import re
import json
import pickle
import numpy as np
import faiss
import ollama
from tqdm import tqdm
from datasets import load_dataset
from config import DATA_DIR, EXPERIMENTS_DIR

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_NAME  = "gfissore/arxiv-abstracts-2021"
DATASET_SPLIT = "train"
NUM_DOCS_20K  = 20_000
CHUNK_SIZE    = 400     # identical to original
CHUNK_OVERLAP = 50      # identical to original
OLLAMA_EMBED  = "nomic-embed-text"

OUT_CORPUS    = os.path.join(DATA_DIR, "corpus_20k.json")
OUT_DIR       = os.path.join(EXPERIMENTS_DIR, "baseline_20k")
OUT_INDEX     = os.path.join(OUT_DIR, "baseline_20k.faiss")
OUT_CHUNKS    = os.path.join(OUT_DIR, "baseline_20k_chunks.pkl")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: Cleaning (identical to step1_load_dataset.py) ─────────────────────

def clean_text(text):
    text = re.sub(r'\$[^$]+\$', '', text)
    text = re.sub(r'\\\w+\{[^}]*\}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Step 2: Load 20k abstracts ────────────────────────────────────────────────

def load_corpus():
    if os.path.exists(OUT_CORPUS):
        print(f"Found existing corpus_20k.json — loading from disk...")
        with open(OUT_CORPUS) as f:
            docs = json.load(f)
        print(f"Loaded {len(docs)} documents")
        return docs

    print(f"Loading {NUM_DOCS_20K} abstracts from {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, trust_remote_code=True)

    docs = []
    for item in tqdm(dataset, desc="Loading abstracts"):
        abstract = item.get("abstract", "") or item.get("text", "")
        title    = item.get("title", "")

        if not abstract or len(abstract) < 100:
            continue

        cleaned = clean_text(abstract)
        if len(cleaned) < 100:
            continue

        docs.append({
            "id":    item.get("id", str(len(docs))),
            "title": clean_text(title),
            "text":  cleaned
        })

        if len(docs) >= NUM_DOCS_20K:
            break

    with open(OUT_CORPUS, "w") as f:
        json.dump(docs, f, indent=2)

    print(f"Saved {len(docs)} documents to {OUT_CORPUS}")
    return docs

# ── Step 3: Chunk ─────────────────────────────────────────────────────────────

def chunk_documents(docs):
    chunks = []
    for doc in docs:
        text  = doc["text"]
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunks.append({
                "chunk_id": f"{doc['id']}_{len(chunks)}",
                "doc_id":   doc["id"],
                "title":    doc["title"],
                "text":     text[start:end]
            })
            if end == len(text):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ── Step 4: Embed ─────────────────────────────────────────────────────────────

def embed_chunks(chunks):
    print(f"\nEmbedding {len(chunks):,} chunks with {OLLAMA_EMBED}...")
    print("Progress saved every 1000 chunks — safe to interrupt and resume.\n")

    # Resume support — check for partial embeddings
    partial_path = os.path.join(OUT_DIR, "embeddings_partial.npy")
    start_idx    = 0
    embeddings   = []

    if os.path.exists(partial_path):
        embeddings = list(np.load(partial_path))
        start_idx  = len(embeddings)
        print(f"Resuming from chunk {start_idx:,}")

    for i, chunk in enumerate(tqdm(chunks[start_idx:], initial=start_idx,
                                    total=len(chunks), desc="Embedding")):
        response = ollama.embeddings(model=OLLAMA_EMBED, prompt=chunk["text"])
        embeddings.append(response["embedding"])

        # Save partial progress every 1000 chunks
        actual_idx = start_idx + i + 1
        if actual_idx % 1000 == 0:
            np.save(partial_path, np.array(embeddings, dtype="float32"))

    emb_array = np.array(embeddings, dtype="float32")

    # Clean up partial file
    if os.path.exists(partial_path):
        os.remove(partial_path)

    return emb_array

# ── Step 5: Build and save FAISS index ────────────────────────────────────────

def build_and_save_index(embeddings, chunks):
    print(f"\nBuilding FAISS index — {len(embeddings):,} vectors, dim={embeddings.shape[1]}")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, OUT_INDEX)
    with open(OUT_CHUNKS, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved index  : {OUT_INDEX}")
    print(f"Saved chunks : {OUT_CHUNKS}")
    print(f"Vectors      : {index.ntotal:,}  |  dim={embeddings.shape[1]}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Verify Ollama before starting
    print("Checking Ollama connection...")
    try:
        test = ollama.embeddings(model=OLLAMA_EMBED, prompt="test")
        assert len(test["embedding"]) > 0
        print(f"Ollama OK — dim={len(test['embedding'])}\n")
    except Exception as e:
        raise RuntimeError(
            f"Ollama not reachable: {e}\n"
            "Run: ollama serve\n"
            "And: ollama pull nomic-embed-text"
        )

    docs   = load_corpus()
    chunks = chunk_documents(docs)
    print(f"\nChunked {len(docs):,} docs into {len(chunks):,} chunks")

    # Skip embedding if index already exists
    if os.path.exists(OUT_INDEX) and os.path.exists(OUT_CHUNKS):
        print(f"\nIndex already exists at {OUT_INDEX} — skipping embedding.")
        print("Delete it manually if you want to rebuild.")
        return

    embeddings = embed_chunks(chunks)
    build_and_save_index(embeddings, chunks)

    print("\nDone. Next steps:")
    print("  1. Add 'baseline_20k' to step4d_eval_bge.py BGE_RUNS")
    print("     to evaluate existing models against the larger corpus")
    print("  2. Optionally generate new training pairs from corpus_20k.json")
    print("     using step3_generate_pairs.py with NUM_DOCS=20000")

if __name__ == "__main__":
    main()
