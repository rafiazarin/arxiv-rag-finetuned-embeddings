import os
import json
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import (
    DATA_DIR, BASELINE_DIR, FINETUNED_DIR,
    FINETUNED_MODEL_PATH, CHUNK_SIZE, CHUNK_OVERLAP
)

# ── Chunking (same logic as baseline — must be identical) ─────────────────────

def chunk_documents(docs):
    chunks = []
    for doc in docs:
        text = doc["text"]
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

# ── Embedding with sentence-transformers ──────────────────────────────────────

def embed_with_st_model(model, chunks, batch_size=64):
    """Embed all chunks using a SentenceTransformer model."""
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # required for cosine via IndexFlatIP
        convert_to_numpy=True
    )
    return embeddings.astype("float32")

# ── FAISS index ───────────────────────────────────────────────────────────────

def build_and_save_index(embeddings, chunks, out_dir, name):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(out_dir, f"{name}.faiss"))
    with open(os.path.join(out_dir, f"{name}_chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {name} index: {index.ntotal} vectors, dim={dim}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load corpus
    with open(os.path.join(DATA_DIR, "corpus.json")) as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} documents.")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # ── Fine-tuned index ──────────────────────────────────────────────────────
    print(f"\nLoading fine-tuned model from {FINETUNED_MODEL_PATH}...")
    ft_model = SentenceTransformer(FINETUNED_MODEL_PATH)

    ft_embeddings = embed_with_st_model(ft_model, chunks)
    build_and_save_index(ft_embeddings, chunks, FINETUNED_DIR, "finetuned")

    # ── Smoke test: compare baseline vs finetuned on one query ───────────────
    print("\n── Smoke test ──────────────────────────────────────────────────")
    query = "What methods are used for training neural language models?"

    # Load baseline index for comparison
    baseline_index = faiss.read_index(
        os.path.join(BASELINE_DIR, "baseline.faiss")
    )
    with open(os.path.join(BASELINE_DIR, "baseline_chunks.pkl"), "rb") as f:
        baseline_chunks = pickle.load(f)

    # Baseline uses Ollama — import from baseline_rag
    from baseline_rag import embed_text as baseline_embed

    # Baseline retrieval
    q_base = np.array([baseline_embed(query)], dtype="float32")
    faiss.normalize_L2(q_base)
    _, base_idxs = baseline_index.search(q_base, 3)

    # Finetuned retrieval
    q_ft = ft_model.encode([query], normalize_embeddings=True).astype("float32")
    _, ft_idxs = faiss.read_index(
        os.path.join(FINETUNED_DIR, "finetuned.faiss")
    ), None
    # reload to be safe
    ft_index = faiss.read_index(os.path.join(FINETUNED_DIR, "finetuned.faiss"))
    _, ft_idxs = ft_index.search(q_ft, 3)

    print(f"\nQuery: '{query}'")
    print("\nBaseline top-1:")
    print(f"  {baseline_chunks[base_idxs[0][0]]['text'][:200]}")
    print("\nFine-tuned top-1:")
    print(f"  {chunks[ft_idxs[0][0]]['text'][:200]}")
    print("\nIf these look different and fine-tuned looks more relevant — model is working.")

if __name__ == "__main__":
    main()