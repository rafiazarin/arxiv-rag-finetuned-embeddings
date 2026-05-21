import os
import json
import pickle
import faiss
import numpy as np
import ollama
from tqdm import tqdm
from config import (
    DATA_DIR, BASELINE_DIR, OLLAMA_LLM, OLLAMA_EMBED,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
)

# ── 1. Chunking ────────────────────────────────────────────────────────────────

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

# ── 2. Embedding ───────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    response = ollama.embeddings(model=OLLAMA_EMBED, prompt=text)
    return response["embedding"]

def embed_chunks(chunks, batch_size=32):
    embeddings = []
    print(f"Embedding {len(chunks)} chunks with {OLLAMA_EMBED}...")
    for chunk in tqdm(chunks):
        emb = embed_text(chunk["text"])
        embeddings.append(emb)
    return np.array(embeddings, dtype="float32")

# ── 3. FAISS index ─────────────────────────────────────────────────────────────

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def save_index(index, chunks, name="baseline"):
    faiss.write_index(index, os.path.join(BASELINE_DIR, f"{name}.faiss"))
    with open(os.path.join(BASELINE_DIR, f"{name}_chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {name} index with {index.ntotal} vectors.")

def load_index(name="baseline", models_dir=None):
    if models_dir is None:
        models_dir = BASELINE_DIR
    index = faiss.read_index(os.path.join(models_dir, f"{name}.faiss"))
    with open(os.path.join(models_dir, f"{name}_chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# ── 4. Retrieval ───────────────────────────────────────────────────────────────

def retrieve(query: str, index, chunks, embed_fn=embed_text, k=TOP_K):
    q_emb = np.array([embed_fn(query)], dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "chunk": chunks[idx],
                "score": float(score)
            })
    return results

# ── 5. Generation ──────────────────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: list) -> str:
    context = "\n\n".join([r["chunk"]["text"] for r in retrieved_chunks])
    prompt = f"""You are a helpful research assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
    response = ollama.generate(model=OLLAMA_LLM, prompt=prompt)
    return response["response"].strip()

# ── 6. Full RAG query ──────────────────────────────────────────────────────────

def rag_query(query: str, index, chunks, embed_fn=embed_text):
    retrieved = retrieve(query, index, chunks, embed_fn)
    answer    = generate_answer(query, retrieved)
    contexts  = [r["chunk"]["text"] for r in retrieved]
    return answer, contexts

# ── 7. Build and save baseline ─────────────────────────────────────────────────

def build_baseline():
    corpus_path = os.path.join(DATA_DIR, "corpus.json")
    with open(corpus_path) as f:
        docs = json.load(f)

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} documents.")

    embeddings = embed_chunks(chunks)
    index      = build_index(embeddings)
    save_index(index, chunks, name="baseline")

    print("\nSanity check — querying: 'What is attention mechanism in transformers?'")
    retrieved = retrieve("What is attention mechanism in transformers?", index, chunks)
    print(f"Top result: {retrieved[0]['chunk']['text'][:200]}")

if __name__ == "__main__":
    build_baseline()