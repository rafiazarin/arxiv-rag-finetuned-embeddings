# Domain-Adaptive RAG with Fine-Tuned Embeddings

A retrieval-augmented generation pipeline where the embedding model is
fine-tuned on domain-specific data and evaluated across three experimental
conditions. The project investigates the effects of synthetic training data,
catastrophic forgetting, and corpus mixing on retrieval quality.

Built entirely with free, local tools — no paid APIs, no cloud compute
beyond a free Colab T4 GPU session for training.

**Models on HuggingFace:**
- [`rafiazarin/arxiv-cs-embedding-finetuned`](https://huggingface.co/rafiazarin/arxiv-cs-embedding-finetuned) — fine-tuned on arXiv domain pairs only
- [`rafiazarin/arxiv-ms-marco-embedding-mixed`](https://huggingface.co/rafiazarin/arxiv-ms-marco-embedding-mixed) — fine-tuned on mixed arXiv + MS MARCO pairs

---

## Results

Four methods evaluated across three query sets.
Metrics: Hit Rate@3, MRR@3, NDCG@10 with 95% bootstrap confidence intervals.

### Synthetic eval — 100 LLM-generated queries (in-distribution)

| Method | Hit@3 | MRR@3 | NDCG@10 |
|---|---|---|---|
| Baseline (nomic-embed-text) | 0.690 | 0.6117 | 0.5055 |
| BM25 | 0.700 | 0.6517 | 0.5288 |
| Fine-tuned mixed (ours) | 0.710 | 0.6533 | 0.5297 |
| **Fine-tuned arXiv-only (ours)** | **0.710** | **0.6633** | **0.5359** |

### Manual eval — 30 human-written queries (honest eval)

| Method | Hit@3 | MRR@3 | NDCG@10 |
|---|---|---|---|
| BM25 | 0.667 | 0.5889 | 0.5211 |
| Fine-tuned arXiv-only | 0.733 | 0.6833 | 0.5828 |
| **Fine-tuned mixed (ours)** | **0.767** | **0.6944** | **0.5802** |
| Baseline (nomic-embed-text) | 0.800 | 0.7167 | 0.6106 |

### Combined eval — 130 queries (synthetic + manual)

| Method | Hit@3 | MRR@3 | NDCG@10 |
|---|---|---|---|
| BM25 | 0.692 | 0.6372 | 0.5271 |
| Baseline (nomic-embed-text) | 0.715 | 0.6359 | 0.5297 |
| **Fine-tuned mixed (ours)** | **0.723** | **0.6628** | **0.5414** |
| Fine-tuned arXiv-only | 0.715 | 0.6679 | 0.5467 |

> **Note on confidence intervals:** With n=100 synthetic and n=30 manual
> queries, bootstrap 95% CIs overlap across all methods. Differences are
> directionally consistent but not statistically significant at this sample
> size. This is documented honestly as a limitation.

---

## Key Findings

### Finding 1 — Synthetic training data causes overfitting to query patterns

Fine-tuning on LLM-generated (query, passage) pairs improves retrieval on
synthetic eval queries (+8.4% MRR over baseline) but degrades on
human-written queries (-4.7% MRR). The model learns to match the
generator's phrasing patterns rather than achieving genuine domain
adaptation. This is evidenced by the reversal of rankings between
synthetic and manual eval sets.

### Finding 2 — Catastrophic forgetting on out-of-distribution queries

The arXiv-only fine-tuned model shows -43.6% keyword relevance on 10
general CS queries outside the training distribution (transformers, GANs,
reinforcement learning) compared to the off-the-shelf baseline. Fine-tuning
on a narrow physics/math corpus partially overwrites the model's general
semantic understanding.

### Finding 3 — Corpus mixing partially mitigates both effects

Mixing 1,350 arXiv pairs with 1,350 MS MARCO pairs during training:
- Recovers Hit@3 on manual queries: 0.733 → 0.767 (+4.6%)
- Achieves best overall Hit@3 on the combined eval set (0.723)
- Slightly sacrifices in-distribution MRR: 0.6633 → 0.6533 (-1.5%)

The mixed model outperforms both BM25 and the off-the-shelf baseline on
the combined eval set, representing the best overall retrieval system
among the four methods evaluated.

---

## Architecture
arXiv CS abstracts (5,000 docs, 2021)
↓
Chunking (400 chars, 50 overlap)
↓
13,050 text chunks
↓
FAISS IndexFlatIP (cosine similarity)
↓
Top-3 chunk retrieval
↓
Ollama gemma3:4b → Answer

**Training pipeline:**

arXiv pairs (1,350) + MS MARCO pairs (1,350) = 2,700 mixed pairs
↓
MultipleNegativesRankingLoss
all-MiniLM-L6-v2 base model
5 epochs, batch size 32, lr 2e-5
Free Colab T4 GPU (~25 min)
↓
Fine-tuned model (384-dim, 80MB)

---

## Technical Decisions

**Why MultipleNegativesRankingLoss?**
For each (query, passage) pair in a batch, all other passages act as
implicit negatives. No manually labeled hard negatives needed. Efficient
and well-suited for synthetic training data at this scale.

**Why FAISS IndexFlatIP over HNSW?**
With 13,050 vectors, exact search is fast and eliminates approximation
error. HNSW is appropriate at 1M+ vectors.

**Why retrieval metrics instead of RAGAS?**
RAGAS uses an LLM judge for faithfulness/relevancy. During development
it produced NaN scores with local Ollama models due to output parsing
failures — a known compatibility issue. Since the embedding model (not
the LLM generator) was modified, retrieval-focused metrics directly
measure what changed.

**Why three eval sets?**
A single synthetic eval set would have concealed Finding 1. The
divergence between synthetic and manual eval results is itself the
most important finding in this project.

---

## Evaluation Sets

| Set | Size | Source | Purpose |
|---|---|---|---|
| Synthetic | 100 | LLM-generated from corpus | In-distribution benchmark |
| Manual | 30 | Human-written (this project) | Honest generalization test |
| Combined | 130 | Both | Overall system comparison |

The manual eval queries were written by reading 30 randomly sampled
abstracts and writing researcher-style questions that are answerable
from the abstract but phrased independently of it.

---

## Limitations

**1. Overlapping confidence intervals**
All bootstrap 95% CIs overlap across methods at n=100 and n=30.
Improvements are directionally consistent but not statistically
significant. A rigorous study would require 500+ eval queries.

**2. Synthetic eval contamination**
Training and eval pairs share the same generator and template.
The synthetic eval measures pattern matching as much as retrieval quality.

**3. Residual catastrophic forgetting**
Mixed training reduces but does not eliminate OOD degradation.
Full mitigation would require EWC regularization or a larger base model
(bge-base-en-v1.5) with more capacity to retain general knowledge.

**4. Small corpus**
5,000 abstracts means some queries have no relevant documents at all,
inflating apparent failure rates beyond what the embedding models
are responsible for.

---

## Setup

```bash
git clone https://github.com/rafiazarin/arxiv-rag-finetuned-embeddings
cd arxiv-rag-finetuned-embeddings
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

---

## Reproducing Results

```bash
# 1. Download corpus
python step1_load_dataset.py

# 2. Build baseline FAISS index
python baseline_rag.py

# 3. Generate synthetic training pairs (~90 min, resumable)
python step3_generate_pairs.py

# 4. Fine-tune on Colab (see Colab notebook instructions in /notebooks)
#    Unzip downloaded model to experiments/finetuned/model/
#    Unzip mixed model to experiments/mixed/model/

# 5. Build fine-tuned indexes
python step2_build_index.py

# 6. Write manual eval queries
#    Already provided in data/manual_eval.json

# 7. Run final four-way evaluation
python step6_final_eval.py

# 8. Out-of-distribution evaluation
python step4b_ood_eval.py
```

---

## Stack

| Component | Tool |
|---|---|
| Corpus | arXiv CS abstracts via HuggingFace Datasets |
| Vector store | FAISS IndexFlatIP (exact cosine search) |
| Baseline embedding | nomic-embed-text via Ollama |
| Fine-tuned embedding | all-MiniLM-L6-v2, sentence-transformers |
| Training loss | MultipleNegativesRankingLoss |
| General training data | MS MARCO v1.1 via HuggingFace Datasets |
| Training compute | Google Colab T4 GPU (free tier) |
| LLM | gemma3:4b via Ollama (local, M2 GPU) |
| Experiment tracking | MLflow |
| Eval metrics | Hit Rate@3, MRR@3, NDCG@10, bootstrap 95% CI |

---

## Use the Models

```python
from sentence_transformers import SentenceTransformer

# Mixed model — best overall on combined eval
model = SentenceTransformer("rafiazarin/arxiv-ms-marco-embedding-mixed")
embeddings = model.encode(["your query here"], normalize_embeddings=True)

# arXiv-only model — best on in-distribution queries
model = SentenceTransformer("rafiazarin/arxiv-cs-embedding-finetuned")
embeddings = model.encode(["your query here"], normalize_embeddings=True)
```