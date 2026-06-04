# Domain-Adaptive RAG with Fine-Tuned Embeddings

A systematic study of embedding model fine-tuning for retrieval-augmented
generation (RAG), covering six training strategies, two domains, a
cross-encoder evaluation fix, and a replication on biomedical text.

The central finding: standard RAG evaluation is vulnerable to two
compounding contamination mechanisms — using the same LLM for both
training data generation and evaluation queries, and using the same
bi-encoder model family as both the model under evaluation and the
relevance judge. Replacing the relevance judge with an independent
cross-encoder eliminates both effects and substantially changes the
conclusions.

Built entirely with free, local tools — no paid APIs, no cloud compute
beyond a free Colab T4 GPU session for fine-tuning.

**Models on HuggingFace:**
- [`rafiazarin/arxiv-cs-embedding-finetuned`](https://huggingface.co/rafiazarin/arxiv-cs-embedding-finetuned) — all-MiniLM fine-tuned on arXiv pairs
- [`rafiazarin/arxiv-ms-marco-embedding-mixed`](https://huggingface.co/rafiazarin/arxiv-ms-marco-embedding-mixed) — all-MiniLM fine-tuned on mixed arXiv + MS MARCO

**Companion paper:** *Fine-Tuning Embedding Models for RAG: Disentangling
Evaluation Contamination from Genuine Domain Adaptation* — manuscript in
preparation, 2025.

---

## Results

### arXiv — Primary Evaluation (n=140 human-written queries)

BGE-base-en-v1.5 variants, evaluated with an independent
**cross-encoder relevance matcher** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
No pairwise difference between the five standard variants is statistically
significant (McNemar p > 0.45; permutation p > 0.90 for all comparisons).

| Model | Training Data | Hit@3 | MRR | NDCG@10 |
|---|---|---|---|---|
| BGE-base | arXiv only (gemma) | **0.9071** | 0.8405 | 0.6717 |
| BGE-base | Mixed 50/50 | 0.9000 | **0.8464** | **0.6743** |
| BGE-base | arXiv only (mistral) | 0.9000 | 0.8429 | 0.6743 |
| BGE-base | Staged MARCO→arXiv | 0.9000 | 0.8417 | 0.6712 |
| BGE-base | Staged arXiv→MARCO | 0.9000 | 0.8405 | 0.6725 |
| BGE-base | Hard negatives ⚠️ | 0.7143 | 0.6131 | 0.5156 |
| — | — | — | — | — |
| nomic-embed-text † | none (no FT) | 0.9600 | 0.8911 | 0.7043 |
| BM25 † | n/a | 0.9133 | 0.8767 | 0.6798 |

> † Evaluated in a prior run using all-MiniLM-L6-v2 cosine matcher
> (threshold 0.75) — not directly comparable to cross-encoder rows above.
> Hard-negatives model is significantly worse than all others (p < 0.001).

### arXiv — Evaluation Mode Comparison (BGE arXiv-only, cross-encoder)

Shows that contamination from the prior bi-encoder matcher is fully
eliminated — synthetic queries now score *lower* than human queries,
as expected.

| Eval Set | Query Source | n | Hit@3 | MRR | NDCG@10 |
|---|---|---|---|---|---|
| Manual | Human-written | 140 | 0.9071 | 0.8405 | 0.6717 |
| Synthetic (held-out) | gemma3:4b | 100 | 0.8900 | 0.8050 | 0.6494 |
| Extreme ablation | Training pairs (direct) | 100 | 0.9300 | 0.8850 | 0.6618 |

> Under the prior bi-encoder matcher, the fine-tuned model appeared to gain
> +6.8% MRR on synthetic eval vs. the no-FT baseline while losing −1.2% on
> manual — a ranking reversal characteristic of evaluation contamination.
> This effect vanishes with the cross-encoder.

### PubMed — Domain Adaptation (n=90 human-written queries)

Both models evaluated with the same cross-encoder matcher — a fair,
direct comparison. Fine-tuning yields a large, statistically significant
improvement when the base model is genuinely out-of-distribution.

| Model | Training Data | Hit@3 | MRR | NDCG@10 | McNemar | Perm(MRR) |
|---|---|---|---|---|---|---|
| BGE-base (fine-tuned) | PubMed (gemma) | **0.8778** | **0.7722** | **0.5631** | p < 0.001 | p = 0.002 |
| nomic-embed-text | none (no FT) | 0.6889 | 0.5889 | 0.4722 | — | — |

---

## Key Findings

### Finding 1 — Two evaluation contamination mechanisms, not one

Standard RAG evaluation is contaminated in two compounding ways:
1. **Generator contamination** — using the same LLM to generate training
   pairs and evaluation queries makes the fine-tuned model appear better
   on eval than it is.
2. **Matcher contamination** — using the same bi-encoder model family as
   both the model under evaluation and the relevance judge inflates match
   rates by scoring retrieved content through the same representational lens.

Both effects are eliminated by using a cross-encoder for relevance
judgement. After this fix, the apparent +6.8% MRR advantage of the
fine-tuned model on synthetic eval disappears entirely.

### Finding 2 — Fine-tuning shows no significant benefit in-domain (arXiv)

With clean evaluation, all five standard BGE fine-tuning strategies
(arXiv-only, mixed, mistral pairs, staged variants) produce statistically
identical results. No training strategy, data source, or mixing ratio
produces a significant improvement over any other (p > 0.45 across all
comparisons, n=140). When the base model is already in-distribution,
domain-specific fine-tuning data adds negligible value.

### Finding 3 — Fine-tuning provides genuine improvement out-of-domain (PubMed)

On biomedical text, where nomic-embed-text's general pre-training data
underrepresents domain vocabulary, fine-tuning on 1,300 gemma-generated
PubMed pairs yields +18.9 pp Hit@3 (0.878 vs. 0.689, p < 0.001). The
improvement is *larger* on human-written queries than on synthetic ones —
the opposite of a contamination pattern. This confirms genuine domain
adaptation rather than evaluation artifact.

### Finding 4 — Hard negative mining collapses in narrow-domain corpora

ANCE-style hard negative mining produces Hit@3 = 0.714, significantly
below every other variant (p < 0.001). In a narrow corpus (arXiv
physics/math), topically similar passages are common, making them likely
to be relevant positives. Treating them as hard negatives during
MultipleNegativesRankingLoss training penalises the model for retrieving
on-topic content, inducing representational collapse. Hard negative mining
requires either a diverse corpus or a cross-encoder filter to screen false
negatives before training.

---

## Architecture

```
arXiv abstracts (5k or 20k docs)    PubMed abstracts (2k docs)
         ↓                                    ↓
   Chunking (400 tokens, 50 overlap)
         ↓                   ↓
   FAISS IndexFlatIP     Fine-tuned BGE-base-en-v1.5
   (cosine similarity)       (domain-specific pairs)
         ↓
   Top-3 chunk retrieval
         ↓
   Ollama gemma3:4b → Answer
```

**Fine-tuning pipeline (arXiv):**
```
gemma3:4b synthetic pairs (3,000)   MS MARCO pairs (1,500)
             ↓                              ↓
      MultipleNegativesRankingLoss — 6 training variants
      BGE-base-en-v1.5 base model
      Google Colab T4 GPU / M2 local
             ↓
      Evaluated: 140 human-written queries
      Matcher: cross-encoder/ms-marco-MiniLM-L-6-v2
      Stats: McNemar test (Hit@3) + permutation test (MRR)
```

---

## Experimental Variants

| Variant | Training Data | Pairs | Notes |
|---|---|---|---|
| BGE arXiv-only | arXiv gemma3:4b | 2,700 | Primary in-domain baseline |
| BGE mixed 50/50 | arXiv + MS MARCO | 2,700 | Highest MRR on arXiv |
| BGE mistral | arXiv mistral:latest | 2,700 | Tests generator independence |
| BGE staged gen→dom | MARCO then arXiv | 2×1,350 | Sequential staging |
| BGE staged dom→gen | arXiv then MARCO | 2×1,350 | Reverse staging |
| BGE hard negatives | Mined triplets | 2,699 | Collapsed (false negatives) |
| BGE PubMed | PubMed gemma3:4b | 1,300 | Out-of-domain replication |

---

## Evaluation Methodology

**Relevance matching** determines whether a retrieved passage is relevant
to a ground-truth positive. Two approaches were compared:

| Matcher | Model | Threshold | Notes |
|---|---|---|---|
| Bi-encoder (prior) | all-MiniLM-L6-v2 cosine | ≥ 0.75 | Susceptible to matcher contamination |
| Cross-encoder (this work) | cross-encoder/ms-marco-MiniLM-L-6-v2 | logit ≥ 0 | Independent architecture; no circularity |

**Significance testing:**
- McNemar's test with continuity correction for paired binary Hit@3 outcomes
- 10,000-permutation two-sided test for MRR differences
- 95% bootstrap confidence intervals (1,000 resamples) on all metrics

**Evaluation sets:**

| Set | Domain | n | Source | Purpose |
|---|---|---|---|---|
| arXiv manual | arXiv | 140 | Human-written | Primary honest eval |
| arXiv synthetic | arXiv | 100 | gemma3:4b (held-out) | Contamination demonstration |
| arXiv extreme | arXiv | 100 | Training pairs (direct) | Upper contamination bound |
| PubMed manual | PubMed | 90 | Human-written | Domain adaptation eval |
| PubMed synthetic | PubMed | 100 | gemma3:4b (held-out) | Cross-domain contamination check |
| 20k arXiv manual | arXiv 20k | 140 | Human-written | Corpus scale generalisation |

---

## Limitations

1. **Wide confidence intervals** — n=140 arXiv queries yields bootstrap
   CIs of ±0.03–0.05 on MRR. Most differences between fine-tuning variants
   are not statistically significant at this sample size. At least 300
   queries per domain are recommended for detecting effects smaller than 5 pp.

2. **Unequal matchers for nomic/BM25** — These baselines were evaluated
   with the prior all-MiniLM matcher; BGE variants use the cross-encoder.
   A fully fair comparison would require re-evaluating nomic under the
   cross-encoder, which is left for future work.

3. **Single training pass per variant** — No hyperparameter search was
   performed. Training configuration (5 epochs, batch size 64, lr 2e-5)
   was held constant across all variants.

4. **Hard negative false-negative problem** — The hard-negative collapse
   is attributed to false negatives in the narrow-domain corpus, not to
   a fundamental failure of the approach. Cross-encoder filtering of
   mined negatives before training is expected to recover performance.

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
# 1. Download arXiv corpus (5k docs)
python step1_load_dataset.py

# 2. Build baseline FAISS index (nomic-embed-text)
python step2_build_index.py

# 3. Generate synthetic training pairs — gemma3:4b (~90 min, resumable)
python step3_generate_pairs.py

# 4. Generate mistral training pairs
python step3d_generate_pairs_mistral.py

# 5. Mine hard negative triplets
python step3b_mine_negatives.py

# 6. Scale corpus to 20k (runs overnight on M2)
python step8_scale_corpus.py

# 7. Fine-tune BGE variants on Colab (see /notebooks)
#    Upload training pairs + val set; unzip model to experiments/<variant>/

# 8. Evaluate all arXiv BGE variants
python step4d_eval_bge.py --eval manual       # 140 human queries
python step4d_eval_bge.py --eval synthetic    # 100 held-out synthetic
python step4d_eval_bge.py --eval extreme      # contamination ablation
python step4d_eval_bge.py --eval manual --corpus 20k  # 20k corpus

# 9. PubMed domain replication
python step10b_build_pubmed_index.py
python step10c_generate_pubmed_pairs.py
# Fine-tune via step10d_train_pubmed_bge.ipynb (Colab)
python step10e_eval_pubmed.py --eval manual
python step10e_eval_pubmed.py --eval synthetic
```

---

## Stack

| Component | Tool |
|---|---|
| Corpora | arXiv CS abstracts + PubMed abstracts via HuggingFace Datasets |
| Vector store | FAISS IndexFlatIP (exact cosine, L2-normalised) |
| Baseline embedding | nomic-embed-text via Ollama |
| Fine-tuned embedding | BGE-base-en-v1.5, sentence-transformers |
| Early experiments | all-MiniLM-L6-v2, sentence-transformers |
| Training loss | MultipleNegativesRankingLoss |
| General training data | MS MARCO v1.1 via HuggingFace Datasets |
| Synthetic data LLMs | gemma3:4b and mistral:latest via Ollama |
| Relevance matching | cross-encoder/ms-marco-MiniLM-L-6-v2 (CrossEncoder) |
| Training compute | Google Colab T4 GPU (free tier) + Apple M2 local |
| LLM for generation | gemma3:4b via Ollama (local, M2 GPU) |
| Experiment tracking | MLflow |
| Eval metrics | Hit@3, MRR, NDCG@10 with 95% bootstrap CI |
| Significance tests | McNemar (Hit@3), permutation test (MRR) |
