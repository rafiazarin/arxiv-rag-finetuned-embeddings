# Fine-Tuning Embedding Models for RAG: What Actually Helps

A study of fine-tuning embedding models for retrieval-augmented generation (RAG)
on two domains: arXiv abstracts and PubMed abstracts. Built with free tools
(local Mac + free Colab T4).

**Status (Sep 2026):**
- **PubMed results were re-run from scratch** in one controlled run. Model and code are public.
- **arXiv results come from the original run (May–Jun 2026).** Those models and the arXiv
  training pairs were lost in a hardware failure, so they cannot be re-run or checked.
  They are reported as-is, with caveats.

**Model on HuggingFace:** [`rafiazarin/bge-base-pubmed-finetuned`](https://huggingface.co/rafiazarin/bge-base-pubmed-finetuned)

---

## Main result — PubMed (re-run, Sep 2026)

90 human-written queries over 7,627 chunks from 2,000 PubMed abstracts.
All models evaluated in the same run with the same relevance matcher.
Results file: [`experiments/pubmed_step11_scores.json`](experiments/pubmed_step11_scores.json).

| Model | Hit@3 | MRR (95% CI) | NDCG@10 |
|---|---|---|---|
| BGE-base, fine-tuned (5 epochs) | 0.8667 | 0.7704 (0.6925–0.8481) | 0.5609 |
| BGE-base, fine-tuned (epoch-1 checkpoint) | 0.8778 | 0.7722 (0.6944–0.8389) | 0.5620 |
| BGE-base, no fine-tuning | 0.8444 | 0.7222 (0.6481–0.7944) | 0.5378 |
| nomic-embed-text, no fine-tuning | 0.7778 | 0.6519 (0.5667–0.7370) | 0.5083 |

Paired significance tests (McNemar for Hit@3, paired permutation for MRR):

| Comparison | Hit@3 p | MRR p |
|---|---|---|
| Fine-tuned (5 ep) vs. untuned BGE | 0.6171 | 0.0153 |
| Untuned BGE vs. nomic | 0.0771 | 0.0323 |
| Fine-tuned (5 ep) vs. nomic | 0.0133 | 0.0005 |
| Fine-tuned 5 ep vs. epoch 1 | 1.0000 | 1.0000 |

**What this shows:**
1. Most of the gain over nomic comes from **choosing BGE**, not from fine-tuning.
2. Fine-tuning adds a **small, consistent ranking gain**: mean MRR +0.0445 across 3 seeds (p = 0.0229, see seed study below). There is **no significant Hit@3 gain**.
3. Training beyond epoch 1 added nothing measurable.

### Seed study (pre-registered)

The primary test was fixed before running: mean MRR of 3 fine-tuned seeds vs. untuned BGE,
paired permutation test on per-query reciprocal rank averaged over seeds.
All models evaluated in one Colab run. Results: [`experiments/pubmed_seed_study.json`](experiments/pubmed_seed_study.json).
Notebook: [`notebooks/step12_pubmed_seed_study.ipynb`](notebooks/step12_pubmed_seed_study.ipynb).

| Model | Hit@3 | MRR | Hit@3 p vs. untuned | MRR p vs. untuned |
|---|---|---|---|---|
| Fine-tuned, seed 42 (published) | 0.8667 | 0.7704 | 0.6171 | 0.0153 |
| Fine-tuned, seed 1 | 0.8667 | 0.7704 | 0.6171 | 0.0153 |
| Fine-tuned, seed 2 | 0.8667 | 0.7593 | 0.6171 | 0.0699 |
| BGE-base, no fine-tuning | 0.8444 | 0.7222 | — | — |

**Primary result:** mean MRR over the 3 seeds 0.7667 (SD 0.0064) vs. 0.7222 untuned, p = 0.0229.
Hit@3 did not change significantly for any seed.

Seeds 1 and 42 have different weights but identical top-3 rankings on all 90 queries.
The Colab run reproduced the Mac numbers for the published model and untuned BGE exactly.

**Caveats:** n = 90; relevance judged by an automatic matcher; the seed-2 model alone is not
significant on MRR (p = 0.0699).

---

## arXiv results (original run, May–Jun 2026 — not re-verifiable)

140 human-written queries over 13,050 chunks from 5,000 arXiv abstracts.
Same cross-encoder matcher as above. Results file:
[`experiments/bge_scores_manual.json`](experiments/bge_scores_manual.json).

| BGE-base variant | Training data | Hit@3 | MRR (95% CI) | NDCG@10 |
|---|---|---|---|---|
| arXiv only | 2,700 gemma3:4b pairs | 0.9071 | 0.8405 (0.7869–0.8941) | 0.6717 |
| Mixed 50/50 | 1,350 arXiv + 1,350 MS MARCO | 0.9000 | 0.8464 (0.7893–0.9000) | 0.6743 |
| Mistral pairs | 2,700 mistral pairs | 0.9000 | 0.8429 (0.7869–0.8976) | 0.6743 |
| Staged MARCO → arXiv | 1,350 + 1,350, 3 + 3 epochs | 0.9000 | 0.8417 (0.7821–0.8940) | 0.6712 |
| Staged arXiv → MARCO | 1,350 + 1,350, 3 + 3 epochs | 0.9000 | 0.8405 (0.7821–0.8929) | 0.6725 |
| Hard negatives (TripletLoss) | 2,499 mined triplets | 0.7143 | 0.6131 (0.5404–0.6845) | 0.5156 |

**Observations:**
- The five standard variants perform almost identically; their 95% CIs overlap heavily.
- The hard-negatives model is clearly worse; its MRR CI does not overlap the others.
  One possible cause: in a narrow corpus, mined "hard negatives" are often actually
  relevant (false negatives). This was not tested. It also used a different loss
  (TripletLoss), which is a confound.
- On a 20k-abstract corpus, all variants dropped slightly
  ([`bge_scores_manual_20k.json`](experiments/bge_scores_manual_20k.json)).

**Why these can't be taken further:**
- There was **no untuned-BGE baseline**, so this run cannot show whether fine-tuning helped at all.
- Model selection saved a new checkpoint only when validation *improved*. On PubMed,
  validation was saturated at 1.0 from epoch 1, so only the epoch-1 checkpoint was saved.
  The arXiv mistral model used the same 10-query validation setup, so it was **likely an
  epoch-1 checkpoint too**. Other arXiv variants used 200 synthetic validation pairs;
  whether they saturated is unknown.
- Significance tests in this run compared each model only against arXiv-only, used an
  unpaired permutation test, and per-query scores were not saved.
- The models and training pairs were lost, so none of this can be re-run.

---

## Earlier observation: evaluation contamination (motivation, not a controlled result)

Early experiments used an all-MiniLM-L6-v2 model fine-tuned on gemma3:4b pairs, judged by
a bi-encoder similarity matcher. Against untuned nomic-embed-text it scored:
- **+0.068 MRR** on 100 held-out gemma-generated queries (0.8600 vs. 0.7917)
- **−0.012 MRR** on 150 human-written queries (0.8789 vs. 0.8911)

([`experiments/finetuned/full_scores_synthetic.json`](experiments/finetuned/full_scores_synthetic.json),
[`experiments/finetuned/full_scores.json`](experiments/finetuned/full_scores.json))

This pattern suggested that evaluating on queries from the same LLM that generated the
training data inflates results, and motivated switching to human-written queries and an
independent cross-encoder matcher. It is **not** a controlled test: the model families and
matchers differ between conditions, and the synthetic query set was lost, so it cannot be re-run.

---

## Method

- **Corpora.** arXiv: first 5,000 records of `gfissore/arxiv-abstracts-2021` with no field
  filter. The first record is arXiv:0704.0001, a particle-physics paper, so the corpus is
  not CS-only; the field mix was not measured. PubMed: first 2,000 abstracts of
  `ccdv/pubmed-summarization` (train split) that pass a length filter, shuffled with seed 42.
- **Chunking.** 400 characters, 50-character overlap. FAISS exact cosine search.
- **Training.** MultipleNegativesRankingLoss (except hard negatives: TripletLoss),
  batch 32, lr 2e-5, 100 warmup steps, 5 epochs (staged: 3 + 3).
- **Relevance.** For each query, a retrieved chunk counts as relevant if
  `cross-encoder/ms-marco-MiniLM-L-6-v2` scores it ≥ 0 against the query's source abstract.
- **Metrics.** Hit@3, MRR@3, NDCG@10, with 1,000-sample bootstrap 95% CIs.

## Limitations

1. Relevance is judged by an automatic matcher, not human labels. The matcher is an MS MARCO
   query–passage model used here to compare passage to passage; this use was not validated.
2. Small evaluation sets (90 and 140 queries) give wide confidence intervals.
3. Training queries are LLM-generated.
4. Queries were encoded without BGE's query instruction prefix, and nomic without its
   task prefixes.
5. The nomic PubMed score changed between the original run (0.6889 Hit@3) and the re-run
   (0.7778) on identical data, most likely because of a newer Ollama version. Only numbers
   from the same run are compared above.
6. `step11_eval_pubmed_baselines.py` currently expects both fine-tuned models as local folders.

## Reproduce the PubMed results

```bash
git clone https://github.com/rafiazarin/arxiv-rag-finetuned-embeddings
cd arxiv-rag-finetuned-embeddings
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.12.0 sentence-transformers==5.5.1 transformers==5.9.0 faiss-cpu==1.13.2 \
            numpy==2.4.6 scipy==1.17.1 datasets==4.8.5 ollama==0.6.2 tqdm==4.67.3
ollama pull nomic-embed-text           # with `ollama serve` running

python restore_pubmed_pool.py          # rebuilds data/pubmed_pool.json, verified against repo data
python step10b_build_pubmed_index.py   # nomic baseline index
# Train in Colab: notebooks/step10d_train_pubmed_bge.ipynb, then unzip into experiments/
python step11_eval_pubmed_baselines.py
```

`requirements.txt` lists the full original environment; the command above installs only what
the PubMed pipeline needs.
