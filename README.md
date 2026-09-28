# Fine-Tuning Embedding Models for RAG: Does It Actually Help?

A study of fine-tuning embedding models for retrieval-augmented generation (RAG)
on PubMed and arXiv abstracts, with a focus on whether the evaluation itself can be trusted.
Built with free tools (local Mac + free Colab T4).

**Summary (PubMed, 90 human-written queries):**
1. **The base model choice has the clearest effect.** Untuned BGE-base beats nomic-embed-text on MRR
   in every setting tested (p = 0.0011 on 2,000 abstracts, p < 0.0001 on 20,000, p = 0.0323 under the
   automatic judge). Hit@3 was significant only on the 20,000-abstract corpus (p = 0.0033).
2. **Fine-tuning on 1,300 synthetic pairs did not significantly help on the objective metric.**
   Pre-registered tests: p = 0.299 on 2,000 abstracts, and p = 0.1158 on 20,000 abstracts, where the
   fine-tuned model scored lower (MRR 0.7611 vs. 0.8019). Only the automatic cross-encoder judge
   showed a gain (+0.0445 MRR, 3 seeds, p = 0.0229).
3. **The automatic judge agreed poorly with human labels** (Cohen's kappa 0.2 on 60 items),
   so results that rely on it alone should not be trusted.

**Status (Sep 2026):** PubMed results were re-run from scratch; model, code and all result files
are public. arXiv results come from the original run (May–Jun 2026); those models and training
pairs were lost in a hardware failure and cannot be re-run.

**Model on HuggingFace:** [`rafiazarin/bge-base-pubmed-finetuned`](https://huggingface.co/rafiazarin/bge-base-pubmed-finetuned)

---

## PubMed results (re-run, Sep 2026)

90 human-written queries over 7,627 chunks (2,000 PubMed abstracts, 400-character chunks).
Within each table, all models were evaluated in the same run.

### Metric 1 — objective: retrieved chunk comes from the source abstract

The primary test (fine-tuned 5 ep vs. untuned BGE, MRR) was fixed before running.
Results: [`experiments/pubmed_source_match_scores.json`](experiments/pubmed_source_match_scores.json).

| Model | Hit@3 | MRR (95% CI) | NDCG@10 |
|---|---|---|---|
| BGE-base, fine-tuned (5 epochs) | 0.9778 | 0.9722 (0.9389–1.0) | 0.8198 |
| BGE-base, fine-tuned (epoch-1 checkpoint) | 0.9889 | 0.9685 (0.9352–0.9944) | 0.8274 |
| BGE-base, no fine-tuning | 0.9889 | 0.9519 (0.9166–0.9833) | 0.8177 |
| nomic-embed-text, no fine-tuning | 0.9444 | 0.8685 (0.8037–0.9223) | 0.7404 |

| Comparison (paired tests) | Hit@3 p (McNemar) | MRR p (permutation) |
|---|---|---|
| **Fine-tuned vs. untuned BGE (primary)** | 1.0000 | **0.2990** |
| Untuned BGE vs. nomic | 0.1336 | 0.0011 |
| Fine-tuned vs. nomic | 0.3711 | 0.0001 |
| Fine-tuned 5 ep vs. epoch 1 | 1.0000 | 0.7512 |

Untuned BGE already finds the source abstract in the top 3 for 98.9% of queries, so there is
little room for fine-tuning to help on this metric (ceiling effect). A chunk from the source
abstract does not always contain the answer.

### Metric 1 on a 10x larger corpus (ceiling check)

Same queries and relevance rule, searched over 20,000 abstracts (75,862 chunks; a superset of the
original 2,000). The primary test (fine-tuned vs. untuned BGE, MRR) was fixed before running.
Script: [`step15_scale_pubmed.py`](step15_scale_pubmed.py); results:
[`experiments/pubmed_20k_source_match_scores.json`](experiments/pubmed_20k_source_match_scores.json).

| Model | Hit@3 | MRR (95% CI) | NDCG@10 |
|---|---|---|---|
| BGE-base, no fine-tuning | 0.8667 | 0.8019 (0.7241–0.8778) | 0.6086 |
| BGE-base, fine-tuned (5 epochs) | 0.8222 | 0.7611 (0.6796–0.8389) | 0.5830 |
| nomic-embed-text, no fine-tuning | 0.7333 | 0.6519 (0.5611–0.7408) | 0.5316 |

| Comparison (paired tests) | Hit@3 p (McNemar) | MRR p (permutation) |
|---|---|---|
| **Fine-tuned vs. untuned BGE (primary)** | 0.3428 | **0.1158** |
| Untuned BGE vs. nomic | 0.0033 | < 0.0001 |
| Fine-tuned vs. nomic | 0.0433 | 0.0026 |

The larger corpus removes the ceiling (untuned Hit@3 drops from 0.9889 to 0.8667). Fine-tuning still
shows no significant gain, and its point estimate is lower than the untuned model's.

### Metric 2 — automatic judge: cross-encoder relevance

A retrieved chunk counts as relevant if `cross-encoder/ms-marco-MiniLM-L-6-v2` scores it ≥ 0
against the source abstract. Results: [`experiments/pubmed_step11_scores.json`](experiments/pubmed_step11_scores.json)
(reproduced exactly from the published model: [`pubmed_step11_scores_hub.json`](experiments/pubmed_step11_scores_hub.json)).

| Model | Hit@3 | MRR (95% CI) | NDCG@10 |
|---|---|---|---|
| BGE-base, fine-tuned (5 epochs) | 0.8667 | 0.7704 (0.6925–0.8481) | 0.5609 |
| BGE-base, fine-tuned (epoch-1 checkpoint) | 0.8778 | 0.7722 (0.6944–0.8389) | 0.5620 |
| BGE-base, no fine-tuning | 0.8444 | 0.7222 (0.6481–0.7944) | 0.5378 |
| nomic-embed-text, no fine-tuning | 0.7778 | 0.6519 (0.5667–0.7370) | 0.5083 |

| Comparison (paired tests) | Hit@3 p | MRR p |
|---|---|---|
| Fine-tuned vs. untuned BGE | 0.6171 | 0.0153 |
| Untuned BGE vs. nomic | 0.0771 | 0.0323 |
| Fine-tuned vs. nomic | 0.0133 | 0.0005 |
| Fine-tuned 5 ep vs. epoch 1 | 1.0000 | 1.0000 |

**Seed study (pre-registered).** Primary test: mean MRR of 3 fine-tuned seeds vs. untuned BGE,
paired permutation on seed-averaged per-query reciprocal rank. One Colab run.
Results: [`experiments/pubmed_seed_study.json`](experiments/pubmed_seed_study.json);
notebook: [`notebooks/step12_pubmed_seed_study.ipynb`](notebooks/step12_pubmed_seed_study.ipynb).

| Model | Hit@3 | MRR | Hit@3 p vs. untuned | MRR p vs. untuned |
|---|---|---|---|---|
| Fine-tuned, seed 42 (published) | 0.8667 | 0.7704 | 0.6171 | 0.0153 |
| Fine-tuned, seed 1 | 0.8667 | 0.7704 | 0.6171 | 0.0153 |
| Fine-tuned, seed 2 | 0.8667 | 0.7593 | 0.6171 | 0.0699 |
| BGE-base, no fine-tuning | 0.8444 | 0.7222 | — | — |

Mean MRR over the 3 seeds: 0.7667 (SD 0.0064) vs. 0.7222 untuned, p = 0.0229. Seeds 1 and 42
have different weights but identical top-3 rankings on all 90 queries.

### Checking the automatic judge against human labels

From 324 unique query–chunk pairs (top-3 results of both BGE models), 30 the judge called
relevant and 30 it called not relevant were sampled (seed 42), shuffled, and labeled by the author
without seeing the judge's decision. Question: does the chunk contain the information that
answers the query? Script: [`step13_matcher_validation.py`](step13_matcher_validation.py);
results: [`experiments/matcher_validation.json`](experiments/matcher_validation.json).

| | Human: yes | Human: no |
|---|---|---|
| Judge: relevant | 14 | 16 |
| Judge: not relevant | 8 | 22 |

Precision 0.4667, negative predictive value 0.7333, Cohen's kappa 0.2, agreement re-weighted to
the pool 0.6379. Disagreement was concentrated in chunks from the source abstract (agreement 19/42)
rather than chunks from other abstracts (17/18): the judge mostly detects "same abstract", not
"answers the query". Caveats: one annotator, who could see the source abstract; 60 items.

---

## arXiv results (original run, May–Jun 2026 — not re-verifiable)

140 human-written queries over 13,050 chunks from 5,000 arXiv abstracts, scored with the
cross-encoder judge (Metric 2 above). Results file:
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
- The hard-negatives model is clearly worse; its MRR CI does not overlap the others. One possible
  cause is false negatives in a narrow corpus (untested); it also used a different loss (TripletLoss).
- On a 20k-abstract corpus, all variants dropped slightly
  ([`bge_scores_manual_20k.json`](experiments/bge_scores_manual_20k.json)).

**Why these can't be taken further:** no untuned-BGE baseline; they rely on the cross-encoder
judge, which agreed poorly with human labels on PubMed; model selection may have kept only
epoch-1 checkpoints (the mistral model used the same 10-query validation that saturated on PubMed);
significance tests compared each model only with arXiv-only, used an unpaired test, and per-query
scores were not saved; the models and training pairs were lost.

---

## Earlier observation: evaluation contamination (motivation, not a controlled result)

Early experiments used an all-MiniLM-L6-v2 model fine-tuned on gemma3:4b pairs, judged by a
bi-encoder similarity matcher. Against untuned nomic-embed-text it scored +0.068 MRR on 100
held-out gemma-generated queries (0.8600 vs. 0.7917) but −0.012 MRR on 150 human-written queries
(0.8789 vs. 0.8911)
([`full_scores_synthetic.json`](experiments/finetuned/full_scores_synthetic.json),
[`full_scores.json`](experiments/finetuned/full_scores.json)).
This motivated switching to human-written queries. It is not a controlled test: model families and
matchers differ between conditions, and the synthetic query set was lost.

---

## Method

- **Corpora.** PubMed: first 2,000 abstracts of `ccdv/pubmed-summarization` (train split) passing
  a length filter, shuffled with seed 42. arXiv: first 5,000 records of `gfissore/arxiv-abstracts-2021`
  with no field filter; the first record is arXiv:0704.0001 (particle physics), so the corpus is not
  CS-only; field mix not measured.
- **Chunking.** 400 characters, 50-character overlap. FAISS exact cosine search.
- **Training.** MultipleNegativesRankingLoss (hard negatives: TripletLoss), batch 32, lr 2e-5,
  100 warmup steps, 5 epochs (staged: 3 + 3). PubMed: 1,300 gemma3:4b-generated pairs.
- **Metrics.** Hit@3, MRR@3, NDCG@10, 1,000-sample bootstrap 95% CIs; McNemar (Hit@3) and paired
  sign-flip permutation (MRR) tests.

## Limitations

1. Neither relevance metric equals "answers the query": the automatic judge agreed poorly with human
   labels, and the objective metric counts any chunk of the source abstract.
2. Human labels come from one annotator on 60 items.
3. Small evaluation sets (90 and 140 queries); the PubMed queries were written from the abstracts,
   which makes the source abstract easy to retrieve (ceiling effect on Metric 1 at 2,000
   abstracts; the 20,000-abstract check addresses this).
4. Training queries are LLM-generated.
5. Queries were encoded without BGE's query instruction prefix, and nomic without its task prefixes.
6. The nomic PubMed score changed between the original run and the re-run on identical data, most
   likely because of a newer Ollama version. Only numbers from the same run are compared.

## Reproduce the PubMed results

```bash
git clone https://github.com/rafiazarin/arxiv-rag-finetuned-embeddings
cd arxiv-rag-finetuned-embeddings
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.12.0 sentence-transformers==5.5.1 transformers==5.9.0 faiss-cpu==1.13.2 \
            numpy==2.4.6 scipy==1.17.1 datasets==4.8.5 ollama==0.6.2 tqdm==4.67.3
ollama pull nomic-embed-text                        # with `ollama serve` running

python restore_pubmed_pool.py                       # rebuilds data/pubmed_pool.json, verified
python step10b_build_pubmed_index.py                # nomic baseline index
python step11_eval_pubmed_baselines.py --from-hub   # Metric 2, uses the published model
python step13_matcher_validation.py score           # judge vs. human labels (labels in data/)
# Metric 1 (step14_source_match_eval.py, step15_scale_pubmed.py) and the epoch-1 comparison need local models:
# run notebooks/step10d_train_pubmed_bge.ipynb in Colab and unzip into experiments/.
# Seed study: notebooks/step12_pubmed_seed_study.ipynb (Colab).
```

`requirements.txt` lists the full original environment; the command above installs only what the
PubMed pipeline needs.
