# Original arXiv study (May–Jun 2026) — not re-verifiable

These results come from the first version of this project. The fine-tuned models and the arXiv training pairs
were lost in a hardware failure in Sep 2026, so none of this can be re-run. It is kept for transparency.
The current, re-run study is in the [main README](../README.md).

## arXiv results

140 human-written queries over 13,050 chunks from 5,000 arXiv abstracts, scored with the cross-encoder judge
(Metric 2 in the main README). Results file: [`experiments/bge_scores_manual.json`](../experiments/bge_scores_manual.json).

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
- The hard-negatives model is clearly worse; its MRR CI does not overlap the others. One possible cause is false
  negatives in a narrow corpus (untested); it also used a different loss (TripletLoss).
- On a 20k-abstract corpus, all variants dropped slightly
  ([`bge_scores_manual_20k.json`](../experiments/bge_scores_manual_20k.json)).

**Why these can't be taken further:** no untuned-BGE baseline; they rely on the cross-encoder judge, which agreed
poorly with human labels on PubMed; model selection may have kept only epoch-1 checkpoints (the mistral model used
the same 10-query validation that saturated on PubMed); significance tests compared each model only with
arXiv-only, used an unpaired test, and per-query scores were not saved; the models and training pairs were lost.

**Corpus.** First 5,000 records of `gfissore/arxiv-abstracts-2021` with no field filter; the first record is
arXiv:0704.0001 (particle physics), so the corpus is not CS-only; field mix not measured.
**Training.** MultipleNegativesRankingLoss (hard negatives: TripletLoss), batch 32, lr 2e-5, 100 warmup steps,
5 epochs (staged: 3 + 3).

## Early observation: evaluation contamination (motivation, not a controlled result)

Early experiments used an all-MiniLM-L6-v2 model fine-tuned on gemma3:4b pairs, judged by a bi-encoder similarity
matcher. Against untuned nomic-embed-text it scored +0.068 MRR on 100 held-out gemma-generated queries
(0.8600 vs. 0.7917) but −0.012 MRR on 150 human-written queries (0.8789 vs. 0.8911)
([`full_scores_synthetic.json`](../experiments/finetuned/full_scores_synthetic.json),
[`full_scores.json`](../experiments/finetuned/full_scores.json)).
This motivated switching to human-written queries. It is not a controlled test: model families and matchers
differ between conditions, and the synthetic query set was lost.
