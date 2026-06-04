"""
step10e_eval_pubmed.py

Evaluates PubMed models — mirrors step4d_eval_bge.py for the biomedical domain.

Compares:
  - Baseline nomic-embed-text (no fine-tuning)
  - BGE-base fine-tuned on PubMed pairs

On two eval sets:
  - synthetic: last 100 pairs from pubmed_training_pairs.json (gemma-generated)
  - manual:    pubmed_manual_eval90.json (90 human-written queries)

The contamination pattern should replicate: fine-tuned model wins on
synthetic, baseline wins or ties on manual.

Usage:
    python step10e_eval_pubmed.py --eval manual
    python step10e_eval_pubmed.py --eval synthetic
    python step10e_eval_pubmed.py --eval combined
"""

import os
import json
import argparse
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from scipy import stats
from baseline_rag import embed_text as nomic_embed, load_index
from config import DATA_DIR, EXPERIMENTS_DIR, BASELINE_DIR, TOP_K

# ── Models ────────────────────────────────────────────────────────────────────

PUBMED_RUNS = {
    "PubMed BGE-base (fine-tuned)": {
        "model_dir":  "pubmed_bge/model",
        "index_file": "pubmed_bge/pubmed_bge.faiss",
        "chunks_src": "pubmed_baseline/pubmed_baseline_chunks.pkl",
        "type":       "bge"
    },
    "PubMed Baseline (nomic, no FT)": {
        "index_file": "pubmed_baseline/pubmed_baseline.faiss",
        "chunks_src": "pubmed_baseline/pubmed_baseline_chunks.pkl",
        "type":       "nomic"
    },
}

# ── Metrics (identical to step4d) ─────────────────────────────────────────────

def hit_at_k(retrieved, relevant, k):
    return int(any(r in relevant for r in retrieved[:k]))

def reciprocal_rank(retrieved, relevant, k):
    for i, r in enumerate(retrieved[:k], 1):
        if r in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    dcg  = sum(1.0 / np.log2(i + 2) for i, r in enumerate(retrieved[:k]) if r in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def bootstrap_ci(scores, n_resamples=1000, ci=0.95):
    scores = np.array(scores)
    means  = [np.mean(np.random.choice(scores, size=len(scores), replace=True))
              for _ in range(n_resamples)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return round(float(lo), 4), round(float(hi), 4)

def mcnemar_test(hits_a, hits_b):
    """McNemar's test for paired binary Hit@k outcomes (with continuity correction)."""
    hits_a, hits_b = np.array(hits_a), np.array(hits_b)
    b = int(np.sum((hits_a == 1) & (hits_b == 0)))
    c = int(np.sum((hits_a == 0) & (hits_b == 1)))
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return round(float(1 - stats.chi2.cdf(chi2, df=1)), 4)

def permutation_test(scores_a, scores_b, n_permutations=10000):
    """Two-sided permutation test for difference in means (MRR, NDCG)."""
    a, b     = np.array(scores_a), np.array(scores_b)
    observed = abs(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a      = len(a)
    rng      = np.random.default_rng(42)
    count    = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        if abs(np.mean(perm[:n_a]) - np.mean(perm[n_a:])) >= observed:
            count += 1
    return round(float(count / n_permutations), 4)

# ── Match function (identical to step4d) ──────────────────────────────────────
# Cross-encoder avoids circularity — no bi-encoder model under evaluation
# shares this architecture. ms-marco raw logit > 0 = match.

_match_model = None

def get_match_model():
    global _match_model
    if _match_model is None:
        _match_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _match_model

def is_match(retrieved_text, ground_truth, threshold=0.0):
    try:
        model = get_match_model()
        score = model.predict([(ground_truth, retrieved_text)])[0]
        return float(score) >= threshold
    except Exception:
        a = set(retrieved_text.lower().split())
        b = set(ground_truth.lower().split())
        return len(a & b) / len(a) > 0.5 if a else False

# ── Eval set loader ───────────────────────────────────────────────────────────

def load_eval_pairs(eval_type):
    pairs = []
    if eval_type in ("synthetic", "combined"):
        path = os.path.join(DATA_DIR, "pubmed_training_pairs.json")
        with open(path) as f:
            all_pairs = json.load(f)
        pairs += all_pairs[-100:]
        print(f"Synthetic: last 100 of pubmed_training_pairs.json")

    if eval_type in ("manual", "combined"):
        path = os.path.join(DATA_DIR, "pubmed_manual_eval90.json")
        with open(path) as f:
            pairs += json.load(f)
        print(f"Manual: pubmed_manual_eval90.json ({len(pairs)} queries)")
    return pairs

# ── Core evaluator ────────────────────────────────────────────────────────────

def evaluate(embed_fn, index, chunks, eval_pairs, k=TOP_K):
    hit_scores, rr_scores, ndcg_scores = [], [], []

    for pair in tqdm(eval_pairs):
        q_emb = np.array([embed_fn(pair["query"])], dtype="float32")
        faiss.normalize_L2(q_emb)
        _, idxs    = index.search(q_emb, 10)
        retrieved  = [chunks[i]["text"] for i in idxs[0] if i >= 0]
        relevant   = {t for t in retrieved if is_match(t, pair["positive"])}
        relevant.add(pair["positive"])

        hit_scores.append(hit_at_k(retrieved, relevant, k))
        rr_scores.append(reciprocal_rank(retrieved, relevant, k))
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, 10))

    return {
        "hit_rate":    round(float(np.mean(hit_scores)),  4),
        "hit_rate_ci": bootstrap_ci(hit_scores),
        "mrr":         round(float(np.mean(rr_scores)),   4),
        "mrr_ci":      bootstrap_ci(rr_scores),
        "ndcg_10":     round(float(np.mean(ndcg_scores)), 4),
        "ndcg_10_ci":  bootstrap_ci(ndcg_scores),
        "n_queries":   len(eval_pairs),
        "k":           k,
        # Per-query vectors for significance tests — stripped before JSON save
        "_hit_scores": hit_scores,
        "_rr_scores":  rr_scores,
    }

# ── Pretty print ──────────────────────────────────────────────────────────────

def print_results(all_scores, eval_type):
    col = 35
    n   = list(all_scores.values())[0]["n_queries"]
    print(f"\n{'='*72}")
    print(f"  PUBMED EVAL  |  {eval_type}  |  n={n}  |  95% CI bootstrap")
    print(f"{'='*72}")
    print(f"  {'Model':<{col}} {'Hit@3':>8} {'MRR':>8} {'NDCG@10':>10}")
    print(f"  {'-'*65}")
    for label, s in sorted(all_scores.items(), key=lambda x: -x[1]["mrr"]):
        print(f"  {label:<{col}} {s['hit_rate']:>8.4f} {s['mrr']:>8.4f} {s['ndcg_10']:>10.4f}")
        lo_h, hi_h = s["hit_rate_ci"]
        lo_m, hi_m = s["mrr_ci"]
        lo_n, hi_n = s["ndcg_10_ci"]
        print(f"  {'':>{col}} {f'[{lo_h},{hi_h}]':>8} {f'[{lo_m},{hi_m}]':>8} {f'[{lo_n},{hi_n}]':>10}")
    print(f"{'='*72}\n")

def print_significance(all_scores, reference_label="PubMed Baseline (nomic, no FT)"):
    """Pairwise significance tests vs. the no-FT baseline."""
    if reference_label not in all_scores:
        print(f"  (Reference '{reference_label}' not found — skipping significance table)\n")
        return
    ref = all_scores[reference_label]
    col = 35
    print(f"  Significance vs. '{reference_label}'")
    print(f"  {'Model':<{col}} {'McNemar(Hit)':>14} {'Perm(MRR)':>12}")
    print(f"  {'-'*65}")
    for label, s in sorted(all_scores.items(), key=lambda x: -x[1]["mrr"]):
        if label == reference_label:
            continue
        p_hit = mcnemar_test(s["_hit_scores"], ref["_hit_scores"])
        p_mrr = permutation_test(s["_rr_scores"], ref["_rr_scores"])
        sig_h = " *" if p_hit < 0.05 else "  "
        sig_m = " *" if p_mrr < 0.05 else "  "
        print(f"  {label:<{col}} {p_hit:>12.4f}{sig_h} {p_mrr:>10.4f}{sig_m}")
    print(f"  (* p < 0.05 two-sided)\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", choices=["synthetic", "manual", "combined"],
                        default="manual")
    args       = parser.parse_args()
    eval_type  = args.eval
    eval_pairs = load_eval_pairs(eval_type)
    print(f"\nEval set: {len(eval_pairs)} queries ({eval_type})\n")

    all_scores = {}

    for label, cfg in PUBMED_RUNS.items():
        chunks_path = os.path.join(EXPERIMENTS_DIR, cfg["chunks_src"])
        index_path  = os.path.join(EXPERIMENTS_DIR, cfg["index_file"])

        for p, name in [(chunks_path, "chunks"), (index_path, "index")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing {name}: {p}")

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        index = faiss.read_index(index_path)

        if cfg["type"] == "nomic":
            embed_fn = nomic_embed
        else:
            model_dir = os.path.join(EXPERIMENTS_DIR, cfg["model_dir"])
            model     = SentenceTransformer(model_dir)
            embed_fn  = lambda t: model.encode(t, normalize_embeddings=True).tolist()

        print(f"\nEvaluating: {label}")
        all_scores[label] = evaluate(embed_fn, index, chunks, eval_pairs)

    print_results(all_scores, eval_type)
    print_significance(all_scores)

    # Strip per-query vectors before saving
    saveable = {k: {m: v for m, v in s.items() if not m.startswith("_")}
                for k, s in all_scores.items()}
    out_path = os.path.join(EXPERIMENTS_DIR, f"pubmed_scores_{eval_type}.json")
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
