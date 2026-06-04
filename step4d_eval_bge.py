"""
step4d_eval_bge.py  (updated for step 9)

Evaluates bge-base model variants on manual_eval140.json (140 queries).
The 10 held-out validation queries (manual_val.json) are excluded from eval
to avoid contamination with training-time validation.

Usage:
    python step4d_eval_bge.py --eval manual
    python step4d_eval_bge.py --eval synthetic
    python step4d_eval_bge.py --eval extreme        # contamination ablation
    python step4d_eval_bge.py --eval combined
    python step4d_eval_bge.py --eval manual --corpus 20k
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
from config import DATA_DIR, EXPERIMENTS_DIR, TOP_K

# ── Models to evaluate ────────────────────────────────────────────────────────
BGE_RUNS = {
    "BGE-base (arXiv only)": {
        "model_dir":  "bge_arxiv100/model",
        "index_file": "bge_arxiv100/bge_arxiv100.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
    "BGE-base (mixed 50/50)": {
        "model_dir":  "bge_mixed50/model",
        "index_file": "bge_mixed50/bge_mixed50.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
    "BGE-base (hard negatives)": {
        "model_dir":  "bge_hardneg/model",
        "index_file": "bge_hardneg/bge_hardneg.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
    "BGE-base (mistral pairs)": {
        "model_dir":  "bge_mistral/model",
        "index_file": "bge_mistral/bge_mistral.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
    "Staged gen->dom (MARCO then arXiv)": {
        "model_dir":  "staged_gen_dom/model",
        "index_file": "staged_gen_dom/staged_gen_dom.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
    "Staged dom->gen (arXiv then MARCO)": {
        "model_dir":  "staged_dom_gen/model",
        "index_file": "staged_dom_gen/staged_dom_gen.faiss",
        "chunks_src": "baseline/baseline_chunks.pkl",
    },
}

# ── Static nomic baseline ─────────────────────────────────────────────────────
# Hardcoded from prior eval on manual_eval140.json (n=140).
# Matcher used: all-MiniLM-L6-v2 cosine @ 0.75 — differs from the cross-encoder
# used in BGE runs above. See paper footnote.
NOMIC_STATIC = {
    "hit_rate": 0.9600,
    "mrr":      0.8911,
    "ndcg_10":  0.7043,
}

# ── Metrics ───────────────────────────────────────────────────────────────────

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

# ── Match function ────────────────────────────────────────────────────────────
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

def load_eval_pairs(eval_type="manual"):
    pairs = []
    if eval_type == "extreme":
        # Extreme contamination ablation: use the first 100 training pairs directly.
        # Fine-tuned models trained on these should hit near 1.0 — makes the
        # contamination mechanism undeniable to a skeptical reviewer.
        with open(os.path.join(DATA_DIR, "training_pairs.json")) as f:
            pairs = json.load(f)[:100]
        print("EXTREME: first 100 training pairs (direct training data)")
        return pairs

    if eval_type in ("synthetic", "combined"):
        with open(os.path.join(DATA_DIR, "training_pairs.json")) as f:
            pairs += json.load(f)[-100:]

    if eval_type in ("manual", "combined"):
        path140 = os.path.join(DATA_DIR, "manual_eval140.json")
        path150 = os.path.join(DATA_DIR, "manual_eval.json")
        if os.path.exists(path140):
            with open(path140) as f:
                pairs += json.load(f)
            print("Using manual_eval140.json (140 queries, 10 held out for val)")
        elif os.path.exists(path150):
            with open(path150) as f:
                pairs += json.load(f)
            print("Using manual_eval.json (150 queries)")
        else:
            print("Warning: no manual eval file found.")
    return pairs

# ── Core evaluator ────────────────────────────────────────────────────────────

def evaluate(model_dir, index, chunks, eval_pairs, k=TOP_K):
    model    = SentenceTransformer(model_dir)
    embed_fn = lambda t: model.encode(t, normalize_embeddings=True).tolist()

    hit_scores, rr_scores, ndcg_scores = [], [], []

    for pair in tqdm(eval_pairs, desc=os.path.basename(model_dir)):
        q_emb = np.array([embed_fn(pair["query"])], dtype="float32")
        faiss.normalize_L2(q_emb)
        _, idxs   = index.search(q_emb, 10)
        retrieved = [chunks[i]["text"] for i in idxs[0] if i >= 0]
        relevant  = {t for t in retrieved if is_match(t, pair["positive"])}
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
    print(f"  BGE-BASE EVAL  |  {eval_type}  |  n={n}  |  95% CI bootstrap")
    print(f"{'='*72}")
    print(f"  {'Model':<{col}} {'Hit@3':>8} {'MRR':>8} {'NDCG@10':>10}")
    print(f"  {'-'*65}")
    for label, s in sorted(all_scores.items(), key=lambda x: -x[1]["mrr"]):
        print(f"  {label:<{col}} {s['hit_rate']:>8.4f} {s['mrr']:>8.4f} {s['ndcg_10']:>10.4f}")
        lo_h, hi_h = s["hit_rate_ci"]
        lo_m, hi_m = s["mrr_ci"]
        lo_n, hi_n = s["ndcg_10_ci"]
        print(f"  {'':>{col}} {f'[{lo_h},{hi_h}]':>8} {f'[{lo_m},{hi_m}]':>8} {f'[{lo_n},{hi_n}]':>10}")

    if eval_type == "manual" and NOMIC_STATIC:
        print(f"  {'-'*65}")
        lbl = "Baseline nomic-embed-text (no FT) †"
        s   = NOMIC_STATIC
        print(f"  {lbl:<{col}} {s['hit_rate']:>8.4f} {s['mrr']:>8.4f} {s['ndcg_10']:>10.4f}")
        print(f"  {'':>{col}} {'[prior run]':>8}")
        print(f"\n  † nomic evaluated with all-MiniLM cosine matcher; BGE rows use cross-encoder.")

    print(f"{'='*72}\n")

def print_significance(all_scores, reference_label="BGE-base (arXiv only)"):
    """Pairwise significance tests vs. a reference model."""
    if reference_label not in all_scores:
        print(f"  (Reference '{reference_label}' not found — skipping)\n")
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

# ── 20k corpus index builder ──────────────────────────────────────────────────

def get_or_build_20k_index(label, cfg, chunks_20k):
    model_slug = cfg["model_dir"].replace("/model", "")
    index_path = os.path.join(EXPERIMENTS_DIR, model_slug, f"{model_slug}_20k.faiss")
    model_dir  = os.path.join(EXPERIMENTS_DIR, cfg["model_dir"])

    if os.path.exists(index_path):
        print(f"  Loading cached 20k index: {index_path}")
        return faiss.read_index(index_path)

    print(f"  Building 20k index for '{label}' (~1-2 min)...")
    model      = SentenceTransformer(model_dir)
    texts      = [c["text"] for c in chunks_20k]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True,
                              convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"  Saved: {index_path}  ({index.ntotal} vectors)")
    return index

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",
                        choices=["synthetic", "manual", "combined", "extreme"],
                        default="manual",
                        help="'extreme' loads first 100 training pairs for contamination ablation")
    parser.add_argument("--corpus", choices=["5k", "20k"], default="5k",
                        help="'20k' builds/loads per-model indexes automatically")
    args       = parser.parse_args()
    eval_type  = args.eval
    eval_pairs = load_eval_pairs(eval_type)
    print(f"\nEval set: {len(eval_pairs)} queries ({eval_type})")
    print(f"Corpus:   {args.corpus}\n")

    chunks_20k = None
    if args.corpus == "20k":
        chunks_20k_path = os.path.join(EXPERIMENTS_DIR,
                                       "baseline_20k", "baseline_20k_chunks.pkl")
        if not os.path.exists(chunks_20k_path):
            raise FileNotFoundError(f"20k chunks not found: {chunks_20k_path}")
        with open(chunks_20k_path, "rb") as f:
            chunks_20k = pickle.load(f)
        print(f"Loaded 20k chunks: {len(chunks_20k)} passages\n")

    all_scores = {}

    for label, cfg in BGE_RUNS.items():
        model_dir   = os.path.join(EXPERIMENTS_DIR, cfg["model_dir"])
        chunks_path = os.path.join(EXPERIMENTS_DIR, cfg["chunks_src"])

        for p, name in [(model_dir, "model"), (chunks_path, "chunks")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing {name}: {p}")

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        if args.corpus == "20k":
            index  = get_or_build_20k_index(label, cfg, chunks_20k)
            chunks = chunks_20k
        else:
            index_path = os.path.join(EXPERIMENTS_DIR, cfg["index_file"])
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Missing index: {index_path}")
            index = faiss.read_index(index_path)

        print(f"\nEvaluating: {label}")
        all_scores[label] = evaluate(model_dir, index, chunks, eval_pairs)

    print_results(all_scores, eval_type)
    print_significance(all_scores)

    saveable   = {k: {m: v for m, v in s.items() if not m.startswith("_")}
                  for k, s in all_scores.items()}
    corpus_tag = f"_{args.corpus}" if args.corpus != "5k" else ""
    out_path   = os.path.join(EXPERIMENTS_DIR,
                              f"bge_scores_{eval_type}{corpus_tag}.json")
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
