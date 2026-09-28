"""
step13_matcher_validation.py

Checks the automatic relevance matcher (cross-encoder/ms-marco-MiniLM-L-6-v2,
logit >= 0) against human judgments on PubMed.

PRE-REGISTERED (decided before labeling):
  - Pool: top-3 chunks retrieved for the 90 manual queries by untuned BGE-base
    and the published fine-tuned model (unique query-chunk pairs). Chunks that
    are identical to the source abstract are excluded (always counted relevant).
  - Sample: 30 pairs the matcher calls relevant + 30 it calls not relevant,
    random seed 42, shown in shuffled order, matcher decision hidden.
  - Human question: does this chunk contain the information that answers the query?
  - Reported: precision (matcher-yes that the human says yes), NPV (matcher-no
    that the human says no), agreement re-weighted to the pool, Cohen's kappa on
    the sample, all with counts. 'Unsure' labels are reported and excluded.

Usage:
    python step13_matcher_validation.py sample   # builds data/matcher_validation_sample.json
    python step13_matcher_validation.py label    # label interactively (resumable)
    python step13_matcher_validation.py score    # results -> experiments/matcher_validation.json
Do not open the sample file before labeling: it contains the matcher's decisions.
"""
import os
import sys
import json
import pickle
import random
import textwrap
from config import DATA_DIR, EXPERIMENTS_DIR

SAMPLE_PATH = os.path.join(DATA_DIR, "matcher_validation_sample.json")
LABELS_PATH = os.path.join(DATA_DIR, "matcher_validation_labels.json")
OUT_PATH    = os.path.join(EXPERIMENTS_DIR, "matcher_validation.json")
N_PER_SIDE  = 30
SEED        = 42
E = lambda *p: os.path.join(EXPERIMENTS_DIR, *p)


def cmd_sample():
    import faiss
    from sentence_transformers import SentenceTransformer
    import step10e_eval_pubmed as pub

    if os.path.exists(SAMPLE_PATH):
        print(f"Sample already exists: {SAMPLE_PATH} (not overwriting).")
        return
    pairs = pub.load_eval_pairs("manual")
    with open(E("pubmed_baseline", "pubmed_baseline_chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    setups = [("BAAI/bge-base-en-v1.5", E("bge_base_noft", "pubmed.faiss")),
              ("rafiazarin/bge-base-pubmed-finetuned", E("pubmed_bge_hub", "pubmed_bge_hub.faiss"))]
    pool = set()
    for model_name, index_path in setups:
        print(f"Retrieving top-3 with {model_name}...")
        model = SentenceTransformer(model_name)
        index = faiss.read_index(index_path)
        for qi, p in enumerate(pairs):
            q = model.encode([p["query"]], normalize_embeddings=True).astype("float32")
            _, idxs = index.search(q, 3)
            for ci in idxs[0]:
                pool.add((qi, int(ci)))

    print(f"Scoring {len(pool)} unique query-chunk pairs with the matcher...")
    matcher = pub.get_match_model()
    items = []
    for qi, ci in sorted(pool):
        gt, text = pairs[qi]["positive"], chunks[ci]["text"]
        if text == gt:
            continue
        score = float(matcher.predict([(gt, text)])[0])
        items.append({"query_idx": qi, "chunk_idx": ci, "query": pairs[qi]["query"],
                      "source_abstract": gt, "chunk": text,
                      "matcher_score": score, "matcher_relevant": score >= 0.0,
                      "chunk_from_source": text in gt})

    pos = [x for x in items if x["matcher_relevant"]]
    neg = [x for x in items if not x["matcher_relevant"]]
    rng = random.Random(SEED)
    sample = rng.sample(pos, min(N_PER_SIDE, len(pos))) + rng.sample(neg, min(N_PER_SIDE, len(neg)))
    rng.shuffle(sample)
    out = {"pool_size": len(items), "pool_matcher_positive": len(pos),
           "pool_matcher_negative": len(neg), "sample": sample}
    with open(SAMPLE_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Pool: {len(items)} pairs ({len(pos)} matcher-relevant, {len(neg)} not)")
    print(f"Saved sample of {len(sample)} to {SAMPLE_PATH}")


def wrap(t):
    return textwrap.fill(t, width=90, initial_indent="  ", subsequent_indent="  ")


def cmd_label():
    sample = json.load(open(SAMPLE_PATH))["sample"]
    labels = json.load(open(LABELS_PATH)) if os.path.exists(LABELS_PATH) else {}
    print("Question: does the CHUNK contain the information that answers the QUERY?")
    print("Keys: y = yes | n = no | u = unsure | q = quit (progress is saved)\n")
    for i, item in enumerate(sample):
        if str(i) in labels:
            continue
        print("=" * 94)
        print(f"Item {i + 1}/{len(sample)}   (labeled so far: {len(labels)})")
        print("\nQUERY:\n" + wrap(item["query"]))
        print("\nSOURCE ABSTRACT (the query was written from this):\n" + wrap(item["source_abstract"]))
        print("\nCHUNK (retrieved):\n" + wrap(item["chunk"]))
        while True:
            ans = input("\nDoes the chunk answer the query? [y/n/u/q]: ").strip().lower()
            if ans in ("y", "n", "u", "q"):
                break
        if ans == "q":
            break
        labels[str(i)] = ans
        with open(LABELS_PATH, "w") as f:
            json.dump(labels, f, indent=2)
    print(f"\nSaved {len(labels)}/{len(sample)} labels to {LABELS_PATH}")


def cmd_score():
    data = json.load(open(SAMPLE_PATH))
    sample = data["sample"]
    labels = json.load(open(LABELS_PATH))
    if len(labels) < len(sample):
        print(f"Only {len(labels)}/{len(sample)} labeled. Finish labeling first.")
        return
    rows = [(s["matcher_relevant"], labels[str(i)], s["chunk_from_source"]) for i, s in enumerate(sample)]
    unsure = sum(1 for _, h, _ in rows if h == "u")
    rows = [(m, h == "y", src) for m, h, src in rows if h != "u"]

    tp = sum(1 for m, h, _ in rows if m and h)
    fp = sum(1 for m, h, _ in rows if m and not h)
    tn = sum(1 for m, h, _ in rows if not m and not h)
    fn = sum(1 for m, h, _ in rows if not m and h)
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    p_obs = (tp + tn) / n
    p_exp = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else float("nan")
    w_pos = data["pool_matcher_positive"] / data["pool_size"]
    weighted = w_pos * precision + (1 - w_pos) * npv

    def split(from_src):
        r = [(m, h) for m, h, s in rows if s == from_src]
        return {"n": len(r), "agree": sum(1 for m, h in r if m == h)}

    out = {"n_labeled": n, "n_unsure": unsure,
           "confusion": {"matcher_yes_human_yes": tp, "matcher_yes_human_no": fp,
                         "matcher_no_human_no": tn, "matcher_no_human_yes": fn},
           "precision": round(precision, 4), "npv": round(npv, 4),
           "sample_agreement": round(p_obs, 4), "cohens_kappa": round(kappa, 4),
           "pool_matcher_positive_rate": round(w_pos, 4),
           "agreement_reweighted_to_pool": round(weighted, 4),
           "chunks_from_source_abstract": split(True),
           "chunks_from_other_abstracts": split(False)}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    cmds = {"sample": cmd_sample, "label": cmd_label, "score": cmd_score}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(1)
    cmds[sys.argv[1]]()
