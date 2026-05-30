"""
step5_train_mixed.py

Fetches MS MARCO pairs, mixes with arXiv pairs at configurable ratios,
trains a separate model for each ratio, saves to experiments/mixed_<ratio>/model

Usage:
    python step5_train_mixed.py              # trains all ratios
    python step5_train_mixed.py --ratio 80   # trains only 80/20
"""
import os
import json
import argparse
import random
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from config import DATA_DIR, EXPERIMENTS_DIR, TRAIN_SIZE
#%pip install 'accelerate>=1.1.0'
# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL    = "all-MiniLM-L6-v2"   # same base as your arXiv-only model
MARCO_CACHE   = os.path.join(DATA_DIR, "marco_pairs.json")
RATIOS        = [80, 70, 50]          # arXiv % — so 80 means 80% arXiv, 20% MARCO
EPOCHS        = 5
BATCH_SIZE    = 32
WARMUP_STEPS  = 100
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)

# ── Step 1: Load arXiv pairs ──────────────────────────────────────────────────

def load_arxiv_pairs():
    path = os.path.join(DATA_DIR, "training_pairs.json")
    with open(path) as f:
        all_pairs = json.load(f)
    # Never touch the last 100 — those are your held-out eval set
    train_pairs = all_pairs[:-100]
    print(f"Loaded {len(train_pairs)} arXiv training pairs (held-out 100 for eval)")
    return train_pairs

# ── Step 2: Fetch MS MARCO pairs ─────────────────────────────────────────────

def load_marco_pairs(n=1000):
    """
    Fetch n query-passage pairs from MS MARCO.
    Cached locally so you only download once.
    """
    if os.path.exists(MARCO_CACHE):
        with open(MARCO_CACHE) as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} MS MARCO pairs from cache")
        return pairs

    print(f"Downloading MS MARCO pairs from HuggingFace (~few minutes)...")
    dataset = load_dataset(
        "ms_marco",
        "v1.1",
        split="train",
        trust_remote_code=True
    )

    pairs = []
    for item in dataset:
        query = item["query"]
        # MS MARCO has multiple passages; take the first positive one
        passages = item.get("passages", {})
        texts    = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])

        for text, selected in zip(texts, is_selected):
            if selected == 1 and len(text) > 100:
                pairs.append({"query": query, "positive": text})
                break

        if len(pairs) >= n:
            break

    with open(MARCO_CACHE, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Saved {len(pairs)} MS MARCO pairs to {MARCO_CACHE}")
    return pairs

# ── Step 3: Mix pairs at a given ratio ────────────────────────────────────────

def mix_pairs(arxiv_pairs, marco_pairs, arxiv_pct):
    """
    arxiv_pct: integer 0-100, % of final set that is arXiv
    Total size matches TRAIN_SIZE from config.
    """
    marco_pct   = 100 - arxiv_pct
    n_arxiv     = int(TRAIN_SIZE * arxiv_pct / 100)
    n_marco     = TRAIN_SIZE - n_arxiv

    # Sample without replacement; shuffle before slicing for reproducibility
    arxiv_sample = random.sample(arxiv_pairs, min(n_arxiv, len(arxiv_pairs)))
    marco_sample = random.sample(marco_pairs, min(n_marco, len(marco_pairs)))

    mixed = arxiv_sample + marco_sample
    random.shuffle(mixed)

    print(f"Mixed dataset: {len(arxiv_sample)} arXiv + {len(marco_sample)} MARCO "
          f"= {len(mixed)} total ({arxiv_pct}/{marco_pct} split)")
    return mixed

# ── Step 4: Train ─────────────────────────────────────────────────────────────

def train(pairs, out_dir, ratio_label):
    os.makedirs(out_dir, exist_ok=True)

    # Convert to InputExample format
    examples = [
        InputExample(texts=[p["query"], p["positive"]])
        for p in pairs
    ]

    model     = SentenceTransformer(BASE_MODEL)
    loader    = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    loss      = MultipleNegativesRankingLoss(model)

    # Validation: use your held-out arXiv eval pairs as queries
    # (these are never in training regardless of ratio)
    val_path = os.path.join(DATA_DIR, "training_pairs.json")
    with open(val_path) as f:
        all_pairs = json.load(f)
    val_pairs = all_pairs[-100:]

    queries   = {str(i): p["query"]    for i, p in enumerate(val_pairs)}
    corpus    = {str(i): p["positive"] for i, p in enumerate(val_pairs)}
    relevant  = {str(i): {str(i)}      for i in range(len(val_pairs))}

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant,
        name=f"val-{ratio_label}"
    )

    total_steps = len(loader) * EPOCHS

    print(f"\nTraining {ratio_label} model — {len(pairs)} pairs, "
          f"{EPOCHS} epochs, {total_steps} steps...")

    model.fit(
        train_objectives=[(loader, loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        evaluation_steps=len(loader),   # eval every epoch
        output_path=out_dir,
        save_best_model=True,
        show_progress_bar=True
    )

    print(f"Saved {ratio_label} model to {out_dir}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratio", type=int, choices=RATIOS, default=None,
        help="arXiv %% to train (e.g. 80 = 80/20 split). Omit to train all."
    )
    args = parser.parse_args()

    arxiv_pairs = load_arxiv_pairs()
    marco_pairs = load_marco_pairs(n=1500)  # fetch 1500, sample from these

    ratios_to_run = [args.ratio] if args.ratio else RATIOS

    for arxiv_pct in ratios_to_run:
        ratio_label = f"arxiv{arxiv_pct}_marco{100 - arxiv_pct}"
        out_dir     = os.path.join(EXPERIMENTS_DIR, f"mixed_{arxiv_pct}", "model")

        print(f"\n{'='*60}")
        print(f"Training ratio: {arxiv_pct}% arXiv / {100-arxiv_pct}% MARCO")
        print(f"Output: {out_dir}")
        print(f"{'='*60}")

        mixed = mix_pairs(arxiv_pairs, marco_pairs, arxiv_pct)
        train(mixed, out_dir, ratio_label)

    print("\nAll training runs complete.")
    print("Next: run step5b_eval_mixed.py to evaluate each ratio.")

if __name__ == "__main__":
    main()