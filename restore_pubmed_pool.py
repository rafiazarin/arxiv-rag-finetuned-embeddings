"""
restore_pubmed_pool.py

Rebuilds data/pubmed_pool.json exactly as step10a_write_pubmed_queries.py did
(same dataset, same filters, same order, same seed-42 shuffle).

Uses streaming, so the full dataset (2.3 GB) is never downloaded to disk.
Only saves if every PubMed positive already on GitHub is found in the pool.

Usage:
    python restore_pubmed_pool.py
"""
import os
import json
import random
from datasets import load_dataset
from config import DATA_DIR

SEED = 42
N    = 2000
OUT  = os.path.join(DATA_DIR, "pubmed_pool.json")
CHECK_FILES = ["pubmed_training_pairs.json", "pubmed_manual_eval.json"]


def build_pool(config_name):
    ds = load_dataset("ccdv/pubmed-summarization", config_name,
                      split="train", streaming=True)
    pool = []
    for item in ds:
        abstract = (item.get("abstract") or "").strip()
        if len(abstract) < 200 or len(abstract) > 2000:   # same as step10a
            continue
        if len(abstract.split()) < 50:                     # same as step10a
            continue
        pool.append({"text": abstract})
        if len(pool) >= N:
            break
    random.seed(SEED)                                      # same as step10a
    random.shuffle(pool)
    return pool


def verify(pool):
    texts = {p["text"] for p in pool}
    ok = True
    for fname in CHECK_FILES:
        with open(os.path.join(DATA_DIR, fname)) as f:
            pairs = json.load(f)
        found = sum(p["positive"] in texts for p in pairs)
        print(f"  {fname}: {found}/{len(pairs)} positives found")
        ok = ok and found == len(pairs)
    return ok


def main():
    for config_name in ["section", "document"]:
        print(f"Trying subset '{config_name}' (streaming)...")
        pool = build_pool(config_name)
        print(f"  pool size: {len(pool)}")
        if verify(pool):
            with open(OUT, "w") as f:
                json.dump(pool, f, indent=2)
            print(f"MATCH - saved {len(pool)} abstracts to {OUT}")
            return
        print("  no full match with this subset")
    print("MISMATCH - nothing saved.")


if __name__ == "__main__":
    main()
