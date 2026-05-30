"""
step9_split_manual_eval.py

Splits manual_eval.json into:
  - data/manual_val.json    — 10 randomly selected queries for training validation
  - data/manual_eval140.json — remaining 140 queries for final eval

Run this ONCE. All future Colab training runs use manual_val.json for the
evaluator instead of synthetic pairs. Final eval runs use manual_eval140.json.

Usage:
    python step9_split_manual_eval.py
"""

import os
import json
import random
from config import DATA_DIR

SEED     = 42
VAL_SIZE = 10

random.seed(SEED)

# Load full manual eval set
manual_path = os.path.join(DATA_DIR, "manual_eval.json")
with open(manual_path) as f:
    all_queries = json.load(f)

print(f"Loaded {len(all_queries)} manual eval queries")

# Random split
indices  = list(range(len(all_queries)))
val_idx  = set(random.sample(indices, VAL_SIZE))
eval_idx = [i for i in indices if i not in val_idx]

val_queries  = [all_queries[i] for i in sorted(val_idx)]
eval_queries = [all_queries[i] for i in eval_idx]

# Save
val_path  = os.path.join(DATA_DIR, "manual_val.json")
eval_path = os.path.join(DATA_DIR, "manual_eval140.json")

with open(val_path, "w") as f:
    json.dump(val_queries, f, indent=2)

with open(eval_path, "w") as f:
    json.dump(eval_queries, f, indent=2)

print(f"Saved {len(val_queries)}  queries -> data/manual_val.json    (for Colab validation)")
print(f"Saved {len(eval_queries)} queries -> data/manual_eval140.json (for final eval)")

print(f"\nExample val query: {val_queries[0]['query']}")
print(f"\nIMPORTANT: From now on:")
print(f"  - All Colab training notebooks use manual_val.json for the evaluator")
print(f"  - All eval scripts use manual_eval140.json instead of manual_eval.json")
