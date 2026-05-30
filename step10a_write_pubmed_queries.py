"""
step10a_write_pubmed_queries.py

Interactive tool for writing manual eval queries from PubMed abstracts.
Displays one abstract at a time. You type the query. Progress is saved
after every entry so it's safe to quit and resume anytime.

Controls:
    Type your query and press Enter  — saves and moves to next abstract
    s + Enter                        — skip this abstract (too technical)
    q + Enter                        — quit and save progress

Output: data/pubmed_manual_eval.json
Format: [{query, positive}]  — same as manual_eval.json

Usage:
    python step10a_write_pubmed_queries.py

Goal: 150 queries. Do 10-15 per session.
"""

import os
import json
import random
from datasets import load_dataset
from config import DATA_DIR

# ── Config ────────────────────────────────────────────────────────────────────

TARGET        = 150
SEED          = 42
OUTPUT_PATH   = os.path.join(DATA_DIR, "pubmed_manual_eval.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "pubmed_abstracts_seen.json")
DATASET_NAME  = "ccdv/pubmed-summarization"

# ── Load dataset ──────────────────────────────────────────────────────────────

def load_pubmed_abstracts(n=2000):
    """
    Load a pool of PubMed abstracts to pick from.
    Cached locally after first download.
    """
    cache_path = os.path.join(DATA_DIR, "pubmed_pool.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            pool = json.load(f)
        print(f"Loaded {len(pool)} abstracts from local cache.")
        return pool

    print(f"Downloading PubMed abstracts from HuggingFace (~2 min)...")
    dataset = load_dataset(DATASET_NAME, split="train",
                           trust_remote_code=True, streaming=False)

    pool = []
    for item in dataset:
        abstract = item.get("abstract", "").strip()
        # Filter: must be readable length, not just a title
        if len(abstract) < 200 or len(abstract) > 2000:
            continue
        # Filter: skip abstracts that are mostly numbers/tables
        word_count = len(abstract.split())
        if word_count < 50:
            continue
        pool.append({"text": abstract})
        if len(pool) >= n:
            break

    random.seed(SEED)
    random.shuffle(pool)

    with open(cache_path, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"Saved {len(pool)} abstracts to cache.")
    return pool

# ── Load existing progress ────────────────────────────────────────────────────

def load_progress():
    pairs = []
    seen  = set()

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            pairs = json.load(f)

    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            seen = set(json.load(f))

    return pairs, seen

# ── Save progress ─────────────────────────────────────────────────────────────

def save_progress(pairs, seen):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(pairs, f, indent=2)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(list(seen), f)

# ── Display helpers ───────────────────────────────────────────────────────────

def print_header():
    print("\n" + "="*70)
    print("  PUBMED MANUAL EVAL QUERY WRITER")
    print("  Controls: type query + Enter | 's' to skip | 'q' to quit")
    print("="*70)

def print_abstract(abstract, index, total_done):
    print(f"\n{'─'*70}")
    print(f"  Abstract #{index+1}  |  Queries done: {total_done}/{TARGET}")
    print(f"{'─'*70}")
    # Word-wrap the abstract at 70 chars for readability
    words  = abstract.split()
    line   = ""
    for word in words:
        if len(line) + len(word) + 1 > 68:
            print(f"  {line}")
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        print(f"  {line}")
    print(f"{'─'*70}")

def validate_query(query, abstract):
    """Basic quality checks with warnings."""
    warnings = []
    if len(query) < 15:
        warnings.append("too short (under 15 chars)")
    if len(query) > 200:
        warnings.append("too long (over 200 chars)")
    # Check for direct copying — flag if 5+ consecutive words from abstract
    abstract_words = abstract.lower().split()
    query_words    = query.lower().split()
    for i in range(len(query_words) - 4):
        window = query_words[i:i+5]
        for j in range(len(abstract_words) - 4):
            if abstract_words[j:j+5] == window:
                warnings.append("possible direct copy from abstract")
                break
    return warnings

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print_header()

    abstracts        = load_pubmed_abstracts(n=2000)
    pairs, seen      = load_progress()

    if pairs:
        print(f"\nResuming — {len(pairs)} queries already saved.")

    # Filter out already-seen abstracts
    unseen = [a for i, a in enumerate(abstracts) if i not in seen]

    if len(unseen) == 0:
        print("All abstracts in pool have been seen. Increase pool size.")
        return

    print(f"\nPool: {len(unseen)} unseen abstracts available.")
    print(f"Goal: {TARGET - len(pairs)} more queries to reach {TARGET}.\n")
    input("Press Enter to start...")

    for abs_idx, abstract_item in enumerate(unseen):
        if len(pairs) >= TARGET:
            print(f"\nTarget reached — {TARGET} queries saved.")
            break

        abstract = abstract_item["text"]
        original_idx = abstracts.index(abstract_item)

        print_abstract(abstract, abs_idx, len(pairs))

        while True:
            try:
                raw = input("\n  Your query: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nInterrupted — saving progress...")
                save_progress(pairs, seen)
                return

            if raw.lower() == "q":
                print(f"\nQuitting — saved {len(pairs)} queries.")
                save_progress(pairs, seen)
                return

            if raw.lower() == "s":
                print("  Skipped.")
                seen.add(original_idx)
                save_progress(pairs, seen)
                break

            if not raw:
                print("  Empty input — type your query, 's' to skip, 'q' to quit.")
                continue

            # Add '?' if missing
            query = raw if raw.endswith("?") else raw + "?"

            # Quality checks
            warnings = validate_query(query, abstract)
            if warnings:
                print(f"  ⚠ Warning: {', '.join(warnings)}")
                confirm = input("  Keep it anyway? (y/n): ").strip().lower()
                if confirm != "y":
                    print("  Try again.")
                    continue

            # Save
            pairs.append({"query": query, "positive": abstract})
            seen.add(original_idx)
            save_progress(pairs, seen)
            print(f"  ✓ Saved ({len(pairs)}/{TARGET})")
            break

    print(f"\nSession complete — {len(pairs)} queries saved to {OUTPUT_PATH}")
    remaining = TARGET - len(pairs)
    if remaining > 0:
        print(f"  {remaining} more needed to reach {TARGET}. Run script again to continue.")
    else:
        print(f"  Target reached. Ready for PubMed pipeline.")

if __name__ == "__main__":
    main()
