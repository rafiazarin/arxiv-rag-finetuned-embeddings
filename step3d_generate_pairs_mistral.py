"""
step3d_generate_pairs_mistral.py

Generates 3000 synthetic query-passage pairs using mistral:latest instead of
gemma3:4b. Everything else is identical to step3_generate_pairs.py — same
prompt, same corpus, same temperature, same quality filters.

Scientific purpose:
    If evaluation contamination is real, a model trained on gemma-generated
    pairs should score higher on gemma-generated eval queries than a model
    trained on mistral-generated pairs, even when both are evaluated on the
    same human-written queries. This directly tests whether the contamination
    effect is generator-specific.

Output: data/training_pairs_mistral.json

Usage:
    python step3d_generate_pairs_mistral.py

Runtime: ~60-90 minutes on M2. Progress saved every 100 pairs — safe to stop
and resume.
"""

import os
import json
import random
import ollama
from tqdm import tqdm
from config import DATA_DIR, NUM_PAIRS

# ── Only change vs step3_generate_pairs.py ────────────────────────────────────
MISTRAL_MODEL = "mistral:latest"
OUTPUT_FILE   = "training_pairs_mistral.json"

# ── Identical prompt to gemma3:4b run ─────────────────────────────────────────
PROMPT_TEMPLATE = """You are creating training data for an information retrieval system.

Given the following research abstract passage, generate ONE specific question that:
- Can be answered directly from this passage
- Is specific (not too general)
- Sounds like something a researcher would search for

Passage:
{passage}

Output ONLY the question, nothing else. No preamble, no explanation."""

# ── Identical generation logic ─────────────────────────────────────────────────

def generate_question(passage: str) -> str | None:
    try:
        response = ollama.generate(
            model=MISTRAL_MODEL,
            prompt=PROMPT_TEMPLATE.format(passage=passage),
            options={"temperature": 0.7, "num_predict": 80}
        )
        question = response["response"].strip()
        if len(question) < 15 or len(question) > 200:
            return None
        if not question.endswith("?"):
            question = question.split("\n")[0]
        return question
    except Exception as e:
        print(f"Error: {e}")
        return None

# ── Main ──────────────────────────────────────────────────────────────────────

def generate_pairs():
    corpus_path = os.path.join(DATA_DIR, "corpus.json")
    with open(corpus_path) as f:
        docs = json.load(f)

    passages = [d["text"] for d in docs if len(d["text"]) > 150]
    random.shuffle(passages)

    out_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    pairs = []

    # Resume if interrupted
    if os.path.exists(out_path):
        with open(out_path) as f:
            pairs = json.load(f)
        print(f"Resuming from {len(pairs)} existing pairs.")

    print(f"Generating {NUM_PAIRS} pairs using {MISTRAL_MODEL}...")
    print("Progress saved every 100 pairs — safe to stop and resume.\n")

    for passage in tqdm(passages):
        if len(pairs) >= NUM_PAIRS:
            break

        question = generate_question(passage)
        if question:
            pairs.append({"query": question, "positive": passage})

        if len(pairs) % 100 == 0 and len(pairs) > 0:
            with open(out_path, "w") as f:
                json.dump(pairs, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nSaved {len(pairs)} pairs to {out_path}")
    print(f"\nExample pair:")
    print(f"  Query    : {pairs[0]['query']}")
    print(f"  Positive : {pairs[0]['positive'][:150]}...")
    print(f"\nNext: upload training_pairs_mistral.json to Colab and retrain bge-base.")

if __name__ == "__main__":
    generate_pairs()
