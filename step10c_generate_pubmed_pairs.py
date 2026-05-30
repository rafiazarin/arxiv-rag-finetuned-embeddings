"""
step10c_generate_pubmed_pairs.py

Generates 1500 synthetic query-passage pairs from PubMed abstracts
using gemma3:4b — identical prompt and logic to step3_generate_pairs.py.

Split:
    First 1300 pairs  → training
    Pairs 1300-1400   → validation (100)
    Last  100 pairs   → held-out synthetic eval (never touched during training)

Output: data/pubmed_training_pairs.json
Runtime: ~60-90 min on M2. Progress saved every 100 pairs.

Usage:
    python step10c_generate_pubmed_pairs.py
"""

import os
import json
import random
import ollama
from tqdm import tqdm
from config import DATA_DIR

OLLAMA_LLM  = "gemma3:4b"
NUM_PAIRS   = 1500
OUTPUT_PATH = os.path.join(DATA_DIR, "pubmed_training_pairs.json")
SEED        = 42

random.seed(SEED)

# Identical prompt to step3_generate_pairs.py
PROMPT_TEMPLATE = """You are creating training data for an information retrieval system.

Given the following research abstract passage, generate ONE specific question that:
- Can be answered directly from this passage
- Is specific (not too general)
- Sounds like something a researcher would search for

Passage:
{passage}

Output ONLY the question, nothing else. No preamble, no explanation."""

def generate_question(passage: str):
    try:
        response = ollama.generate(
            model=OLLAMA_LLM,
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

def main():
    pool_path = os.path.join(DATA_DIR, "pubmed_pool.json")
    with open(pool_path) as f:
        docs = json.load(f)

    passages = [d["text"] for d in docs if len(d["text"]) > 150]
    random.shuffle(passages)

    pairs = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            pairs = json.load(f)
        print(f"Resuming from {len(pairs)} existing pairs.")

    print(f"Generating {NUM_PAIRS} pairs using {OLLAMA_LLM}...")
    print("Progress saved every 100 pairs.\n")

    for passage in tqdm(passages):
        if len(pairs) >= NUM_PAIRS:
            break
        question = generate_question(passage)
        if question:
            pairs.append({"query": question, "positive": passage})
        if len(pairs) % 100 == 0 and len(pairs) > 0:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(pairs, f, indent=2)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nSaved {len(pairs)} pairs to {OUTPUT_PATH}")
    print(f"Split: 1300 train | 100 val | 100 held-out synthetic eval")
    print(f"Next: upload to Colab and run step10d_train_pubmed_bge.ipynb")

if __name__ == "__main__":
    main()
