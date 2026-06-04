import os
import json
import random
import ollama
from tqdm import tqdm
from config import DATA_DIR, OLLAMA_LLM, NUM_PAIRS

PROMPT_TEMPLATE = """You are creating training data for an information retrieval system.

Given the following research abstract passage, generate ONE specific question that:
- Can be answered directly from this passage
- Is specific (not too general)
- Sounds like something a researcher would search for

Passage:
{passage}

Output ONLY the question, nothing else. No preamble, no explanation."""

def generate_question(passage: str) -> str | None:
    try:
        response = ollama.generate(
            model=OLLAMA_LLM,
            prompt=PROMPT_TEMPLATE.format(passage=passage),
            options={"temperature": 0.7, "num_predict": 80}
        )
        question = response["response"].strip()
        # Basic quality filter
        if len(question) < 15 or len(question) > 200:
            return None
        if not question.endswith("?"):
            question = question.split("\n")[0]  # take first line
        return question
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_pairs():
    corpus_path = os.path.join(DATA_DIR, "corpus.json")
    with open(corpus_path) as f:
        docs = json.load(f)
    
    # Use document texts directly (not chunks) for richer context
    passages = [d["text"] for d in docs if len(d["text"]) > 150]
    random.shuffle(passages)
    
    pairs = []
    print(f"Generating {NUM_PAIRS} training pairs using {OLLAMA_LLM}...")
    print("This will take ~60-90 minutes on M2. You can stop and resume — progress is saved every 100 pairs.\n")
    
    out_path = os.path.join(DATA_DIR, "training_pairs.json")
    
    # Load existing progress if resuming
    if os.path.exists(out_path):
        with open(out_path) as f:
            pairs = json.load(f)
        print(f"Resuming from {len(pairs)} existing pairs.")
    
    for i, passage in enumerate(tqdm(passages)):
        if len(pairs) >= NUM_PAIRS:
            break
            
        question = generate_question(passage)
        if question:
            pairs.append({
                "query": question,
                "positive": passage
            })
        
        # Save every 100 pairs so you don't lose progress
        if len(pairs) % 100 == 0 and len(pairs) > 0:
            with open(out_path, "w") as f:
                json.dump(pairs, f, indent=2)
    
    # Final save
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)
    
    print(f"\nSaved {len(pairs)} training pairs to {out_path}")
    print(f"Sample pair:")
    print(f"  Query:    {pairs[0]['query']}")
    print(f"  Positive: {pairs[0]['positive'][:150]}...")

if __name__ == "__main__":
    generate_pairs()
    
    
