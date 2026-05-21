import os
import json
import re
from datasets import load_dataset
from config import DATASET_NAME, DATASET_SPLIT, NUM_DOCS, DATA_DIR

def clean_text(text):
    """Remove LaTeX, extra whitespace, and non-ASCII."""
    text = re.sub(r'\$[^$]+\$', '', text)       # remove inline math
    text = re.sub(r'\\\w+\{[^}]*\}', '', text)  # remove LaTeX commands
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_save():
    print(f"Loading {NUM_DOCS} documents from {DATASET_NAME}...")
    
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        trust_remote_code=True
    )
    
    docs = []
    for item in dataset:
        abstract = item.get("abstract", "") or item.get("text", "")
        title = item.get("title", "")
        
        if not abstract or len(abstract) < 100:
            continue
            
        cleaned = clean_text(abstract)
        if len(cleaned) < 100:
            continue
            
        docs.append({
            "id": item.get("id", str(len(docs))),
            "title": clean_text(title),
            "text": cleaned
        })
        
        if len(docs) >= NUM_DOCS:
            break
    
    # Save locally
    out_path = os.path.join(DATA_DIR, "corpus.json")
    with open(out_path, "w") as f:
        json.dump(docs, f, indent=2)
    
    print(f"Saved {len(docs)} documents to {out_path}")
    print(f"\nSample document:")
    print(f"  Title: {docs[0]['title'][:80]}")
    print(f"  Text:  {docs[0]['text'][:200]}...")

if __name__ == "__main__":
    load_and_save()