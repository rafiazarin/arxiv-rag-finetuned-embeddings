import json
import os
from config import FINETUNED_DIR

def main():
    scores_path = os.path.join(FINETUNED_DIR, "comparison_scores.json")

    if not os.path.exists(scores_path):
        print("Run step4_evaluate.py first.")
        return

    with open(scores_path) as f:
        scores = json.load(f)

    # Table header
    col1, col2, col3 = 35, 12, 12
    divider = "-" * (col1 + col2 + col3 + 6)

    print("\n" + "=" * (col1 + col2 + col3 + 6))
    print("  RAG RETRIEVAL EVALUATION — arXiv CS Abstracts")
    print(f"  Metric: Hit Rate@{3} and MRR@{3} on 100 held-out queries")
    print("=" * (col1 + col2 + col3 + 6))
    print(f"  {'Method':<{col1}} {'Hit Rate':>{col2}} {'MRR':>{col3}}")
    print(divider)

    # Sort by hit_rate ascending so fine-tuned shows last (most impactful)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["hit_rate"])

    for method, s in sorted_scores:
        marker = "  ◀ ours" if "Fine-tuned" in method else ""
        print(f"  {method:<{col1}} {s['hit_rate']:>{col2}.4f} {s['mrr']:>{col3}.4f}{marker}")

    print(divider)

    # Delta vs baseline
    if "Baseline (nomic-embed-text)" in scores and "Fine-tuned (ours)" in scores:
        base = scores["Baseline (nomic-embed-text)"]
        ft   = scores["Fine-tuned (ours)"]
        hr_delta  = round(ft["hit_rate"] - base["hit_rate"], 4)
        mrr_delta = round(ft["mrr"]      - base["mrr"],      4)
        hr_pct    = round((hr_delta / base["hit_rate"]) * 100, 1) if base["hit_rate"] else 0
        mrr_pct   = round((mrr_delta / base["mrr"])     * 100, 1) if base["mrr"]      else 0

        print(f"\n  Improvement over baseline:")
        print(f"    Hit Rate : {hr_delta:+.4f}  ({hr_pct:+.1f}%)")
        print(f"    MRR      : {mrr_delta:+.4f}  ({mrr_pct:+.1f}%)")

    print("=" * (col1 + col2 + col3 + 6) + "\n")

if __name__ == "__main__":
    main()
    
