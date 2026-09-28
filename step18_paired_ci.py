"""
step18_paired_ci.py

Adds two analyses to the PubMed results, using saved per-query scores only:
  1. 95% paired bootstrap CI (10,000 resamples, seed 42) for the MRR difference,
     fine-tuned (5 ep) minus untuned BGE, in each evaluation setting.
  2. Train/test overlap: how many evaluation questions have a source abstract that
     was also a training positive (first 1,300 pairs), and the MRR difference split
     by seen / unseen abstracts (exploratory, not tested).
Output: experiments/paired_ci_and_overlap.json
Usage:  python step18_paired_ci.py
"""
import json
import numpy as np

FT, NO = "BGE-base (PubMed FT, 5 ep)", "BGE-base (no FT)"
SETTINGS = {"objective_2k": "experiments/pubmed_source_match_scores.json",
            "objective_20k": "experiments/pubmed_20k_source_match_scores.json",
            "automatic_judge_2k": "experiments/pubmed_step11_scores.json"}

train = json.load(open("data/pubmed_training_pairs.json"))[:1300]
evalq = json.load(open("data/pubmed_manual_eval90.json"))
train_pos = {p["positive"] for p in train}
seen = np.array([p["positive"] in train_pos for p in evalq])
out = {"n_eval": len(evalq), "n_eval_source_in_training": int(seen.sum()), "settings": {}}
print(f"Eval questions whose source abstract was a training positive: {seen.sum()}/{len(evalq)}")

rng = np.random.default_rng(42)
for name, path in SETTINGS.items():
    s = json.load(open(path))["scores"]
    ft, no = np.array(s[FT]["per_query_rr"]), np.array(s[NO]["per_query_rr"])
    d = ft - no
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    r = {"mrr_diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
         "seen_diff": round(float(d[seen].mean()), 4), "unseen_diff": round(float(d[~seen].mean()), 4)}
    out["settings"][name] = r
    print(f"{name:<20} MRR diff {r['mrr_diff']:+.4f}  95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
          f"  | seen {r['seen_diff']:+.4f}  unseen {r['unseen_diff']:+.4f}")

with open("experiments/paired_ci_and_overlap.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved to experiments/paired_ci_and_overlap.json")
