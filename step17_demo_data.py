"""
step17_demo_data.py

Builds the data files for the in-browser search demo, and measures how much the
demo's compression changes results compared with the full-precision evaluation.

Demo setup : 8-bit ONNX query encoders (what the browser runs) + int8 chunk vectors.
Full setup : PyTorch fp32 query encoders + fp32 chunk vectors (the FAISS indexes).
Reported per model over the 90 manual queries: query-embedding cosine (demo vs full),
identical top-3 rate, top-3 overlap, and Metric-1 MRR@3 for full vs demo.

Outputs: ~/Projects/pubmed-search-demo/data/ (chunks.json, emb_noft.i8, emb_ft.i8, meta.json)
         experiments/demo_quantization_check.json
Usage:   python step17_demo_data.py
"""
import os
import json
import pickle
import numpy as np
import faiss
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import step10e_eval_pubmed as pub
import step11_eval_pubmed_baselines as s11

OUT = os.path.expanduser("~/Projects/pubmed-search-demo/data")
REPORT = s11.E("demo_quantization_check.json")
MODELS = {
    "noft": {"st": "BAAI/bge-base-en-v1.5", "onnx_repo": "rafiazarin/bge-base-en-v1.5-onnx",
             "index": s11.E("bge_base_noft", "pubmed.faiss")},
    "ft": {"st": "rafiazarin/bge-base-pubmed-finetuned", "onnx_repo": "rafiazarin/bge-base-pubmed-finetuned",
           "index": s11.E("pubmed_bge_hub", "pubmed_bge_hub.faiss")},
}


def onnx_encoder(repo):
    path = hf_hub_download(repo, "onnx/model_quantized.onnx")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(repo)
    names = {i.name for i in sess.get_inputs()}

    def enc(text):
        t = tok([text], truncation=True, max_length=512, return_tensors="np")
        h = sess.run(None, {k: t[k].astype(np.int64) for k in t if k in names})[0][:, 0]
        return (h / np.linalg.norm(h, axis=1, keepdims=True))[0].astype(np.float32)
    return enc


def main():
    pairs = pub.load_eval_pairs("manual")
    with open(s11.need(s11.E("pubmed_baseline", "pubmed_baseline_chunks.pkl")), "rb") as f:
        chunks = pickle.load(f)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "chunks.json"), "w") as f:
        json.dump([c["text"] for c in chunks], f)

    def rr(top, positive):
        return next((1.0 / (i + 1) for i, c in enumerate(top) if chunks[c]["text"] in positive), 0.0)

    report = {}
    for key, cfg in MODELS.items():
        print(f"\n== {key}: {cfg['st']}")
        index = faiss.read_index(s11.need(cfg["index"]))
        full = index.reconstruct_n(0, index.ntotal).astype(np.float32)
        q8 = np.clip(np.round(full * 127), -127, 127).astype(np.int8)
        q8.tofile(os.path.join(OUT, f"emb_{key}.i8"))
        demo = q8.astype(np.float32) / 127
        st = SentenceTransformer(cfg["st"])
        enc = onnx_encoder(cfg["onnx_repo"])
        cos, same, overlap, rr_full, rr_demo = [], [], [], [], []
        for p in pairs:
            qf = st.encode(p["query"], normalize_embeddings=True)
            qd = enc(p["query"])
            cos.append(float(qf @ qd))
            tf = list(np.argsort(-(full @ qf))[:3])
            td = list(np.argsort(-(demo @ qd))[:3])
            same.append(tf == td)
            overlap.append(len(set(tf) & set(td)) / 3)
            rr_full.append(rr(tf, p["positive"]))
            rr_demo.append(rr(td, p["positive"]))
        report[key] = {"query_cosine_mean": round(float(np.mean(cos)), 4),
                       "query_cosine_min": round(float(np.min(cos)), 4),
                       "identical_top3_rate": round(float(np.mean(same)), 4),
                       "top3_overlap_mean": round(float(np.mean(overlap)), 4),
                       "metric1_mrr_full": round(float(np.mean(rr_full)), 4),
                       "metric1_mrr_demo": round(float(np.mean(rr_demo)), 4)}
        print(json.dumps(report[key], indent=2))

    meta = {"n_chunks": len(chunks), "dim": 768, "int8_scale": 127,
            "models": {k: {"onnx_repo": v["onnx_repo"], "file": f"emb_{k}.i8"} for k, v in MODELS.items()}}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDemo data in {OUT}\nReport saved to {REPORT}")


if __name__ == "__main__":
    main()
