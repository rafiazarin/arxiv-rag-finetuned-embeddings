import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")

EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
BASELINE_DIR    = os.path.join(EXPERIMENTS_DIR, "baseline")
FINETUNED_DIR   = os.path.join(EXPERIMENTS_DIR, "finetuned")

# Dataset
DATASET_NAME  = "gfissore/arxiv-abstracts-2021"
NUM_DOCS      = 5000
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50

# Models
OLLAMA_LLM   = "gemma3:4b"
OLLAMA_EMBED = "nomic-embed-text"
FINETUNED_MODEL_PATH = os.path.join(FINETUNED_DIR, "model")

# RAG
TOP_K = 3

# Training
NUM_PAIRS        = 3000
TRAIN_SIZE       = 2700
VAL_SIZE         = 200
# last 100 pairs = held-out eval, never touched during training

# Eval
NUM_EVAL_SAMPLES = 100     # always the last 100 pairs
EVAL_SEED        = 42