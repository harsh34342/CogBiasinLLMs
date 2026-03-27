"""
config.py
---------
Central configuration for the bias framework.
Edit MODEL_NAME and EXPERIMENT settings before running.
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Default model — change this to any HuggingFace model you want to test.
# For quick testing use "facebook/opt-125m"
# For paper-comparable results use "mistralai/Mistral-7B-Instruct-v0.2"
#   or "meta-llama/Llama-2-7b-chat-hf" (requires HF token)
MODEL_NAME = "facebook/opt-1.3b"

# If using Llama-2 (gated model), set your HuggingFace token here or via env var:
# export HUGGINGFACE_TOKEN=hf_your_token_here
HF_TOKEN = None  # or "hf_xxxx"

# Device: "auto" lets accelerate choose best device (GPU > CPU)
DEVICE = "auto"

# Max new tokens for model generation
MAX_NEW_TOKENS = 64

# Temperature for generation (0 = deterministic, higher = more random)
TEMPERATURE = 0.0   # deterministic for reproducibility

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Number of student profiles to use per experiment
# Paper used 5449 anchoring prompts; start small for testing
NUM_STUDENTS_ANCHORING = 10     # number of unique students in a sequence
NUM_ORDER_PERMUTATIONS = 6      # how many order shuffles to test (paper: all permutations)

NUM_STUDENTS_FRAMING = 20       # number of student profiles for framing test
FRAMING_REPEATS = 1             # repeats per student (paper used 1)

# Typical admission rate (from paper: 30%)
BASE_ADMISSION_RATE = 0.30

# Random seed for reproducibility
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# MITIGATION STRATEGIES TO RUN
# ─────────────────────────────────────────────────────────────────────────────

# Options: "baseline", "awareness", "contrastive", "counterfactual", "selfhelp"
MITIGATIONS_TO_RUN = [
    "baseline",
    "awareness",
    "contrastive",
    "counterfactual",
    "selfhelp",
]

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
