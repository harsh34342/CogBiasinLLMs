# Multi-Agent LLM Cognitive Bias Framework
## Anchoring & Framing Bias Detection + Mitigation

Replicates and extends the BiasBuster paper (Echterhoff et al., 2024) using a
multi-agent pipeline built on HuggingFace Transformers.

---

## Setup

```bash
pip install transformers torch accelerate sentencepiece pandas matplotlib seaborn tqdm
```

For GPU (recommended):
```bash
pip install transformers torch accelerate sentencepiece pandas matplotlib seaborn tqdm
# torch installs with CUDA automatically if your drivers support it
```

---

## Project Structure

```
bias_framework/
├── main.py                  # Entry point — run everything from here
├── config.py                # Model names, experiment settings
├── agents/
│   ├── base_agent.py        # Base LLM agent class (wraps HuggingFace model)
│   ├── decision_agent.py    # Admissions decision-making agent
│   ├── debiasing_agent.py   # Self-help debiasing agent
│   └── evaluator_agent.py   # Bias evaluation/scoring agent
├── biases/
│   ├── anchoring.py         # Anchoring bias prompts + experiment runner
│   └── framing.py           # Framing bias prompts + experiment runner
├── mitigation/
│   ├── zero_shot.py         # Awareness prompting
│   ├── few_shot.py          # Contrastive + counterfactual prompting
│   └── self_help.py         # Self-help debiasing (LLM rewrites own prompt)
├── data/
│   └── student_profiles.py  # Synthetic student profile generator
├── results/
│   └── (auto-generated CSVs and plots go here)
├── evaluate.py              # Metrics: confidence, delta rates, bias scores
└── visualize.py             # Plots matching paper Figure 3 style
```

---

## Running Experiments

```bash
# Run full pipeline (both biases, all mitigations)
python main.py

# Run only anchoring bias
python main.py --bias anchoring

# Run only framing bias
python main.py --bias framing

# Run with a specific model
python main.py --model "facebook/opt-1.3b"

# Run with mitigation strategy
python main.py --bias framing --mitigation selfhelp

# Run all mitigations for both biases
python main.py --mitigation all
```

---

## Models Supported (open-source, HuggingFace)

| Model | Size | Notes |
|-------|------|-------|
| `facebook/opt-125m` | 125M | Fast, for testing |
| `facebook/opt-1.3b` | 1.3B | Good baseline |
| `facebook/opt-6.7b` | 6.7B | Matches paper's Llama-2-7B scale |
| `meta-llama/Llama-2-7b-chat-hf` | 7B | Requires HF token + approval |
| `meta-llama/Llama-2-13b-chat-hf` | 13B | Requires HF token + approval |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | Good alternative, no gating |
| `tiiuae/falcon-7b-instruct` | 7B | Good alternative |

Set your preferred model in `config.py` or via `--model` flag.

---

## Output

Results are saved to `results/`:
- `anchoring_results.csv` — per-student decision confidence scores
- `framing_results.csv` — admit/reject rates per framing condition
- `bias_summary.csv` — aggregated bias scores across conditions
- `plots/` — bar charts and distribution plots
