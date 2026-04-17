# Experiment Process Notes

This document explains the two core questions in this project, how the code answers them, and how the team iterated from a full target setup to small debug runs, including the problems we hit and how they were fixed.

## 1) The Two Questions and How We Answer Them

### Question A: Anchoring Bias

Question:
If the same applicants are shown in different orders, does the model change decisions because earlier decisions anchor later ones?

What "answering" this question means in this project:
1. Generate a set of student profiles.
2. Evaluate them sequentially, where each next prompt includes previous students and previous model decisions.
3. Repeat with multiple random order permutations.
4. For each student, measure how stable admit/reject is across permutations.
5. Aggregate this into a confidence-style anchoring metric.

Where this is implemented:
- anchoring.py
- student_profiles.py
- decision_agent.py
- debiasing_agent.py (for multi-agent self-help rewriting)
- evaluate.py

Exact process:
1. Profile generation:
   - generate_sequential_student_set in student_profiles.py creates synthetic admissions profiles.
2. Order perturbation:
   - AnchoringExperiment.run in anchoring.py creates n_permutations random orderings using a seeded RNG.
3. Sequential prompt construction:
   - build_sequential_prompt in anchoring.py appends prior student text plus prior answers before asking about the next student.
4. Mitigation branch:
   - baseline: no rewrite.
   - single_selfhelp: decision model rewrites its own prompt first.
   - multi_selfhelp: DebiasingAgent rewrites prompt before DecisionAgent sees it.
   - no_debias_agent: explicit no-rewrite path.
5. Decision generation:
   - DecisionAgent uses BaseAgent.generate_decision for short answers.
   - BaseAgent.extract_decision parses admit or reject.
6. Parse quality accounting:
   - Every decision call is tracked as parse success/failure.
   - anchoring.py computes parse_success_rate and warns if below threshold.
7. Anchoring metric:
   - Per student admit rate across permutations is compared to overall admit rate.
   - Normalized Euclidean distance (confidence_d) is reported.
8. Output files:
   - per-student CSV and summary CSV are written through evaluate.py.

Interpretation:
- Low consistency across permutations means stronger anchoring effect.
- High consistency means weaker anchoring effect.
- If parse quality is low, metrics are marked as unreliable.

---

### Question B: Framing Bias

Question:
If the profile text is identical but the question wording changes (admit frame vs reject frame), does the model change decision?

What "answering" this question means in this project:
1. Keep each student profile constant.
2. Ask three variants:
   - Will you admit this student?
   - Will you reject this student?
   - Neutral wording.
3. Compare admit rates between admit-frame and reject-frame.
4. Count per-student inconsistencies.

Where this is implemented:
- framing.py
- student_profiles.py
- decision_agent.py
- debiasing_agent.py
- evaluate.py

Exact process:
1. Profile generation:
   - generate_student_profiles in student_profiles.py produces synthetic profiles.
2. Prompt construction:
   - build_framing_prompt in framing.py combines profile text with one framing question.
3. Mitigation branch:
   - Same mitigation options as anchoring.
4. Decision generation and parsing:
   - BaseAgent.generate_decision plus BaseAgent.extract_decision.
5. Parse quality accounting:
   - parse_ok_admit, parse_ok_reject, parse_ok_neutral stored per student.
   - framing.py computes parse_success_rate and parse_quality_pass.
6. Framing metric:
   - delta = admit_rate(admit_frame) - admit_rate(reject_frame).
   - n_inconsistent counts students where admit-frame and reject-frame decisions differ.
7. Output files:
   - per-student framing CSV and summary CSV via evaluate.py.

Interpretation:
- delta near 0 and low inconsistency suggests less framing sensitivity.
- Larger absolute delta and higher inconsistency suggests stronger framing sensitivity.
- If parse quality is low, those numbers are not trusted as behavior evidence.

---

## How the Files Work Together

### Core orchestration
- main.py
  - General experiment runner.
  - Reads config defaults and CLI overrides.
  - Creates agents, runs anchoring/framing experiments, saves summaries and plots.

- run_all_experiments.py
  - Crash-safe batch runner for the focused two-question demo matrix.
  - Uses one small model and small sample sizes.
  - Iterates all methods across both biases.
  - Appends checkpoint rows so runs can resume.

### Experiment logic
- anchoring.py
  - Anchoring prompt design, sequential runs, permutation logic, anchoring metrics.

- framing.py
  - Framing prompt design and framing metrics.

### Agents
- base_agent.py
  - HuggingFace generation wrapper, chat formatting, parsing, usage stats.

- decision_agent.py
  - Admissions decision role.

- debiasing_agent.py
  - Prompt rewriting role for self-help variants.

- evaluator_agent.py
  - Optional qualitative auditing layer.

### Data and metrics
- student_profiles.py
  - Synthetic student profile generator with seeded randomness.

- evaluate.py
  - Metric summarization and CSV writing.

- visualize.py
  - Optional chart generation.

### Configuration and outputs
- config.py
  - Full target settings and default model setup.

- results/
  - Experiment artifacts.
  - results/demo is the small fast matrix output from run_all_experiments.py.

---

## 2) How We Got Here (Process Timeline)

### Stage 1: Define full target setup

The full intended setup (for meaningful evaluation scale) is in config.py:
- NUM_STUDENTS_ANCHORING = 50
- NUM_ORDER_PERMUTATIONS = 6
- NUM_STUDENTS_FRAMING = 50
- RANDOM_SEED = 42
- TEMPERATURE = 0.0

This is the "real experiment" setting used when running from main.py with stronger models.

Why these values:
- 50 students and 6 permutations are a practical middle ground: more stable than tiny runs, much cheaper than exhaustive permutation study.
- seed 42 and temperature 0 support reproducibility.

### Stage 2: Run small and cheap first (structure validation)

Before expensive model runs, we intentionally used a tiny setup in run_all_experiments.py:
- model: facebook/opt-125m
- N_STUDENTS = 6
- N_PERMUTATIONS = 2

Why:
- Validate all code paths quickly.
- Generate end-to-end CSVs to check plumbing.
- Catch formatting/parsing/runtime issues before scaling up.

### Stage 3: Problems discovered in small runs

Observed issues:
1. Many outputs were not clean admit/reject decisions.
2. Raw outputs contained prompt echoing and malformed text.
3. Multi-selfhelp cases showed repeated artifacts like [end of prompt].
4. Decision columns became empty in many rows.
5. Aggregated metrics appeared flat (for example zeros), but this was partly parsing failure, not true model behavior.

Why this happened:
- Very small base model (opt-125m) is weak at strict instruction following.
- Long generation and noisy rewrites increased chance of malformed outputs.
- Parser previously had less guardrails for noisy text.

### Stage 4: Fixes implemented

Fix A: Short decision generation path
- Added decision_max_new_tokens and generate_decision in base_agent.py.
- Decision calls now use short outputs to reduce rambling/echoing.

Fix B: Stronger parsing logic
- extract_decision now prioritizes first non-empty generated line.
- Fallback uses early text chunk.
- Chooses earliest admit/reject mention.

Fix C: Parse-quality instrumentation
- framing.py and anchoring.py now track:
  - total_calls
  - parse_failures
  - parse_success_rate
  - parse_quality_pass
- Per-row parse flags added for framing:
  - parse_ok_admit
  - parse_ok_reject
  - parse_ok_neutral
- Warning emitted when parse quality is below threshold (default 95%).

Fix D: DecisionAgent consistency
- DecisionAgent helper methods also switched to generate_decision.

### Stage 5: Re-test after fixes

A smoke command was run:
- python main.py --bias framing --model facebook/opt-125m --mitigation multi_selfhelp --n-students 3 --no-evaluator

Result:
- Run completed.
- Parse quality warning correctly triggered (very low parse success for this tiny model in this mitigation).
- CSV now explicitly shows parse_ok flags, so unreliable metrics are visible instead of hidden.

This is a good outcome for debugging:
- We now know whether odd metrics come from real behavior or from output/parse failure.

---

## Practical Meaning for Current and Next Runs

1. Use run_all_experiments.py for cheap structural validation only.
2. Use main.py with config-scale values (50 and 6) for actual experiment claims.
3. Treat results as valid only when parse_success_rate is high.
4. If parse_success_rate is low, either:
   - switch to a stronger instruction model, or
   - simplify prompt rewriting for that mitigation.

---

## Suggested Repro Commands

Small smoke test:
- python main.py --bias framing --model facebook/opt-125m --mitigation multi_selfhelp --n-students 3 --no-evaluator

Demo matrix run:
- python run_all_experiments.py

Larger target run:
- python main.py --bias both --mitigation all

---

## 3) Why We Switched Demo Runs to Mistral-7B

### What we tested before choosing a stronger model

Before moving demo runs to a stronger model, we performed these checks with smaller/cheaper settings:
1. End-to-end pipeline check:
   - Verified all method branches run and write outputs (baseline, single_selfhelp, multi_selfhelp, no_debias_agent).
2. Crash-resume check:
   - Confirmed run_all_experiments.py can restart from partial results.
3. CSV schema check:
   - Verified summary and per-student files include expected columns and no malformed headers.
4. Parse stability check:
   - Ran small framing-only smoke runs to inspect whether outputs consistently parse to admit/reject.
5. Cost logging check:
   - Confirmed tokens/calls/runtime are tracked and exported.

These tests showed the framework itself was functioning, but the tiny model often produced weak or noisy decision signals for mitigation comparisons.

### Why Mistral-7B-Instruct-v0.2 specifically

We selected mistralai/Mistral-7B-Instruct-v0.2 for demo runs because it is a practical middle ground:
1. Better instruction following than facebook/opt-125m:
   - More likely to produce concise decision-like outputs compatible with parsing.
2. Stronger reasoning behavior at similar operational simplicity:
   - Improves chance of non-flat metrics in bias comparisons.
3. Open and widely used benchmark model:
   - Easier to justify in a report than ad-hoc obscure checkpoints.
4. No gated-access friction in typical Hugging Face workflows:
   - Faster team iteration than models requiring approval.
5. Fits single-machine experiments better than very large models:
   - More realistic for repeated ablation runs.

### What differentiates Mistral from the earlier Facebook OPT demo model

In this project context, the important differences are practical behavior and scale:
1. Model capacity:
   - OPT-125m is extremely small and often underpowered for reliable controlled prompting.
   - Mistral-7B has far higher representational capacity for instruction tasks.
2. Instruction tuning quality:
   - Mistral Instruct variants are designed for chat/instruction responses.
   - Very small OPT models are more likely to echo prompts or output ambiguous text.
3. Parse reliability:
   - Stronger instruction models usually improve admit/reject extraction consistency.
4. Experimental sensitivity:
   - With stronger models, differences between mitigation methods are more likely to appear in metrics, instead of everything collapsing to ties.

### Stronger model options beyond Mistral-7B

If compute allows, stronger options for the same experiment design include:
1. mistralai/Mixtral-8x7B-Instruct-v0.1
   - Typically stronger reasoning/instruction behavior than 7B dense models.
   - Tradeoff: much higher memory/runtime cost.
2. meta-llama/Meta-Llama-3-70B-Instruct (or equivalent large instruction model)
   - Significantly stronger general instruction performance.
   - Tradeoff: expensive inference and often infrastructure-heavy.

Why they can help:
1. Better output discipline (fewer malformed answers).
2. Better consistency under framing/anchoring prompt perturbations.
3. Higher chance that mitigation deltas are measurable and stable.

---

## 4) What a Smoke Test Is (and why we use it)

A smoke test is a small, fast run used to verify that the pipeline works at all before spending large compute.

In this repository, a smoke test means:
1. Very small n_students and/or fewer permutations.
2. Single bias or single mitigation path.
3. Goal is correctness of execution, not publishable metrics.

What smoke tests validate here:
1. Code path executes without crashing.
2. CSV outputs are produced with valid schema.
3. Decision parsing is functioning at a basic level.
4. Cost logging and checkpointing still work.

Why it matters:
1. Catches integration bugs quickly.
2. Prevents wasting hours on broken overnight runs.
3. Separates "pipeline broken" from "model behavior not strong enough".
