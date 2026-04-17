# One-Page Experiment Summary

## Project Goal

This project evaluates cognitive bias in LLM admissions decisions using a multi-agent pipeline. We focus on two questions:

1. Anchoring question:
Does decision order change outcomes for the same student?

2. Framing question:
Does wording alone ("admit" vs "reject" phrasing) change outcomes for the same student profile?

---

## How We Answer the Two Questions

### Anchoring (Order Sensitivity)

Process:
1. Generate synthetic student profiles.
2. Run sequential evaluation where each new prompt includes prior students plus prior decisions.
3. Repeat across multiple random order permutations.
4. Compute per-student stability and overall anchoring metric.

Key metric:
- Normalized confidence distance based on per-student admit rate versus overall admit rate.

Main files:
- anchoring.py
- student_profiles.py
- decision_agent.py
- debiasing_agent.py
- evaluate.py

### Framing (Wording Sensitivity)

Process:
1. Keep each student profile fixed.
2. Ask three prompt variants:
   - admit frame
   - reject frame
   - neutral frame
3. Compare outcomes for the same student across frames.
4. Compute framing gap and inconsistency counts.

Key metrics:
- delta = admit_rate(admit_frame) - admit_rate(reject_frame)
- n_inconsistent = students whose admit/reject frame decisions disagree.

Main files:
- framing.py
- student_profiles.py
- decision_agent.py
- debiasing_agent.py
- evaluate.py

---

## How the System Fits Together

- main.py:
General runner for full experiments and larger settings.

- run_all_experiments.py:
Small, crash-safe matrix runner for quick validation.

- base_agent.py:
Shared HuggingFace generation, prompt formatting, and decision parsing.

- decision_agent.py:
Produces admissions decisions.

- debiasing_agent.py:
Rewrites prompts for self-help mitigation variants.

- evaluator_agent.py:
Optional qualitative auditing layer.

- evaluate.py:
Summaries and CSV exports.

- results/:
Stores per-student and summary outputs.

---

## Parameter Strategy and Iteration History

### Full target settings (for substantive runs)

Defined in config.py:
- anchoring students: 50
- anchoring permutations: 6
- framing students: 50
- seed: 42
- temperature: 0.0

Why:
- Better statistical stability than tiny tests.
- Still computationally feasible.

### Small validation settings (for debug/smoke tests)

Used in run_all_experiments.py:
- model: facebook/opt-125m
- students: 6
- permutations: 2

Why:
- Fast end-to-end path testing.
- Cheap way to catch pipeline and formatting problems before larger runs.

---

## Problems We Hit

1. Many outputs were not clean admit/reject strings.
2. Small-model generations often echoed prompt text or produced malformed text.
3. Multi-selfhelp rewrite outputs sometimes contained repeated prompt markers.
4. Decision fields were often empty in CSVs.
5. Summary metrics looked flat (for example zeros), but some of that was parse failure, not true model behavior.

---

## Fixes We Implemented

1. Short decision generation path:
- Added a dedicated short-token decision call to reduce rambling and prompt echo.

2. Stronger parsing:
- Parse first non-empty line first, then fallback to early text chunk.
- Pick earliest admit/reject token match.

3. Parse-quality instrumentation:
- Track total_calls, parse_failures, parse_success_rate, parse_quality_pass.
- Add per-row parse flags for framing outputs.
- Warn when parse rate is below threshold.

4. Consistency update:
- DecisionAgent helper paths also switched to short decision generation.

---

## Current Interpretation Rule

Experiment metrics are trusted only when parse_success_rate is high.

If parse_success_rate is low:
1. Treat metrics as unreliable for bias claims.
2. Use stronger instruction-tuned models.
3. Or simplify rewrite prompts/mitigation path.

---

## Practical Command Flow

1. Smoke test:
python main.py --bias framing --model facebook/opt-125m --mitigation multi_selfhelp --n-students 3 --no-evaluator

2. Demo matrix:
python run_all_experiments.py

3. Full run (target scale):
python main.py --bias both --mitigation all
