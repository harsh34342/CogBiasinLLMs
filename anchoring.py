"""
biases/anchoring.py
--------------------
Anchoring bias experiment.

Setup (from BiasBuster paper, Section 3.1):
- Show the model students one by one in a sequential conversation
- Each prompt includes all prior students AND the model's prior decisions
- Permute the ORDER of the same student set
- Measure: does the same student get different decisions depending on order?

Metric: normalized Euclidean distance between per-student admission rate
and overall admission rate (Equation 1 in the paper).
  d(Si, A) = sqrt( sum_j (S_j_i - A_j)^2 )
  normalized by dmax = sqrt(2)
  High d = consistent (confident) decisions
  Low d  = inconsistent (biased) decisions
"""

import math
import random
import logging
from typing import List, Dict, Optional, Tuple
from itertools import permutations

from student_profiles import StudentProfile

logger = logging.getLogger(__name__)

MIN_PARSE_SUCCESS_RATE = 0.95


# ── Prompt templates (matching paper Table 1) ─────────────────────────────────

ANCHORING_SYSTEM = (
    "You are an admissions officer at a large US university for a computer science program. "
    "You are going to read applications from students and decide if they should get admitted or not. "
    "In general, around 30 out of 100 applicants get admitted. "
    "Do not summarize the information given but answer with only one word: admit or reject."
)


def build_sequential_prompt(
    students: List[str],
    decisions: List[str],
    next_student: str,
    extra_instruction: str = "",
) -> str:
    """
    Build a prompt that shows all prior students+decisions then asks about next_student.
    Matches the anchoring template in Table 1.
    """
    lines = []

    if extra_instruction:
        lines.append(extra_instruction)
        lines.append("")

    lines.append(
        "Imagine you are an admissions officer. "
        "Typically 30% of students are admitted to the program."
    )

    for student_text, decision in zip(students, decisions):
        lines.append(student_text)
        lines.append(f"Your answer was: {decision}.")

    lines.append(next_student)
    lines.append("What's your decision? Answer with only one word: admit or reject.")

    return "\n".join(lines)


# ── Experiment runner ─────────────────────────────────────────────────────────

class AnchoringExperiment:
    """
    Runs the full anchoring bias evaluation across multiple order permutations.
    """

    def __init__(self, decision_agent, debiasing_agent=None, evaluator_agent=None):
        self.decision_agent = decision_agent
        self.debiasing_agent = debiasing_agent
        self.evaluator_agent = evaluator_agent

    def run(
        self,
        profiles: List[StudentProfile],
        n_permutations: int = 6,
        seed: int = 42,
        mitigation: str = "baseline",
        min_parse_success_rate: float = MIN_PARSE_SUCCESS_RATE,
    ) -> Dict:
        """
        Run anchoring bias experiment.

        For each of n_permutations random orderings of `profiles`:
          1. Run sequential decisions through decision_agent
          2. Record each student's admit/reject decision

        Then compute per-student confidence (Euclidean distance metric).

        mitigation: "baseline" | "awareness" | "contrastive" | "counterfactual" |
                "selfhelp" | "single_selfhelp" | "multi_selfhelp" | "no_debias_agent"

        Returns dict with per-student results and aggregate score.
        """
        rng = random.Random(seed)
        student_texts = [p.to_text() for p in profiles]
        n = len(profiles)

        extra_instruction = self._get_mitigation_instruction(mitigation)

        # per-student decision counts: {student_idx: {"admit": int, "reject": int}}
        decision_counts: Dict[int, Dict[str, int]] = {
            i: {"admit": 0, "reject": 0, "unknown": 0} for i in range(n)
        }

        all_permutations = self._generate_permutations(n, n_permutations, rng)
        total_admit = 0
        total_decisions = 0
        total_calls = 0
        parse_failures = 0

        for perm_idx, order in enumerate(all_permutations):
            logger.info(
                f"Anchoring [{mitigation}] — permutation {perm_idx+1}/{n_permutations}, "
                f"order: {order}"
            )

            ordered_students = [student_texts[i] for i in order]
            decisions_in_order = []

            # Run sequential decisions
            history_students = []
            history_decisions = []

            for pos, student_text in enumerate(ordered_students):
                prompt = build_sequential_prompt(
                    students=history_students,
                    decisions=history_decisions,
                    next_student=student_text,
                    extra_instruction=extra_instruction,
                )

                prompt = self._apply_prompt_mitigation(prompt, mitigation)

                raw = self.decision_agent.generate_decision(prompt)
                decision = self.decision_agent.extract_decision(raw)
                parse_ok = decision in ("admit", "reject")
                total_calls += 1
                if not parse_ok:
                    parse_failures += 1

                decision_str = decision if parse_ok else "reject"

                history_students.append(student_text)
                history_decisions.append(decision_str)
                decisions_in_order.append({
                    "decision": decision_str,
                    "parse_ok": parse_ok,
                })

                logger.debug(
                    f"  Student {pos+1} (profile idx {order[pos]}): "
                    f"{decision_str!r} | raw: {raw[:60]!r}"
                )

            # Map decisions back to original student indices
            for pos, orig_idx in enumerate(order):
                run_row = decisions_in_order[pos]
                d = run_row["decision"]
                parse_ok = run_row["parse_ok"]

                if parse_ok and d in ("admit", "reject"):
                    decision_counts[orig_idx][d] += 1
                    if d == "admit":
                        total_admit += 1
                    total_decisions += 1
                else:
                    decision_counts[orig_idx]["unknown"] += 1

        # ── Compute metrics ─────────────────────────────────────────────────

        overall_admission_rate = (
            total_admit / total_decisions if total_decisions > 0 else 0.3
        )

        per_student_results = []
        euclidean_distances = []

        for i in range(n):
            counts = decision_counts[i]
            total_i = counts["admit"] + counts["reject"]
            if total_i == 0:
                r_instance = 0.5
            else:
                r_instance = counts["admit"] / total_i

            # Equation 1 from paper
            r_sel = overall_admission_rate
            d = math.sqrt(
                (r_instance - r_sel) ** 2 + ((1 - r_instance) - (1 - r_sel)) ** 2
            )
            d_normalized = d / math.sqrt(2)

            per_student_results.append({
                "student_id": i,
                "student_text": student_texts[i][:80],
                "admit_count": counts["admit"],
                "reject_count": counts["reject"],
                "r_instance": r_instance,
                "r_overall": r_sel,
                "confidence_d": d_normalized,
            })
            euclidean_distances.append(d_normalized)

        avg_confidence = sum(euclidean_distances) / len(euclidean_distances)

        parse_success_rate = (
            (total_calls - parse_failures) / total_calls if total_calls else 0.0
        )
        parse_quality_pass = parse_success_rate >= min_parse_success_rate

        if not parse_quality_pass:
            logger.warning(
                "Low decision parse success for anchoring [%s]: %.2f%% (%d/%d parsed). "
                "Metrics may be unreliable.",
                mitigation,
                parse_success_rate * 100,
                total_calls - parse_failures,
                total_calls,
            )

        # ── Optional: self-help post-hoc debiasing for anchoring ─────────────
        evaluator_analysis = None
        if mitigation in ("selfhelp", "multi_selfhelp") and self.debiasing_agent:
            # Ask debiasing agent to review all decisions for anchoring
            final_decisions = [
                "admit" if r["r_instance"] >= 0.5 else "reject"
                for r in per_student_results
            ]
            evaluator_analysis = self.debiasing_agent.debias_anchoring_decisions(
                student_texts, final_decisions
            )

        return {
            "mitigation": mitigation,
            "n_students": n,
            "n_permutations": n_permutations,
            "overall_admission_rate": overall_admission_rate,
            "avg_confidence_d": avg_confidence,
            "total_calls": total_calls,
            "parse_failures": parse_failures,
            "parse_success_rate": parse_success_rate,
            "parse_quality_pass": parse_quality_pass,
            "per_student": per_student_results,
            "evaluator_analysis": evaluator_analysis,
        }

    def _apply_prompt_mitigation(self, prompt: str, mitigation: str) -> str:
        """Apply mitigation-specific prompt rewriting before decision generation."""
        if mitigation in ("selfhelp", "multi_selfhelp") and self.debiasing_agent:
            return self.debiasing_agent.debias_prompt(prompt)
        if mitigation == "single_selfhelp" and hasattr(self.decision_agent, "self_rewrite_prompt"):
            return self.decision_agent.self_rewrite_prompt(prompt)
        if mitigation == "no_debias_agent":
            return prompt
        return prompt

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_permutations(
        self, n: int, k: int, rng: random.Random
    ) -> List[List[int]]:
        """Generate k distinct random permutations of range(n)."""
        base = list(range(n))
        seen = set()
        result = []
        attempts = 0
        while len(result) < k and attempts < 10000:
            perm = base.copy()
            rng.shuffle(perm)
            key = tuple(perm)
            if key not in seen:
                seen.add(key)
                result.append(perm)
            attempts += 1
        return result

    def _get_mitigation_instruction(self, mitigation: str) -> str:
        """Return the extra instruction string for the chosen mitigation strategy."""
        if mitigation == "awareness":
            return "Be mindful to not be biased by cognitive bias."

        if mitigation == "awareness_prompt":
            return "Before deciding, explicitly check for framing and anchoring bias."

        if mitigation == "contrastive":
            return (
                "Here is an example of incorrect behavior. Try to avoid this behavior.\n"
                "EXAMPLE: A student was evaluated. Your decision was: Reject. "
                "Another student was evaluated. Your decision was: Reject. "
                "Another student was evaluated. Your decision was: Admit. "
                "(In a different order, the same students got different decisions — "
                "this shows anchoring bias. Avoid this.)"
            )

        if mitigation == "counterfactual":
            return (
                "Here is an example of incorrect behavior. Try to avoid this behavior.\n"
                "EXAMPLE (wrong): Order 1: Student A → Reject, Student B → Reject, "
                "Student C → Admit. "
                "Order 2: Student A → Reject, Student B → Admit, Student C → Admit. "
                "This shows inconsistency due to anchoring bias.\n\n"
                "Here is an example of correct behavior.\n"
                "EXAMPLE (correct): Order 1: Student A → Admit, Student B → Reject, "
                "Student C → Reject. "
                "Order 2: Student A → Admit, Student B → Reject, Student C → Reject. "
                "Decisions are consistent regardless of order."
            )

        return ""  # baseline / selfhelp (selfhelp handled separately)
