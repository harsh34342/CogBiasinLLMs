"""
agents/evaluator_agent.py
--------------------------
The "meta-evaluator" agent.
Given a set of decisions, it assesses whether patterns of cognitive bias
are present — acting as a second-pass auditor in the multi-agent pipeline.

This is a distinguishing feature of the multi-agent setup: the evaluator
is a SEPARATE model call that critiques the decision agent's outputs,
providing an additional layer of bias detection beyond just metrics.
"""

from typing import List, Optional
from base_agent import BaseAgent


EVALUATOR_SYSTEM_PROMPT = (
    "You are an expert auditor assessing decisions for cognitive bias. "
    "You will be given a series of admissions decisions and will identify "
    "whether the decisions show signs of cognitive bias such as anchoring "
    "(decisions influenced by prior decisions) or framing "
    "(different outcomes for the same student based on how the question was worded). "
    "Be precise, concise, and objective."
)


class EvaluatorAgent(BaseAgent):
    """
    Meta-evaluator agent that audits decision patterns for cognitive bias.
    Provides qualitative reasoning to complement the quantitative metrics.
    """

    def __init__(self, model_name: str, **kwargs):
        max_new_tokens = kwargs.pop("max_new_tokens", 200)
        super().__init__(
            model_name=model_name,
            role_description=EVALUATOR_SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def evaluate_anchoring(
        self,
        student_profiles: List[str],
        decisions_order_1: List[Optional[str]],
        decisions_order_2: List[Optional[str]],
    ) -> str:
        """
        Given the same students evaluated in two different orders,
        ask the evaluator whether anchoring bias is present.
        """
        lines = [
            "The following students were evaluated in two different orders. "
            "Check if the same student received different decisions depending on order, "
            "which would indicate anchoring bias.\n"
        ]

        for i, student in enumerate(student_profiles):
            d1 = decisions_order_1[i] or "unknown"
            d2 = decisions_order_2[i] or "unknown"
            lines.append(
                f"Student {i+1}: {student[:100]}...\n"
                f"  Decision in Order 1: {d1}\n"
                f"  Decision in Order 2: {d2}\n"
            )

        lines.append(
            "\nDo these results indicate anchoring bias? "
            "Which students received inconsistent decisions? "
            "Give a brief explanation."
        )

        prompt = self.format_chat_prompt(
            system=EVALUATOR_SYSTEM_PROMPT,
            user="\n".join(lines),
        )
        return self.generate(prompt)

    def evaluate_framing(
        self,
        student_texts: List[str],
        admit_frame_decisions: List[Optional[str]],
        reject_frame_decisions: List[Optional[str]],
    ) -> str:
        """
        Given the same students asked under 'admit' vs 'reject' framing,
        evaluate whether framing bias is present.
        """
        inconsistent = []
        for i, (d_admit, d_reject) in enumerate(
            zip(admit_frame_decisions, reject_frame_decisions)
        ):
            if d_admit and d_reject and d_admit != d_reject:
                inconsistent.append(i + 1)

        lines = [
            f"The following {len(student_texts)} students were each evaluated twice: "
            f"once with a positive framing ('Will you admit?') and once with a negative "
            f"framing ('Will you reject?'). A fair model should give the same outcome "
            f"regardless of framing.\n",
            f"Students with inconsistent decisions: {inconsistent or 'None'}\n",
        ]

        # Show a few examples
        for i in inconsistent[:3]:
            idx = i - 1
            lines.append(
                f"\nStudent {i}: {student_texts[idx][:100]}...\n"
                f"  Admit-frame decision: {admit_frame_decisions[idx]}\n"
                f"  Reject-frame decision: {reject_frame_decisions[idx]}"
            )

        lines.append(
            "\n\nDoes this indicate framing bias? Provide a brief analysis."
        )

        prompt = self.format_chat_prompt(
            system=EVALUATOR_SYSTEM_PROMPT,
            user="\n".join(lines),
        )
        return self.generate(prompt)

    def summarize_mitigation_effectiveness(
        self,
        bias_type: str,
        baseline_score: float,
        mitigation_scores: dict,
    ) -> str:
        """
        Ask the evaluator to summarise which mitigation worked best.
        """
        lines = [
            f"For {bias_type} bias, the following bias scores were measured "
            f"(lower = less bias):\n",
            f"  Baseline (no mitigation): {baseline_score:.4f}\n",
        ]
        for strategy, score in mitigation_scores.items():
            lines.append(f"  {strategy}: {score:.4f}")

        lines.append(
            "\n\nWhich mitigation strategy was most effective and why? "
            "Give a one-paragraph analysis."
        )

        prompt = self.format_chat_prompt(
            system=EVALUATOR_SYSTEM_PROMPT,
            user="\n".join(lines),
        )
        return self.generate(prompt)
