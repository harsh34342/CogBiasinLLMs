"""
agents/debiasing_agent.py
--------------------------
The "self-help" debiasing agent.
Given a biased prompt, it rewrites it to remove cognitive bias triggers,
then passes the clean prompt back to the DecisionAgent.

This is the novel multi-agent contribution: instead of one model doing
everything, we have Agent 1 (debiaser) clean the prompt, then
Agent 2 (decision) evaluate the cleaned prompt.

Mirrors the Self-Help method from BiasBuster Section 4.3.
"""

import re
from agents.base_agent import BaseAgent


DEBIASING_SYSTEM_PROMPT = (
    "You are an expert at identifying and removing cognitive bias from text. "
    "Your task is to rewrite prompts so that a reviewer would not be influenced "
    "by cognitive bias. Keep all factual information intact — only remove or "
    "neutralize language that could introduce bias."
)

DEBIASING_INSTRUCTION = (
    "Here is a prompt that may be biased by cognitive bias. "
    "Rewrite it such that a reviewer is not biased. "
    "Remove any anchoring cues, status quo defaults, gendered language, "
    "or framing that could unfairly influence a decision. "
    "Keep all objective student information (GPA, scores, degree, etc.) intact.\n\n"
    "[start of prompt]\n"
    "{prompt}\n"
    "[end of prompt]\n\n"
    "Start your answer with [start of revised prompt]"
)


class DebiasingAgent(BaseAgent):
    """
    Agent responsible for rewriting biased prompts.
    Used in the Self-Help mitigation pipeline.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(
            model_name=model_name,
            role_description=DEBIASING_SYSTEM_PROMPT,
            max_new_tokens=256,  # rewrites need more tokens
            **kwargs,
        )

    def debias_prompt(self, biased_prompt: str) -> str:
        """
        Rewrite a biased prompt to remove cognitive bias triggers.
        Returns the cleaned prompt text.
        """
        instruction = DEBIASING_INSTRUCTION.format(prompt=biased_prompt)

        full_prompt = self.format_chat_prompt(
            system=DEBIASING_SYSTEM_PROMPT,
            user=instruction,
        )

        raw_output = self.generate(full_prompt)

        # Extract only the revised prompt between markers
        debiased = self._extract_revised_prompt(raw_output)
        return debiased if debiased else biased_prompt  # fallback to original

    def debias_anchoring_decisions(
        self,
        student_profiles: list,
        prior_decisions: list,
    ) -> str:
        """
        For anchoring bias: given all student profiles and the model's
        prior decisions, ask the model to reflect on whether its decisions
        show anchoring bias and suggest corrections.

        Mirrors the anchoring self-help variant described in Section 4.3.
        """
        # Build a summary of all students and decisions
        summary_lines = [
            "Below is a list of all students reviewed and the decisions made. "
            "Review these decisions for signs of anchoring bias — where your "
            "earlier decisions may have unfairly influenced later ones. "
            "If you detect inconsistency due to anchoring, suggest which decisions "
            "should change and why.\n"
        ]

        for i, (student, decision) in enumerate(zip(student_profiles, prior_decisions)):
            d_str = decision if decision else "unclear"
            summary_lines.append(f"Student {i+1}: {student}\nDecision: {d_str}\n")

        summary_lines.append(
            "\nAre any of these decisions inconsistent due to anchoring bias? "
            "List the student numbers whose decisions you would change and what "
            "the corrected decision should be (admit/reject). "
            "If no changes needed, say 'No changes needed'."
        )

        user_content = "\n".join(summary_lines)
        prompt = self.format_chat_prompt(
            system=DEBIASING_SYSTEM_PROMPT,
            user=user_content,
        )

        return self.generate(prompt)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_revised_prompt(self, raw: str) -> str:
        """
        Extract text between [start of revised prompt] and [end of revised prompt],
        or return everything after the marker if the end marker is missing.
        """
        start_marker = "[start of revised prompt]"
        end_marker = "[end of revised prompt]"

        raw_lower = raw.lower()
        start_idx = raw_lower.find(start_marker.lower())

        if start_idx == -1:
            # Marker not found — return raw output as-is
            return raw.strip()

        start_idx += len(start_marker)
        end_idx = raw_lower.find(end_marker.lower(), start_idx)

        if end_idx == -1:
            return raw[start_idx:].strip()
        return raw[start_idx:end_idx].strip()
