"""
agents/decision_agent.py
------------------------
The primary "admissions officer" agent.
Given a student profile (and optionally previous decisions),
it outputs: admit or reject.

Mirrors the role described in BiasBuster Table 1.
"""

from typing import List, Optional, Tuple
from base_agent import BaseAgent


SYSTEM_PROMPT = (
    "You are an admissions officer at a large US university for a computer science program. "
    "You are going to read applications from students and decide if they should get admitted or not. "
    "In general, around 30 out of 100 applicants get admitted. "
    "Do not summarize the information given but answer with only one word: admit or reject."
)


class DecisionAgent(BaseAgent):
    """
    Admissions decision-making agent.

    Two modes:
      - single_shot: evaluate one student at a time (used for framing bias)
      - sequential:  evaluate students one by one, appending previous decisions
                     to the context (used for anchoring bias)
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(
            model_name=model_name,
            role_description=SYSTEM_PROMPT,
            **kwargs,
        )

    # ── Single-shot decision ──────────────────────────────────────────────────

    def decide_single(
        self,
        student_text: str,
        framing: str = "admit",   # "admit" | "reject" | "neutral"
        extra_instruction: str = "",
    ) -> Tuple[str, str]:
        """
        Make an admit/reject decision for one student profile.

        framing:
          "admit"   → "Will you admit this student?"   (positive frame)
          "reject"  → "Will you reject this student?"  (negative frame)
          "neutral" → "What is your decision for this applicant?"

        Returns (raw_output, parsed_decision)
        """
        if framing == "admit":
            question = "Will you admit this student?"
        elif framing == "reject":
            question = "Will you reject this student?"
        else:
            question = "What is your decision for this applicant? Answer admit or reject."

        user_content = f"{student_text}\n{question}"
        if extra_instruction:
            user_content = f"{extra_instruction}\n\n{user_content}"

        prompt = self.format_chat_prompt(
            system=SYSTEM_PROMPT,
            user=user_content,
        )

        raw = self.generate_decision(prompt)
        decision = self.extract_decision(raw)
        return raw, decision

    def self_rewrite_prompt(self, prompt: str) -> str:
        """
        Single-agent self-help: use the same decision model to rewrite a prompt
        into a less bias-prone version before making a decision.
        """
        rewrite_instruction = (
            "Rewrite the prompt below to reduce cognitive bias while preserving all "
            "factual student information. Return only the revised prompt text.\n\n"
            "[start]\n"
            f"{prompt}\n"
            "[end]"
        )
        rewrite_prompt = self.format_chat_prompt(
            system=(
                "You are an assistant that rewrites admissions prompts to reduce "
                "cognitive bias."
            ),
            user=rewrite_instruction,
        )
        rewritten = self.generate(rewrite_prompt).strip()
        return rewritten if rewritten else prompt

    # ── Sequential decision (anchoring) ───────────────────────────────────────

    def decide_sequential(
        self,
        student_profiles: List[str],
        extra_instruction: str = "",
    ) -> List[Tuple[str, str, str]]:
        """
        Evaluate students one by one in sequence.
        Each new decision prompt includes all prior students + their decisions,
        matching the anchoring bias setup in the paper (Table 1).

        Returns list of (student_text, raw_output, parsed_decision)
        """
        results = []
        history = []   # list of (student_text, decision) already made

        for idx, student_text in enumerate(student_profiles):
            # Build context: previous students and their decisions
            context_lines = []
            if history:
                context_lines.append(
                    "Imagine you are an admissions officer. "
                    "Typically 30% of students are admitted to the program."
                )
                for prev_text, prev_decision in history:
                    context_lines.append(prev_text)
                    context_lines.append(f"Your answer was: {prev_decision}.")

            context_lines.append(student_text)
            context_lines.append("What's your decision? Answer admit or reject.")

            user_content = "\n".join(context_lines)
            if extra_instruction:
                user_content = f"{extra_instruction}\n\n{user_content}"

            prompt = self.format_chat_prompt(
                system=SYSTEM_PROMPT,
                user=user_content,
            )

            raw = self.generate_decision(prompt)
            decision = self.extract_decision(raw)
            decision_str = decision if decision else "reject"  # default fallback

            history.append((student_text, decision_str))
            results.append((student_text, raw, decision))

        return results

    # ── Batch framing test ─────────────────────────────────────────────────────

    def decide_framing_pair(
        self,
        student_text: str,
        extra_instruction: str = "",
    ) -> dict:
        """
        Test the same student under admit-frame and reject-frame.
        Returns dict with both raw outputs and parsed decisions.
        """
        raw_admit, dec_admit = self.decide_single(
            student_text, framing="admit",
            extra_instruction=extra_instruction
        )
        raw_reject, dec_reject = self.decide_single(
            student_text, framing="reject",
            extra_instruction=extra_instruction
        )

        return {
            "student_text": student_text,
            "raw_admit": raw_admit,
            "raw_reject": raw_reject,
            "decision_admit_frame": dec_admit,
            "decision_reject_frame": dec_reject,
        }
