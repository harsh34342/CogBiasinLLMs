"""
agents/base_agent.py
--------------------
Base LLM agent — wraps a HuggingFace text-generation pipeline.
All specialised agents (decision, debiasing, evaluator) inherit from this.
"""

import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Wraps a HuggingFace text-generation model into a callable agent.
    Lazy-loads the model on first use to avoid import-time GPU allocation.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        hf_token: Optional[str] = None,
        role_description: str = "You are a helpful assistant.",
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
        self.role_description = role_description

        self._pipeline = None   # lazy-loaded
        self._tokenizer = None

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _load(self):
        """Load the model and tokenizer on first call."""
        if self._pipeline is not None:
            return

        from transformers import pipeline, AutoTokenizer
        import torch

        logger.info(f"Loading model: {self.model_name}")

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        token_kwargs = {"token": self.hf_token} if self.hf_token else {}

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **token_kwargs,
        )

        # Ensure pad token exists (needed for batched inference)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device_map = self.device
        if self.device == "auto":
            device_map = "auto" if torch.cuda.is_available() else None

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self._tokenizer,
            torch_dtype=dtype,
            device_map=device_map,
            **token_kwargs,
        )

        logger.info(f"Model loaded: {self.model_name}")

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """
        Run a single prompt through the model and return the generated text
        (excluding the input prompt).
        """
        self._load()

        do_sample = self.temperature > 0.0

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
            return_full_text=False,   # return only new tokens
        )

        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        outputs = self._pipeline(prompt, **gen_kwargs)
        raw = outputs[0]["generated_text"].strip()
        return raw

    # ── Chat-style prompt formatting ──────────────────────────────────────────

    def format_chat_prompt(self, system: str, user: str) -> str:
        """
        Format a system+user message pair into a prompt string.
        Handles both chat-tuned models (llama/mistral) and base models (OPT).
        """
        model_lower = self.model_name.lower()

        # Llama-2 chat format
        if "llama-2" in model_lower and "chat" in model_lower:
            return (
                f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
            )

        # Mistral instruct format
        if "mistral" in model_lower or "mixtral" in model_lower:
            return (
                f"<s>[INST] {system}\n\n{user} [/INST]"
            )

        # Falcon instruct format
        if "falcon" in model_lower:
            return (
                f"System: {system}\nUser: {user}\nAssistant:"
            )

        # Generic / OPT / base models — simple concatenation
        return f"{system}\n\n{user}\n"

    # ── Utility ───────────────────────────────────────────────────────────────

    def extract_decision(self, text: str) -> Optional[str]:
        """
        Parse 'admit' or 'reject' from model output.
        Returns 'admit', 'reject', or None if unparseable.
        """
        t = text.lower().strip()

        # Direct keyword match
        if re.search(r'\badmit\b', t):
            return "admit"
        if re.search(r'\breject\b', t):
            return "reject"

        # Fallback: first word
        first = t.split()[0] if t.split() else ""
        if "admit" in first:
            return "admit"
        if "reject" in first:
            return "reject"

        return None

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"
