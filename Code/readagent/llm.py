"""
LLM client wrapper for Mistral API.
Handles all LLM inference calls used throughout the pipeline.
"""

import time
import logging
from typing import Optional

from mistralai.client import Mistral

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around the Mistral API for consistent LLM calls."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = Mistral(
            api_key=config.api_key,
            timeout_ms=120000,  # 120 seconds timeout
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system-level instruction.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            The generated text response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        retries = 3
        for attempt in range(retries):
            try:
                response = self.client.chat.complete(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                )

                # Track token usage
                if response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                self.call_count += 1

                result = response.choices[0].message.content
                logger.debug(
                    f"LLM call #{self.call_count}: "
                    f"input={response.usage.prompt_tokens if response.usage else '?'}, "
                    f"output={response.usage.completion_tokens if response.usage else '?'}"
                )
                return result

            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"LLM call failed after {retries} attempts: {e}")

    def get_usage_stats(self) -> dict:
        """Return token usage statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "call_count": self.call_count,
        }

    def reset_stats(self):
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
