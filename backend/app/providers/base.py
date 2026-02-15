"""
Base provider interface for LLM providers.
"""

import traceback
from abc import ABC, abstractmethod

from backend.app.models import ModelName, ProviderResponse


class BaseProvider(ABC):
    """Abstract base class that every LLM provider must implement."""

    @abstractmethod
    def generate(self, prompt: str, model: ModelName) -> ProviderResponse:
        """
        Send a prompt to the specified model and return the response.

        Args:
            prompt: The user prompt to send.
            model:  Which model to use for generation.

        Returns:
            ProviderResponse with the generated text, token usage,
            latency, and cost.
        """
        ...

    @abstractmethod
    def supports_model(self, model: ModelName) -> bool:
        """Return True if this provider can serve the given model."""
        ...

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_per_1k_input: float,
        cost_per_1k_output: float,
    ) -> float:
        """Compute the dollar cost for a request given token counts and rates."""
        try:
            input_cost = (input_tokens / 1000) * cost_per_1k_input
            output_cost = (output_tokens / 1000) * cost_per_1k_output
            return round(input_cost + output_cost, 6)
        except Exception:
            traceback.print_exc()
            raise
