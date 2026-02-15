"""
Provider manager — resolves the correct provider for a given model
and dispatches the generation request.
"""

import traceback

from backend.app.models import ModelName, ProviderResponse
from backend.app.providers.base import BaseProvider
from backend.app.providers.openai_provider import OpenAIProvider
from backend.app.providers.anthropic_provider import AnthropicProvider


class ProviderManager:
    """
    Lazily initialises providers and routes generation requests
    to the correct one based on the model name.
    """

    def __init__(self) -> None:
        try:
            self._providers: list[BaseProvider] = [
                OpenAIProvider(),
                AnthropicProvider(),
            ]
        except Exception:
            traceback.print_exc()
            raise

    def generate(self, prompt: str, model: ModelName) -> ProviderResponse:
        """
        Find the provider that supports the given model and generate a response.
        """
        try:
            for provider in self._providers:
                if provider.supports_model(model):
                    return provider.generate(prompt, model)

            raise ValueError(
                f"No provider found for model: {model.value}. "
                f"Available providers: {[type(p).__name__ for p in self._providers]}"
            )
        except Exception:
            traceback.print_exc()
            raise
