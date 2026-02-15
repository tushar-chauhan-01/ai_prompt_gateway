"""
OpenAI provider — handles GPT-4o and GPT-4o-mini via the OpenAI API.
"""

import os
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from backend.app.models import ModelName, ProviderName, ProviderResponse
from backend.app.providers.base import BaseProvider
from backend.app.router import MODEL_REGISTRY

_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_ENV_PATH)

# Map our ModelName enum to the actual OpenAI API model IDs
_MODEL_ID_MAP: dict[ModelName, str] = {
    ModelName.GPT_4O_MINI: "gpt-4o-mini",
    ModelName.GPT_4O: "gpt-4o",
}


class OpenAIProvider(BaseProvider):
    """Real OpenAI API provider for GPT-4o and GPT-4o-mini."""

    def __init__(self) -> None:
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
            self._client = OpenAI(api_key=api_key)
        except Exception:
            traceback.print_exc()
            raise

    def supports_model(self, model: ModelName) -> bool:
        """Return True if model is a GPT variant we support."""
        try:
            return model in _MODEL_ID_MAP
        except Exception:
            traceback.print_exc()
            raise

    def generate(self, prompt: str, model: ModelName) -> ProviderResponse:
        """
        Send prompt to OpenAI and return a ProviderResponse with real
        generated text, token usage, measured latency, and calculated cost.
        """
        try:
            if not self.supports_model(model):
                raise ValueError(f"OpenAIProvider does not support model: {model.value}")

            api_model_id = _MODEL_ID_MAP[model]
            model_info = MODEL_REGISTRY[model]

            start = time.perf_counter()

            response = self._client.chat.completions.create(
                model=api_model_id,
                max_tokens=1024,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            latency_ms = int((time.perf_counter() - start) * 1000)

            # Extract response data
            response_text = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens

            # Calculate cost
            cost = self._calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_per_1k_input=model_info.cost_per_1k_input_tokens,
                cost_per_1k_output=model_info.cost_per_1k_output_tokens,
            )

            return ProviderResponse(
                model=model,
                provider=ProviderName.OPENAI,
                response_text=response_text,
                tokens_used=total_tokens,
                latency_ms=latency_ms,
                simulated_cost=cost,
            )
        except Exception:
            traceback.print_exc()
            raise
