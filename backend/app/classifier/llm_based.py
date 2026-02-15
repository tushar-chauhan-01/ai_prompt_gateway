"""
LLM-based prompt classifier.

Sends the user prompt to a real LLM (Claude Haiku or GPT-4o-mini) and asks
it to return a structured JSON classification with complexity_score,
task_type, reasoning, and confidence.

Supports two providers controlled by the CLASSIFIER_LLM_PROVIDER env var:
  - "anthropic"  → Claude 3.5 Haiku  (default)
  - "openai"     → GPT-4o-mini
"""

import json
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv

from backend.app.models import ClassificationResult, ClassifierMode, TaskType

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_ENV_PATH)

_CLASSIFIER_PROVIDER = os.getenv("CLASSIFIER_LLM_PROVIDER", "anthropic").lower()

# ---------------------------------------------------------------------------
# Valid task types (for the prompt sent to the LLM)
# ---------------------------------------------------------------------------

_VALID_TASK_TYPES = [t.value for t in TaskType]

# ---------------------------------------------------------------------------
# System prompt shared by both providers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""You are a prompt complexity classifier for an AI gateway.

Given a user prompt, analyze it and return a JSON object with exactly these fields:

{{
  "complexity_score": <integer 1-10>,
  "task_type": "<one of: {', '.join(_VALID_TASK_TYPES)}>",
  "reasoning": "<1-2 sentence explanation of why you assigned this score and type>",
  "confidence": <float 0.0-1.0>
}}

Scoring guidelines:
- 1-3: Simple factual questions, definitions, basic translations, yes/no questions
- 4-6: Moderate tasks like standard code generation, creative writing, straightforward analysis
- 7-10: Complex multi-step reasoning, advanced math, system design, nuanced long-form analysis

Task type definitions:
- simple_qa: Factual questions, definitions, lookups
- translation: Language translation requests
- code: Programming, debugging, code generation
- analysis: Comparing, evaluating, critiquing, reviewing
- creative: Poetry, stories, essays, artistic writing
- math: Calculations, proofs, equations, statistics
- reasoning: Logic, philosophy, thought experiments, complex explanations
- general: Anything that doesn't fit the above categories

IMPORTANT: Return ONLY the raw JSON object, no markdown fences, no extra text."""


# ---------------------------------------------------------------------------
# Provider: Anthropic (Claude)
# ---------------------------------------------------------------------------

def _classify_with_anthropic(prompt: str) -> dict:
    """Call Claude Haiku to classify the prompt. Returns parsed JSON dict."""
    try:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file."
            )

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Classify this prompt:\n\n{prompt}"}
            ],
        )

        raw_text = message.content[0].text.strip()
        return _parse_llm_response(raw_text)
    except Exception:
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Provider: OpenAI (GPT-4o-mini)
# ---------------------------------------------------------------------------

def _classify_with_openai(prompt: str) -> dict:
    """Call GPT-4o-mini to classify the prompt. Returns parsed JSON dict."""
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            temperature=0.3,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this prompt:\n\n{prompt}"},
            ],
        )

        raw_text = response.choices[0].message.content.strip()
        return _parse_llm_response(raw_text)
    except Exception:
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Response parsing & validation
# ---------------------------------------------------------------------------

def _parse_llm_response(raw_text: str) -> dict:
    """
    Parse the raw LLM text into a validated dict.
    Handles markdown fences, extra whitespace, and invalid values gracefully.
    """
    try:
        # Strip markdown code fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        data = json.loads(cleaned)

        # Validate and clamp complexity_score
        score = int(data.get("complexity_score", 5))
        score = max(1, min(10, score))

        # Validate task_type
        raw_type = str(data.get("task_type", "general")).lower().strip()
        if raw_type not in _VALID_TASK_TYPES:
            raw_type = "general"

        # Validate confidence
        confidence = float(data.get("confidence", 0.8))
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        # Validate reasoning
        reasoning = str(data.get("reasoning", "LLM classification"))
        if not reasoning.strip():
            reasoning = "LLM classification (no reasoning provided)"

        return {
            "complexity_score": score,
            "task_type": raw_type,
            "reasoning": reasoning,
            "confidence": confidence,
        }
    except json.JSONDecodeError as e:
        traceback.print_exc()
        raise ValueError(
            f"LLM returned invalid JSON. Raw response: {raw_text!r}"
        ) from e
    except Exception:
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(prompt: str) -> ClassificationResult:
    """
    Classify a prompt by sending it to a real LLM (Claude Haiku or GPT-4o-mini).

    The provider is selected via the CLASSIFIER_LLM_PROVIDER env var.
    Returns a ClassificationResult with the LLM's assessment.
    """
    try:
        if _CLASSIFIER_PROVIDER == "openai":
            data = _classify_with_openai(prompt)
        else:
            data = _classify_with_anthropic(prompt)

        return ClassificationResult(
            complexity_score=data["complexity_score"],
            task_type=TaskType(data["task_type"]),
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            classifier_mode=ClassifierMode.LLM_BASED,
        )
    except Exception:
        traceback.print_exc()
        raise
