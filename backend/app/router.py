"""
Core routing engine for the AI Gateway.

Takes a ClassificationResult (complexity score + task type) and selects the
optimal model, building a step-by-step reasoning chain that the UI can display.
"""

import traceback

from backend.app.models import (
    ClassificationResult,
    ModelInfo,
    ModelName,
    ProviderName,
    ReasoningStep,
    RoutingDecision,
    TaskType,
)


# ---------------------------------------------------------------------------
# Model registry — static metadata for every available model
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[ModelName, ModelInfo] = {
    ModelName.GPT_4O_MINI: ModelInfo(
        name=ModelName.GPT_4O_MINI,
        provider=ProviderName.OPENAI,
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=300,
        strengths=["fast", "cheap", "simple tasks", "translations"],
        max_context_tokens=128_000,
    ),
    ModelName.GPT_4O: ModelInfo(
        name=ModelName.GPT_4O,
        provider=ProviderName.OPENAI,
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800,
        strengths=["top-tier reasoning", "complex math", "advanced code"],
        max_context_tokens=128_000,
    ),
    ModelName.CLAUDE_35_SONNET: ModelInfo(
        name=ModelName.CLAUDE_35_SONNET,
        provider=ProviderName.ANTHROPIC,
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=700,
        strengths=["nuanced analysis", "long-form writing", "creative writing", "multilingual"],
        max_context_tokens=200_000,
    ),
}


# ---------------------------------------------------------------------------
# Routing table — maps (complexity tier, task type) → model
# ---------------------------------------------------------------------------

# Complexity tiers
_LOW = "low"        # 1-3
_MEDIUM = "medium"  # 4-6
_HIGH = "high"      # 7-10


def _get_tier(score: int) -> str:
    """Map a 1-10 complexity score to a tier label."""
    try:
        if score <= 3:
            return _LOW
        elif score <= 6:
            return _MEDIUM
        else:
            return _HIGH
    except Exception:
        traceback.print_exc()
        raise


# (tier, task_type) → ModelName
# Explicit mappings; anything not listed falls through to tier defaults.
_ROUTING_TABLE: dict[tuple[str, TaskType], ModelName] = {
    # --- LOW complexity (1-3) → GPT-4o-mini for everything ---
    (_LOW, TaskType.SIMPLE_QA):    ModelName.GPT_4O_MINI,
    (_LOW, TaskType.TRANSLATION):  ModelName.GPT_4O_MINI,
    (_LOW, TaskType.GENERAL):      ModelName.GPT_4O_MINI,
    (_LOW, TaskType.CREATIVE):     ModelName.GPT_4O_MINI,
    (_LOW, TaskType.CODE):         ModelName.GPT_4O_MINI,
    (_LOW, TaskType.ANALYSIS):     ModelName.GPT_4O_MINI,
    (_LOW, TaskType.MATH):         ModelName.GPT_4O_MINI,
    (_LOW, TaskType.REASONING):    ModelName.GPT_4O_MINI,

    # --- MEDIUM complexity (4-6) → split between mini and Sonnet ---
    (_MEDIUM, TaskType.CODE):        ModelName.GPT_4O_MINI,
    (_MEDIUM, TaskType.ANALYSIS):    ModelName.CLAUDE_35_SONNET,
    (_MEDIUM, TaskType.MATH):        ModelName.GPT_4O_MINI,
    (_MEDIUM, TaskType.CREATIVE):    ModelName.CLAUDE_35_SONNET,
    (_MEDIUM, TaskType.TRANSLATION): ModelName.CLAUDE_35_SONNET,
    (_MEDIUM, TaskType.SIMPLE_QA):   ModelName.GPT_4O_MINI,
    (_MEDIUM, TaskType.GENERAL):     ModelName.GPT_4O_MINI,
    (_MEDIUM, TaskType.REASONING):   ModelName.CLAUDE_35_SONNET,

    # --- HIGH complexity (7-10) → GPT-4o and Sonnet ---
    (_HIGH, TaskType.REASONING):    ModelName.GPT_4O,
    (_HIGH, TaskType.MATH):         ModelName.GPT_4O,
    (_HIGH, TaskType.CODE):         ModelName.GPT_4O,
    (_HIGH, TaskType.ANALYSIS):     ModelName.CLAUDE_35_SONNET,
    (_HIGH, TaskType.CREATIVE):     ModelName.CLAUDE_35_SONNET,
    (_HIGH, TaskType.GENERAL):      ModelName.CLAUDE_35_SONNET,
    (_HIGH, TaskType.TRANSLATION):  ModelName.CLAUDE_35_SONNET,
    (_HIGH, TaskType.SIMPLE_QA):    ModelName.GPT_4O,
}

# Tier-level defaults (fallback if a specific task_type is missing)
_TIER_DEFAULTS: dict[str, ModelName] = {
    _LOW:    ModelName.GPT_4O_MINI,
    _MEDIUM: ModelName.GPT_4O_MINI,
    _HIGH:   ModelName.GPT_4O,
}


# ---------------------------------------------------------------------------
# Reasoning chain builder
# ---------------------------------------------------------------------------

def _build_reasoning_chain(
    classification: ClassificationResult,
    tier: str,
    chosen_model: ModelName,
    model_info: ModelInfo,
) -> list[ReasoningStep]:
    """Construct the step-by-step reasoning chain for the routing decision."""
    try:
        steps: list[ReasoningStep] = []

        # Step 1 — Classification summary
        steps.append(ReasoningStep(
            step=1,
            description=(
                f"Prompt classified as '{classification.task_type.value}' "
                f"with complexity {classification.complexity_score}/10 "
                f"(confidence: {classification.confidence}) "
                f"using {classification.classifier_mode.value} classifier."
            ),
        ))

        # Step 2 — Tier assignment
        tier_ranges = {_LOW: "1-3", _MEDIUM: "4-6", _HIGH: "7-10"}
        steps.append(ReasoningStep(
            step=2,
            description=(
                f"Complexity {classification.complexity_score} falls in the "
                f"{tier.upper()} tier (range {tier_ranges[tier]})."
            ),
        ))

        # Step 3 — Model selection rationale
        steps.append(ReasoningStep(
            step=3,
            description=(
                f"For {tier.upper()} complexity + '{classification.task_type.value}' tasks, "
                f"routing to {chosen_model.value} ({model_info.provider.value}). "
                f"Strengths: {', '.join(model_info.strengths)}."
            ),
        ))

        # Step 4 — Cost & latency context
        avg_cost = (model_info.cost_per_1k_input_tokens + model_info.cost_per_1k_output_tokens) / 2
        baseline = MODEL_REGISTRY[ModelName.GPT_4O]
        baseline_avg = (baseline.cost_per_1k_input_tokens + baseline.cost_per_1k_output_tokens) / 2

        if chosen_model == ModelName.GPT_4O:
            cost_note = "This is the premium baseline model — no cost savings on this request."
        else:
            savings = ((baseline_avg - avg_cost) / baseline_avg) * 100 if baseline_avg > 0 else 0
            cost_note = (
                f"Estimated ~${avg_cost:.4f}/1k tokens vs "
                f"${baseline_avg:.4f}/1k (GPT-4o baseline) — "
                f"~{savings:.0f}% cost reduction."
            )

        steps.append(ReasoningStep(
            step=4,
            description=cost_note,
        ))

        # Step 5 — Latency note
        steps.append(ReasoningStep(
            step=5,
            description=(
                f"Expected latency: ~{model_info.avg_latency_ms}ms "
                f"(baseline GPT-4o: ~{baseline.avg_latency_ms}ms)."
            ),
        ))

        return steps
    except Exception:
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route(classification: ClassificationResult) -> RoutingDecision:
    """
    Given a ClassificationResult, select the optimal model and return
    a full RoutingDecision with reasoning chain, cost, and latency.
    """
    try:
        tier = _get_tier(classification.complexity_score)

        # Look up model from routing table, fall back to tier default
        key = (tier, classification.task_type)
        chosen_model = _ROUTING_TABLE.get(key, _TIER_DEFAULTS[tier])

        model_info = MODEL_REGISTRY[chosen_model]

        # Build reasoning chain
        reasoning_chain = _build_reasoning_chain(
            classification, tier, chosen_model, model_info,
        )

        # Average cost per 1k tokens (input + output blend)
        avg_cost = (model_info.cost_per_1k_input_tokens + model_info.cost_per_1k_output_tokens) / 2

        return RoutingDecision(
            model=chosen_model,
            provider=model_info.provider,
            reasoning_chain=reasoning_chain,
            estimated_cost_per_1k_tokens=round(avg_cost, 6),
            estimated_latency_ms=model_info.avg_latency_ms,
        )
    except Exception:
        traceback.print_exc()
        raise


def get_all_models() -> list[ModelInfo]:
    """Return metadata for every model in the registry (for GET /models)."""
    try:
        return list(MODEL_REGISTRY.values())
    except Exception:
        traceback.print_exc()
        raise


def get_baseline_model() -> ModelInfo:
    """Return the baseline (most expensive) model info for cost comparisons."""
    try:
        return MODEL_REGISTRY[ModelName.GPT_4O]
    except Exception:
        traceback.print_exc()
        raise
