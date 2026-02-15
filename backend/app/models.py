"""
Pydantic schemas for the AI Gateway.

Covers: classification, routing, provider responses, logging, stats,
and all API request / response envelopes.
"""

import traceback
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """Recognised prompt task categories."""
    SIMPLE_QA = "simple_qa"
    TRANSLATION = "translation"
    CODE = "code"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    MATH = "math"
    REASONING = "reasoning"
    GENERAL = "general"


class ClassifierMode(str, Enum):
    """Which classification strategy to use."""
    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"


class ModelName(str, Enum):
    """LLM models the gateway can route to."""
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    CLAUDE_35_SONNET = "claude-3.5-sonnet"


class ProviderName(str, Enum):
    """Cloud providers behind each model."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# ---------------------------------------------------------------------------
# Classification schemas
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """Output produced by either classifier."""
    complexity_score: int = Field(
        ..., ge=1, le=10,
        description="Prompt complexity on a 1-10 scale",
    )
    task_type: TaskType
    reasoning: str = Field(
        ..., min_length=1,
        description="Human-readable explanation of how the score was derived",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Classifier confidence (1.0 for rule-based)",
    )
    classifier_mode: ClassifierMode

    @field_validator("complexity_score")
    @classmethod
    def validate_complexity_range(cls, v: int) -> int:
        try:
            if not 1 <= v <= 10:
                raise ValueError("complexity_score must be between 1 and 10")
            return v
        except Exception:
            traceback.print_exc()
            raise


# ---------------------------------------------------------------------------
# Routing schemas
# ---------------------------------------------------------------------------

class ReasoningStep(BaseModel):
    """One step inside the router's reasoning chain."""
    step: int
    description: str


class RoutingDecision(BaseModel):
    """Full routing result returned by the router."""
    model: ModelName
    provider: ProviderName
    reasoning_chain: list[ReasoningStep] = Field(
        default_factory=list,
        description="Step-by-step explanation of the routing decision",
    )
    estimated_cost_per_1k_tokens: float = Field(
        ..., ge=0.0,
        description="Estimated cost in USD per 1 000 tokens",
    )
    estimated_latency_ms: int = Field(
        ..., ge=0,
        description="Estimated response latency in milliseconds",
    )


# ---------------------------------------------------------------------------
# Model / provider metadata
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Static metadata for a single model."""
    name: ModelName
    provider: ProviderName
    cost_per_1k_input_tokens: float = Field(..., ge=0.0)
    cost_per_1k_output_tokens: float = Field(..., ge=0.0)
    avg_latency_ms: int = Field(..., ge=0)
    strengths: list[str] = Field(default_factory=list)
    max_context_tokens: int = Field(default=128_000, ge=0)


# ---------------------------------------------------------------------------
# Provider response
# ---------------------------------------------------------------------------

class ProviderResponse(BaseModel):
    """What a (mock) provider returns after generation."""
    model: ModelName
    provider: ProviderName
    response_text: str
    tokens_used: int = Field(..., ge=0)
    latency_ms: int = Field(..., ge=0)
    simulated_cost: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# API request / response envelopes
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    """POST /route — incoming request body."""
    prompt: str = Field(
        ..., min_length=1, max_length=10_000,
        description="The user prompt to classify and route",
    )
    classifier_mode: ClassifierMode = Field(
        default=ClassifierMode.RULE_BASED,
        description="Classification strategy to use",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_blank(cls, v: str) -> str:
        try:
            stripped = v.strip()
            if not stripped:
                raise ValueError("prompt must not be blank or whitespace-only")
            return stripped
        except Exception:
            traceback.print_exc()
            raise


class RouteResponse(BaseModel):
    """POST /route — full response including everything the UI needs."""
    request_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
    )
    prompt: str
    classification: ClassificationResult
    routing: RoutingDecision
    response: ProviderResponse
    cost_comparison: "CostComparison"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CostComparison(BaseModel):
    """Shows how much was saved vs always using the most expensive model."""
    chosen_model: ModelName
    chosen_cost: float = Field(..., ge=0.0)
    baseline_model: ModelName = Field(
        default=ModelName.GPT_4O,
        description="Most expensive model used as the cost baseline",
    )
    baseline_cost: float = Field(..., ge=0.0)
    savings_percent: float = Field(
        ..., ge=0.0, le=100.0,
        description="Percentage saved compared to baseline",
    )

    @field_validator("savings_percent")
    @classmethod
    def validate_savings_range(cls, v: float) -> float:
        try:
            if not 0.0 <= v <= 100.0:
                raise ValueError("savings_percent must be between 0 and 100")
            return round(v, 2)
        except Exception:
            traceback.print_exc()
            raise


# Rebuild RouteResponse now that CostComparison is defined
RouteResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Logging schemas
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    """One row in the routing-history log."""
    request_id: str
    timestamp: datetime
    prompt_snippet: str = Field(
        ..., max_length=120,
        description="Truncated prompt for display in tables",
    )
    classifier_mode: ClassifierMode
    complexity_score: int = Field(..., ge=1, le=10)
    task_type: TaskType
    routed_model: ModelName
    latency_ms: int = Field(..., ge=0)
    cost: float = Field(..., ge=0.0)

    @staticmethod
    def from_route_response(resp: RouteResponse) -> "LogEntry":
        """Build a compact log entry from a full route response."""
        try:
            snippet = resp.prompt[:117] + "..." if len(resp.prompt) > 120 else resp.prompt
            return LogEntry(
                request_id=resp.request_id,
                timestamp=resp.timestamp,
                prompt_snippet=snippet,
                classifier_mode=resp.classification.classifier_mode,
                complexity_score=resp.classification.complexity_score,
                task_type=resp.classification.task_type,
                routed_model=resp.routing.model,
                latency_ms=resp.response.latency_ms,
                cost=resp.response.simulated_cost,
            )
        except Exception:
            traceback.print_exc()
            raise


# ---------------------------------------------------------------------------
# Stats / analytics schemas
# ---------------------------------------------------------------------------

class ModelUsageStat(BaseModel):
    """Usage count and cost for a single model."""
    model: ModelName
    request_count: int = Field(..., ge=0)
    total_cost: float = Field(..., ge=0.0)
    avg_latency_ms: float = Field(..., ge=0.0)


class GatewayStats(BaseModel):
    """GET /stats — aggregated analytics."""
    total_requests: int = Field(..., ge=0)
    total_cost: float = Field(..., ge=0.0)
    total_baseline_cost: float = Field(
        ..., ge=0.0,
        description="What it would have cost using the baseline model for everything",
    )
    total_savings: float = Field(..., ge=0.0)
    savings_percent: float = Field(..., ge=0.0, le=100.0)
    model_usage: list[ModelUsageStat] = Field(default_factory=list)
    avg_complexity: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Average complexity score across all requests",
    )

    @field_validator("savings_percent")
    @classmethod
    def validate_stats_savings(cls, v: float) -> float:
        try:
            if not 0.0 <= v <= 100.0:
                raise ValueError("savings_percent must be between 0 and 100")
            return round(v, 2)
        except Exception:
            traceback.print_exc()
            raise
