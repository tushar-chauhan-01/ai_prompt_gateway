"""
FastAPI application — AI Gateway entry point.

Endpoints:
  POST /route   — classify + route + generate response (main endpoint)
  GET  /models  — list available models with pricing
  GET  /logs    — get routing history
  GET  /stats   — aggregated cost savings, model usage distribution
  GET  /health  — simple health check
"""

import traceback

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.app.cache import ResponseCache
from backend.app.logger import RequestLogger
from backend.app.models import (
    ClassifierMode,
    CostComparison,
    GatewayStats,
    LogEntry,
    ModelInfo,
    ModelName,
    RouteRequest,
    RouteResponse,
)
from backend.app.router import get_all_models, get_baseline_model, route
from backend.app.providers.manager import ProviderManager
from backend.app.classifier import rule_based as rb_classifier
from backend.app.classifier import llm_based as llm_classifier

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Gateway",
    description="Intelligent LLM router that selects the optimal model based on prompt complexity.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared singletons
# ---------------------------------------------------------------------------

_cache = ResponseCache()
_logger = RequestLogger()
_provider_manager = ProviderManager()


# ---------------------------------------------------------------------------
# POST /route
# ---------------------------------------------------------------------------

@app.post("/route", response_model=RouteResponse)
def route_prompt(request: RouteRequest) -> RouteResponse:
    """
    Main endpoint: classify the prompt, route to the optimal model,
    generate a response, and return everything the UI needs.
    """
    try:
        # 1. Check cache
        cached = _cache.get(request.prompt, request.classifier_mode)
        if cached is not None:
            return cached

        # 2. Classify
        if request.classifier_mode == ClassifierMode.LLM_BASED:
            classification = llm_classifier.classify(request.prompt)
        else:
            classification = rb_classifier.classify(request.prompt)

        # 3. Route
        routing = route(classification)

        # 4. Generate response via real provider
        provider_response = _provider_manager.generate(request.prompt, routing.model)

        # 5. Cost comparison vs baseline (GPT-4o)
        baseline_info = get_baseline_model()
        baseline_avg_cost = (
            baseline_info.cost_per_1k_input_tokens
            + baseline_info.cost_per_1k_output_tokens
        ) / 2
        baseline_cost = (provider_response.tokens_used / 1000) * baseline_avg_cost

        chosen_cost = provider_response.simulated_cost
        if baseline_cost > 0 and chosen_cost < baseline_cost:
            savings_pct = round(((baseline_cost - chosen_cost) / baseline_cost) * 100, 2)
        else:
            savings_pct = 0.0

        cost_comparison = CostComparison(
            chosen_model=routing.model,
            chosen_cost=round(chosen_cost, 6),
            baseline_model=ModelName.GPT_4O,
            baseline_cost=round(baseline_cost, 6),
            savings_percent=savings_pct,
        )

        # 6. Assemble response
        response = RouteResponse(
            prompt=request.prompt,
            classification=classification,
            routing=routing,
            response=provider_response,
            cost_comparison=cost_comparison,
        )

        # 7. Cache it
        _cache.put(request.prompt, request.classifier_mode, response)

        # 8. Log it
        _logger.log(response)

        return response

    except ValueError as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@app.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    """Return metadata for all available models with pricing info."""
    try:
        return get_all_models()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ---------------------------------------------------------------------------
# GET /logs
# ---------------------------------------------------------------------------

@app.get("/logs", response_model=list[LogEntry])
def get_logs(
    limit: int = Query(default=50, ge=1, le=200, description="Max entries to return"),
    offset: int = Query(default=0, ge=0, description="Entries to skip"),
) -> list[LogEntry]:
    """Return routing history in reverse-chronological order."""
    try:
        return _logger.get_logs(limit=limit, offset=offset)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------

@app.get("/stats", response_model=GatewayStats)
def get_stats() -> GatewayStats:
    """Return aggregated cost savings, model usage distribution, and more."""
    try:
        return _logger.get_stats()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    """Simple health check endpoint."""
    try:
        return {
            "status": "healthy",
            "models_available": len(get_all_models()),
            "cache_stats": _cache.stats(),
            "total_requests_logged": _logger.count,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
