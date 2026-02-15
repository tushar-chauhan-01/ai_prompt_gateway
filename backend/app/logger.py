"""
In-memory request/routing logger.

Stores LogEntry records for every routed request and computes
aggregated GatewayStats on demand.
"""

import threading
import traceback
from collections import defaultdict

from backend.app.models import (
    CostComparison,
    GatewayStats,
    LogEntry,
    ModelName,
    ModelUsageStat,
    RouteResponse,
)
from backend.app.router import get_baseline_model


class RequestLogger:
    """Thread-safe in-memory store for routing log entries and statistics."""

    def __init__(self) -> None:
        try:
            self._entries: list[LogEntry] = []
            self._lock = threading.Lock()
        except Exception:
            traceback.print_exc()
            raise

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log(self, response: RouteResponse) -> LogEntry:
        """
        Record a completed route response.
        Returns the created LogEntry.
        """
        try:
            entry = LogEntry.from_route_response(response)
            with self._lock:
                self._entries.append(entry)
            return entry
        except Exception:
            traceback.print_exc()
            raise

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_logs(self, limit: int = 50, offset: int = 0) -> list[LogEntry]:
        """
        Return log entries in reverse-chronological order (newest first).
        Supports pagination via limit/offset.
        """
        try:
            with self._lock:
                reversed_entries = list(reversed(self._entries))
                return reversed_entries[offset : offset + limit]
        except Exception:
            traceback.print_exc()
            raise

    def get_stats(self) -> GatewayStats:
        """Compute aggregated gateway statistics from all logged entries."""
        try:
            with self._lock:
                entries = list(self._entries)

            if not entries:
                return GatewayStats(
                    total_requests=0,
                    total_cost=0.0,
                    total_baseline_cost=0.0,
                    total_savings=0.0,
                    savings_percent=0.0,
                    model_usage=[],
                    avg_complexity=0.0,
                )

            baseline_model_info = get_baseline_model()
            baseline_avg_cost_per_1k = (
                baseline_model_info.cost_per_1k_input_tokens
                + baseline_model_info.cost_per_1k_output_tokens
            ) / 2

            total_cost = 0.0
            total_baseline_cost = 0.0
            total_complexity = 0

            # Per-model accumulators
            model_counts: dict[ModelName, int] = defaultdict(int)
            model_costs: dict[ModelName, float] = defaultdict(float)
            model_latencies: dict[ModelName, list[int]] = defaultdict(list)

            for entry in entries:
                total_cost += entry.cost
                total_complexity += entry.complexity_score

                # Estimate baseline cost: use the same token count but at
                # baseline model pricing. We approximate tokens from the
                # actual cost and the routed model's pricing.
                # Simpler approach: assume same tokens → scale by price ratio.
                total_baseline_cost += self._estimate_baseline_cost(
                    entry, baseline_avg_cost_per_1k,
                )

                model_counts[entry.routed_model] += 1
                model_costs[entry.routed_model] += entry.cost
                model_latencies[entry.routed_model].append(entry.latency_ms)

            total_requests = len(entries)
            total_savings = max(0.0, total_baseline_cost - total_cost)
            savings_percent = (
                (total_savings / total_baseline_cost * 100)
                if total_baseline_cost > 0
                else 0.0
            )
            # Clamp to 100 to satisfy the validator
            savings_percent = min(savings_percent, 100.0)

            avg_complexity = total_complexity / total_requests

            model_usage = []
            for model in model_counts:
                latencies = model_latencies[model]
                model_usage.append(ModelUsageStat(
                    model=model,
                    request_count=model_counts[model],
                    total_cost=round(model_costs[model], 6),
                    avg_latency_ms=round(sum(latencies) / len(latencies), 2),
                ))

            return GatewayStats(
                total_requests=total_requests,
                total_cost=round(total_cost, 6),
                total_baseline_cost=round(total_baseline_cost, 6),
                total_savings=round(total_savings, 6),
                savings_percent=round(savings_percent, 2),
                model_usage=model_usage,
                avg_complexity=round(avg_complexity, 2),
            )
        except Exception:
            traceback.print_exc()
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_baseline_cost(
        entry: LogEntry,
        baseline_avg_cost_per_1k: float,
    ) -> float:
        """
        Estimate what a request would have cost using the baseline model.
        Uses a simple ratio: if we know the actual cost and the model's
        avg rate, we can derive approximate tokens and re-price at baseline.
        """
        try:
            from backend.app.router import MODEL_REGISTRY

            model_info = MODEL_REGISTRY.get(entry.routed_model)
            if model_info is None:
                return entry.cost

            model_avg_cost_per_1k = (
                model_info.cost_per_1k_input_tokens
                + model_info.cost_per_1k_output_tokens
            ) / 2

            if model_avg_cost_per_1k <= 0:
                return entry.cost

            # Approximate token count from actual cost
            approx_tokens = (entry.cost / model_avg_cost_per_1k) * 1000
            baseline_cost = (approx_tokens / 1000) * baseline_avg_cost_per_1k

            return round(baseline_cost, 6)
        except Exception:
            traceback.print_exc()
            raise

    def clear(self) -> None:
        """Remove all log entries."""
        try:
            with self._lock:
                self._entries.clear()
        except Exception:
            traceback.print_exc()
            raise

    @property
    def count(self) -> int:
        """Return total number of logged entries."""
        try:
            with self._lock:
                return len(self._entries)
        except Exception:
            traceback.print_exc()
            raise
