"""
Rule-based prompt classifier.

Uses heuristics — token count, keyword patterns, structural cues — to
determine the task type and complexity score (1-10) of a user prompt.
"""

import re
import traceback

from backend.app.models import ClassificationResult, ClassifierMode, TaskType


# ---------------------------------------------------------------------------
# Keyword / pattern banks  (order matters — first match wins for task type)
# ---------------------------------------------------------------------------

_CODE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(def |class |import |function |const |let |var |=>|async |await )\b", re.I),
    re.compile(r"\b(python|javascript|typescript|java|rust|golang|c\+\+|sql|html|css|react|django|flask|fastapi)\b", re.I),
    re.compile(r"\b(write|build|create|implement|code|debug|fix|refactor|optimise|optimize)\b.*\b(function|class|api|app(lication)?|script|program|module|endpoint|server|database|query|service|system)\b", re.I),
    re.compile(r"\b(bug|error|exception|traceback|stack\s*trace|segfault|compile|runtime)\b", re.I),
    re.compile(r"```"),
]

_MATH_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(solve|calculate|compute|derive|integrate|differentiate|prove|equation|formula)\b", re.I),
    re.compile(r"\b(algebra|calculus|geometry|trigonometry|probability|statistics|linear\s*algebra|matrix|matrices)\b", re.I),
    re.compile(r"[0-9]+\s*[\+\-\*/\^]\s*[0-9]+"),
    re.compile(r"\b(sum|product|factorial|logarithm|sqrt|sin|cos|tan)\b", re.I),
]

_CREATIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(write|compose|create|draft)\b.*\b(poem|story|essay|song|lyrics|haiku|limerick|narrative|fiction|blog\s*post|article)\b", re.I),
    re.compile(r"\b(creative|imaginative|poetic|artistic|metaphor|rhyme)\b", re.I),
    re.compile(r"\b(once upon a time|in a world where|dear diary)\b", re.I),
]

_ANALYSIS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(analy[sz]e|compare|contrast|evaluate|assess|critique|review|examine|investigate|discuss)\b", re.I),
    re.compile(r"\b(pros?\s+(and|&)\s+cons?|trade\s*-?\s*offs?|implications?|impact)\b", re.I),
    re.compile(r"\b(explain|describe|elaborate)\b.*\b(how|why|difference|relationship|impact)\b", re.I),
]

_TRANSLATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(translat(e|ion)|convert)\b.*\b(to|into|from)\b.*\b(english|spanish|french|german|chinese|japanese|korean|hindi|arabic|portuguese|russian|italian)\b", re.I),
    re.compile(r"\b(in\s+(english|spanish|french|german|chinese|japanese|korean|hindi|arabic|portuguese|russian|italian))\b", re.I),
]

_REASONING_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(reason|logic|deduc|induc|infer|hypothe|thought\s*experiment|paradox|dilemma)\b", re.I),
    re.compile(r"\b(step\s*by\s*step|chain\s*of\s*thought|think\s*through|work\s*through)\b", re.I),
    re.compile(r"\b(quantum|relativity|philosophy|epistemology|ontology|consciousness)\b", re.I),
    re.compile(r"\b(explain\s+(why|how)\b.*\b(complex|advanced|nuanced|detailed))\b", re.I),
]

_SIMPLE_QA_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(what|who|when|where|which|how\s+many|how\s+much|is|are|was|were|do|does|did|can|could)\b", re.I),
    re.compile(r"\b(define|meaning\s+of|what\s+is|who\s+is|capital\s+of)\b", re.I),
]

# Complexity-boosting signals
_COMPLEXITY_BOOSTERS: list[tuple[re.Pattern, int, str]] = [
    (re.compile(r"\b(step\s*by\s*step|detailed|comprehensive|thorough|in\s*-?\s*depth)\b", re.I), 2, "requests detailed/thorough treatment"),
    (re.compile(r"\b(compare|contrast|trade\s*-?\s*offs?|pros?\s+(and|&)\s+cons?)\b", re.I), 1, "involves comparison/trade-off analysis"),
    (re.compile(r"\b(explain|why|how\s+does|how\s+do)\b", re.I), 1, "asks for explanation"),
    (re.compile(r"\b(multiple|several|many|various|different)\b", re.I), 1, "references multiple items"),
    (re.compile(r"\b(advanced|complex|difficult|challenging|hard)\b", re.I), 2, "explicitly mentions high difficulty"),
    (re.compile(r"\b(error\s*handling|edge\s*case|security|authentication|authoriz)\b", re.I), 1, "mentions robustness concerns"),
    (re.compile(r"\b(architect|design\s*pattern|system\s*design|scalab|distributed)\b", re.I), 2, "involves architecture/design"),
]

# Complexity-reducing signals
_COMPLEXITY_REDUCERS: list[tuple[re.Pattern, int, str]] = [
    (re.compile(r"\b(simple|basic|easy|quick|brief|short)\b", re.I), 1, "explicitly simple/basic"),
    (re.compile(r"\b(yes\s+or\s+no|true\s+or\s+false)\b", re.I), 2, "binary question"),
    (re.compile(r"^.{1,30}$"), 1, "very short prompt"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_token_count(prompt: str) -> int:
    """Rough token estimate: ~1 token per 4 characters or 0.75 words."""
    try:
        word_count = len(prompt.split())
        char_count = len(prompt)
        return max(int((word_count * 0.75 + char_count / 4) / 2), 1)
    except Exception:
        traceback.print_exc()
        raise


def _count_matches(patterns: list[re.Pattern], text: str) -> int:
    """Return how many patterns from the list match the text."""
    try:
        return sum(1 for p in patterns if p.search(text))
    except Exception:
        traceback.print_exc()
        raise


def _detect_task_type(prompt: str) -> tuple[TaskType, str]:
    """
    Detect the primary task type from keyword patterns.
    Returns (TaskType, reasoning_fragment).
    """
    try:
        checks: list[tuple[list[re.Pattern], TaskType, str]] = [
            (_CODE_PATTERNS, TaskType.CODE, "code-related keywords detected"),
            (_MATH_PATTERNS, TaskType.MATH, "math/calculation keywords detected"),
            (_TRANSLATION_PATTERNS, TaskType.TRANSLATION, "translation request detected"),
            (_CREATIVE_PATTERNS, TaskType.CREATIVE, "creative writing keywords detected"),
            (_REASONING_PATTERNS, TaskType.REASONING, "complex reasoning keywords detected"),
            (_ANALYSIS_PATTERNS, TaskType.ANALYSIS, "analytical keywords detected"),
            (_SIMPLE_QA_PATTERNS, TaskType.SIMPLE_QA, "simple question pattern detected"),
        ]

        best_type = TaskType.GENERAL
        best_reason = "no strong keyword signals; classified as general"
        best_hits = 0

        for patterns, task_type, reason in checks:
            hits = _count_matches(patterns, prompt)
            if hits > best_hits:
                best_hits = hits
                best_type = task_type
                best_reason = reason

        return best_type, best_reason
    except Exception:
        traceback.print_exc()
        raise


def _compute_base_complexity(task_type: TaskType, token_count: int) -> tuple[int, list[str]]:
    """
    Assign a base complexity score from the task type and prompt length.
    Returns (base_score, list_of_reasoning_fragments).
    """
    try:
        reasons: list[str] = []

        # Base score by task type
        base_scores: dict[TaskType, int] = {
            TaskType.SIMPLE_QA: 2,
            TaskType.TRANSLATION: 3,
            TaskType.GENERAL: 3,
            TaskType.CREATIVE: 5,
            TaskType.CODE: 5,
            TaskType.ANALYSIS: 5,
            TaskType.MATH: 6,
            TaskType.REASONING: 7,
        }
        score = base_scores.get(task_type, 3)
        reasons.append(f"base score {score} for task type '{task_type.value}'")

        # Length adjustment
        if token_count > 200:
            score += 2
            reasons.append(f"long prompt (~{token_count} tokens, +2)")
        elif token_count > 80:
            score += 1
            reasons.append(f"medium-length prompt (~{token_count} tokens, +1)")
        else:
            reasons.append(f"short prompt (~{token_count} tokens, +0)")

        return score, reasons
    except Exception:
        traceback.print_exc()
        raise


def _apply_adjustments(
    prompt: str,
    score: int,
    reasons: list[str],
) -> tuple[int, list[str]]:
    """Apply booster and reducer patterns, mutating the reasons list."""
    try:
        for pattern, delta, reason in _COMPLEXITY_BOOSTERS:
            if pattern.search(prompt):
                score += delta
                reasons.append(f"+{delta}: {reason}")

        for pattern, delta, reason in _COMPLEXITY_REDUCERS:
            if pattern.search(prompt):
                score -= delta
                reasons.append(f"-{delta}: {reason}")

        return score, reasons
    except Exception:
        traceback.print_exc()
        raise


def _clamp(value: int, lo: int = 1, hi: int = 10) -> int:
    """Clamp an integer into [lo, hi]."""
    try:
        return max(lo, min(hi, value))
    except Exception:
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(prompt: str) -> ClassificationResult:
    """
    Classify a prompt using rule-based heuristics.

    Returns a ClassificationResult with complexity_score (1-10),
    detected task_type, and a human-readable reasoning string.
    """
    try:
        # 1. Token estimation
        token_count = _estimate_token_count(prompt)

        # 2. Task-type detection
        task_type, type_reason = _detect_task_type(prompt)

        # 3. Base complexity
        score, reasons = _compute_base_complexity(task_type, token_count)
        reasons.insert(0, type_reason)

        # 4. Boosters / reducers
        score, reasons = _apply_adjustments(prompt, score, reasons)

        # 5. Clamp
        final_score = _clamp(score)
        if final_score != score:
            reasons.append(f"clamped from {score} to {final_score}")

        reasoning = " | ".join(reasons)

        return ClassificationResult(
            complexity_score=final_score,
            task_type=task_type,
            reasoning=reasoning,
            confidence=1.0,
            classifier_mode=ClassifierMode.RULE_BASED,
        )
    except Exception:
        traceback.print_exc()
        raise
