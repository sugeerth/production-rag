"""Safety guardrails and fallback behavior."""

import re
from dataclasses import dataclass

import config
from rag.retrieval import RetrievalResult


@dataclass
class SafetyResult:
    is_safe: bool
    reason: str = ""
    message: str = ""


# Patterns that indicate potentially unsafe queries
BLOCKED_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+(instructions|prompts)",
    r"you\s+are\s+now\s+(?:a|an)\s+(?:different|new)",
    r"pretend\s+(?:you|to)\s+(?:are|be)",
    r"override\s+(?:your|system)\s+(?:instructions|prompt)",
    r"disregard\s+(?:your|all|previous)",
]


def check_query_safety(query: str) -> SafetyResult:
    """Check if the query contains prompt injection attempts."""
    query_lower = query.lower().strip()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query_lower):
            return SafetyResult(
                is_safe=False,
                reason="prompt_injection",
                message="I cannot process this query as it appears to contain instructions that could compromise the system's integrity. Please rephrase your question.",
            )

    if len(query) > 5000:
        return SafetyResult(
            is_safe=False,
            reason="query_too_long",
            message="Your query is too long. Please shorten it and try again.",
        )

    return SafetyResult(is_safe=True)


def check_retrieval_confidence(retrieval_result: RetrievalResult) -> SafetyResult:
    """Check if retrieval results are confident enough to generate an answer."""
    if not retrieval_result.chunks:
        return SafetyResult(
            is_safe=False,
            reason="no_results",
            message="I couldn't find any relevant documents to answer your question. Please try rephrasing your question or ensure relevant documents have been uploaded.",
        )

    if retrieval_result.top_score < config.ABSTENTION_THRESHOLD:
        return SafetyResult(
            is_safe=False,
            reason="low_confidence",
            message="I found some documents but none seem closely related to your question. The retrieved content may not be reliable enough to provide a good answer. Please try rephrasing your question or upload more relevant documents.",
        )

    return SafetyResult(is_safe=True)


def check_safety(query: str, retrieval_result: RetrievalResult) -> SafetyResult:
    """Run all safety checks."""
    # Check query safety
    query_check = check_query_safety(query)
    if not query_check.is_safe:
        return query_check

    # Check retrieval confidence
    retrieval_check = check_retrieval_confidence(retrieval_result)
    if not retrieval_check.is_safe:
        return retrieval_check

    return SafetyResult(is_safe=True)
