"""Evaluation metrics for each RAG component."""

import time
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime

import config

_log_executor = ThreadPoolExecutor(max_workers=1)


@dataclass
class QueryLog:
    """Log entry for a single query."""
    timestamp: str
    query: str
    query_rewritten: str
    num_chunks_retrieved: int
    retrieval_scores: list[float]
    top_score: float
    answer_length: int
    citations: list[str]
    latency_ms: float
    status: str
    user_feedback: str = ""  # thumbs_up, thumbs_down, or empty


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    avg_retrieval_score: float = 0.0
    avg_chunks_used: float = 0.0
    avg_citations_per_answer: float = 0.0
    abstention_rate: float = 0.0
    error_rate: float = 0.0
    positive_feedback_rate: float = 0.0
    negative_feedback_rate: float = 0.0


# In-memory log store (in production, use a database)
_query_logs: list[QueryLog] = []
LOG_FILE = os.path.join(config.DATA_DIR, "query_logs.jsonl")


def log_query(query: str, query_rewritten: str, retrieval_scores: list[float],
              top_score: float, answer: str, citations: list[str],
              latency_ms: float, status: str) -> QueryLog:
    """Log a query and its results."""
    entry = QueryLog(
        timestamp=datetime.now().isoformat(),
        query=query,
        query_rewritten=query_rewritten,
        num_chunks_retrieved=len(retrieval_scores),
        retrieval_scores=retrieval_scores,
        top_score=top_score,
        answer_length=len(answer.split()),
        citations=citations,
        latency_ms=latency_ms,
        status=status,
    )
    _query_logs.append(entry)

    # Non-blocking file write
    def _write_log(log_data: str):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        try:
            with open(LOG_FILE, "a") as f:
                f.write(log_data + "\n")
        except Exception:
            pass

    _log_executor.submit(_write_log, json.dumps(asdict(entry)))

    return entry


def record_feedback(query_index: int, feedback: str):
    """Record user feedback for a query."""
    if 0 <= query_index < len(_query_logs):
        _query_logs[query_index].user_feedback = feedback


def get_metrics() -> EvalMetrics:
    """Compute aggregate evaluation metrics."""
    if not _query_logs:
        return EvalMetrics()

    total = len(_query_logs)
    latencies = [q.latency_ms for q in _query_logs]
    scores = [q.top_score for q in _query_logs if q.retrieval_scores]
    chunks = [q.num_chunks_retrieved for q in _query_logs]
    citations = [len(q.citations) for q in _query_logs]
    abstentions = sum(1 for q in _query_logs if q.status in ("blocked", "low_confidence", "no_results"))
    errors = sum(1 for q in _query_logs if q.status == "error")
    positive = sum(1 for q in _query_logs if q.user_feedback == "thumbs_up")
    negative = sum(1 for q in _query_logs if q.user_feedback == "thumbs_down")
    feedback_total = positive + negative

    return EvalMetrics(
        total_queries=total,
        avg_latency_ms=sum(latencies) / total if total else 0,
        avg_retrieval_score=sum(scores) / len(scores) if scores else 0,
        avg_chunks_used=sum(chunks) / total if total else 0,
        avg_citations_per_answer=sum(citations) / total if total else 0,
        abstention_rate=abstentions / total if total else 0,
        error_rate=errors / total if total else 0,
        positive_feedback_rate=positive / feedback_total if feedback_total else 0,
        negative_feedback_rate=negative / feedback_total if feedback_total else 0,
    )


def get_recent_logs(n: int = 20) -> list[dict]:
    """Get the N most recent query logs."""
    recent = _query_logs[-n:]
    return [asdict(log) for log in reversed(recent)]


def load_logs_from_file():
    """Load logs from the persistent log file."""
    global _query_logs
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        _query_logs.append(QueryLog(**data))
        except Exception:
            pass
