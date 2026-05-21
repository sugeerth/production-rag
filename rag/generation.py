"""Generation: prompt assembly and LLM generation via Ollama."""

import re

import httpx

import config
from rag.retrieval import RetrievalResult
from rag.guardrails import check_safety, SafetyResult

# Pre-compiled citation extraction pattern
_CITATION_PATTERN = re.compile(r'\[Source:\s*([^\]]+)\]')

# Ollama connection pool singleton
_ollama_client = None


def get_ollama_client() -> httpx.AsyncClient:
    """Get or create a reusable Ollama HTTP client with connection pooling."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _ollama_client


async def close_ollama_client():
    """Close the Ollama client (call on shutdown)."""
    global _ollama_client
    if _ollama_client is not None:
        await _ollama_client.aclose()
        _ollama_client = None


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context documents. Follow these rules strictly:

1. ONLY use information from the provided context to answer the question.
2. If the context does not contain enough information to answer, say "I don't have enough information in the available documents to answer this question."
3. ALWAYS cite your sources using [Source: filename] format after each claim.
4. Do NOT make up or hallucinate information not present in the context.
5. If multiple sources provide conflicting information, mention all perspectives and note the conflict.
6. Be concise but thorough. Provide direct answers first, then supporting details.
7. If the question is ambiguous, state your interpretation before answering."""


def assemble_context(retrieval_result: RetrievalResult) -> str:
    """Assemble retrieved chunks into a context string with citations."""
    if not retrieval_result.chunks:
        return "No relevant documents found."

    context_parts = []
    seen_ids = set()
    seen_token_sets = []

    for i, chunk in enumerate(retrieval_result.chunks):
        # Deduplicate by chunk_id
        cid = chunk.get("chunk_id")
        if cid and cid in seen_ids:
            continue
        if cid:
            seen_ids.add(cid)

        # Near-duplicate detection via token-set overlap
        tokens = set(chunk["text"].lower().split())
        is_near_dup = False
        for prev_tokens in seen_token_sets:
            if not tokens or not prev_tokens:
                continue
            overlap = len(tokens & prev_tokens) / min(len(tokens), len(prev_tokens))
            if overlap > 0.85:
                is_near_dup = True
                break
        if is_near_dup:
            continue
        seen_token_sets.append(tokens)

        source = chunk.get("metadata", {}).get("source", "Unknown")
        chunk_idx = chunk.get("metadata", {}).get("chunk_index", "?")
        score = chunk.get("combined_score", chunk.get("score", 0))

        context_parts.append(
            f"[Document {i+1} | Source: {source} | Section {chunk_idx} | Relevance: {score:.2f}]\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_prompt(query: str, context: str) -> str:
    """Build the final prompt for the LLM."""
    return f"""Context Documents:
{context}

---

Question: {query}

Please provide a comprehensive answer based on the context documents above. Cite sources using [Source: filename] format."""


async def generate_answer(query: str, retrieval_result: RetrievalResult) -> dict:
    """Generate a grounded answer using Ollama."""

    # Check guardrails
    safety = check_safety(query, retrieval_result)
    if not safety.is_safe:
        return {
            "answer": safety.message,
            "citations": [],
            "confidence": 0.0,
            "retrieval_score": retrieval_result.top_score,
            "status": "blocked",
            "reason": safety.reason,
        }

    # Assemble context
    context = assemble_context(retrieval_result)

    # Build prompt
    user_prompt = build_prompt(query, context)

    # Call Ollama using pooled client
    try:
        client = get_ollama_client()
        response = await client.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": config.TEMPERATURE,
                    "num_predict": 1024,
                },
            },
        )
        response.raise_for_status()
        result = response.json()
        answer = result["message"]["content"]
    except httpx.ConnectError:
        return {
            "answer": "Error: Cannot connect to Ollama. Please ensure Ollama is running (ollama serve).",
            "citations": [],
            "confidence": 0.0,
            "retrieval_score": retrieval_result.top_score,
            "status": "error",
            "reason": "ollama_unavailable",
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "citations": [],
            "confidence": 0.0,
            "retrieval_score": retrieval_result.top_score,
            "status": "error",
            "reason": str(e),
        }

    # Extract citations from answer
    citation_pattern = _CITATION_PATTERN.findall(answer)
    citations = list(set(citation_pattern))

    # Build source list from retrieved chunks
    sources = []
    for chunk in retrieval_result.chunks:
        meta = chunk.get("metadata", {})
        sources.append({
            "source": meta.get("source", "Unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "score": chunk.get("combined_score", chunk.get("score", 0)),
            "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
        })

    return {
        "answer": answer,
        "citations": citations,
        "sources": sources,
        "confidence": retrieval_result.top_score,
        "retrieval_score": retrieval_result.top_score,
        "query_rewritten": retrieval_result.query_rewritten,
        "status": "success",
        "chunks_used": len(retrieval_result.chunks),
    }
