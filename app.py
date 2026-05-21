"""FastAPI application for the RAG system."""

import os
import time
import shutil
from dataclasses import asdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

import config
from rag.ingestion import ingest_directory, ingest_document, PARSERS
from rag.indexing import index_chunks, get_index_stats, clear_index
from rag.retrieval import retrieve
from rag.generation import generate_answer
from rag.evaluation import (
    log_query, get_metrics, get_recent_logs,
    record_feedback, load_logs_from_file,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: preload models, warm caches. Shutdown: close clients."""
    load_logs_from_file()
    # Preload embedding model
    from rag.embeddings import get_model
    get_model()
    # Warm BM25 index from existing ChromaDB data
    from rag.indexing import _rebuild_bm25
    _rebuild_bm25()
    # Initialize Ollama connection pool
    from rag.generation import get_ollama_client
    get_ollama_client()
    yield
    # Cleanup
    from rag.generation import close_ollama_client
    await close_ollama_client()


app = FastAPI(
    title="RAG System",
    description="Retrieval-Augmented Generation over private documents",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(config.BASE_DIR, "static")), name="static")


# --- Models ---

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True


class FeedbackRequest(BaseModel):
    query_index: int
    feedback: str  # "thumbs_up" or "thumbs_down"


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    return FileResponse(os.path.join(config.BASE_DIR, "static", "index.html"))


@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    """Main Q&A endpoint."""
    start = time.time()

    # Retrieve
    retrieval_result = retrieve(
        query=request.query,
        top_k=request.top_k,
        use_reranking=request.use_reranking,
    )

    # Generate
    result = await generate_answer(request.query, retrieval_result)

    latency_ms = (time.time() - start) * 1000

    # Log
    log_query(
        query=request.query,
        query_rewritten=retrieval_result.query_rewritten,
        retrieval_scores=retrieval_result.retrieval_scores,
        top_score=retrieval_result.top_score,
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        latency_ms=latency_ms,
        status=result.get("status", "unknown"),
    )

    result["latency_ms"] = round(latency_ms, 1)
    return result


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in PARSERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {list(PARSERS.keys())}",
        )

    # Save file
    os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)
    filepath = os.path.join(config.DOCUMENTS_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest and index
    try:
        chunks = ingest_document(filepath)
        result = index_chunks(chunks)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "index_stats": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest")
async def ingest_all():
    """Ingest all documents from the documents directory."""
    chunks, stats = ingest_directory()
    if chunks:
        index_result = index_chunks(chunks)
    else:
        index_result = {"indexed": 0}

    return {
        "status": "success",
        "stats": asdict(stats),
        "index": index_result,
    }


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    index_stats = get_index_stats()
    eval_metrics = get_metrics()

    # List documents
    docs = []
    if os.path.isdir(config.DOCUMENTS_DIR):
        for f in sorted(os.listdir(config.DOCUMENTS_DIR)):
            ext = os.path.splitext(f)[1].lower()
            if ext in PARSERS:
                filepath = os.path.join(config.DOCUMENTS_DIR, f)
                size = os.path.getsize(filepath)
                docs.append({"name": f, "size": size, "type": ext})

    return {
        "index": index_stats,
        "metrics": asdict(eval_metrics),
        "documents": docs,
        "config": {
            "ollama_model": config.OLLAMA_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_rerank": config.TOP_K_RERANK,
            "hybrid_alpha": config.HYBRID_ALPHA,
        },
    }


@app.get("/api/logs")
async def get_logs(n: int = 20):
    """Get recent query logs."""
    return {"logs": get_recent_logs(n)}


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for a query."""
    record_feedback(request.query_index, request.feedback)
    return {"status": "recorded"}


@app.post("/api/clear")
async def clear_all():
    """Clear all indexed data."""
    clear_index()
    return {"status": "cleared"}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and re-index."""
    filepath = os.path.join(config.DOCUMENTS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Document not found")
    os.remove(filepath)
    # Re-index everything
    clear_index()
    chunks, stats = ingest_directory()
    if chunks:
        index_chunks(chunks)
    return {"status": "deleted", "filename": filename}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
