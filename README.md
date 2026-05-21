# Production RAG

> Production-grade Retrieval-Augmented Generation over your private documents — hybrid search, reranking, and traceable, cited answers.

## Overview

A production-quality Retrieval-Augmented Generation system for answering natural-language questions over private documents. It combines hybrid BM25 + vector retrieval with cross-encoder-style reranking, and returns answers with inline citations and full traceability back to source chunks. Built for low-latency interactive use with hallucination guardrails and built-in evaluation and monitoring.

## Features

- Hybrid retrieval: BM25 lexical search + dense vector search fused via Reciprocal Rank Fusion
- Reranking stage to lift the most relevant chunks before generation
- Inline citations (`[Source: filename]`) with full answer-to-source traceability
- Multi-format ingestion: PDF, Markdown, DOCX, HTML, plain text
- Incremental re-indexing on document upload
- Safety guardrails: prompt-injection detection and confidence-based abstention
- Built-in evaluation suite and a live monitoring dashboard

## Tech Stack

- **Language:** Python
- **API / Server:** FastAPI, Uvicorn
- **Vector store:** ChromaDB (HNSW)
- **Lexical search:** BM25 (`rank-bm25`)
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM:** Ollama (llama3.2)

---

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure Ollama is running with llama3.2
ollama pull llama3.2
ollama serve  # if not already running

# 3. Launch the application
python app.py

# 4. Open http://localhost:8000
```

The app ships with 3 sample documents in `documents/`. Upload more via the UI or drop files into the `documents/` folder and click "Re-index All Documents".

---

## ML System Design

### Scenario

Design a RAG system that answers user questions using a private corpus (internal docs, PDFs, knowledge base articles).

**Requirements:**
- Natural-language Q&A over private documents
- Handle frequent document updates (new/changed docs)
- Provide citations and traceability to sources
- Low latency for interactive use (target p95 < 2s)
- Reduce hallucinations; answers grounded in retrieved context

### Scope & Assumptions

| Parameter | Value |
|---|---|
| Corpus scale | 10^5 - 10^7 chunks |
| Formats | PDF, Markdown, DOCX, HTML, plain text |
| Update frequency | Near-real-time (upload triggers re-indexing) |
| Query types | Fact lookup, multi-section synthesis, summarization |
| Latency budget | p95 < 2s retrieval, < 10s end-to-end with generation |
| Output | Natural language answer + source citations |

---

## End-to-End Architecture

```
                        OFFLINE PIPELINE
  ┌──────────┐    ┌────────┐    ┌─────────┐    ┌──────────┐    ┌───────────┐
  │ Document  │───>│ Parser │───>│ Chunker │───>│ Embedder │───>│ Vector DB │
  │ Sources   │    │        │    │         │    │          │    │ + BM25    │
  └──────────┘    └────────┘    └─────────┘    └──────────┘    └───────────┘
  PDF/MD/DOCX/     Format-       Semantic +     MiniLM-L6       ChromaDB
  HTML/TXT         specific      overlap        384-dim         HNSW index

                        ONLINE PIPELINE
  ┌───────┐    ┌─────────┐    ┌────────┐    ┌────────┐    ┌─────────┐    ┌────────┐
  │ Query │───>│ Rewrite │───>│ Hybrid │───>│Reranker│───>│ Context │───>│Ollama  │───> Answer
  │       │    │ +Expand │    │ Search │    │        │    │Assembly │    │LLM     │    +Citations
  └───────┘    └─────────┘    └────────┘    └────────┘    └─────────┘    └────────┘
  User input    Abbreviation   Vector+BM25   Embedding     Dedupe +      llama3.2
                expansion      RRF fusion    re-score      compress      T=0.1
```

### Data Flow

1. **User uploads document** -> Parser extracts text -> Chunker splits into segments -> Embedder creates vectors -> Stored in ChromaDB + BM25 index
2. **User asks question** -> Query rewritten/expanded -> Hybrid search (vector + BM25 via RRF) -> Top-20 candidates reranked to top-5 -> Context assembled with citations -> Ollama generates grounded answer -> Safety checks applied -> Answer with sources returned

---

## Component Details

### 1. Document Ingestion & Preprocessing (`rag/ingestion.py`)

**Parsing:**
- Format-specific parsers for PDF (PyPDF2), DOCX (python-docx), HTML (BeautifulSoup), Markdown, plain text
- Structure preservation: headings, page numbers, section boundaries
- Text cleaning: whitespace normalization, control character removal, artifact cleanup

**Chunking strategy:**
- **Semantic-first**: Split by markdown headings/sections to preserve topical coherence
- **Sentence-boundary splitting**: Large sections split at sentence boundaries (not mid-word)
- **Size**: ~300 tokens per chunk with ~50 token overlap (configurable)
- **Minimum filter**: Chunks below 50 tokens are discarded
- **Metadata**: Each chunk tagged with `source`, `filepath`, `file_type`, `chunk_index`, `total_chunks`, `modified`, `word_count`

**Pitfalls addressed:**
- Too-small chunks lose context; too-large chunks reduce retrieval precision
- PDF extraction noise handled by cleaning pipeline
- Duplicate/boilerplate text handled by deduplication in context assembly

### 2. Embedding Strategy (`rag/embeddings.py`)

**Model:** `all-MiniLM-L6-v2` (384 dimensions)
- Good balance of speed and quality
- Normalized embeddings for cosine similarity
- Batch encoding for ingestion efficiency
- Lazy model loading to reduce startup time

**Design decisions:**
- L2-normalized embeddings so cosine similarity = dot product
- Single model for both documents and queries (symmetric embedding)
- Batch size of 64 for throughput during ingestion

### 3. Indexing (`rag/indexing.py`)

**Vector index:** ChromaDB with HNSW (cosine space)
- Persistent storage on disk
- Upsert support for incremental updates
- Metadata filtering capability

**Lexical index:** BM25Okapi
- Built from all indexed chunks
- Rebuilt on each indexing operation
- Handles exact keyword matching that vector search may miss

**Hybrid search:** Reciprocal Rank Fusion (RRF)
- `RRF_score(d) = alpha * 1/(k + rank_vector(d)) + (1-alpha) * 1/(k + rank_bm25(d))`
- Default alpha = 0.7 (favors vector search)
- k = 60 (standard RRF constant)

### 4. Retrieval & Reranking (`rag/retrieval.py`)

**Query understanding:**
- Abbreviation expansion (API, ML, NLP, RAG, DB, UI, LLM)
- Question mark removal for search optimization
- (Production: LLM-based query rewriting for multi-query retrieval)

**Retrieval:**
- Top-20 hybrid search candidates
- Configurable alpha between vector and BM25 weights

**Reranking:**
- Re-embed query and all candidates
- Compute direct embedding similarity as rerank score
- Combined score: `0.3 * original_score + 0.7 * rerank_score`
- Select top-5 after reranking

### 5. Generation & Grounding (`rag/generation.py`)

**Context assembly:**
- Deduplicate near-identical chunks (first 100 chars)
- Format with document number, source filename, section, relevance score
- Separated by `---` dividers for clarity

**System prompt enforces:**
- Answer ONLY from provided context
- Cite sources using `[Source: filename]` format
- Say "I don't know" when information is missing
- Note conflicting sources
- Be concise but thorough

**LLM configuration:**
- Ollama with llama3.2 (configurable via `OLLAMA_MODEL` env var)
- Temperature: 0.1 (low for factual consistency)
- Max prediction tokens: 1024

### 6. Safety & Guardrails (`rag/guardrails.py`)

**Prompt injection detection:**
- Pattern matching for common injection attempts ("ignore previous instructions", "pretend you are", etc.)
- Query length limit (5000 chars)

**Retrieval confidence guardrails:**
- Abstention threshold: if best retrieval score < 0.25, refuse to answer
- No-results handling: explicit "no relevant documents found" message

**Fallback behavior:**
- Graceful error messages for Ollama connection failures
- Descriptive abstention messages guiding users to rephrase or upload relevant docs

---

## Evaluation Plan

### A. Ingestion/Chunking Quality

| Metric | Method | Target |
|---|---|---|
| Chunk coherence | Manual audit (sample 50 chunks) | >90% rated "coherent" |
| Duplication rate | Automated near-duplicate detection | <5% duplicate chunks |
| Avg chunk size | Automated measurement | 200-400 tokens |
| Parser error rate | Count failures per format | <1% failure rate |
| Metadata accuracy | Spot-check source/timestamp | 100% accurate |

### B. Retrieval Quality

| Metric | Method | Target |
|---|---|---|
| Recall@5 | Labeled (query, relevant_chunk) pairs | >80% |
| Recall@20 | Same labeled dataset | >95% |
| MRR | Rank of first relevant result | >0.7 |
| nDCG@5 | Graded relevance labels | >0.75 |
| Latency (retrieval only) | Timer on hybrid search | p95 < 200ms |

**Creating labels:** Generate synthetic Q/A pairs from documents using an LLM, with human spot-checks.

### C. Reranking Quality

| Metric | Method | Target |
|---|---|---|
| nDCG@5 improvement | Compare before/after reranking | >10% relative improvement |
| Precision@1 improvement | Same | >15% relative improvement |
| Reranking latency | Timer | <100ms |

### D. Generation Quality

| Metric | Method | Target |
|---|---|---|
| Answer correctness | Human rating (1-5 scale) or LLM-as-judge | >4.0 avg |
| Faithfulness/grounding | Check each claim against context | >95% claims grounded |
| Citation accuracy | Verify cited sources contain the claimed info | >90% citations valid |
| Readability | Human rating | >4.0 avg |
| Hallucination rate | Claims not in any retrieved context | <5% |

### E. End-to-End Metrics

| Metric | Method | Target |
|---|---|---|
| Task success rate | User can get correct answer | >80% |
| User satisfaction | Thumbs up/down | >75% positive |
| End-to-end latency | Timer | p95 < 10s |
| Abstention rate | Count refusals / total queries | 10-20% (balanced) |
| Cost per query | Track compute usage | Within budget |

---

## Online Monitoring & Continuous Improvement

### Logging (built-in)

Every query logs:
- Original query, rewritten query
- Retrieved chunk IDs and scores
- Reranking scores
- Final answer text and length
- Citation count
- Latency (ms)
- Status (success/error/blocked)
- User feedback (thumbs up/down)

### Live Metrics Dashboard

The **Monitoring** tab shows:
- Total indexed chunks
- Total queries served
- Average latency, retrieval score, chunks used, citations
- Abstention rate
- Positive feedback rate
- Recent query logs table

### Continuous Improvement Loop

1. **Collect failure cases** from user feedback (thumbs down) and low-confidence queries
2. **Add to evaluation set** for regression testing
3. **Tune parameters:**
   - Chunk size and overlap
   - Top-k retrieval and reranking counts
   - Hybrid alpha (vector vs BM25 weight)
   - Retrieval confidence thresholds
4. **Expand domain coverage:**
   - Add synonym/abbreviation expansions
   - Upload additional documents for coverage gaps
   - Consider domain-adapted embedding models
5. **Monitor for drift:**
   - Embedding distribution shifts over time
   - Retrieval score distributions
   - Document freshness and staleness

---

## Project Structure

```
RAG/
├── app.py                  # FastAPI application (API + static files)
├── config.py               # All configuration parameters
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── rag/
│   ├── __init__.py
│   ├── ingestion.py        # Document parsing, cleaning, chunking
│   ├── embeddings.py       # Embedding model management
│   ├── indexing.py         # ChromaDB vector store + BM25 index
│   ├── retrieval.py        # Query rewriting, hybrid search, reranking
│   ├── generation.py       # Context assembly + Ollama LLM generation
│   ├── guardrails.py       # Safety checks + fallback behavior
│   └── evaluation.py       # Query logging + metrics computation
├── static/
│   ├── index.html          # Web UI
│   ├── style.css           # Styling
│   └── app.js              # Frontend logic
├── documents/              # Document corpus (upload files here)
│   ├── rag_system_guide.md
│   ├── machine_learning_basics.md
│   └── vector_databases.md
└── data/                   # Persistent storage (ChromaDB, logs)
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the web UI |
| `POST` | `/api/query` | Ask a question `{query, top_k, use_reranking}` |
| `POST` | `/api/upload` | Upload a document (multipart form) |
| `POST` | `/api/ingest` | Re-index all documents in `documents/` |
| `GET` | `/api/stats` | System stats, metrics, document list, config |
| `GET` | `/api/logs?n=20` | Recent query logs |
| `POST` | `/api/feedback` | Submit feedback `{query_index, feedback}` |
| `POST` | `/api/clear` | Clear all indexed data |
| `DELETE` | `/api/documents/{filename}` | Delete a document and re-index |

## Configuration

All parameters in `config.py` or via environment variables:

| Parameter | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model for generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | 300 | Target tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Token overlap between chunks |
| `TOP_K_RETRIEVAL` | 20 | Initial retrieval candidates |
| `TOP_K_RERANK` | 5 | Final results after reranking |
| `HYBRID_ALPHA` | 0.7 | Vector weight (1-alpha for BM25) |
| `TEMPERATURE` | 0.1 | LLM generation temperature |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.3 | Minimum relevance score |
| `ABSTENTION_THRESHOLD` | 0.25 | Score below which system abstains |

## Common Edge Cases Handled

- **Conflicting sources**: System prompt instructs LLM to present multiple perspectives
- **No relevant documents**: Graceful abstention with helpful message
- **Prompt injection**: Pattern-based detection and blocking
- **Ollama unavailable**: Clear error message with instructions
- **Unsupported file types**: Explicit error listing supported formats
- **Empty chunks**: Filtered out during ingestion
- **Duplicate chunks**: Deduplicated during context assembly
