# RAG System Design Guide

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI architecture pattern that enhances Large Language Models (LLMs) by grounding their responses in retrieved documents from a private knowledge base. Instead of relying solely on the model's pre-trained knowledge, RAG systems first retrieve relevant passages from a corpus, then use those passages as context for generating an answer.

The key benefits of RAG include:
- **Reduced hallucinations**: Answers are grounded in actual documents
- **Up-to-date information**: The knowledge base can be updated independently of the model
- **Traceability**: Every answer can cite its sources
- **Domain specificity**: Works with private/proprietary documents

## Architecture Overview

A RAG system consists of two main pipelines:

### Offline Pipeline (Ingestion)
1. **Document Collection**: Gather documents from various sources (S3, Drive, Confluence, Git repos)
2. **Parsing**: Convert documents to plain text while preserving structure
3. **Chunking**: Split text into retrievable units with metadata
4. **Embedding**: Convert chunks to vector representations
5. **Indexing**: Store vectors and metadata in a searchable index

### Online Pipeline (Query)
1. **Query Understanding**: Parse, rewrite, and expand the user's question
2. **Retrieval**: Find relevant chunks using vector and keyword search
3. **Reranking**: Re-score candidates for better precision
4. **Context Assembly**: Prepare retrieved passages with citations
5. **Generation**: Use an LLM to synthesize a grounded answer
6. **Post-processing**: Apply safety filters and formatting

## Document Ingestion

### Parsing Strategies
Different document formats require specialized parsers:
- **PDF**: Use PyPDF2 or pdfplumber; preserve page numbers for citations
- **Markdown/Text**: Direct text extraction with heading detection
- **DOCX**: python-docx for Word documents
- **HTML**: BeautifulSoup with boilerplate removal

### Chunking Best Practices
Chunking is critical for retrieval quality:
- **Semantic chunking**: Split by headings/sections first, then by sentences
- **Chunk size**: 200-400 tokens is typical; too small loses context, too large reduces retrieval precision
- **Overlap**: 10-20% overlap prevents information loss at boundaries
- **Metadata**: Attach source file, section heading, page number, timestamps

## Embedding Strategy

### Model Selection
- **all-MiniLM-L6-v2**: Good balance of speed and quality (384 dimensions)
- **BGE-large**: Higher quality but slower (1024 dimensions)
- **Domain-adapted models**: Fine-tune on domain vocabulary if needed

### Indexing
- Use HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search
- ChromaDB or FAISS for local deployment
- Pinecone, Weaviate, or Qdrant for production scale

## Retrieval Strategies

### Hybrid Search
Combine vector (semantic) search with BM25 (keyword) search:
- Vector search captures semantic meaning
- BM25 excels at exact keyword matching
- Reciprocal Rank Fusion (RRF) merges both result lists

### Reranking
After initial retrieval, rerank candidates for better precision:
- Cross-encoder models score query-passage pairs directly
- More expensive but more accurate than bi-encoder similarity
- Typically rerank top-20 candidates down to top-5

## Generation and Grounding

### Prompt Engineering
The system prompt should:
- Instruct the model to ONLY use provided context
- Require citations in a consistent format
- Include an abstention instruction for unsupported questions

### Hallucination Mitigation
- Set low temperature (0.1-0.3) for factual consistency
- Implement confidence thresholds on retrieval scores
- Use abstention: refuse to answer when context is insufficient

## Evaluation Framework

### Per-Component Metrics
1. **Chunking**: Coherence, duplication rate, size distribution
2. **Retrieval**: Recall@K, MRR, nDCG
3. **Reranking**: Precision improvement, latency overhead
4. **Generation**: Correctness, faithfulness, citation quality

### End-to-End Metrics
- Task success rate
- User satisfaction (thumbs up/down)
- Latency (p50, p95)
- Abstention rate
- Cost per query

## Common Edge Cases

- **Conflicting sources**: Present multiple perspectives with citations
- **Multi-hop questions**: Use iterative retrieval (retrieve -> draft -> retrieve again)
- **Very long documents**: Hierarchical retrieval (doc -> section -> chunk)
- **Stale documents**: Track modification timestamps, prefer recent sources
