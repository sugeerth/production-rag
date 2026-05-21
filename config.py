"""RAG System Configuration."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking
CHUNK_SIZE = 300  # tokens (approx words)
CHUNK_OVERLAP = 50  # tokens overlap between chunks
MIN_CHUNK_SIZE = 50  # minimum chunk size to keep

# Retrieval
TOP_K_RETRIEVAL = 20  # initial retrieval count
TOP_K_RERANK = 5  # after reranking
HYBRID_ALPHA = 0.7  # weight for vector search (1-alpha for BM25)

# Generation
MAX_CONTEXT_TOKENS = 3000
TEMPERATURE = 0.1

# Guardrails
RETRIEVAL_SCORE_THRESHOLD = 0.15  # minimum score to consider a result relevant
ABSTENTION_THRESHOLD = 0.10  # if best score below this, abstain

# Collection name
COLLECTION_NAME = "rag_documents"
