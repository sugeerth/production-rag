# Vector Databases: A Comprehensive Guide

## What is a Vector Database?

A vector database is a specialized database designed to store, index, and query high-dimensional vector embeddings efficiently. Unlike traditional databases that search by exact matches, vector databases find the most similar vectors using distance metrics like cosine similarity, Euclidean distance, or dot product.

## Why Vector Databases?

Traditional databases are optimized for structured data with exact matches. But AI applications need to:
- Find semantically similar text passages
- Search for similar images
- Match user preferences to items
- Retrieve relevant context for LLMs

Vector databases solve this with Approximate Nearest Neighbor (ANN) search, trading a small amount of accuracy for orders-of-magnitude speedup.

## Key Indexing Algorithms

### HNSW (Hierarchical Navigable Small World)
- Builds a multi-layer graph structure
- Fast query time: O(log N)
- High recall (typically >95%)
- Higher memory usage
- Best for: read-heavy workloads with moderate dataset sizes

### IVF (Inverted File Index)
- Partitions vectors into clusters using k-means
- Only searches relevant clusters at query time
- Lower memory than HNSW
- Requires training on representative data
- Best for: large datasets with memory constraints

### Product Quantization (PQ)
- Compresses vectors by splitting into sub-vectors and quantizing
- Dramatic memory reduction (often 10-50x)
- Some accuracy loss
- Often combined with IVF (IVF-PQ)
- Best for: very large datasets (billions of vectors)

## Popular Vector Databases

### ChromaDB
- Open-source, lightweight, easy to get started
- Python-native API
- Built-in embedding functions
- Good for prototyping and small-to-medium scale
- Persistent and in-memory modes

### Pinecone
- Fully managed cloud service
- Automatic scaling and high availability
- Metadata filtering
- Good for production workloads
- Pricing based on pod type and storage

### Weaviate
- Open-source with cloud option
- GraphQL API
- Built-in vectorization modules
- Hybrid search (vector + keyword)
- Multi-modal support

### Qdrant
- Open-source, written in Rust
- High performance
- Rich filtering capabilities
- Payload (metadata) indexing
- gRPC and REST APIs

### FAISS (Facebook AI Similarity Search)
- Library, not a database (no built-in persistence)
- Extremely fast for batch queries
- GPU support
- Best for research and custom implementations

## Hybrid Search

Combining vector search with traditional keyword search:

### BM25 (Best Matching 25)
- Classic information retrieval algorithm
- Based on term frequency and document length
- Excels at exact keyword matching
- Complements semantic vector search

### Reciprocal Rank Fusion (RRF)
A simple but effective method to combine rankings from multiple search methods:
```
RRF_score(d) = sum(1 / (k + rank_i(d)))
```
Where k is a constant (typically 60) and rank_i is the rank from search method i.

## Best Practices

### Choosing Embedding Dimensions
- Higher dimensions capture more nuance but cost more storage/compute
- 384 dimensions (MiniLM) is a good starting point
- 768-1024 for higher quality requirements
- Use dimensionality reduction (PCA) if needed

### Metadata Filtering
- Store metadata alongside vectors for filtering
- Pre-filter before vector search for efficiency
- Common filters: document type, date range, access control

### Batch vs Real-time Ingestion
- Batch: More efficient for large document collections
- Real-time: Necessary for frequently updated content
- Use change detection to avoid re-embedding unchanged documents

### Monitoring
- Track query latency and recall
- Monitor index size and memory usage
- Watch for embedding drift over time
- Set up alerts for degraded retrieval quality
