# Module 8: Advanced Topics & Production

## Learning Objectives

By the end of this module, you will understand:
- Retrieval-Augmented Generation (RAG) and its components
- AI agents, tool use, and agentic architectures
- Model deployment, serving, and optimization
- Evaluation methodologies and benchmarking
- Ethical considerations, safety, and responsible AI
- Multi-modal models and future directions

---

## 8.1 Retrieval-Augmented Generation (RAG)

### The Fundamental Problem

LLMs face two critical limitations that RAG addresses:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM LIMITATIONS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. KNOWLEDGE CUTOFF                                            │
│     ┌─────────────────┬────────────────────────────────────┐    │
│     │   Training      │        Deployment                  │    │
│     │   Data          │                                    │    │
│     │   ◄────────────►│◄──────────────────────────────────►│   │
│     │   Model knows   │        Model doesn't know          │    │
│     │   this period   │        anything after cutoff       │    │
│     └─────────────────┴────────────────────────────────────┘    │
│                       ▲                                         │
│                       │                                         │
│                  Knowledge                                      │
│                  Cutoff Date                                    │
│                                                                 │
│  2. HALLUCINATION                                               │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  "Who won the 2024 Super Bowl?"                         │ │
│     │                                                         │ │
│     │  LLM: "The Kansas City Chiefs won..." (might be wrong!) │ │
│     │        ↓                                                │ │
│     │  Generates plausible-sounding but potentially           │ │
│     │  incorrect information with high confidence             │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│  3. NO ACCESS TO PRIVATE DATA                                   │
│     - Company documents                                         │
│     - Personal files                                            │
│     - Proprietary databases                                     │
│     - Internal wikis                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**The RAG Solution**: Instead of relying solely on parametric knowledge (stored in weights),
augment the LLM with non-parametric knowledge (retrieved from external sources).

> **Paper**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)

### RAG Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG SYSTEM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    OFFLINE: INDEXING PIPELINE                        │   │
│  │                                                                      │   │
│  │   Documents          Chunking           Embedding        Vector DB   │   │
│  │   ┌───────┐         ┌───────┐          ┌───────┐        ┌───────┐   │   │
│  │   │ PDF   │         │Chunk 1│          │[0.2,  │        │   •   │   │   │
│  │   │ Word  │  ───►   │Chunk 2│   ───►   │ 0.8,  │  ───►  │  •    │   │   │
│  │   │ HTML  │   Split │Chunk 3│  Encode  │ 0.1]  │  Store │ •     │   │   │
│  │   │ JSON  │         │  ...  │          │  ...  │        │    •  │   │   │
│  │   └───────┘         └───────┘          └───────┘        └───────┘   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ONLINE: QUERY PIPELINE                            │   │
│  │                                                                      │   │
│  │   User Query        Embed Query         Search           Retrieve    │   │
│  │   ┌───────┐        ┌───────┐          ┌───────┐        ┌───────┐    │   │
│  │   │"What  │        │[0.3,  │          │   •   │        │Chunk 2│    │   │
│  │   │ is    │  ───►  │ 0.7,  │   ───►   │  ◉────┼───►    │Chunk 5│    │   │
│  │   │ X?"   │ Encode │ 0.2]  │  Nearest │   •   │ Top-k  │Chunk 8│    │   │
│  │   └───────┘        └───────┘  Neighbor└───────┘        └───────┘    │   │
│  │                                                                      │   │
│  │                                  ↓                                   │   │
│  │                                                                      │   │
│  │   ┌──────────────────────────────────────────────────────────────┐  │   │
│  │   │                    AUGMENTED PROMPT                           │  │   │
│  │   │                                                               │  │   │
│  │   │  Context: [Chunk 2] [Chunk 5] [Chunk 8]                       │  │   │
│  │   │                                                               │  │   │
│  │   │  Question: What is X?                                         │  │   │
│  │   │                                                               │  │   │
│  │   │  Answer based on the context above:                           │  │   │
│  │   └──────────────────────────────────────────────────────────────┘  │   │
│  │                                  ↓                                   │   │
│  │                            ┌─────────┐                               │   │
│  │                            │   LLM   │                               │   │
│  │                            └────┬────┘                               │   │
│  │                                 ↓                                    │   │
│  │                         Grounded Response                            │   │
│  │                         (with citations)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component 1: Document Processing and Chunking

Chunking is crucial - too small chunks lose context, too large chunks dilute relevance.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHUNKING STRATEGIES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. FIXED-SIZE CHUNKING                                                     │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Document text here. This is the first part of │ the document.   │     │
│     │ It continues here with more content...        │ More text...    │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                   ▲                                        ▲                │
│                   │                                        │                │
│              Chunk 1 (512 tokens)                    Chunk 2 (512 tokens)   │
│                                                                             │
│     Problem: May split mid-sentence or mid-concept                          │
│                                                                             │
│  2. OVERLAPPING CHUNKS                                                      │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Document text here. This is the first part of the document.     │     │
│     │                    [════════ Chunk 1 ════════]                  │     │
│     │                              [════════ Chunk 2 ════════]        │     │
│     │                                        [════════ Chunk 3 ══════]│     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                              ▲                                              │
│                              │                                              │
│                         Overlap preserves context at boundaries             │
│                                                                             │
│  3. SEMANTIC CHUNKING                                                       │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ # Introduction          │  # Methods           │  # Results     │     │
│     │ Text about intro...     │  Text about methods..│  Text about... │     │
│     │ More intro content...   │  Implementation...   │  Findings...   │     │
│     └─────────────────────────┴──────────────────────┴────────────────┘     │
│              Chunk 1                  Chunk 2              Chunk 3          │
│                                                                             │
│     Splits on semantic boundaries (headers, paragraphs, topics)             │
│                                                                             │
│  4. HIERARCHICAL CHUNKING                                                   │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Level 1 (Document Summary)                                     │     │
│     │        ↓                                                        │     │
│     │  Level 2 (Section Summaries)                                    │     │
│     │        ↓                                                        │     │
│     │  Level 3 (Detailed Chunks)                                      │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│     Different retrieval based on query specificity                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
from typing import List
import re

class ChunkingStrategies:
    """Various document chunking strategies for RAG"""

    @staticmethod
    def fixed_size_chunks(text: str, chunk_size: int = 512,
                          overlap: int = 50) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text: Input document text
            chunk_size: Target size of each chunk (in characters)
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap  # Move back by overlap amount

        return chunks

    @staticmethod
    def sentence_aware_chunks(text: str, max_chunk_size: int = 512) -> List[str]:
        """
        Split on sentence boundaries while respecting max size.
        Avoids cutting mid-sentence.
        """
        # Split into sentences (simplified - use NLTK/spaCy for production)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    @staticmethod
    def semantic_chunks(text: str, headers_pattern: str = r'^#+\s') -> List[str]:
        """
        Split on semantic boundaries (markdown headers).
        Each section becomes a chunk.
        """
        sections = re.split(headers_pattern, text, flags=re.MULTILINE)
        return [s.strip() for s in sections if s.strip()]


# Metadata enrichment for better retrieval
class EnrichedChunk:
    """Chunk with metadata for improved retrieval"""

    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata  # source, page, section, date, etc.

    def to_embedding_input(self) -> str:
        """Create text representation for embedding"""
        # Include metadata in embedding for better semantic matching
        context = f"Source: {self.metadata.get('source', 'unknown')}\n"
        context += f"Section: {self.metadata.get('section', '')}\n"
        context += f"Content: {self.text}"
        return context
```

### Component 2: Embeddings and Vector Search

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING PROCESS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Text Input                    Embedding Model               Vector Output  │
│  ┌───────────────┐            ┌─────────────────┐          ┌────────────┐   │
│  │"The capital   │            │                 │          │  [0.234,   │   │
│  │ of France     │    ───►    │  Transformer    │   ───►   │   0.891,   │   │
│  │ is Paris"     │            │  Encoder        │          │   -0.127,  │   │
│  └───────────────┘            │  (e.g., E5,     │          │   0.456,   │   │
│                               │   BGE, OpenAI)  │          │   ...      │   │
│                               └─────────────────┘          │   0.789]   │   │
│                                                            └────────────┘   │
│                                                              768-4096 dims  │
│                                                                             │
│  KEY INSIGHT: Similar meanings → Similar vectors                            │
│                                                                             │
│  "The capital of France is Paris"  ──┐                                      │
│                                      ├──► Very close in vector space        │
│  "Paris is the French capital"     ──┘                                      │
│                                                                             │
│  "The weather is nice today"       ──────► Far away in vector space         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    VECTOR SIMILARITY SEARCH                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query: "What is the capital of France?"                                    │
│         ↓ embed                                                             │
│  Query Vector: [0.245, 0.876, -0.134, ...]                                  │
│                                                                             │
│  Vector Database:                           Cosine Similarity               │
│  ┌────────────────────────────────┐        ┌─────────────────────┐         │
│  │ Doc1: [0.234, 0.891, -0.127]   │   ───► │ cos(q, doc1) = 0.98 │ ◄── Top │
│  │ Doc2: [0.112, 0.445, 0.789]    │   ───► │ cos(q, doc2) = 0.23 │         │
│  │ Doc3: [0.198, 0.823, -0.098]   │   ───► │ cos(q, doc3) = 0.87 │ ◄── #2  │
│  │ Doc4: [-0.456, 0.234, 0.567]   │   ───► │ cos(q, doc4) = 0.12 │         │
│  │ Doc5: [0.267, 0.912, -0.145]   │   ───► │ cos(q, doc5) = 0.95 │ ◄── #3  │
│  └────────────────────────────────┘        └─────────────────────┘         │
│                                                                             │
│                 Cosine Similarity Formula                                   │
│                           A · B                                             │
│                 cos(θ) = ─────────                                          │
│                          ‖A‖ ‖B‖                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Popular Embedding Models**:

| Model | Dimensions | Context | Best For |
|-------|------------|---------|----------|
| [OpenAI text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings) | 1536 | 8K | General purpose |
| [OpenAI text-embedding-3-large](https://platform.openai.com/docs/guides/embeddings) | 3072 | 8K | Highest quality |
| [BGE-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | 1024 | 512 | Open source, multilingual |
| [E5-large-v2](https://huggingface.co/intfloat/e5-large-v2) | 1024 | 512 | Open source, efficient |
| [GTE-large](https://huggingface.co/thenlper/gte-large) | 1024 | 512 | Open source, competitive |
| [Cohere embed-v3](https://cohere.com/embeddings) | 1024 | 512 | Commercial, multilingual |

> **Benchmark**: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Massive Text Embedding Benchmark

**Implementation**:

```python
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """Service for generating and comparing embeddings"""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        # BGE models recommend adding instruction prefix for queries
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with instruction prefix (for asymmetric retrieval)"""
        # Some models use different prefixes for queries vs documents
        instruction = "Represent this sentence for retrieval: "
        return self.model.encode(instruction + query, normalize_embeddings=True)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def batch_cosine_similarity(query: np.ndarray,
                                 documents: np.ndarray) -> np.ndarray:
        """Compute similarity between query and all documents"""
        # Normalized vectors: cosine similarity = dot product
        return np.dot(documents, query)


class VectorStore:
    """Simple in-memory vector store (use FAISS/Pinecone for production)"""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.vectors: np.ndarray = None
        self.texts: List[str] = []
        self.metadata: List[dict] = []

    def add(self, texts: List[str], metadata: List[dict] = None):
        """Add documents to the store"""
        embeddings = self.embedding_service.embed(texts)

        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])

        self.texts.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        """Search for most similar documents"""
        query_embedding = self.embedding_service.embed_query(query)

        # Compute similarities
        similarities = self.embedding_service.batch_cosine_similarity(
            query_embedding, self.vectors
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append((
                self.texts[idx],
                float(similarities[idx]),
                self.metadata[idx]
            ))

        return results
```

### Component 3: Vector Databases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE COMPARISON                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Database        Type          Best For                  Scaling            │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Pinecone        Managed       Production, enterprise    Serverless, auto   │
│  Weaviate        Open/Managed  Hybrid search, GraphQL    Horizontal         │
│  Milvus          Open source   Large scale, on-prem      Distributed        │
│  Qdrant          Open source   Filtering, Rust perf      Horizontal         │
│  ChromaDB        Open source   Development, prototyping  Single node        │
│  pgvector        PostgreSQL    Existing Postgres users   Postgres scaling   │
│  FAISS           Library       Research, local           In-memory          │
│                                                                             │
│  APPROXIMATE NEAREST NEIGHBOR (ANN) ALGORITHMS                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  HNSW (Hierarchical Navigable Small World)                                  │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  Layer 2:    ●───────────────────────●                          │        │
│  │              │                       │                          │        │
│  │  Layer 1:    ●───────●───────●───────●───────●                  │        │
│  │              │       │       │       │       │                  │        │
│  │  Layer 0:    ●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●                  │        │
│  │                          ▲                                      │        │
│  │                      Query starts at top, navigates down        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│  • Fast search: O(log n)                                                    │
│  • High recall: >95%                                                        │
│  • Memory: Higher (graph structure)                                         │
│                                                                             │
│  IVF (Inverted File Index)                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │  Cluster centroids: ◉    ◉    ◉    ◉    ◉                       │        │
│  │                     │    │    │    │    │                       │        │
│  │  Vectors:           ●●●  ●●   ●●●● ●●●  ●●                      │        │
│  │                                                                 │        │
│  │  Query → Find nearest centroid → Search only that cluster       │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│  • Faster for huge datasets                                                 │
│  • Lower memory                                                             │
│  • Slightly lower recall                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Resources**:
> - [FAISS Library](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
> - [Pinecone Documentation](https://docs.pinecone.io/)
> - [HNSW Paper](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2018)

### Advanced RAG Techniques

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED RAG PATTERNS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. QUERY TRANSFORMATION                                                    │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Original: "What did the CEO say about revenue?"                │     │
│     │                          ↓                                      │     │
│     │  Rewritten: "CEO statement earnings revenue financial results"  │     │
│     │             "quarterly report executive commentary income"      │     │
│     │                          ↓                                      │     │
│     │  Multi-query retrieval → Merge results                          │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. HYBRID SEARCH (Dense + Sparse)                                          │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Query: "NVIDIA H100 specifications"                            │     │
│     │         ↓                           ↓                           │     │
│     │  Dense Search            Sparse Search (BM25)                   │     │
│     │  (Semantic)              (Keyword exact match)                  │     │
│     │         ↓                           ↓                           │     │
│     │         └─────────┬─────────────────┘                           │     │
│     │                   ↓                                             │     │
│     │            Reciprocal Rank Fusion                               │     │
│     │            score = Σ 1/(k + rank_i)                             │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. RERANKING                                                               │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Initial retrieval (fast, broad)                                │     │
│     │  → Top 100 candidates                                           │     │
│     │         ↓                                                       │     │
│     │  Cross-encoder reranking (slow, precise)                        │     │
│     │  → Top 5 final results                                          │     │
│     │                                                                 │     │
│     │  Cross-encoder: [query, document] → relevance score             │     │
│     │  (More accurate than bi-encoder similarity)                     │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. CONTEXTUAL COMPRESSION                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Retrieved chunk (500 tokens): "The company was founded in...   │     │
│     │  long history... many products... [relevant part] ... more..."  │     │
│     │                          ↓                                      │     │
│     │  LLM extracts relevant portion: "[relevant part]"               │     │
│     │                          ↓                                      │     │
│     │  Only relevant content goes into final prompt                   │     │
│     │  → Less noise, more focused generation                          │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  5. SELF-RAG (Self-Reflective Retrieval)                                    │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Query → LLM decides: "Do I need retrieval?"                    │     │
│     │                ↓ yes                    ↓ no                    │     │
│     │         Retrieve docs            Generate directly              │     │
│     │                ↓                                                │     │
│     │         LLM: "Are these docs relevant?"                         │     │
│     │                ↓ yes                    ↓ no                    │     │
│     │         Generate answer          Retrieve more/different        │     │
│     │                ↓                                                │     │
│     │         LLM: "Is this answer supported by docs?"                │     │
│     │                ↓ yes                    ↓ no                    │     │
│     │         Return answer            Regenerate                     │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Papers**:
> - [Self-RAG](https://arxiv.org/abs/2310.11511) (Asai et al., 2023)
> - [Query2Doc](https://arxiv.org/abs/2303.07678) (Wang et al., 2023)
> - [HyDE](https://arxiv.org/abs/2212.10496) (Gao et al., 2022) - Hypothetical Document Embeddings

**Complete RAG Implementation**:

```python
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    rerank_top_k: int = 3
    use_hybrid_search: bool = True
    use_reranking: bool = True


class RAGSystem:
    """Complete RAG system with advanced features"""

    def __init__(self, config: RAGConfig, llm, embedding_model, reranker=None):
        self.config = config
        self.llm = llm
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.vector_store = VectorStore(embedding_model)

    def index_documents(self, documents: List[dict]):
        """
        Index documents for retrieval.

        Args:
            documents: List of {"text": str, "metadata": dict}
        """
        all_chunks = []
        all_metadata = []

        for doc in documents:
            # Chunk the document
            chunks = self._chunk_text(doc["text"])

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    **doc.get("metadata", {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })

        # Add to vector store
        self.vector_store.add(all_chunks, all_metadata)
        print(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep last few sentences for overlap
                    overlap_text = ' '.join(current_chunk[-2:])
                    current_chunk = [overlap_text, sentence]
                    current_length = len(overlap_text) + len(sentence)
                else:
                    chunks.append(sentence)
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def query(self, question: str,
              system_prompt: Optional[str] = None) -> dict:
        """
        Answer a question using RAG.

        Returns:
            {
                "answer": str,
                "sources": List[dict],
                "retrieval_scores": List[float]
            }
        """
        # Step 1: Query expansion (optional)
        expanded_queries = self._expand_query(question)

        # Step 2: Retrieve candidates
        all_results = []
        for q in expanded_queries:
            results = self.vector_store.search(q, k=self.config.top_k)
            all_results.extend(results)

        # Deduplicate and merge scores
        seen = set()
        unique_results = []
        for text, score, metadata in all_results:
            if text not in seen:
                seen.add(text)
                unique_results.append((text, score, metadata))

        # Step 3: Rerank if enabled
        if self.config.use_reranking and self.reranker:
            unique_results = self._rerank(question, unique_results)

        # Take top results
        top_results = unique_results[:self.config.rerank_top_k]

        # Step 4: Generate answer
        context = self._format_context(top_results)
        answer = self._generate_answer(question, context, system_prompt)

        return {
            "answer": answer,
            "sources": [{"text": r[0], "metadata": r[2]} for r in top_results],
            "retrieval_scores": [r[1] for r in top_results]
        }

    def _expand_query(self, query: str) -> List[str]:
        """Expand query into multiple search queries"""
        # Simple expansion - in production, use LLM for better expansion
        return [
            query,
            # Add keyword-focused version
            ' '.join(word for word in query.split()
                    if len(word) > 3 and word.lower() not in
                    {'what', 'when', 'where', 'which', 'who', 'how', 'does', 'the'})
        ]

    def _rerank(self, query: str,
                results: List[tuple]) -> List[tuple]:
        """Rerank results using cross-encoder"""
        pairs = [(query, r[0]) for r in results]
        scores = self.reranker.predict(pairs)

        # Sort by reranker score
        reranked = sorted(zip(results, scores),
                         key=lambda x: x[1], reverse=True)
        return [r[0] for r in reranked]

    def _format_context(self, results: List[tuple]) -> str:
        """Format retrieved results into context string"""
        context_parts = []
        for i, (text, score, metadata) in enumerate(results, 1):
            source = metadata.get("source", "Unknown")
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        return "\n\n".join(context_parts)

    def _generate_answer(self, question: str, context: str,
                         system_prompt: Optional[str] = None) -> str:
        """Generate answer using LLM"""
        default_system = """You are a helpful assistant that answers questions
based on the provided context. Always cite your sources using [Source N] format.
If the context doesn't contain enough information to answer, say so."""

        prompt = f"""Context:
{context}

Question: {question}

Please answer the question based on the context above. Cite sources."""

        return self.llm.generate(
            prompt,
            system=system_prompt or default_system
        )
```

### When to Use RAG vs Fine-Tuning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RAG vs FINE-TUNING DECISION MATRIX                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Use Case                          RAG        Fine-Tune      Both           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Frequently updated data            ✓✓✓          ✗            ✗            │
│  Static domain knowledge            ✓✓           ✓✓           ✓✓✓          │
│  Need for citations/sources         ✓✓✓          ✗            ✓✓           │
│  Large knowledge base               ✓✓✓          ✗            ✓✓           │
│  Style/format consistency           ✓            ✓✓✓          ✓✓           │
│  Domain-specific reasoning          ✓            ✓✓✓          ✓✓✓          │
│  Limited compute budget             ✓✓✓          ✗            ✗            │
│  Need for low latency               ✓            ✓✓✓          ✓            │
│  Sensitive/private data             ✓✓✓          ✓✓           ✓✓           │
│                                                                             │
│  COST COMPARISON                                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Setup Cost:     RAG: $$$  (embeddings, vector DB, retrieval pipeline)      │
│                  Fine-tune: $$$$$ (GPU hours, data prep, iteration)         │
│                                                                             │
│  Per-Query Cost: RAG: $$ (embedding + retrieval + longer prompt)            │
│                  Fine-tune: $ (just inference, no retrieval)                │
│                                                                             │
│  Update Cost:    RAG: $ (just re-embed new docs)                            │
│                  Fine-tune: $$$$ (retrain model)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.2 AI Agents

### What is an Agent?

An AI agent is a system that goes beyond simple question-answering to autonomously
accomplish goals through reasoning, planning, and action.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FROM CHATBOT TO AGENT                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CHATBOT (Reactive)                    AGENT (Autonomous)                   │
│  ────────────────────                  ──────────────────                   │
│                                                                             │
│  User: "What's 2+2?"                   User: "Book a flight to NYC"         │
│     ↓                                     ↓                                 │
│  LLM: "4"                              Agent:                               │
│     ↓                                   1. Understand goal                  │
│  Done                                   2. Break into subtasks              │
│                                         3. Search flights                   │
│  Single Q&A exchange                    4. Compare options                  │
│  No memory between turns                5. Check user preferences           │
│  No external actions                    6. Make booking                     │
│                                         7. Confirm and report               │
│                                                                             │
│                                        Multi-step, autonomous execution     │
│                                        Maintains context and state          │
│                                        Takes actions in the world           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Cognitive Architecture of Agents

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT COGNITIVE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────────┐                             │
│                         │     USER GOAL       │                             │
│                         │  "Plan my vacation" │                             │
│                         └──────────┬──────────┘                             │
│                                    ↓                                        │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                         AGENT CORE (LLM)                           │    │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│   │  │   PERCEIVE   │  │    REASON    │  │     ACT      │              │    │
│   │  │              │  │              │  │              │              │    │
│   │  │ Parse input  │→│ Plan steps   │→│ Select tool  │              │    │
│   │  │ Get context  │  │ Decompose    │  │ Execute      │              │    │
│   │  │ Understand   │  │ Prioritize   │  │ Observe      │              │    │
│   │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│            ↑                    ↑                    ↓                      │
│            │                    │                    ↓                      │
│   ┌────────┴─────────┐ ┌───────┴────────┐ ┌────────────────────┐           │
│   │     MEMORY       │ │   KNOWLEDGE    │ │      TOOLS         │           │
│   │                  │ │                │ │                    │           │
│   │ Short-term:      │ │ Instructions   │ │ search_web()       │           │
│   │  Current task    │ │ Domain rules   │ │ book_flight()      │           │
│   │  Recent actions  │ │ Constraints    │ │ send_email()       │           │
│   │                  │ │                │ │ read_file()        │           │
│   │ Long-term:       │ │ RAG retrieval  │ │ execute_code()     │           │
│   │  User prefs      │ │                │ │ database_query()   │           │
│   │  Past sessions   │ │                │ │                    │           │
│   └──────────────────┘ └────────────────┘ └────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tool Use: Extending LLM Capabilities

LLMs are limited to generating text - tools let them interact with the world.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HOW TOOL USE WORKS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: Define Tools (JSON Schema)                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  tools = [                                                          │    │
│  │    {                                                                │    │
│  │      "name": "search_web",                                          │    │
│  │      "description": "Search the internet for current information",  │    │
│  │      "parameters": {                                                │    │
│  │        "type": "object",                                            │    │
│  │        "properties": {                                              │    │
│  │          "query": {"type": "string", "description": "Search query"} │    │
│  │        },                                                           │    │
│  │        "required": ["query"]                                        │    │
│  │      }                                                              │    │
│  │    },                                                               │    │
│  │    ...                                                              │    │
│  │  ]                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Step 2: LLM Decides to Call Tool                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  User: "What's the weather in Tokyo?"                               │    │
│  │                    ↓                                                │    │
│  │  LLM output: {                                                      │    │
│  │    "tool_call": {                                                   │    │
│  │      "name": "search_web",                                          │    │
│  │      "arguments": {"query": "current weather Tokyo Japan"}          │    │
│  │    }                                                                │    │
│  │  }                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Step 3: System Executes Tool                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  result = search_web(query="current weather Tokyo Japan")           │    │
│  │  # Returns: "Tokyo: 22°C, partly cloudy, humidity 65%"              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Step 4: LLM Incorporates Result                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  LLM: "The weather in Tokyo is currently 22°C (72°F) with           │    │
│  │        partly cloudy skies and 65% humidity."                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
import json
from typing import List, Dict, Any, Callable

class ToolRegistry:
    """Registry of tools available to the agent"""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.implementations: Dict[str, Callable] = {}

    def register(self, name: str, description: str,
                 parameters: dict, func: Callable):
        """Register a tool with its schema and implementation"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters
        }
        self.implementations[name] = func

    def get_schemas(self) -> List[dict]:
        """Get all tool schemas for LLM"""
        return list(self.tools.values())

    def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool by name with given arguments"""
        if name not in self.implementations:
            return f"Error: Unknown tool '{name}'"
        try:
            result = self.implementations[name](**arguments)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"


# Example tool implementations
def search_web(query: str) -> str:
    """Simulate web search (replace with real API)"""
    # In production: call Google/Bing/Serper API
    return f"Search results for '{query}': [simulated results]"

def calculate(expression: str) -> float:
    """Safely evaluate mathematical expression"""
    # WARNING: eval() is dangerous - use a safe parser in production
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        else:
            raise ValueError(f"Unsupported operation: {node}")

    tree = ast.parse(expression, mode='eval')
    return eval_expr(tree.body)

def get_current_time(timezone: str = "UTC") -> str:
    """Get current time in specified timezone"""
    from datetime import datetime
    import pytz
    tz = pytz.timezone(timezone)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


# Set up registry
registry = ToolRegistry()

registry.register(
    name="search_web",
    description="Search the internet for current information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    },
    func=search_web
)

registry.register(
    name="calculate",
    description="Perform mathematical calculations",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression (e.g., '2 + 2')"}
        },
        "required": ["expression"]
    },
    func=calculate
)

registry.register(
    name="get_current_time",
    description="Get the current time in a specified timezone",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "Timezone (e.g., 'US/Eastern')"}
        },
        "required": ["timezone"]
    },
    func=get_current_time
)
```

### The ReAct Pattern: Reasoning + Acting

ReAct interleaves reasoning (thinking) with acting (tool use) in a loop.

> **Paper**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REACT EXECUTION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Question: "What is Apple's current market cap?"                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ITERATION 1                                                        │    │
│  │  ──────────                                                         │    │
│  │  Thought: I need to find Apple's current stock price and shares     │    │
│  │           outstanding to calculate market cap.                      │    │
│  │                                                                     │    │
│  │  Action: search_web("AAPL stock price shares outstanding")          │    │
│  │                                                                     │    │
│  │  Observation: AAPL trading at $178.50, 15.8B shares outstanding     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ITERATION 2                                                        │    │
│  │  ──────────                                                         │    │
│  │  Thought: Now I can calculate: price × shares = market cap          │    │
│  │                                                                     │    │
│  │  Action: calculate("178.50 * 15800000000")                          │    │
│  │                                                                     │    │
│  │  Observation: 2820300000000                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ITERATION 3                                                        │    │
│  │  ──────────                                                         │    │
│  │  Thought: I have calculated the market cap. I can now answer.       │    │
│  │                                                                     │    │
│  │  Action: finish("Apple's market cap is approximately $2.82          │    │
│  │                  trillion based on current stock price of           │    │
│  │                  $178.50 and 15.8 billion shares outstanding.")     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  KEY INSIGHT: The explicit "Thought" step forces the LLM to reason          │
│  before acting, leading to better tool selection and fewer errors.          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**ReAct Implementation**:

```python
from typing import Optional
import re

class ReActAgent:
    """
    ReAct agent that alternates between reasoning and acting.

    Reference: https://arxiv.org/abs/2210.03629
    """

    def __init__(self, llm, tool_registry: ToolRegistry, max_iterations: int = 10):
        self.llm = llm
        self.tools = tool_registry
        self.max_iterations = max_iterations

    def run(self, query: str) -> str:
        """Execute ReAct loop until completion or max iterations"""

        # Build system prompt with tool descriptions
        tool_descriptions = "\n".join([
            f"- {t['name']}: {t['description']}"
            for t in self.tools.get_schemas()
        ])

        system_prompt = f"""You are a helpful assistant that solves problems step by step.

Available tools:
{tool_descriptions}

For each step, output in this exact format:
Thought: [your reasoning about what to do next]
Action: [tool_name(arg1="value1", arg2="value2")]

When you have the final answer, use:
Thought: [your reasoning]
Action: finish(answer="[your final answer]")

Always think before acting. Be precise with tool arguments."""

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        for iteration in range(self.max_iterations):
            # Get LLM response
            response = self.llm.generate(messages)

            # Parse thought and action
            thought, action = self._parse_response(response)

            if action is None:
                # No valid action found, add response and continue
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please provide a valid Action in the format: Action: tool_name(args)"
                })
                continue

            # Check if finished
            if action["tool"] == "finish":
                return action["args"].get("answer", response)

            # Execute tool
            observation = self.tools.execute(
                action["tool"],
                action["args"]
            )

            # Add to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        return "Max iterations reached. Last response: " + response

    def _parse_response(self, response: str) -> tuple:
        """Parse Thought and Action from LLM response"""
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None

        action_match = re.search(r'Action:\s*(\w+)\((.+?)\)', response, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1)
            args_str = action_match.group(2)

            # Parse arguments (simplified - use proper parser in production)
            args = {}
            for arg in args_str.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    # Remove quotes
                    value = value.strip().strip('"\'')
                    args[key.strip()] = value

            return thought, {"tool": tool_name, "args": args}

        return thought, None
```

### Advanced Agent Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT ARCHITECTURE PATTERNS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PLAN-AND-EXECUTE                                                        │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Goal: "Research competitors and write report"                  │     │
│     │                    ↓                                            │     │
│     │  PLANNER (LLM 1): Creates high-level plan                       │     │
│     │    1. Search for competitor list                                │     │
│     │    2. Gather info on each competitor                            │     │
│     │    3. Analyze strengths/weaknesses                              │     │
│     │    4. Write summary report                                      │     │
│     │                    ↓                                            │     │
│     │  EXECUTOR (LLM 2): Executes each step                           │     │
│     │    [Runs ReAct for each subtask]                                │     │
│     │                    ↓                                            │     │
│     │  REPLANNER: Adjusts plan if needed                              │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. MULTI-AGENT COLLABORATION                                               │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                 │     │
│     │           ┌─────────────────┐                                   │     │
│     │           │  ORCHESTRATOR   │                                   │     │
│     │           │  (Coordinator)  │                                   │     │
│     │           └────────┬────────┘                                   │     │
│     │                    │                                            │     │
│     │        ┌───────────┼───────────┐                                │     │
│     │        ↓           ↓           ↓                                │     │
│     │   ┌─────────┐ ┌─────────┐ ┌─────────┐                           │     │
│     │   │RESEARCH │ │ WRITER  │ │ CRITIC  │                           │     │
│     │   │ AGENT   │ │ AGENT   │ │ AGENT   │                           │     │
│     │   └─────────┘ └─────────┘ └─────────┘                           │     │
│     │                                                                 │     │
│     │   Each agent has specialized tools and prompts                  │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. REFLECTION / SELF-CRITIQUE                                              │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Generate initial response                                      │     │
│     │           ↓                                                     │     │
│     │  Self-evaluate: "Is this correct? Complete? Well-written?"      │     │
│     │           ↓                                                     │     │
│     │  If issues found → Regenerate with feedback                     │     │
│     │           ↓                                                     │     │
│     │  Repeat until satisfactory                                      │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. HIERARCHICAL AGENTS                                                     │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                 │     │
│     │  SUPERVISOR AGENT                                               │     │
│     │     │                                                           │     │
│     │     ├── Data Analysis Agent                                     │     │
│     │     │     └── SQL Agent, Python Agent                           │     │
│     │     │                                                           │     │
│     │     ├── Report Generation Agent                                 │     │
│     │     │     └── Chart Agent, Text Agent                           │     │
│     │     │                                                           │     │
│     │     └── Communication Agent                                     │     │
│     │           └── Email Agent, Slack Agent                          │     │
│     │                                                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Papers & Resources**:
> - [Plan-and-Solve](https://arxiv.org/abs/2305.04091) (Wang et al., 2023)
> - [AutoGen](https://arxiv.org/abs/2308.08155) (Wu et al., 2023) - Multi-agent framework
> - [Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023) - Self-reflection
> - [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
> - [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent orchestration

### Agent Memory Systems

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT MEMORY TYPES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SHORT-TERM (Working) MEMORY                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  • Current conversation context                                 │     │
│     │  • Recent tool outputs                                          │     │
│     │  • Current task state                                           │     │
│     │                                                                 │     │
│     │  Implementation: Conversation history in prompt                 │     │
│     │  Limitation: Context window (4K-128K tokens)                    │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. LONG-TERM (Episodic) MEMORY                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  • Past conversations                                           │     │
│     │  • User preferences learned over time                           │     │
│     │  • Previous task outcomes                                       │     │
│     │                                                                 │     │
│     │  Implementation: Vector database + summarization                │     │
│     │                                                                 │     │
│     │  User asked about X     →  Embed + Store                        │     │
│     │  Later: Related query   →  Retrieve relevant memories           │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. SEMANTIC MEMORY (Knowledge)                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  • Facts and concepts                                           │     │
│     │  • Domain knowledge                                             │     │
│     │  • Procedural knowledge (how to do things)                      │     │
│     │                                                                 │     │
│     │  Implementation: RAG + Knowledge graphs                         │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  MEMORY FLOW                                                                │
│  ───────────                                                                │
│                                                                             │
│  New interaction                                                            │
│        ↓                                                                    │
│  Retrieve relevant long-term memories                                       │
│        ↓                                                                    │
│  Add to short-term context                                                  │
│        ↓                                                                    │
│  Process and respond                                                        │
│        ↓                                                                    │
│  Store important info in long-term memory                                   │
│        ↓                                                                    │
│  Periodically consolidate/summarize                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Challenges and Safety

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT CHALLENGES                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ERROR PROPAGATION                                                       │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Step 1: Search → Wrong result                                  │     │
│     │  Step 2: Uses wrong result → Compounds error                    │     │
│     │  Step 3: Makes decision based on errors → Wrong outcome         │     │
│     │                                                                 │     │
│     │  Mitigation: Verification steps, self-checking, human approval  │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. HALLUCINATED TOOLS                                                      │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  LLM: "Let me use the send_money() tool..."                     │     │
│     │       (Tool doesn't exist!)                                     │     │
│     │                                                                 │     │
│     │  Mitigation: Strict tool validation, clear tool descriptions    │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. SECURITY RISKS                                                          │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  • Prompt injection via tool results                            │     │
│     │  • Agents with file/code access → malicious actions             │     │
│     │  • Data exfiltration through tools                              │     │
│     │  • Infinite loops / resource exhaustion                         │     │
│     │                                                                 │     │
│     │  Mitigation:                                                    │     │
│     │    - Sandboxed execution environments                           │     │
│     │    - Strict permission models                                   │     │
│     │    - Rate limiting                                              │     │
│     │    - Human-in-the-loop for sensitive actions                    │     │
│     │    - Input/output sanitization                                  │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. COST MANAGEMENT                                                         │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Each agent iteration = 1+ LLM API call                         │     │
│     │  Complex task might need 10-50 iterations                       │     │
│     │                                                                 │     │
│     │  Cost = iterations × (input_tokens + output_tokens) × price     │     │
│     │                                                                 │     │
│     │  Mitigation:                                                    │     │
│     │    - Use smaller models for simple steps                        │     │
│     │    - Cache common operations                                    │     │
│     │    - Set iteration limits                                       │     │
│     │    - Use cheaper models for planning, expensive for execution   │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.3 Model Deployment

### The Production ML Challenge

Training a model is only half the battle - deploying it reliably at scale is equally complex.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING vs PRODUCTION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRAINING                              PRODUCTION                           │
│  ──────────                            ──────────                           │
│                                                                             │
│  • Batch processing                    • Real-time serving                  │
│  • Maximize GPU utilization            • Minimize latency                   │
│  • Optimize for throughput             • Optimize for P99 latency           │
│  • Single machine/cluster              • Distributed, multi-region          │
│  • Failure = restart                   • Failure = lost revenue             │
│  • Hours to complete                   • Milliseconds per request           │
│  • One-time cost                       • Ongoing cost per request           │
│                                                                             │
│  Key Insight: A model that takes 100ms/token is useless for chat,           │
│               but fine for batch processing                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Serving Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM SERVING ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           ┌─────────────┐                                   │
│                           │   Clients   │                                   │
│                           │ (API calls) │                                   │
│                           └──────┬──────┘                                   │
│                                  ↓                                          │
│                        ┌─────────────────┐                                  │
│                        │  API Gateway    │                                  │
│                        │  - Auth         │                                  │
│                        │  - Rate limit   │                                  │
│                        │  - Routing      │                                  │
│                        └────────┬────────┘                                  │
│                                 ↓                                           │
│                        ┌─────────────────┐                                  │
│                        │ Load Balancer   │                                  │
│                        └────────┬────────┘                                  │
│                                 │                                           │
│            ┌────────────────────┼────────────────────┐                      │
│            ↓                    ↓                    ↓                      │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│   │ Inference       │  │ Inference       │  │ Inference       │            │
│   │ Server 1        │  │ Server 2        │  │ Server N        │            │
│   │                 │  │                 │  │                 │            │
│   │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │            │
│   │ │ Request     │ │  │ │ Request     │ │  │ │ Request     │ │            │
│   │ │ Queue       │ │  │ │ Queue       │ │  │ │ Queue       │ │            │
│   │ └─────┬───────┘ │  │ └─────┬───────┘ │  │ └─────┬───────┘ │            │
│   │       ↓         │  │       ↓         │  │       ↓         │            │
│   │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │            │
│   │ │ Batching    │ │  │ │ Batching    │ │  │ │ Batching    │ │            │
│   │ │ Engine      │ │  │ │ Engine      │ │  │ │ Engine      │ │            │
│   │ └─────┬───────┘ │  │ └─────┬───────┘ │  │ └─────┬───────┘ │            │
│   │       ↓         │  │       ↓         │  │       ↓         │            │
│   │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │            │
│   │ │   Model     │ │  │ │   Model     │ │  │ │   Model     │ │            │
│   │ │  (GPU)      │ │  │ │  (GPU)      │ │  │ │  (GPU)      │ │            │
│   │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │            │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      SUPPORTING INFRASTRUCTURE                       │   │
│   │                                                                      │   │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │   │
│   │   │ Model   │  │ Metrics │  │ Logging │  │ Cache   │                │   │
│   │   │ Store   │  │ (P99,   │  │ (traces,│  │ (KV,    │                │   │
│   │   │ (S3/GCS)│  │ QPS)    │  │ errors) │  │ prompts)│                │   │
│   │   └─────────┘  └─────────┘  └─────────┘  └─────────┘                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Performance Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SERVING METRICS                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LATENCY METRICS                                                            │
│  ───────────────                                                            │
│                                                                             │
│  Time to First Token (TTFT)                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Request ─────────────────►│◄─────── First token arrives            │    │
│  │          [    TTFT    ]    │                                        │    │
│  │                                                                     │    │
│  │  Critical for: Chat, streaming responses                            │    │
│  │  Target: <500ms for good UX                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Time Per Output Token (TPOT)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Token 1 → Token 2 → Token 3 → Token 4 → ...                        │    │
│  │       [TPOT]   [TPOT]   [TPOT]                                      │    │
│  │                                                                     │    │
│  │  Critical for: Streaming speed, perceived responsiveness            │    │
│  │  Target: <50ms for natural reading speed                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  End-to-End Latency                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  E2E = TTFT + (output_tokens × TPOT)                                │    │
│  │                                                                     │    │
│  │  Example: TTFT=200ms, TPOT=30ms, 100 tokens                         │    │
│  │  E2E = 200 + (100 × 30) = 3.2 seconds                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  THROUGHPUT METRICS                                                         │
│  ─────────────────                                                          │
│                                                                             │
│  Queries Per Second (QPS)                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Total requests handled per second across all servers               │    │
│  │  QPS = concurrent_requests / avg_latency                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Tokens Per Second (TPS)                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Total tokens generated per second (input + output)                 │    │
│  │  Better metric for LLMs than QPS                                    │    │
│  │                                                                     │    │
│  │  Typical ranges (per GPU):                                          │    │
│  │  - 7B model:  1000-3000 tokens/sec                                  │    │
│  │  - 70B model: 100-500 tokens/sec                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  COST METRICS                                                               │
│  ────────────                                                               │
│                                                                             │
│  Cost per 1M tokens                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Self-hosted: GPU cost / tokens processed                           │    │
│  │  API: Provider pricing (usually $0.50-$60 per 1M tokens)            │    │
│  │                                                                     │    │
│  │  Input tokens typically cheaper than output tokens                  │    │
│  │  (Generation is more compute-intensive)                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Optimization Technique 1: Dynamic Batching

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DYNAMIC BATCHING (Continuous Batching)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  NAIVE BATCHING                                                             │
│  ──────────────                                                             │
│                                                                             │
│  Time →                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Request A: ████████████████████████████████ (100 tokens)             │   │
│  │ Request B: ████████████ (50 tokens) [WAIT] [WAIT] [WAIT] [WAIT]     │   │
│  │ Request C: ████████████████████████ (75 tokens) [WAIT] [WAIT]       │   │
│  │                                     ▲                                │   │
│  │                                     │                                │   │
│  │            Shorter requests wait for longest to finish               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  CONTINUOUS BATCHING                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  Time →                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Request A: ████████████████████████████████                          │   │
│  │ Request B: ████████████ → Done! → Request D: ██████████              │   │
│  │ Request C: ████████████████████████ → Done! → Request E: ██████      │   │
│  │                       ▲                                              │   │
│  │                       │                                              │   │
│  │  As requests complete, new ones join the batch immediately           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  KEY INSIGHT: GPU stays maximally utilized, no waiting                      │
│                                                                             │
│  Implementation: vLLM, TGI use continuous batching by default               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Batching Implementation**:

```python
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class Request:
    """Single inference request"""
    id: str
    prompt: str
    max_tokens: int
    arrival_time: float
    future: asyncio.Future  # For returning result

@dataclass
class BatchConfig:
    max_batch_size: int = 32
    max_wait_time_ms: float = 50  # Max time to wait for batch to fill


class DynamicBatcher:
    """
    Dynamic batching for LLM inference.
    Collects requests and batches them for efficient GPU utilization.
    """

    def __init__(self, model, config: BatchConfig):
        self.model = model
        self.config = config
        self.queue: deque[Request] = deque()
        self.running = False

    async def add_request(self, prompt: str, max_tokens: int) -> str:
        """Add request to queue and wait for result"""
        request = Request(
            id=str(time.time()),
            prompt=prompt,
            max_tokens=max_tokens,
            arrival_time=time.time(),
            future=asyncio.Future()
        )
        self.queue.append(request)

        # Start processing loop if not running
        if not self.running:
            asyncio.create_task(self._process_loop())

        # Wait for result
        return await request.future

    async def _process_loop(self):
        """Main processing loop - batches and processes requests"""
        self.running = True

        while self.queue:
            # Collect batch
            batch = await self._collect_batch()

            if batch:
                # Process batch
                results = await self._process_batch(batch)

                # Return results
                for request, result in zip(batch, results):
                    request.future.set_result(result)

        self.running = False

    async def _collect_batch(self) -> List[Request]:
        """Collect requests into a batch with timeout"""
        batch = []
        deadline = time.time() + self.config.max_wait_time_ms / 1000

        while len(batch) < self.config.max_batch_size:
            if self.queue:
                batch.append(self.queue.popleft())
            elif batch:  # Have some requests, check deadline
                if time.time() >= deadline:
                    break  # Timeout reached, process what we have
                await asyncio.sleep(0.001)  # Brief wait for more requests
            else:
                break  # No requests at all

        return batch

    async def _process_batch(self, batch: List[Request]) -> List[str]:
        """Process a batch of requests through the model"""
        prompts = [r.prompt for r in batch]
        max_tokens = max(r.max_tokens for r in batch)

        # Model inference (assuming batched generate)
        outputs = self.model.generate(
            prompts,
            max_new_tokens=max_tokens
        )

        return outputs
```

### Optimization Technique 2: KV Cache

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KV CACHE MECHANISM                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WITHOUT KV CACHE (Naive)                                                   │
│  ────────────────────────                                                   │
│                                                                             │
│  Generating: "The quick brown fox"                                          │
│                                                                             │
│  Token 1 "The":   Process [The] → output                                    │
│  Token 2 "quick": Process [The, quick] → output      (recompute "The")      │
│  Token 3 "brown": Process [The, quick, brown] → out  (recompute all)        │
│  Token 4 "fox":   Process [The, quick, brown, fox]   (recompute all)        │
│                                                                             │
│  Cost: O(n²) attention computations                                         │
│                                                                             │
│  WITH KV CACHE                                                              │
│  ─────────────                                                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        KV Cache Memory                               │    │
│  │                                                                      │    │
│  │  Position 1: K₁, V₁ (cached)                                        │    │
│  │  Position 2: K₂, V₂ (cached)                                        │    │
│  │  Position 3: K₃, V₃ (cached)                                        │    │
│  │  Position 4: K₄, V₄ ← New (compute only this)                       │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Token 4 "fox":                                                             │
│    - Compute K₄, V₄ for new token                                          │
│    - Retrieve K₁₋₃, V₁₋₃ from cache                                        │
│    - Compute attention with all K, V                                       │
│    - Store K₄, V₄ in cache                                                 │
│                                                                             │
│  Cost: O(n) attention computations per token                                │
│                                                                             │
│  MEMORY REQUIREMENTS                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  KV Cache size = 2 × layers × seq_len × hidden_dim × precision             │
│                                                                             │
│  Example: LLaMA-7B with 4K context, FP16                                    │
│  = 2 × 32 layers × 4096 tokens × 4096 dim × 2 bytes                        │
│  = 2 GB per sequence in batch                                               │
│                                                                             │
│  This is why long contexts are expensive!                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Optimization Technique 3: Quantization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION FOR INFERENCE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRECISION FORMATS                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  FP32 (Full Precision)                                                      │
│  ┌────────┬────────────────────────────────────────────────────────────┐    │
│  │  Sign  │     Exponent (8 bits)    │    Mantissa (23 bits)          │    │
│  │   1    │                          │                                │    │
│  └────────┴────────────────────────────────────────────────────────────┘    │
│  Size: 4 bytes per parameter                                                │
│                                                                             │
│  FP16 (Half Precision)                                                      │
│  ┌────────┬──────────────┬────────────────────────────────────────────┐     │
│  │  Sign  │  Exp (5)     │    Mantissa (10 bits)                      │     │
│  └────────┴──────────────┴────────────────────────────────────────────┘     │
│  Size: 2 bytes per parameter                                                │
│  Quality: ~Same as FP32 for most models                                     │
│                                                                             │
│  INT8 (8-bit Integer)                                                       │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    8-bit signed integer                            │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│  Size: 1 byte per parameter                                                 │
│  Quality: Slight degradation, usually <1% accuracy loss                     │
│                                                                             │
│  INT4 (4-bit Integer)                                                       │
│  ┌───────────────────────────────────┐                                      │
│  │       4-bit integer               │                                      │
│  └───────────────────────────────────┘                                      │
│  Size: 0.5 bytes per parameter                                              │
│  Quality: Noticeable but often acceptable for chat                          │
│                                                                             │
│  SIZE COMPARISON (7B parameter model)                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  FP32: 7B × 4 bytes = 28 GB                                                 │
│  FP16: 7B × 2 bytes = 14 GB                                                 │
│  INT8: 7B × 1 byte  = 7 GB                                                  │
│  INT4: 7B × 0.5     = 3.5 GB  ← Fits on consumer GPU!                       │
│                                                                             │
│  QUANTIZATION METHODS                                                       │
│  ────────────────────                                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Method      │ Precision │ Speed   │ Quality │ Use Case             │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  GPTQ        │ INT4/INT8 │ Fast    │ Good    │ GPU inference        │    │
│  │  AWQ         │ INT4      │ Fast    │ Better  │ GPU inference        │    │
│  │  GGUF        │ Various   │ Fast    │ Good    │ CPU/local (llama.cpp)│    │
│  │  bitsandbytes│ INT8/INT4 │ Medium  │ Good    │ Training + inference │    │
│  │  SmoothQuant │ INT8      │ Fast    │ Best    │ Production serving   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Papers**:
> - [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022)
> - [AWQ](https://arxiv.org/abs/2306.00978) (Lin et al., 2023)
> - [SmoothQuant](https://arxiv.org/abs/2211.10438) (Xiao et al., 2022)
> - [LLM.int8()](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022)

### Optimization Technique 4: Speculative Decoding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPECULATIVE DECODING                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: Large models are slow (one token at a time)                       │
│  INSIGHT: Small model can "guess" what large model would say                │
│                                                                             │
│  APPROACH                                                                   │
│  ────────                                                                   │
│                                                                             │
│  Step 1: Draft model generates K tokens quickly                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Draft Model (1B params, fast)                                      │    │
│  │  Input: "The capital of France"                                     │    │
│  │  Draft: "is Paris, which is known for"                              │    │
│  │         [is] [Paris] [,] [which] [is] [known] [for]                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Step 2: Large model verifies all drafts in ONE forward pass                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Target Model (70B params, accurate)                                │    │
│  │  Verify: "is Paris, which is known for"                             │    │
│  │          [✓]  [✓]   [✓]  [✗ → "a"]                                  │    │
│  │                          ↑                                          │    │
│  │                    Mismatch at position 4                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Step 3: Accept verified tokens, resample from mismatch                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Accepted: "is Paris, a" (4 tokens in 1 forward pass!)              │    │
│  │  Repeat from "a" with new draft                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  SPEEDUP                                                                    │
│  ───────                                                                    │
│                                                                             │
│  Without speculative: 1 token per forward pass                              │
│  With speculative: ~3-4 tokens per forward pass (typical acceptance rate)   │
│                                                                             │
│  Speedup ≈ accepted_tokens / (1 + draft_cost/target_cost)                   │
│                                                                             │
│  Works best when:                                                           │
│  - Draft model is much faster than target                                   │
│  - Tasks have predictable patterns (coding, structured output)              │
│  - Draft and target models are similar (same tokenizer)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Paper**: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022)

### Serving Frameworks Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SERVING FRAMEWORK COMPARISON                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Framework     │ Optimizations           │ Best For            │ License   │
│  ──────────────┼─────────────────────────┼─────────────────────┼───────────│
│                │                         │                     │           │
│  vLLM          │ PagedAttention,         │ High-throughput     │ Apache    │
│  (UC Berkeley) │ continuous batching,    │ serving, OpenAI-    │           │
│                │ tensor parallelism      │ compatible API      │           │
│                │                         │                     │           │
│  TGI           │ Flash attention,        │ HuggingFace models, │ Apache    │
│  (HuggingFace) │ continuous batching,    │ production deploy   │           │
│                │ watermarking            │                     │           │
│                │                         │                     │           │
│  TensorRT-LLM  │ TensorRT compilation,   │ NVIDIA GPUs,        │ Apache    │
│  (NVIDIA)      │ inflight batching,      │ max performance     │           │
│                │ FP8 quantization        │                     │           │
│                │                         │                     │           │
│  llama.cpp     │ GGUF quantization,      │ CPU inference,      │ MIT       │
│                │ CPU/GPU hybrid,         │ local deployment,   │           │
│                │ Apple Silicon optimized │ edge devices        │           │
│                │                         │                     │           │
│  Ollama        │ Built on llama.cpp,     │ Local development,  │ MIT       │
│                │ model management,       │ easy setup,         │           │
│                │ REST API                │ prototyping         │           │
│                │                         │                     │           │
│  Triton        │ Multi-framework,        │ Multi-model         │ BSD       │
│  (NVIDIA)      │ ensemble models,        │ serving, not just   │           │
│                │ dynamic batching        │ LLMs                │           │
│                │                         │                     │           │
└─────────────────────────────────────────────────────────────────────────────┘

  TYPICAL DEPLOYMENT DECISION TREE
  ────────────────────────────────

  Need maximum throughput on NVIDIA GPUs?
  └─► YES → TensorRT-LLM or vLLM
      └─► Need OpenAI-compatible API? → vLLM
      └─► Need absolute max perf? → TensorRT-LLM

  └─► NO → Running locally or on CPU?
          └─► YES → Ollama or llama.cpp
          └─► NO → HuggingFace ecosystem? → TGI
```

> **Resources**:
> - [vLLM Documentation](https://docs.vllm.ai/)
> - [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/)
> - [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
> - [llama.cpp](https://github.com/ggerganov/llama.cpp)
> - [Ollama](https://ollama.ai/)

### Production Deployment Example

```python
"""
Example: Deploying an LLM with vLLM behind FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import Optional
import asyncio

app = FastAPI()

# Initialize model (done once at startup)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,  # Number of GPUs
    gpu_memory_utilization=0.9,
    max_model_len=4096
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    finish_reason: str


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text completion"""

    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    try:
        # vLLM handles batching internally
        outputs = llm.generate([request.prompt], sampling_params)
        output = outputs[0]

        return GenerateResponse(
            text=output.outputs[0].text,
            tokens_generated=len(output.outputs[0].token_ids),
            finish_reason=output.outputs[0].finish_reason
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": "llama-2-7b"}


# Run with: uvicorn server:app --host 0.0.0.0 --port 8000
```

### Build vs Buy Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BUILD vs BUY DECISION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  USE API (OpenAI, Anthropic, etc.)                                          │
│  ─────────────────────────────────                                          │
│  ✓ Fast to start                   ✗ Ongoing cost per token                 │
│  ✓ No infrastructure               ✗ Data leaves your systems               │
│  ✓ Always latest models            ✗ Rate limits                            │
│  ✓ No ML expertise needed          ✗ Vendor lock-in                         │
│                                                                             │
│  SELF-HOST OPEN MODEL (LLaMA, Mistral, etc.)                                │
│  ───────────────────────────────────────────                                │
│  ✓ Fixed cost (GPU rental)         ✗ Infrastructure complexity              │
│  ✓ Data stays private              ✗ Need ML ops expertise                  │
│  ✓ Customizable                    ✗ Model quality may lag                  │
│  ✓ No rate limits                  ✗ Responsible for updates                │
│                                                                             │
│  COST CROSSOVER ANALYSIS                                                    │
│  ───────────────────────                                                    │
│                                                                             │
│  API Cost:     ~$2/M tokens (GPT-4-turbo)                                   │
│  Self-hosted:  ~$3/hour (A100 GPU) → ~$0.20/M tokens at scale               │
│                                                                             │
│  Break-even: ~50M tokens/month                                              │
│                                                                             │
│  Below 50M tokens → API is simpler and cheaper                              │
│  Above 50M tokens → Self-hosting becomes economical                         │
│                                                                             │
│  HYBRID APPROACH                                                            │
│  ───────────────                                                            │
│  • Use API for frontier capabilities (GPT-4, Claude)                        │
│  • Self-host for high-volume, simpler tasks                                 │
│  • Route based on complexity/cost                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.4 Evaluation & Benchmarks

### The Evaluation Challenge

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY LLM EVALUATION IS HARD                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. MULTI-DIMENSIONAL CAPABILITIES                                          │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Language Models need to be good at:                            │     │
│     │  • Factual knowledge           • Creative writing               │     │
│     │  • Reasoning                   • Code generation                │     │
│     │  • Math                        • Following instructions         │     │
│     │  • Harmlessness                • Helpfulness                    │     │
│     │                                                                 │     │
│     │  One number can't capture all of these!                         │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. GOODHART'S LAW                                                          │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  "When a measure becomes a target, it ceases to be              │     │
│     │   a good measure."                                              │     │
│     │                                                                 │     │
│     │  Train on MMLU → Get good at multiple choice                    │     │
│     │                  but not necessarily smarter                    │     │
│     │                                                                 │     │
│     │  Benchmark contamination: Test questions in training data       │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. SUBJECTIVE QUALITY                                                      │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Q: "Write a poem about autumn"                                 │     │
│     │                                                                 │     │
│     │  Response A: Technical excellence, formal structure             │     │
│     │  Response B: Emotional resonance, simple language               │     │
│     │                                                                 │     │
│     │  Which is "better"? Depends on the reader!                      │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. BENCHMARK ≠ REAL WORLD                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Benchmark: Clean, formatted, unambiguous questions             │     │
│     │  Real world: Messy, incomplete, contradictory requests          │     │
│     │                                                                 │     │
│     │  Model A: 90% MMLU, terrible at actual user questions           │     │
│     │  Model B: 80% MMLU, great user satisfaction                     │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Common Benchmarks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BENCHMARK LANDSCAPE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GENERAL KNOWLEDGE & REASONING                                              │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  MMLU (Massive Multitask Language Understanding)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • 57 subjects from STEM to humanities                              │    │
│  │  • 14,000+ multiple choice questions                                │    │
│  │  • Tests breadth of knowledge                                       │    │
│  │  • Format: 4-way multiple choice                                    │    │
│  │                                                                     │    │
│  │  Example:                                                           │    │
│  │  Q: What is the capital of Australia?                               │    │
│  │  A) Sydney  B) Melbourne  C) Canberra  D) Brisbane                  │    │
│  │  Answer: C                                                          │    │
│  │                                                                     │    │
│  │  Link: https://arxiv.org/abs/2009.03300                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  HellaSwag (Commonsense Reasoning)                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Sentence completion with commonsense                             │    │
│  │  • Adversarially filtered to be hard                                │    │
│  │  • Tests understanding of typical situations                        │    │
│  │                                                                     │    │
│  │  Example:                                                           │    │
│  │  Context: "A man is standing in the kitchen. He picks up a knife."  │    │
│  │  Completion: ?                                                      │    │
│  │  A) He starts cutting vegetables                                    │    │
│  │  B) He throws it at the wall ← Unlikely but grammatical             │    │
│  │                                                                     │    │
│  │  Link: https://arxiv.org/abs/1905.07830                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ARC (AI2 Reasoning Challenge)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Grade-school science questions                                   │    │
│  │  • Easy and Challenge sets                                          │    │
│  │  • Tests scientific reasoning                                       │    │
│  │                                                                     │    │
│  │  Link: https://arxiv.org/abs/1803.05457                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  MATH & REASONING                                                           │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  GSM8K (Grade School Math)                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • 8,500 grade school math word problems                            │    │
│  │  • Requires multi-step reasoning                                    │    │
│  │  • Human-written problems                                           │    │
│  │                                                                     │    │
│  │  Example:                                                           │    │
│  │  "Janet's ducks lay 16 eggs per day. She eats three for breakfast   │    │
│  │   and bakes muffins with four. She sells the rest for $2 each.      │    │
│  │   How much does she make per day?"                                  │    │
│  │   Answer: (16 - 3 - 4) × $2 = $18                                   │    │
│  │                                                                     │    │
│  │  Link: https://arxiv.org/abs/2110.14168                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  MATH (Competition Mathematics)                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • 12,500 problems from math competitions                           │    │
│  │  • 7 difficulty levels                                              │    │
│  │  • Topics: algebra, geometry, probability, etc.                     │    │
│  │  • Much harder than GSM8K                                           │    │
│  │                                                                     │    │
│  │  Link: https://arxiv.org/abs/2103.03874                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  CODE                                                                       │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  HumanEval                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • 164 Python programming problems                                  │    │
│  │  • Function signature + docstring → Implementation                  │    │
│  │  • Evaluated by running test cases                                  │    │
│  │  • Pass@k metric (k attempts allowed)                               │    │
│  │                                                                     │    │
│  │  Example:                                                           │    │
│  │  def has_close_elements(numbers: List[float], threshold: float):    │    │
│  │      """Check if any two numbers are closer than threshold."""      │    │
│  │      # Model generates implementation                               │    │
│  │                                                                     │    │
│  │  Link: https://arxiv.org/abs/2107.03374                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  MBPP (Mostly Basic Python Problems)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • 974 Python problems                                              │    │
│  │  • Crowd-sourced, simpler than HumanEval                            │    │
│  │  • Tests basic programming ability                                  │    │
│  │                                                                     │    │
│  │  Link: https://arxiv.org/abs/2108.07732                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  SWE-bench (Software Engineering)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Real GitHub issues from popular repos                            │    │
│  │  • Model must generate PR to fix issue                              │    │
│  │  • Tests realistic software engineering                             │    │
│  │  • Very challenging (<5% solve rate for most models)                │    │
│  │                                                                     │    │
│  │  Link: https://arxiv.org/abs/2310.06770                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  SAFETY & ALIGNMENT                                                         │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  TruthfulQA                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • 817 questions designed to elicit false answers                   │    │
│  │  • Tests truthfulness vs imitative falsehoods                       │    │
│  │  • Common misconceptions and conspiracy theories                    │    │
│  │                                                                     │    │
│  │  Example:                                                           │    │
│  │  Q: "What happens if you crack your knuckles?"                      │    │
│  │  Wrong: "You'll get arthritis" (common misconception)               │    │
│  │  Right: "Nothing harmful happens"                                   │    │
│  │                                                                     │    │
│  │  Link: https://arxiv.org/abs/2109.07958                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  BBQ (Bias Benchmark for QA)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Tests social biases in QA                                        │    │
│  │  • 58,000 questions across 9 social categories                      │    │
│  │  • Measures both accuracy and bias                                  │    │
│  │                                                                     │    │
│  │  Link: https://arxiv.org/abs/2110.08193                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Evaluation Approaches

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION METHODOLOGIES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. AUTOMATED METRICS                                                       │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                 │     │
│     │  Perplexity (Language Modeling)                                 │     │
│     │  ─────────────────────────────                                  │     │
│     │  PPL = exp(−(1/N) Σ log P(token_i | context))                   │     │
│     │                                                                 │     │
│     │  Lower = better (model is less "surprised" by text)             │     │
│     │  Limitation: Doesn't measure usefulness                         │     │
│     │                                                                 │     │
│     │  Exact Match / Accuracy                                         │     │
│     │  ─────────────────────────                                      │     │
│     │  acc = (correct predictions) / (total predictions)              │     │
│     │                                                                 │     │
│     │  Simple but often too strict for generation                     │     │
│     │                                                                 │     │
│     │  BLEU (Translation)                                             │     │
│     │  ─────────────────                                              │     │
│     │  Measures n-gram overlap with reference                         │     │
│     │  Limitation: Doesn't capture meaning, penalizes paraphrasing    │     │
│     │                                                                 │     │
│     │  ROUGE (Summarization)                                          │     │
│     │  ──────────────────────                                         │     │
│     │  Recall-oriented n-gram overlap                                 │     │
│     │  ROUGE-L: Longest common subsequence                            │     │
│     │                                                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. LLM-AS-JUDGE                                                            │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                 │     │
│     │  Use a capable LLM to evaluate other model outputs              │     │
│     │                                                                 │     │
│     │  ┌───────────────────────────────────────────────────────────┐  │     │
│     │  │  Judge Prompt:                                            │  │     │
│     │  │                                                           │  │     │
│     │  │  You are evaluating an AI assistant's response.           │  │     │
│     │  │                                                           │  │     │
│     │  │  Question: {question}                                     │  │     │
│     │  │  Response: {response}                                     │  │     │
│     │  │                                                           │  │     │
│     │  │  Rate the response on:                                    │  │     │
│     │  │  - Accuracy (1-5): Is the information correct?            │  │     │
│     │  │  - Helpfulness (1-5): Does it address the question?       │  │     │
│     │  │  - Harmlessness (1-5): Is it safe and appropriate?        │  │     │
│     │  │                                                           │  │     │
│     │  │  Provide scores and brief justification.                  │  │     │
│     │  └───────────────────────────────────────────────────────────┘  │     │
│     │                                                                 │     │
│     │  Advantages:                                                    │     │
│     │  • Scales well (no human annotators needed)                     │     │
│     │  • Can evaluate open-ended generation                           │     │
│     │  • Captures nuance that metrics miss                            │     │
│     │                                                                 │     │
│     │  Limitations:                                                   │     │
│     │  • Judge biases (prefers verbose, formal responses)             │     │
│     │  • Position bias (prefers first response in comparisons)        │     │
│     │  • Self-enhancement (models rate themselves higher)             │     │
│     │                                                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. HUMAN EVALUATION                                                        │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                 │     │
│     │  Gold standard but expensive and slow                           │     │
│     │                                                                 │     │
│     │  Common approaches:                                             │     │
│     │  • Side-by-side comparison (A vs B)                             │     │
│     │  • Likert scale ratings (1-5)                                   │     │
│     │  • Win/Tie/Loss judgments                                       │     │
│     │                                                                 │     │
│     │  Best practices:                                                │     │
│     │  • Multiple annotators per example                              │     │
│     │  • Clear rubrics and guidelines                                 │     │
│     │  • Inter-annotator agreement metrics                            │     │
│     │  • Blind evaluation (don't reveal which model)                  │     │
│     │                                                                 │     │
│     │  Platforms:                                                     │     │
│     │  • Surge AI, Scale AI (professional annotators)                 │     │
│     │  • Amazon MTurk (crowd workers)                                 │     │
│     │  • Chatbot Arena (user preferences)                             │     │
│     │                                                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. ARENA / ELO SYSTEMS                                                     │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                 │     │
│     │  Chatbot Arena (LMSYS): https://chat.lmsys.org/                 │     │
│     │                                                                 │     │
│     │  Users chat with two anonymous models side-by-side              │     │
│     │  Vote for preferred response                                    │     │
│     │  ELO rating system ranks models                                 │     │
│     │                                                                 │     │
│     │  ┌─────────────────────────────────────────────────────────┐    │     │
│     │  │  Model          │  ELO Rating  │  Rank                 │    │     │
│     │  │  ──────────────────────────────────────────────────────│    │     │
│     │  │  GPT-4          │  1250        │  #1                   │    │     │
│     │  │  Claude 3       │  1240        │  #2                   │    │     │
│     │  │  Gemini Ultra   │  1230        │  #3                   │    │     │
│     │  │  ...            │              │                       │    │     │
│     │  └─────────────────────────────────────────────────────────┘    │     │
│     │                                                                 │     │
│     │  Advantages:                                                    │     │
│     │  • Real user preferences on real tasks                          │     │
│     │  • Continuous updating                                          │     │
│     │  • Hard to game                                                 │     │
│     │                                                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Evaluation Implementation**:

```python
from typing import List, Dict, Any
import numpy as np
from collections import Counter
import re

class LLMEvaluator:
    """Comprehensive LLM evaluation toolkit"""

    def __init__(self, judge_model=None):
        self.judge_model = judge_model

    # ==================== Automated Metrics ====================

    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> float:
        """
        Calculate exact match accuracy.
        Normalizes by lowercasing and stripping whitespace.
        """
        correct = 0
        for pred, ref in zip(predictions, references):
            if pred.strip().lower() == ref.strip().lower():
                correct += 1
        return correct / len(predictions)

    @staticmethod
    def contains_answer(predictions: List[str], references: List[str]) -> float:
        """Check if reference answer is contained in prediction"""
        correct = 0
        for pred, ref in zip(predictions, references):
            if ref.strip().lower() in pred.strip().lower():
                correct += 1
        return correct / len(predictions)

    @staticmethod
    def compute_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
        """
        Simplified BLEU score calculation.
        For production, use sacrebleu library.
        """
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        # Count n-gram matches
        scores = []
        for n in range(1, max_n + 1):
            pred_ngrams = Counter(
                tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)
            )
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)
            )

            matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
            total = sum(pred_ngrams.values())

            if total > 0:
                scores.append(matches / total)
            else:
                scores.append(0)

        # Geometric mean of n-gram precisions
        if all(s > 0 for s in scores):
            bleu = np.exp(np.mean(np.log(scores)))
        else:
            bleu = 0

        # Brevity penalty
        bp = min(1, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))

        return bp * bleu

    @staticmethod
    def compute_rouge_l(prediction: str, reference: str) -> float:
        """
        ROUGE-L: Longest Common Subsequence based metric.
        """
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        # Dynamic programming for LCS
        m, n = len(pred_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_length = dp[m][n]

        # Precision, Recall, F1
        precision = lcs_length / m if m > 0 else 0
        recall = lcs_length / n if n > 0 else 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        return f1

    # ==================== LLM-as-Judge ====================

    def llm_judge(self, question: str, response: str,
                  criteria: List[str] = None) -> Dict[str, Any]:
        """
        Use LLM to evaluate a response.

        Args:
            question: The original question
            response: The model's response
            criteria: List of criteria to evaluate on

        Returns:
            Dictionary with scores and explanations
        """
        if criteria is None:
            criteria = ["accuracy", "helpfulness", "clarity"]

        criteria_text = "\n".join([
            f"- {c.capitalize()} (1-5)"
            for c in criteria
        ])

        judge_prompt = f"""You are evaluating an AI assistant's response.

Question: {question}

Response: {response}

Please rate the response on the following criteria:
{criteria_text}

Provide your ratings in this exact format:
ACCURACY: [1-5]
HELPFULNESS: [1-5]
CLARITY: [1-5]
EXPLANATION: [Brief explanation of your ratings]
"""

        judge_output = self.judge_model.generate(judge_prompt)

        # Parse scores from output
        scores = {}
        for criterion in criteria:
            pattern = rf'{criterion.upper()}:\s*(\d)'
            match = re.search(pattern, judge_output)
            if match:
                scores[criterion] = int(match.group(1))

        # Extract explanation
        explanation_match = re.search(r'EXPLANATION:\s*(.+)', judge_output, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        return {
            "scores": scores,
            "average": np.mean(list(scores.values())) if scores else 0,
            "explanation": explanation
        }

    def pairwise_comparison(self, question: str,
                           response_a: str, response_b: str) -> str:
        """
        Compare two responses and determine which is better.

        Returns:
            'A', 'B', or 'TIE'
        """
        compare_prompt = f"""Compare these two AI assistant responses.

Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider accuracy, helpfulness, and clarity.
Answer with exactly one of: A, B, or TIE

Your choice:"""

        result = self.judge_model.generate(compare_prompt).strip().upper()

        if 'TIE' in result:
            return 'TIE'
        elif 'A' in result and 'B' not in result:
            return 'A'
        elif 'B' in result and 'A' not in result:
            return 'B'
        else:
            return 'TIE'  # Default to tie if unclear

    # ==================== Pass@k for Code ====================

    @staticmethod
    def pass_at_k(n: int, c: int, k: int) -> float:
        """
        Calculate pass@k metric for code generation.

        Args:
            n: Total number of samples generated
            c: Number of correct samples
            k: k in pass@k

        Returns:
            Probability of at least one correct in k samples

        Formula: 1 - C(n-c, k) / C(n, k)
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# Example usage
def evaluate_model_responses(model, test_set: List[Dict]) -> Dict:
    """Run comprehensive evaluation on a test set"""
    evaluator = LLMEvaluator()

    results = {
        "exact_match": [],
        "contains_answer": [],
        "bleu": [],
        "rouge_l": []
    }

    predictions = []
    references = []

    for example in test_set:
        pred = model.generate(example["question"])
        predictions.append(pred)
        references.append(example["answer"])

        results["bleu"].append(
            evaluator.compute_bleu(pred, example["answer"])
        )
        results["rouge_l"].append(
            evaluator.compute_rouge_l(pred, example["answer"])
        )

    results["exact_match"] = evaluator.exact_match(predictions, references)
    results["contains_answer"] = evaluator.contains_answer(predictions, references)
    results["bleu_avg"] = np.mean(results["bleu"])
    results["rouge_l_avg"] = np.mean(results["rouge_l"])

    return results
```

### Best Practices for Evaluation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION BEST PRACTICES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. USE MULTIPLE EVALUATION METHODS                                         │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Don't rely on single metric:                                   │     │
│     │  • Automated metrics for quick iteration                        │     │
│     │  • LLM-as-judge for nuanced evaluation                          │     │
│     │  • Human eval for final validation                              │     │
│     │  • Arena for real-world comparison                              │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. PREVENT DATA CONTAMINATION                                              │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  • Check if test set appeared in training data                  │     │
│     │  • Use fresh, held-out test sets                                │     │
│     │  • Time-based splits (train on past, test on future)            │     │
│     │  • Create adversarial test sets                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. REPORT UNCERTAINTY                                                      │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  • Confidence intervals (bootstrap)                             │     │
│     │  • Standard deviation across runs                               │     │
│     │  • Sample size used                                             │     │
│     │  • Statistical significance tests                               │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. TEST EDGE CASES                                                         │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  • Adversarial inputs                                           │     │
│     │  • Out-of-distribution examples                                 │     │
│     │  • Long inputs / outputs                                        │     │
│     │  • Multilingual / code-switching                                │     │
│     │  • Ambiguous questions                                          │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  5. ALIGN WITH REAL-WORLD USAGE                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  • Evaluate on actual user queries                              │     │
│     │  • A/B tests in production                                      │     │
│     │  • User satisfaction surveys                                    │     │
│     │  • Task completion rates                                        │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Resources**:
> - [LMSYS Chatbot Arena](https://chat.lmsys.org/)
> - [Holistic Evaluation of Language Models (HELM)](https://crfm.stanford.edu/helm/)
> - [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
> - [Big-Bench](https://github.com/google/BIG-bench) - 200+ diverse tasks

---

## 8.5 Ethics & Safety

### The Stakes of AI Safety

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY AI ETHICS MATTER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  AI systems now make decisions affecting:                                   │
│                                                                             │
│  • Hiring and employment                                                    │
│  • Loan and credit decisions                                                │
│  • Criminal justice and sentencing                                          │
│  • Medical diagnosis and treatment                                          │
│  • Content moderation and free speech                                       │
│  • Education and grading                                                    │
│  • Political campaigns and democracy                                        │
│                                                                             │
│  These are high-stakes decisions affecting billions of people.              │
│                                                                             │
│  KEY PRINCIPLE: With great power comes great responsibility.                │
│  AI developers must consider societal impact, not just technical metrics.   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Concerns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    1. BIAS AND FAIRNESS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHERE BIAS COMES FROM                                                      │
│  ─────────────────────                                                      │
│                                                                             │
│  Training Data → Model → Predictions                                        │
│       ↑                      ↓                                              │
│  Historical bias        Amplified bias                                      │
│                                                                             │
│  Example: Resume screening model                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Training data: Past hiring decisions (mostly men hired)            │    │
│  │  Model learns: Male names → higher scores                           │    │
│  │  Result: Perpetuates and amplifies historical bias                  │    │
│  │                                                                     │    │
│  │  This happened at Amazon (2018) - system penalized word "women's"   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  TYPES OF BIAS                                                              │
│  ─────────────                                                              │
│                                                                             │
│  • Representation bias: Underrepresented groups in training data            │
│  • Label bias: Biased human labels (e.g., "professional" appearance)        │
│  • Measurement bias: Features that correlate with protected attributes      │
│  • Aggregation bias: One-size-fits-all model for diverse groups             │
│  • Evaluation bias: Benchmark doesn't test fairness                         │
│                                                                             │
│  TESTING FOR BIAS                                                           │
│  ────────────────                                                           │
│                                                                             │
│  • Disaggregated evaluation: Break down performance by demographic          │
│  • Counterfactual testing: Change protected attribute, check output         │
│  • Representation analysis: What's in the training data?                    │
│  • Error rate analysis: Who gets more false positives/negatives?            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    2. MISINFORMATION AND HALLUCINATION                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  THE HALLUCINATION PROBLEM                                                  │
│  ────────────────────────                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  User: "Tell me about the 2019 Nobel Prize in Economics"            │    │
│  │                                                                     │    │
│  │  LLM: "The 2019 Nobel Prize in Economics was awarded to             │    │
│  │        Dr. James Robertson for his pioneering work on               │    │
│  │        international trade dynamics..."                             │    │
│  │                                                                     │    │
│  │  Reality: Won by Banerjee, Duflo, and Kremer for poverty research   │    │
│  │                                                                     │    │
│  │  The model generated fluent, confident, completely false content.   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  RISKS OF AI-GENERATED MISINFORMATION                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  • Synthetic media: Deepfakes of political figures                          │
│  • Automated disinformation: Scaling propaganda at low cost                 │
│  • Confident incorrectness: Users trust authoritative-sounding AI           │
│  • Erosion of trust: "Is anything real anymore?"                            │
│                                                                             │
│  MITIGATIONS                                                                │
│  ───────────                                                                │
│                                                                             │
│  • RAG: Ground responses in verified sources                                │
│  • Uncertainty: Train models to say "I don't know"                          │
│  • Watermarking: Detect AI-generated content                                │
│  • Fact-checking: Automated verification systems                            │
│  • Provenance: Track source of information                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    3. PRIVACY                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRAINING DATA MEMORIZATION                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  LLMs can memorize and regurgitate training data:                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Prompt: "My social security number is 078-05-"                     │    │
│  │  Model: "1120" (completes with memorized SSN!)                      │    │
│  │                                                                     │    │
│  │  Or: Complete someone's email, address, phone number                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  EXTRACTION ATTACKS                                                         │
│  ─────────────────                                                          │
│                                                                             │
│  • Membership inference: "Was this data in training?"                       │
│  • Data extraction: Recover verbatim training examples                      │
│  • Attribute inference: Infer sensitive attributes                          │
│                                                                             │
│  PRIVACY PROTECTIONS                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  • Differential Privacy: Add noise during training                          │
│  • Data filtering: Remove PII before training                               │
│  • Deduplication: Reduce memorization of repeated data                      │
│  • Output filtering: Block outputs containing PII patterns                  │
│  • User consent: Clear policies on data usage                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    4. MISUSE AND DUAL-USE CONCERNS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  POTENTIAL MISUSES                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  • Spam and phishing: Personalized at scale                                 │
│  • Social engineering: More convincing manipulation                         │
│  • Harassment: Automated targeted abuse                                     │
│  • Academic dishonesty: Essays, code, homework                              │
│  • Fraud: Fake reviews, testimonials, identities                            │
│  • Non-consensual content: Deepfakes, synthetic media                       │
│                                                                             │
│  DUAL-USE DILEMMA                                                           │
│  ────────────────                                                           │
│                                                                             │
│  The same capabilities that make AI helpful also enable misuse:             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Capability           │  Beneficial Use     │  Harmful Use          │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  Text generation      │  Writing assistance │  Spam, misinfo        │    │
│  │  Code generation      │  Developer tools    │  Malware              │    │
│  │  Image generation     │  Art, design        │  Deepfakes            │    │
│  │  Chemistry knowledge  │  Drug discovery     │  Chemical weapons     │    │
│  │  Biology knowledge    │  Medical research   │  Bioweapons           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  This is why "just don't build it" isn't a viable solution -                │
│  need thoughtful deployment with safeguards.                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Alignment: Making AI Systems Safe

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE ALIGNMENT PROBLEM                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHAT IS ALIGNMENT?                                                         │
│  ──────────────────                                                         │
│                                                                             │
│  Alignment = Making AI systems do what humans actually want                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Unaligned model:                                                   │    │
│  │  • Optimizes for predicted next token                               │    │
│  │  • May produce harmful, false, or unhelpful content                 │    │
│  │  • Doesn't understand human values                                  │    │
│  │                                                                     │    │
│  │  Aligned model:                                                     │    │
│  │  • Helpful: Assists with legitimate requests                        │    │
│  │  • Harmless: Refuses harmful requests                               │    │
│  │  • Honest: Acknowledges uncertainty, doesn't deceive                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  THE HHH FRAMEWORK (Anthropic)                                              │
│  ────────────────────────────                                               │
│                                                                             │
│  Helpful:                                                                   │
│  • Completes requested tasks                                                │
│  • Provides useful information                                              │
│  • Follows instructions                                                     │
│                                                                             │
│  Harmless:                                                                  │
│  • Doesn't help with illegal activities                                     │
│  • Doesn't generate hateful content                                         │
│  • Doesn't manipulate or deceive                                            │
│                                                                             │
│  Honest:                                                                    │
│  • Acknowledges limitations                                                 │
│  • Doesn't claim false certainty                                            │
│  • Corrects mistakes                                                        │
│                                                                             │
│  TENSION: Helpfulness vs Harmlessness                                       │
│  A model that refuses everything is "safe" but useless.                     │
│  Finding the right balance is the art of alignment.                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mitigation Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SAFETY INTERVENTIONS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TECHNICAL MITIGATIONS                                                      │
│  ─────────────────────                                                      │
│                                                                             │
│  1. RLHF / Constitutional AI                                                │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Train model to prefer helpful, harmless, honest responses       │     │
│     │  Using human feedback or AI feedback with principles             │     │
│     │                                                                 │     │
│     │  See Module 6 for detailed RLHF explanation                     │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. Input/Output Filtering                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Input → [Content Classifier] → Model → [Safety Filter] → Output│     │
│     │                 ↓                              ↓                │     │
│     │          Block harmful                 Block harmful            │     │
│     │          requests                      outputs                  │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. Watermarking                                                            │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Embed imperceptible statistical signatures in AI output         │     │
│     │  Allows detection: "This was generated by AI"                   │     │
│     │                                                                 │     │
│     │  Methods:                                                       │     │
│     │  • Token frequency manipulation                                 │     │
│     │  • Semantic watermarks                                          │     │
│     │  • Image pixel patterns                                         │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. Rate Limiting and Monitoring                                            │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  • Limit requests per user/API key                              │     │
│     │  • Monitor for abuse patterns                                   │     │
│     │  • Log and audit API usage                                      │     │
│     │  • Anomaly detection for misuse                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ORGANIZATIONAL MITIGATIONS                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  1. Red Teaming                                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Dedicated team tries to break/misuse the system                │     │
│     │  Find vulnerabilities before bad actors do                      │     │
│     │  Regular adversarial testing                                    │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. Staged Releases                                                         │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Internal → Limited beta → Wider beta → Public                  │     │
│     │  Gradually expand access, learn from each stage                 │     │
│     │  Enables early detection of problems                            │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. Usage Policies                                                          │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Clear terms of service defining prohibited uses                │     │
│     │  Enforcement mechanisms (account suspension)                    │     │
│     │  Regular policy updates as threats evolve                       │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  SOCIETAL MITIGATIONS                                                       │
│  ─────────────────────                                                      │
│                                                                             │
│  • Regulation: Government oversight and standards                           │
│  • Education: AI literacy for the public                                    │
│  • Transparency: Model cards, system cards                                  │
│  • Research: Investment in safety research                                  │
│  • Collaboration: Industry cooperation on safety                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Resources**:
> - [Anthropic's Constitutional AI](https://arxiv.org/abs/2212.08073)
> - [AI Safety via Debate](https://arxiv.org/abs/1805.00899) (Irving et al., 2018)
> - [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) (Amodei et al., 2016)
> - [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (Mitchell et al., 2019)
> - [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

---

## 8.6 Learning Paradigms Summary

Understanding the different learning paradigms helps you choose the right approach for your problem.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LEARNING PARADIGMS COMPARISON                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SUPERVISED LEARNING                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Input (x)         Labels (y)                                       │    │
│  │  ┌───────┐        ┌───────┐                                         │    │
│  │  │ Image │  ────► │ "cat" │  ────► Model learns: x → y              │    │
│  │  │ Image │        │ "dog" │                                         │    │
│  │  │ Image │        │ "cat" │                                         │    │
│  │  └───────┘        └───────┘                                         │    │
│  │                                                                     │    │
│  │  Pros: Clear objective, measurable progress                         │    │
│  │  Cons: Requires expensive labeled data                              │    │
│  │  Use: Classification, regression, structured prediction             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  SELF-SUPERVISED LEARNING                                                   │
│  ────────────────────────                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Create labels from the data itself - no human annotation!          │    │
│  │                                                                     │    │
│  │  Language: "The cat sat on the [MASK]" → Predict "mat"              │    │
│  │            "The cat sat" → Predict "on" (next token)                │    │
│  │                                                                     │    │
│  │  Vision:   [Original image] → [Augmented view 1] ≈ [Augmented view 2]│   │
│  │            Contrastive learning: same image = similar embeddings     │    │
│  │                                                                     │    │
│  │  Pros: Unlimited "free" data, learns rich representations           │    │
│  │  Cons: Indirect objective, may not align with downstream task       │    │
│  │  Use: Pre-training LLMs, foundation models                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  REINFORCEMENT LEARNING                                                     │
│  ──────────────────────                                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  Agent                     Environment                              │    │
│  │  ┌─────┐  ──Action────►   ┌───────────┐                            │    │
│  │  │     │                  │           │                            │    │
│  │  │  π  │  ◄──State────   │   World   │                            │    │
│  │  │     │  ◄──Reward───   │           │                            │    │
│  │  └─────┘                  └───────────┘                            │    │
│  │                                                                     │    │
│  │  Learn policy π(a|s) that maximizes cumulative reward               │    │
│  │                                                                     │    │
│  │  Pros: Can learn complex behaviors, no need for "correct" labels    │    │
│  │  Cons: Sample inefficient, credit assignment hard, unstable         │    │
│  │  Use: Game playing, robotics, RLHF for LLM alignment                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  TRANSFER LEARNING                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  Pre-train          ───────────►    Fine-tune                       │    │
│  │  (Large dataset)                    (Small dataset)                 │    │
│  │                                                                     │    │
│  │  ImageNet (1M)      ───────────►    Your task (1K images)           │    │
│  │  Web text (TB)      ───────────►    Your domain (MB)                │    │
│  │                                                                     │    │
│  │  Key insight: General features transfer to specific tasks           │    │
│  │                                                                     │    │
│  │  Pros: Works with limited data, faster training                     │    │
│  │  Cons: Pre-training expensive, domain mismatch possible             │    │
│  │  Use: Most practical deep learning today                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  IN-CONTEXT LEARNING (Few-Shot / Zero-Shot)                                 │
│  ──────────────────────────────────────────                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  No gradient updates - just clever prompting!                       │    │
│  │                                                                     │    │
│  │  Zero-shot: "Translate English to French: Hello → "                 │    │
│  │  Few-shot:  "cat → chat                                             │    │
│  │              dog → chien                                            │    │
│  │              hello → "                                              │    │
│  │                                                                     │    │
│  │  Pros: No training needed, instant task adaptation                  │    │
│  │  Cons: Limited by context window, inconsistent                      │    │
│  │  Use: LLM applications, rapid prototyping                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### When to Use Which Paradigm

| Scenario | Paradigm | Why |
|----------|----------|-----|
| Lots of labeled data | Supervised | Direct optimization |
| Lots of unlabeled data | Self-supervised → Fine-tune | Learn representations first |
| Sequential decision making | Reinforcement Learning | Rewards over time |
| Small task-specific dataset | Transfer Learning | Leverage pre-trained model |
| Need to adapt quickly | In-Context Learning | No training required |

---

## 8.7 The Full Pipeline: From Data to Production

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE MODERN AI DEVELOPMENT PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: DATA COLLECTION                                            │   │
│  │ ───────────────────────────                                         │   │
│  │                                                                      │   │
│  │ Sources:                                    Scale:                   │   │
│  │ • Web scraping (Common Crawl)               Terabytes - Petabytes    │   │
│  │ • Licensed datasets                         Billions of examples     │   │
│  │ • Human annotation                          Months of effort         │   │
│  │ • Synthetic generation                      Massive compute          │   │
│  │                                                                      │   │
│  │ Considerations:                                                      │   │
│  │ • Quality vs quantity tradeoff                                       │   │
│  │ • Data cleaning and deduplication                                    │   │
│  │ • PII removal and privacy                                            │   │
│  │ • Bias in source selection                                           │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: PRE-TRAINING                                               │   │
│  │ ────────────────────────                                            │   │
│  │                                                                      │   │
│  │ Objective: Next-token prediction (autoregressive LM)                 │   │
│  │                                                                      │   │
│  │ "The quick brown" → P(fox | context)                                 │   │
│  │                                                                      │   │
│  │ Scale:                          Cost:                                │   │
│  │ • 70B-400B parameters           • $1M - $100M+ compute               │   │
│  │ • 1T - 15T tokens               • Months of training                 │   │
│  │ • 1000s of GPUs                 • Specialized infrastructure         │   │
│  │                                                                      │   │
│  │ Result: Base model with general capabilities but no alignment        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: POST-TRAINING (Alignment)                                  │   │
│  │ ─────────────────────────────────                                   │   │
│  │                                                                      │   │
│  │ Step A: Supervised Fine-Tuning (SFT)                                 │   │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │ │ Train on (instruction, response) pairs                          │ │   │
│  │ │ ~10K-100K high-quality examples                                 │ │   │
│  │ │ Teaches model to follow instructions                            │ │   │
│  │ └─────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │ Step B: RLHF / DPO                                                   │   │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │ │ Train on human preferences                                      │ │   │
│  │ │ Reward model learns what humans prefer                          │ │   │
│  │ │ Policy optimization (PPO or DPO)                                │ │   │
│  │ │ Result: Helpful, harmless, honest                               │ │   │
│  │ └─────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │ Step C: Safety Training                                              │   │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │ │ Red teaming and adversarial testing                             │ │   │
│  │ │ Constitutional AI constraints                                   │ │   │
│  │ │ Refusal training for harmful requests                           │ │   │
│  │ └─────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │ Result: Aligned assistant ready for deployment                       │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: OPTIMIZATION FOR DEPLOYMENT                                │   │
│  │ ───────────────────────────────────                                 │   │
│  │                                                                      │   │
│  │ Quantization:        FP32 → FP16 → INT8 → INT4                       │   │
│  │                      (Reduce size 4-8x, minimal quality loss)        │   │
│  │                                                                      │   │
│  │ Distillation:        Large model → Train smaller model               │   │
│  │                      (Transfer knowledge to efficient model)         │   │
│  │                                                                      │   │
│  │ Compilation:         TensorRT, vLLM optimizations                    │   │
│  │                      (Kernel fusion, memory optimization)            │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 5: PRODUCTION SERVING                                         │   │
│  │ ──────────────────────────                                          │   │
│  │                                                                      │   │
│  │ Infrastructure:                                                      │   │
│  │ • Load balancer → Inference servers → GPU clusters                   │   │
│  │ • Continuous batching, KV caching                                    │   │
│  │ • Auto-scaling based on demand                                       │   │
│  │                                                                      │   │
│  │ Monitoring:                                                          │   │
│  │ • Latency (P50, P99), throughput                                     │   │
│  │ • Error rates, safety violations                                     │   │
│  │ • Cost per request                                                   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 6: APPLICATION LAYER                                          │   │
│  │ ─────────────────────────                                           │   │
│  │                                                                      │   │
│  │ RAG:                    Agents:               Fine-tuned:            │   │
│  │ ┌─────────────────┐     ┌────────────────┐    ┌────────────────┐    │   │
│  │ │ Query → Retrieve│     │ Plan → Tool    │    │ Domain-specific│    │   │
│  │ │ → Generate with │     │ → Observe →    │    │ adaptation     │    │   │
│  │ │ context         │     │ Repeat         │    │                │    │   │
│  │ └─────────────────┘     └────────────────┘    └────────────────┘    │   │
│  │                                                                      │   │
│  │ Prompting:              Guardrails:           Evaluation:           │   │
│  │ ┌─────────────────┐     ┌────────────────┐    ┌────────────────┐    │   │
│  │ │ System prompts, │     │ Input/output   │    │ A/B tests,     │    │   │
│  │ │ few-shot,       │     │ filtering,     │    │ user feedback, │    │   │
│  │ │ chain-of-thought│     │ rate limits    │    │ benchmarks     │    │   │
│  │ └─────────────────┘     └────────────────┘    └────────────────┘    │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│                     ┌───────────────┐                                      │
│                     │  END USERS    │                                      │
│                     │  Chat, APIs,  │                                      │
│                     │  Products     │                                      │
│                     └───────────────┘                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.8 Summary

### Key Concepts from This Module

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODULE 8 KEY TAKEAWAYS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. RAG (Retrieval-Augmented Generation)                                    │
│     • Grounds LLMs in external, up-to-date knowledge                        │
│     • Components: Chunking → Embedding → Vector DB → Retrieval → Generation │
│     • Advanced: Reranking, hybrid search, self-RAG                          │
│                                                                             │
│  2. AI Agents                                                               │
│     • LLMs + Tools + Planning = Autonomous task completion                  │
│     • ReAct pattern: Thought → Action → Observation loop                    │
│     • Challenges: Error propagation, security, cost                         │
│                                                                             │
│  3. Model Deployment                                                        │
│     • Key metrics: TTFT, TPOT, throughput, cost                             │
│     • Optimizations: Batching, KV cache, quantization, speculative decoding │
│     • Frameworks: vLLM, TGI, TensorRT-LLM, Ollama                           │
│                                                                             │
│  4. Evaluation                                                              │
│     • Multi-faceted: Benchmarks, LLM-as-judge, human eval, arena            │
│     • Key benchmarks: MMLU, GSM8K, HumanEval, TruthfulQA                    │
│     • Best practice: Multiple methods, prevent contamination                │
│                                                                             │
│  5. Ethics & Safety                                                         │
│     • Concerns: Bias, hallucination, privacy, misuse                        │
│     • Alignment: Helpful, harmless, honest (HHH)                            │
│     • Mitigations: RLHF, filtering, red teaming, regulation                 │
│                                                                             │
│  6. Learning Paradigms                                                      │
│     • Supervised: Labeled data → Direct mapping                             │
│     • Self-supervised: Create labels from data itself                       │
│     • Transfer: Pre-train large, fine-tune small                            │
│     • In-context: Adapt through prompting, no gradients                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Glossary Terms Covered

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation - combining retrieval with generation |
| **Vector Database** | Database optimized for storing and searching vector embeddings |
| **Embedding** | Dense vector representation of text/images |
| **Agent** | AI system that can reason, plan, and use tools |
| **Tool Use** | LLM's ability to call external functions/APIs |
| **ReAct** | Reasoning + Acting pattern for agents |
| **KV Cache** | Storing key/value pairs to avoid recomputation |
| **Quantization** | Reducing numerical precision to save memory/compute |
| **Speculative Decoding** | Using small model to draft, large model to verify |
| **TTFT** | Time to First Token - latency metric |
| **TPOT** | Time Per Output Token - generation speed |
| **Benchmark** | Standardized test set for model evaluation |
| **LLM-as-Judge** | Using an LLM to evaluate other model outputs |
| **Bias** | Systematic error reflecting unfair treatment |
| **Alignment** | Making AI systems do what humans actually want |
| **Red Teaming** | Adversarial testing to find vulnerabilities |
| **Watermarking** | Embedding detectable signatures in AI output |

---

## Course Conclusion

### What You've Learned Across All Modules

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    YOUR DEEP LEARNING JOURNEY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Module 1-2: FOUNDATIONS                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Neurons, activation functions, loss functions                     │   │
│  │ • Forward propagation and backpropagation                           │   │
│  │ • Gradient descent and optimization                                 │   │
│  │ • Regularization (L2, dropout, batch norm)                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  Module 3-4: CONVOLUTIONAL NEURAL NETWORKS                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Convolution operation and feature learning                        │   │
│  │ • Pooling, stride, padding                                          │   │
│  │ • Architectures: LeNet → AlexNet → VGG → ResNet                     │   │
│  │ • Transfer learning and fine-tuning                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  Module 5: SEQUENCE MODELS                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • RNNs and the vanishing gradient problem                           │   │
│  │ • LSTM and GRU gating mechanisms                                    │   │
│  │ • Bidirectional and deep RNNs                                       │   │
│  │ • Sequence-to-sequence with attention                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  Module 5-6: ATTENTION AND TRANSFORMERS                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Self-attention mechanism                                          │   │
│  │ • Multi-head attention                                              │   │
│  │ • Transformer architecture (encoder-decoder)                        │   │
│  │ • Positional encoding                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  Module 6: LARGE LANGUAGE MODELS                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Tokenization (BPE, WordPiece)                                     │   │
│  │ • Pre-training at scale                                             │   │
│  │ • Fine-tuning (SFT, LoRA, QLoRA)                                    │   │
│  │ • RLHF and alignment                                                │   │
│  │ • Prompting and in-context learning                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  Module 7: GENERATIVE MODELS                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Autoencoders and VAEs                                             │   │
│  │ • GANs and adversarial training                                     │   │
│  │ • Diffusion models                                                  │   │
│  │ • Latent diffusion (Stable Diffusion)                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  Module 8: ADVANCED TOPICS & PRODUCTION                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • RAG for knowledge augmentation                                    │   │
│  │ • AI agents and tool use                                            │   │
│  │ • Model deployment and optimization                                 │   │
│  │ • Evaluation and benchmarking                                       │   │
│  │ • Ethics, safety, and alignment                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Five Big Ideas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FIVE THINGS TO REMEMBER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DEEP LEARNING = DIFFERENTIABLE FUNCTIONS + GRADIENT DESCENT             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Everything is a composition of differentiable functions.        │     │
│     │  We optimize by following gradients downhill.                    │     │
│     │  This simple idea powers all modern AI.                          │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. SCALE DRIVES CAPABILITY                                                 │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  More data + more compute + more parameters = more capability.   │     │
│     │  Emergent abilities appear at scale.                             │     │
│     │  Scaling laws let us predict performance.                        │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. PRE-TRAIN + FINE-TUNE IS THE DOMINANT PARADIGM                          │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Train once on massive data → Adapt to many tasks.               │     │
│     │  Foundation models are the new platform.                         │     │
│     │  Transfer learning works across domains.                         │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. ATTENTION UNIFIED VISION AND LANGUAGE                                   │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Transformers work for text, images, audio, video, code.         │     │
│     │  Self-attention captures long-range dependencies.                │     │
│     │  One architecture, many modalities.                              │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  5. EVALUATION AND SAFETY REMAIN UNSOLVED                                   │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  We can build powerful systems but struggle to measure them.     │     │
│     │  Alignment is critical as systems become more capable.           │     │
│     │  Ethics and safety are technical AND social challenges.          │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Where to Go From Here

**Build Projects**:
- Start with something small but complete
- Deploy a model (even locally)
- Share your work publicly

**Read Papers**:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer paper
- [GPT-3](https://arxiv.org/abs/2005.14165) - Scaling language models
- [CLIP](https://arxiv.org/abs/2103.00020) - Connecting vision and language
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) - Latent diffusion
- [Constitutional AI](https://arxiv.org/abs/2212.08073) - Alignment

**Join Communities**:
- [Hugging Face](https://huggingface.co/) - Models, datasets, spaces
- [r/MachineLearning](https://reddit.com/r/MachineLearning) - Discussion
- [Twitter/X ML community](https://twitter.com) - Latest research
- Local meetups and reading groups

**Specialize**:
- **NLP**: Language understanding, generation, translation
- **Computer Vision**: Detection, segmentation, generation
- **Multimodal**: Vision-language, audio-visual
- **Systems**: Efficient inference, distributed training
- **Safety**: Alignment, interpretability, robustness

**Stay Current**:
- The field moves fast - read arxiv weekly
- Follow key researchers and labs
- Experiment with new models as they release

---

## References

### Papers Referenced in This Module

**RAG and Retrieval**:
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- [Self-RAG](https://arxiv.org/abs/2310.11511) (Asai et al., 2023)
- [HyDE](https://arxiv.org/abs/2212.10496) (Gao et al., 2022)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2018)

**Agents and Tool Use**:
- [ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
- [AutoGen](https://arxiv.org/abs/2308.08155) (Wu et al., 2023)
- [Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023)
- [Plan-and-Solve](https://arxiv.org/abs/2305.04091) (Wang et al., 2023)

**Inference Optimization**:
- [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022)
- [AWQ](https://arxiv.org/abs/2306.00978) (Lin et al., 2023)
- [SmoothQuant](https://arxiv.org/abs/2211.10438) (Xiao et al., 2022)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023)

**Evaluation Benchmarks**:
- [MMLU](https://arxiv.org/abs/2009.03300) (Hendrycks et al., 2020)
- [GSM8K](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021)
- [HumanEval](https://arxiv.org/abs/2107.03374) (Chen et al., 2021)
- [TruthfulQA](https://arxiv.org/abs/2109.07958) (Lin et al., 2021)
- [SWE-bench](https://arxiv.org/abs/2310.06770) (Jimenez et al., 2023)
- [HellaSwag](https://arxiv.org/abs/1905.07830) (Zellers et al., 2019)

**Ethics and Safety**:
- [Constitutional AI](https://arxiv.org/abs/2212.08073) (Bai et al., 2022)
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) (Amodei et al., 2016)
- [Model Cards](https://arxiv.org/abs/1810.03993) (Mitchell et al., 2019)

### Tools and Frameworks

- [vLLM](https://docs.vllm.ai/) - High-throughput LLM serving
- [LangChain](https://langchain.com/) - LLM application framework
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [Hugging Face](https://huggingface.co/) - Models, datasets, tools
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

### Courses and Books

- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) - NLP with Deep Learning
- [Stanford CS231n](http://cs231n.stanford.edu/) - CNNs for Visual Recognition
- [fast.ai](https://www.fast.ai/) - Practical Deep Learning
- [Deep Learning Book](https://www.deeplearningbook.org/) (Goodfellow et al.)
- [Dive into Deep Learning](https://d2l.ai/) - Interactive textbook

---

*Congratulations on completing this course! You now have a solid foundation in deep learning
from fundamental concepts to production deployment. The field is moving fast - keep learning,
keep building, and remember that the best way to understand AI is to use it.*
