# Atlas

A modular Retrieval-Augmented Generation (RAG) platform for building intelligent document understanding systems. Atlas ingests unstructured documents, indexes them for semantic retrieval, and generates grounded answers using large language models.

The project is implemented from first principles with a strong emphasis on modularity, extensibility, and production-oriented software architecture.

---

## Features

- PDF document ingestion
- High-fidelity text extraction with PyMuPDF
- Character-based document chunking with configurable overlap
- Dense embeddings using **BAAI/bge-large-en-v1.5**
- FAISS-based vector indexing
- Semantic dense retrieval
- Modular retrieval pipeline
- Persistent vector index and metadata
- Structured logging
- Strongly typed domain models

---

# Architecture

## Current Pipeline

```text
                Documents
                     │
                     ▼
              PDF Parser
                     │
                     ▼
          Character Chunker
                     │
                     ▼
           Embedding Model
      (BAAI/bge-large-en-v1.5)
                     │
                     ▼
            FAISS Vector Store
                     │
                     ▼
            Dense Retriever
                     │
                     ▼
             Prompt Builder
                     │
                     ▼
             Open-source LLM
                     │
                     ▼
        Grounded Response + Sources
```

---

# Core Components

## Document Ingestion

- PDF parsing using **PyMuPDF**
- Page-level document extraction
- Structured `Document` objects

---

## Chunking

Current implementation:

- Character-based chunking
- Configurable chunk size
- Configurable overlap
- Source and page tracking
- Chunk offsets for traceability

---

## Embeddings

Atlas currently uses

- **BAAI/bge-large-en-v1.5**

Features include:

- Batched embedding generation
- FP16 inference
- L2-normalized embeddings
- Configurable batch sizes
---

## Vector Store

Current implementation:

- FAISS `IndexFlatL2`
- Persistent index storage
- Metadata persistence
- Dense similarity search
- Mapping between vector IDs and source chunks

The vector store is implemented behind an abstract interface, allowing alternative backends to be introduced without changing retrieval logic.

---

## Prompt Construction

The retrieval layer is decoupled from prompt generation through a dedicated prompt builder responsible for:

- Context formatting
- Source formatting
- Prompt templating
- LLM-specific prompt construction

---

## LLM Layer

The generation layer is implemented behind an abstract interface to support multiple providers. 

Planned support includes open source LLMs:

- Qwen
- Llama
---

# Design Principles

Atlas follows a modular architecture where every component has a single responsibility.

Current abstractions include:

- `Parser`
- `Chunker`
- `Embedder`
- `VectorStore`
- `Retriever`
- `PromptBuilder`
- `LLM`
- `RAGPipeline`

Each component can be replaced independently without affecting the rest of the system.

---

# Tech Stack

| Component           | Technology             |
| ------------------- | ---------------------- |
| Language            | Python 3.12            |
| Parsing             | PyMuPDF                |
| Embeddings          | BAAI/bge-large-en-v1.5 |
| Vector Search       | FAISS                  |
| Numerical Computing | NumPy                  |
| Logging             | Python Logging         |

---

# Development Roadmap

## Baseline RAG

- [x] PDF parser
- [x] Character chunker
- [x] Embedding pipeline
- [x] FAISS vector store
- [x] Dense retriever
- [x] Prompt builder
- [x] Qwen integration
- [ ] End-to-end RAG pipeline

---

# License

This project is licensed under the MIT License.