# Atlas

Atlas is a modular, local-first enterprise knowledge engine built around retrieval-augmented generation (RAG).

The current version implements the baseline RAG pipeline:

```text
Documents
    ↓
Parser
    ↓
Chunker
    ↓
BGE Embeddings
    ↓
FAISS Vector Store
    ↓
Dense Retrieval
    ↓
Prompt Construction
    ↓
Local LLM
    ↓
Answer
```

The project is intentionally implemented without LangChain so that the core ingestion, embedding, vector-store, retrieval, prompting, and LLM abstractions remain explicit.

> **Current status:** Baseline dense RAG is implemented. Evaluation, hybrid retrieval, BM25, reranking, knowledge graphs, and adaptive retrieval are planned but are not part of this version.

---

## Features

- PDF parsing with PyMuPDF
- Character-based chunking with overlap
- BGE embeddings using `FlagEmbedding`
- FAISS vector search
- Local/open-source LLM generation through a generic LLM interface
- Hugging Face LLM implementation
- Persistent FAISS index and chunk metadata
- Configurable FAISS threading
- CLI-based indexing and querying
- Incremental indexing support through source-document manifests
- Reset/rebuild indexing
- Structured logging

---

## Project structure

```text
atlas/
├── app/
│   ├── cli.py
│   └── ...
├── ingestion/
│   ├── parser.py
│   ├── chunker.py
│   └── indexer.py
├── models/
│   ├── document.py
│   └── chunk.py
├── embeddings/
│   └── embedder.py
├── vector_store/
│   ├── vector_store.py
│   └── faiss_vector_store.py
├── retrieval/
│   └── retriever.py
├── llm/
│   ├── llm.py
│   ├── config.py
│   ├── factory.py
│   └── huggingface_llm.py
├── pipeline/
│   └── pipeline.py
├── data/
│   ├── raw/
│   ├── parsed/
│   ├── chunked/
│   └── index/
└── README.md
```

The exact directory names may differ slightly depending on the current checkout, but the important architectural separation is:

- **ingestion** — document parsing, chunking, indexing
- **models** — domain objects
- **embeddings** — embedding generation
- **vector_store** — vector persistence and similarity search
- **retrieval** — retrieval abstractions
- **llm** — model/provider abstraction
- **pipeline** — orchestration of retrieval + generation
- **app** — CLI/application entry points

---

# Requirements

Recommended environment:

- Python 3.12+ or 3.13+
- macOS/Linux
- CPU is sufficient for the baseline
- Apple Silicon is supported for local development

The current development environment uses an Apple Silicon Mac with a local Hugging Face model.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Models

## Embedding model

The current default embedding model is:

```text
BAAI/bge-base-en-v1.5
```

It produces 768-dimensional embeddings.

The model is downloaded from Hugging Face on first use.

## Generation model

The current local LLM configuration uses:

```text
Qwen/Qwen2.5-0.5B
```

The LLM layer is intentionally abstracted so the model/provider can be changed without changing the RAG pipeline.

---

# Data flow

## 1. Parse

PDF files are parsed page-by-page.

```text
PDF
 ↓
list[Document]
```

A `Document` contains information such as:

```python
Document(
    text="...",
    page=5,
    source="example.pdf",
)
```

## 2. Chunk

Documents are divided into overlapping chunks.

```text
Document
 ↓
list[Chunk]
```

A chunk contains:

```python
Chunk(
    id=...,
    chunk_number=...,
    text="...",
    page=...,
    source="...",
)
```

## 3. Embed

Chunks are converted into dense vectors using BGE.

```text
Chunk
 ↓
Embedding
```

Embeddings are normalized before being stored.

## 4. Index

Embeddings are stored in FAISS along with metadata that allows the vector result to be mapped back to the original chunk.

## 5. Retrieve

A user query is embedded using the same embedding model.

```text
query
 ↓
BGE embedding
 ↓
FAISS
 ↓
top-k chunks
```

## 6. Generate

The retrieved chunks are inserted into a prompt and passed to the local LLM.

The current prompt instructs the model to answer using the retrieved context.

---

# Running Atlas

Run commands from the project directory containing the `app` package.

## Index documents

Build/rebuild the index:

```bash
python -m app.cli index --reset
```

`--reset` removes the existing vector index and metadata and rebuilds it from the available chunked data.

Use this when:

- changing the embedding model
- changing chunking parameters
- changing the underlying corpus substantially
- intentionally rebuilding the entire index

## Incremental indexing

```bash
python -m app.cli index
```

The incremental path uses the indexing manifest to identify documents that have not changed.

Unchanged documents are skipped.

New documents are embedded and added to the existing index.

If an already-indexed document has changed, the current implementation should require a full rebuild rather than attempting unsafe in-place replacement.

---

# Query the RAG pipeline

Start the query interface:

```bash
python -m app.cli query
```

Enter a question when prompted.

For example:

```text
Question: What products does Apple offer?
```

The pipeline performs:

```text
Question
   ↓
Query embedding
   ↓
FAISS retrieval
   ↓
Top-k chunks
   ↓
Prompt construction
   ↓
Local LLM
   ↓
Answer
```

The CLI also prints retrieval information for the returned chunks.

---

# Using the sample test data

The repository can contain a small synthetic dataset under:

```text
data/test_chunked/
```

This data is intentionally small and synthetic so that users can test the retrieval pipeline without downloading a large document corpus.

Example:

```bash
python -m app.cli index --reset
python -m app.cli query
```

Example questions include:

```text
What is Atlas's embedding model?
What is the purpose of the ingestion pipeline?
How does the retrieval pipeline work?
What does the monitoring system track?
What is the local LLM used by Atlas?
```

The sample data is designed so that different questions should retrieve different chunks.

---

# FAISS threading on macOS

Atlas configures FAISS to use a controlled number of OpenMP threads.

For the current Apple Silicon development environment, single-threaded FAISS search is used:

```python
faiss.omp_set_num_threads(1)
```

This avoids a native threading/runtime conflict observed between FAISS and the other native ML libraries loaded by the embedding stack.

This is an environment-level configuration, not a limitation of the RAG architecture. The thread count can be made configurable for other deployment environments.

---

# Persistence

The vector store persists two important pieces of state:

```text
FAISS index
    ↓
vectors

Metadata
    ↓
FAISS vector ID → Chunk
```

The index and metadata must remain consistent:

```text
index.ntotal == number of metadata entries
```

Atlas validates this relationship when loading the persisted store.

The indexing manifest separately tracks source-document state for incremental indexing.

---

# Current architecture

```text
                    ┌─────────────────┐
                    │    PDF/Data     │
                    └────────┬────────┘
                             ↓
                       ┌───────────┐
                       │  Parser   │
                       └─────┬─────┘
                             ↓
                       ┌───────────┐
                       │  Chunker  │
                       └─────┬─────┘
                             ↓
                       ┌───────────┐
                       │  Embedder │
                       │    BGE    │
                       └─────┬─────┘
                             ↓
                       ┌───────────┐
                       │   FAISS   │
                       └─────┬─────┘
                             ↓
                         Retriever
                             ↓
                       Search Results
                             ↓
                      Prompt Builder
                             ↓
                       Local LLM
                             ↓
                          Answer
```

The system is deliberately modular so retrieval and generation components can be replaced independently.

---

# Design principles

### No framework lock-in

Atlas does not depend on LangChain for the core pipeline.

The major components are explicit Python abstractions:

```text
Embedder
VectorStore
Retriever
LLM
PromptBuilder
RAGPipeline
```

### Local-first

The baseline can run using local embedding and generation models.

### Replaceable components

For example:

```text
FAISS
  ↓
another VectorStore implementation
```

or:

```text
Hugging Face LLM
  ↓
another LLM provider
```

without redesigning the entire pipeline.

### Separation of concerns

Indexing and querying are separate operations.

The query path should be read-only with respect to the persisted retrieval index.

---

# Current limitations

The current version is intentionally a baseline.

It does not yet include:

- BM25
- hybrid retrieval
- cross-encoder reranking
- retrieval evaluation
- answer-quality evaluation
- knowledge graph retrieval
- adaptive retrieval
- feedback-driven memory
- FastAPI
- React UI
- distributed vector search
- production authentication/authorization

These are planned extensions rather than dependencies of the baseline.

---

# Roadmap

The planned evolution is:

```text
Baseline Dense RAG
        ↓
Retrieval Evaluation
        ↓
BM25
        ↓
Hybrid Retrieval
        ↓
Cross-Encoder Reranking
        ↓
Context Optimization
        ↓
RAG Evaluation
        ↓
FastAPI / Production API
        ↓
Observability
        ↓
Knowledge Graph Retrieval
        ↓
Adaptive Retrieval
        ↓
Feedback / Memory
```

The next milestone is **retrieval evaluation**, including metrics such as Recall@K and MRR.
