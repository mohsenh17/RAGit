# RAGit

**RAGit** is an open-source project that demonstrates how to build a **Retrieval-Augmented Generation (RAG)** system for understanding and exploring codebases.  
Using **LangChain**, **local embeddings**, and **free LLMs** (e.g., CodeLlama or StarCoder2), this tool can:

- Parse a repository and extract functions, classes, and docstrings.
- Embed and index code snippets for semantic search.
- Answer natural-language questions like:
  - "What does function `foo` do?"
  - "Which file contains the class `MyParser`?"
  - "Is there any function that connects to the database?"

---

## **Key Features**
- **Fully Free & Local** – Uses HuggingFace for embeddings and Ollama for local LLM inference.
- **Vector Search** – Powered by **ChromaDB** or **FAISS**.
- **Extensible** – Add custom parsing or UI layers.

---

## **Roadmap**
1. Basic RAG pipeline (parser → embeddings → retriever → LLM).
2. Conversational mode (multi-turn Q&A).
3. UI for interactive repo queries.

---

## **Tech Stack**
- **LangChain** for RAG orchestration.
- **HuggingFace Sentence-Transformers** for free embeddings.
- **ChromaDB** for vector storage.
- **Ollama** for local LLMs.
- **Python** for code parsing.

## Repository Structure
```bash
RAGit/
│
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── code_parser.py       # Extract functions/classes from repo
│   ├── embedder.py          # Chunk & embed code with sentence-transformers
│   ├── vectorstore.py       # Chroma/FAISS storage and retrieval
│   ├── rag_pipeline.py      # RAG chain (Retriever + LLM)
│   └── main.py              # Entry point: query the codebase
├── examples/
│   └── demo_repo/           # A small repo for testing
└── tests/
    ├── test_parser.py
    ├── test_embedding.py
    └── test_rag_pipeline.py

```