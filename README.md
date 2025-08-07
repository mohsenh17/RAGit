# RAGit

**RAGit** is an open-source project that demonstrates how to build a **Retrieval-Augmented Generation (RAG)** system for understanding and exploring codebases.  
Using **LangChain**, **local embeddings**, and **free LLMs** (CodeLlama), this tool can:

- Parse a repository and extract functions, classes, and docstrings.
- Embed and index code snippets for semantic search.
- Generate commit and PR messages based on code changes.
- Answer natural-language questions like:
  - "What does function `foo` do?"
  - "Which file contains the class `MyParser`?"
  - "Is there any function that connects to the database?"

---

## **Key Features**
- **Fully Free & Local** – Uses HuggingFace for embeddings and Ollama for local LLM inference.
- **Vector Search** – Powered by **ChromaDB** or **FAISS**.
- **Automatic Commit & PR Messages** – Generates helpful summaries of staged changes.
- **Extensible** – Add custom parsing, retrievers, or UI layers.

---

## **Roadmap**
1. Basic RAG pipeline (parser → embeddings → retriever → LLM).
2. **Commit/PR generation from code diffs. ✅**
3. Conversational mode (multi-turn Q&A).
4. UI for interactive repo queries.
5. GitHub Action for automated PR summaries.

---

## **Tech Stack**
- **LangChain** for RAG orchestration.
- **HuggingFace Sentence-Transformers** for free embeddings.
- **ChromaDB** for vector storage.
- **Ollama** for local LLMs.
- **Git** and **diff parsing** for commit/PR analysis.
- **Python** for all components.

---

## **Repository Structure**
```bash
RAGit/
│
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── code_parser.py         # Extract functions/classes from repo
│   ├── embedder.py            # Chunk & embed code with sentence-transformers
│   ├── vectorstore.py         # Chroma/FAISS storage and retrieval
│   ├── rag_pipeline.py        # RAG chain (Retriever + LLM)
│   ├── pr_writer/             # Generate commit/PR messages from diffs
│   │   ├── diff_reader.py     # Extract and parse git diffs
│   │   ├── pr_formatter.py    # Prompt and format commit/PR summaries
│   │   └── utils.py           # Helper functions
│   └── main.py                # Entry point: query the codebase
├── examples/
│   └── demo_repo/             # A small repo for testing
└── tests/
    ├── test_parser.py
    ├── test_embedding.py
    ├── test_rag_pipeline.py
    └── test_pr_writer.py
