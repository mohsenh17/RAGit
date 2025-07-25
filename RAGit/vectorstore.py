from langchain_chroma import Chroma
from langchain.schema import Document
from embedder import CodeEmbedder 
from typing import List
import os

class CodeVectorStore:
    def __init__(self, persist_directory: str = "chroma"):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.embedder = CodeEmbedder()

    def index_snippets(self, snippets: List[str]) -> None:
        documents = [Document(page_content=snippet) for snippet in snippets]
        self.vectorstore = Chroma.from_documents(
            documents,
            embedding=self.embedder.embedding_model,
            persist_directory=self.persist_directory
        )

    def load(self):
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                embedding_function=self.embedder.embedding_model,
                persist_directory=self.persist_directory
            )
        else:
            raise FileNotFoundError(f"No vectorstore found at {self.persist_directory}")

    def search(self, query: str, k: int = 5) -> List[str]:
        if not self.vectorstore:
            raise ValueError("Vectorstore is not loaded. Call load() first.")
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]



if __name__ == "__main__":
    def index_repo(repo_path: str):
        from code_parser import parse_repo

        code_snippets = parse_repo(repo_path)
        vector_store = CodeVectorStore()
        vector_store.index_snippets(code_snippets)
        print(f"Indexed {len(code_snippets)} snippets from {repo_path}.")
    
    repo_path = "./examples/aminoClust"
    index_repo(repo_path)
    
    vector_store = CodeVectorStore()
    vector_store.load()
    
    query = "function to cluster amino acids"
    results = vector_store.search(query)
    
    print(f"Search results for '{query}':")
    for result in results:
        print(result)