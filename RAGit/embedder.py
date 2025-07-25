from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
import os


class CodeEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_snippets(self, snippets: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(snippets)




if __name__ == "__main__":
    def index_repo(repo_path: str):
        from code_parser import parse_repo
        code_snippets = parse_repo(repo_path)
        return code_snippets
    snippets = index_repo("./examples/aminoClust")
    embedder = CodeEmbedder()
    embeddings = embedder.embed_snippets(snippets)
    for snippet, embedding in zip(snippets, embeddings):
        print(f"Snippet: {snippet}\nEmbedding: {embedding[:5]}...")
        print("-----")