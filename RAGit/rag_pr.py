import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
print(str(Path(__file__).resolve().parents[1]))
from pr_writter.diff_reader import GitDiffReader, DiffMode

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from vectorstore import CodeVectorStore

import re
from typing import List

class CodeRAGPipeline:
    def __init__(self, vectorstore_path: str, model_name: str, k: int = 5):
        self.k = k
        self.vectorstore = CodeVectorStore(persist_directory=vectorstore_path)
        self.vectorstore.load()
        self.llm = OllamaLLM(model=model_name)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.vectorstore.as_retriever(search_kwargs={"k": self.k}),
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template="""
You are RAGit, a helpful code assistant created by https://github.com/mohsenh17.

Use the context snippets below to answer the developer's question. 
If nothing relevant shows up in the context, just say "I don't know".

Context:
{context}

{question}

Answer:
""",
                    input_variables=["context", "question"])
            }
        )

    def rewrite_question(self, query: str, num_rewrites: int = 3) -> List[str]:
        prompt = f"""
Given the following developer question:

"{query}"

Generate {num_rewrites} diverse rewritings of this question. 
Try to rephrase it in different ways to improve clarity or ask it from different angles.

Return only the rewrites, one per line.
"""
        response = self.llm.invoke(prompt)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return [line.strip() for line in response.split('\n') if line.strip()]

    def answer_question(self, query: str) -> str:
        rewritten_questions = self.rewrite_question(query)
        formatted_rewrites = "\n".join(f"- {r}" for r in rewritten_questions)
        full_query = (
            f"The developer originally asked:\n{query}\n\n"
            f"We generated different versions of the question to clarify intent:\n{formatted_rewrites}\n\n"
            f"Remember: only answer based on the provided context. "
            f"If the context is unrelated, say 'I don't know.'"
        )
        result = self.qa_chain.invoke({"query": full_query})
        return re.sub(r'<think>.*?</think>', '', result["result"], flags=re.DOTALL).strip()


# === Integration with GitDiffReader ===
def generate_pr_with_rag(repo_path: str, vectorstore_path: str, model_name: str):
    reader = GitDiffReader(repo_path)
    current_branch = reader.get_current_branch()
    print(f"Current branch: {current_branch}")

    # Compare current branch with main
    if current_branch != "main":
        diff_text = reader.get_diff(DiffMode.COMMITTED, "main", current_branch)
    else:
        diff_text = reader.get_diff(DiffMode.COMMITTED, "main~1", "main")

    if not diff_text:
        print("No changes found between current branch and main")
        return None

    file_changes = reader.parse_diff(diff_text)

    # 🔹 Filter only Python files
    file_changes = [fc for fc in file_changes if fc.file_path.endswith(".py")]

    if not file_changes:
        print("No Python file changes found.")
        return None

    total_added = sum(fc.lines_added for fc in file_changes)
    total_deleted = sum(fc.lines_deleted for fc in file_changes)
    print(f"Collected {len(file_changes)} Python files, +{total_added}/-{total_deleted}")

    rag_context = reader.format_for_rag(file_changes)

    # Initialize pipeline
    pipeline = CodeRAGPipeline(vectorstore_path=vectorstore_path, model_name=model_name, k=5)

    # Build PR-writing prompt
    pr_prompt = f"""
You are a helpful assistant that writes GitHub pull request descriptions.

Branch: {current_branch}
Python files changed: {len(file_changes)}, +{total_added}/-{total_deleted}

Code changes:
{rag_context}

Write a professional PR description with:
- Title: concise summary of the main change
- Body: key modifications, motivations, and impact
- Bullet points for multiple changes
- Keep it professional and clear
"""

    # Directly use the LLM
    response = pipeline.llm.invoke(pr_prompt)
    pr_description = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return pr_description


if __name__ == "__main__":
    try:
        pr_description = generate_pr_with_rag(
            repo_path="/home/mohsen/work/yorku/bulkGeneFormer/",
            vectorstore_path="chroma",
            model_name="deepseek-r1:14b"
        )
        if pr_description:
            print("\n=== Generated PR Description ===\n")
            print(pr_description)
        else:
            print("No PR description generated.")
    except Exception as e:
        print(f"Pipeline failed with error: {e}")