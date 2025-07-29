from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import re

from vectorstore import CodeVectorStore


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
You are RAGit a helpful code assistant created by https://github.com/mohsenh17.

Use the context snippets below to answer the developer's question.

Context:
{context}

Question:
{question}

Answer:
""",
                    input_variables=["context", "question"],
                )
            }
        )

    def answer_question(self, query: str) -> str:
        result = self.qa_chain.invoke({"query": query})
        return re.sub(r'<think>.*?</think>', '', result["result"], flags=re.DOTALL).strip()

if __name__ == "__main__":
    print("🔬 Running a quick RAG pipeline test...")
    import re

    try:
        pipeline = CodeRAGPipeline(vectorstore_path="chroma", model_name="deepseek-r1:14b", k=3)
        test_query = "what is your name?"
        answer = pipeline.answer_question(test_query)
        print(f"\nTest Query:\n{test_query}")
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        print(f"Answer:\n{answer}\n")

    except Exception as e:
        print(f"Test failed with error: {e}")
