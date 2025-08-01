from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import re
from typing import List

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

Use the context snippets below to answer the developer's question. if noting shows up in the context and it is seems to be unrelated to the question dont answer the question and just say "I don't know".

Context:
{context}

{question}

Please answer the intended question as clearly and thoroughly as possible. also peint the context provided to you

Answer:
""",
            input_variables=["context", "question"])
            }
        )

    def rewrite_question(self, query: str, num_rewrites: int = 3) -> List[str]:
        prompt = f"""
    Given the following developer question:

    "{query}"

    Generate {num_rewrites} diverse rewritings of this question. Try to rephrase it in different ways to improve clarity or ask it from different angles.

    Return only the rewrites, one per line.
    """
        response = self.llm.invoke(prompt)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        print(f"Rewritten questions:\n{response}\n")
        return [line.strip() for line in response.split('\n') if line.strip()]

    def answer_question(self, query: str) -> str:
        rewritten_questions = self.rewrite_question(query)
        formatted_rewrites = "\n".join(f"- {r}" for r in rewritten_questions)
        full_query = f"The developer originally asked:\n{query}\nWe generated different versions of the question to clarify intent:\n{formatted_rewrites}. remember you are only expected to answer based on the context provided if the context is not related to the question just say I don't know."
        result = self.qa_chain.invoke({
            "query": full_query,
        })
        return re.sub(r'<think>.*?</think>', '', result["result"], flags=re.DOTALL).strip()

if __name__ == "__main__":
    print("Running a quick RAG pipeline test...")
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
