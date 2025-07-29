from rag_pipeline import CodeRAGPipeline

def main():
    rag_pipeline = CodeRAGPipeline(vectorstore_path="chroma", model_name="deepseek-r1:14b", k=5)
    print("RAGit Code Assistant Initialized.")
    print("Ask a question about the codebase (type 'exit' to quit):")


    while True:
        query = input("Your question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting RAGit. Goodbye!")
            break

        try:
            answer = rag_pipeline.answer_question(query)
            print(f"\nAnswer:\n{answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
