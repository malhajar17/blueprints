from src.rag_pipeline import RagPipeline

file_paths = [
    "code/rag/data/bioluminescent-fungi.docx",
    "code/rag/data/a-practical-guide-to-building-agents.pdf",
    "code/rag/data/about.txt",
]
rag_pipeline = RagPipeline(chunk_size=500, chunk_overlap=50, use_tools=False)
rag_pipeline.add_documents(file_paths)

if __name__ == "__main__":

    for question in [
        "What is this demo about?",
        "For which workflows LLM agents are useful?",
        "Where can I find bioluminescent fungis?",
    ]:
        r = rag_pipeline.query(question)
        print(f"Question:\n{question}")
        print(f"Answer:\n{r['messages'][-1].content}")
        print("-" * 40)
