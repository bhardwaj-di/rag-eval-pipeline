from rag.retriever import retrieve
from rag.generator import generate

def answer_question(question: str, ticker: str = None, chat_history: list = []) -> dict:
    print(f"Retrieving relevant chunks for question: '{question}' (ticker={ticker})")
    context_chunks = retrieve(question, top_k=8, ticker=ticker)
    print(f"Retrieved {len(context_chunks)} chunks. Generating answer...")
    answer = generate(question, context_chunks, chat_history)
    return {
        "answer": answer,
        "sources": context_chunks
    }

if __name__ == "__main__":
    ticker = input("Enter ticker (or press Enter to search all): ").strip() or None
    question = input("Enter your question: ").strip()
    result = answer_question(question, ticker=ticker)
    print("\nAnswer:", result)