from retriever import retrieve
from generator import generate

def answer_question(question: str, ticker: str = None) -> str:
    print(f"Retrieving relevant chunks for question: '{question}' (ticker={ticker})")
    context_chunks = retrieve(question, top_k=5, ticker=ticker)
    print(f"Retrieved {len(context_chunks)} chunks. Generating answer...")
    answer = generate(question, context_chunks)
    return answer

if __name__ == "__main__":
    ticker = input("Enter ticker (or press Enter to search all): ").strip() or None
    question = input("Enter your question: ").strip()
    result = answer_question(question, ticker=ticker)
    print("\nAnswer:", result)