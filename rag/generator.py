import requests
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

def generate(question: str, context_chunks: list) -> str:
    context = "\n\n".join(
        f"[{c['ticker']}] {c['text']}" for c in context_chunks
    )
    prompt = f"""You are a helpful assistant for answering questions about SEC filings. Use only the following context to answer the question. If you don't know the answer, say you don't know. Always include the ticker symbol in your answer.\n\nContext:\n{context}\n\nQuestion: {question} \n Answer:""" 

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,
        },
    )
    return response.json()["response"]