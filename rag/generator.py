import os
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "qwen/qwen3-32b"


def generate(question: str, context_chunks: list) -> str:
    context = "\n\n".join(
        f"[{c['ticker']}] {c['text']}" for c in context_chunks
    )
    prompt = f"""You are a helpful assistant for answering questions about SEC filings. Use only the following context to answer the question. Always answer in 2-3 sentences. If you don't know the answer, say you don't know. Always include the ticker symbol in your answer.\n\nContext:\n{context}\n\nQuestion: {question} \n Answer:""" 

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content