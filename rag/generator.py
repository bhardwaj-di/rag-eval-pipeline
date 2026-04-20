import os
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.1-8b-instant"


def generate(question: str, context_chunks: list, chat_history: list = []) -> str:
    context = "\n\n".join(
        f"[{c['ticker']}] {c['text']}" for c in context_chunks
    )
    prompt = f"""You are a helpful assistant for answering questions about SEC 10-K filings for these companies:
- AAPL: Apple Inc.
- GOOGL: Alphabet Inc. (Google)
- META: Meta Platforms Inc. (Facebook)
- MSI: Motorola Solutions Inc.
- NVDA: NVIDIA Corporation

If the question is a greeting, respond in a friendly way and briefly mention you can help with SEC 10-K filings for these companies.
If the question is completely unrelated to SEC filings or these companies, politely say you can only help with questions about their SEC filings.
Otherwise, use only the following context to answer. Always answer in 2-3 sentences. If the answer is not in the context, say you don't know.

Context:
{context}""" 

    messages = [{"role": "system", "content": prompt}]

    for msg in chat_history[-6:]:  # last 3 exchanges
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content