from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

COLLECTION_NAME = "sec_filings"

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30
)

embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBED_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL")
)

def retrieve(query, top_k=5, ticker=None):
    query_embedding = embeddings.embed_query(query)

    if ticker:
        filter = Filter(
            must=[
                FieldCondition(
                    key="ticker",
                    match=MatchValue(value=ticker.upper())
                )
            ]
        )
    else:
        filter = None

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=filter
    )

    return [{"text": r.payload["text"], "ticker": r.payload["ticker"], "source_file": r.payload["source_file"], "score": r.score} for r in results]

