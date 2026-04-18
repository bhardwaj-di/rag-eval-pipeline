from qdrant_client import QdrantClient
from nomic import embed
from dotenv import load_dotenv
import os
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

COLLECTION_NAME = "sec_filings"

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30
)

os.environ["NOMIC_API_KEY"] = os.getenv("NOMIC_API_KEY")

def retrieve(query, top_k=5, ticker=None):
    response = embed.text(
        texts=["search_query: " + query],
        model="nomic-embed-text-v1",
        task_type="search_query"
    )
    query_embedding = response["embeddings"][0]

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

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=filter
    )

    return [
        {
            "text": r.payload["text"],
            "ticker": r.payload["ticker"],
            "source_file": r.payload["source_file"],
            "score": r.score
        }
        for r in results
    ]
