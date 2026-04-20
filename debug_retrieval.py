from qdrant_client import QdrantClient
from nomic import embed
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

# Embed query
response = embed.text(
    texts=["search_query: What was Apple's net income in 2024?"],
    model="nomic-embed-text-v1",
    task_type="search_query"
)
query_vector = response["embeddings"][0]
print(f"Query vector dim: {len(query_vector)}")

# Search without filter
results = client.search(
    collection_name="sec_filings",
    query_vector=query_vector,
    limit=8
)

print("\nTop 8 results (no ticker filter):")
for r in results:
    print(f"  Score: {r.score:.3f} | Ticker: {r.payload['ticker']} | {r.payload['text'][:150]}")
    print()
