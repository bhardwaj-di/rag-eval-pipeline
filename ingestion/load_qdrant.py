import json
import time
from pathlib import Path
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType


load_dotenv()

BASE_DIR = Path(__file__).parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "chunks.json"

COLLECTION_NAME = "sec_filings"
VECTOR_SIZE = 768
BATCH_SIZE = 50 #was 200


def get_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60  # seconds, default is too low
    )

def create_collection(client):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="ticker",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("Payload index on 'ticker' created.")

def load_chunks(path):
    print(f"Loading chunks from {path} ...")
    with open(path, "r") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")
    return chunks

def upload(client, chunks):
    total = len(chunks)
    uploaded = 0
    start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        points = [
            PointStruct(
                id=i + j,
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "ticker": chunk["ticker"],
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
            for j, chunk in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        uploaded += len(batch)
        elapsed = time.time() - start
        print(f"  Uploaded {uploaded}/{total} ({uploaded/elapsed:.1f} chunks/sec)")

    print(f"\nDone. {uploaded} chunks in {elapsed:.1f}s ({elapsed/60:.1f} mins)")

if __name__ == "__main__":
    client = get_client()
    create_collection(client)
    chunks = load_chunks(CHUNKS_FILE)
    upload(client, chunks)