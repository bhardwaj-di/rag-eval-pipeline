import json
import re
import time
import argparse
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nomic import embed
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "sec-edgar-filings"
CHUNKS_FILE = BASE_DIR / "data" / "chunks.json"

NOMIC_EMBED_MODEL = "nomic-embed-text-v1"
EMBED_BATCH_SIZE = 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

def embed_with_retry(chunks, max_retries=3):
    for attempt in range(max_retries):
        try:
            prefixed = ["search_document: " + c for c in chunks]
            response = embed.text(
                texts=prefixed,
                model=NOMIC_EMBED_MODEL,
                task_type="search_document"
            )
            return response["embeddings"]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(5)
            else:
                raise

def extract_narrative(filepath: Path) -> str:
    text = filepath.read_text(encoding="utf-8", errors="ignore")

    documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", text, flags=re.DOTALL)

    narrative_parts = []
    for doc in documents:
        doc_type = re.search(r"<TYPE>([^\n]+)", doc)
        doc_type = doc_type.group(1).strip() if doc_type else ""

        if doc_type in ("10-K", "10-K/A"):
            clean = re.sub(r"<[^>]+>", " ", doc)
            clean = re.sub(r"&#\d+;", " ", clean)
            clean = re.sub(r"&[a-z]+;", " ", clean)

            # Skip XBRL header — jump to where narrative starts
            match = re.search(r'\bPart\s+I\b(?!\s*I)', clean)
            if not match:
                match = re.search(r'\bItem\s+1[\.\s]', clean)
            if match:
                clean = clean[match.start():]

            narrative_parts.append(clean)

    if not narrative_parts:
        return text

    return "\n\n".join(narrative_parts)

def clean_text(text: str) -> str:
    # Remove base64 encoded blobs
    text = re.sub(r"[A-Za-z0-9+/]{100,}", " ", text)
    # Remove XBRL inline tags
    text = re.sub(r"<ix:[^>]+>", " ", text, flags=re.IGNORECASE)
    # Remove lines that are just dashes, underscores or special chars
    text = re.sub(r"^[\-_=*]{3,}$", "", text, flags=re.MULTILINE)
    # Remove lines that look like XBRL data (no spaces, colon-separated)
    text = re.sub(r"^[a-z\-]+:[A-Za-z]+\s+\d.*$", "", text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def process_all_filings(cpu_only=False):
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Mode: CPU only")
    else:
        print("Mode: GPU (if available)")

    all_chunks = []
    files = list(RAW_DIR.rglob("*.txt"))
    print(f"Found {len(files)} filing files\n")

    total_start = time.time()

    for filepath in files:
        parts = filepath.parts
        ticker = parts[-4].upper() if len(parts) >= 4 else "UNKNOWN"
        print(f"Processing {ticker} — {filepath.name}")

        try:
            text = extract_narrative(filepath)
            text = clean_text(text)
            chunks = splitter.split_text(text)
            print(f"  Chunks after extraction: {len(chunks)}")

            print(f"  Embedding {len(chunks)} chunks...")
            file_start = time.time()
            vectors = []
            for i in range(0, len(chunks), EMBED_BATCH_SIZE):
                batch = chunks[i:i + EMBED_BATCH_SIZE]
                vectors.extend(embed_with_retry(batch))
            file_elapsed = time.time() - file_start

            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                all_chunks.append({
                    "text": chunk,
                    "embedding": vector,
                    "ticker": ticker,
                    "source_file": str(filepath),
                    "chunk_index": i,
                })
            print(f"  ✓ {len(chunks)} chunks embedded in {file_elapsed:.1f}s ({len(chunks)/file_elapsed:.1f} chunks/sec)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    total_elapsed = time.time() - total_start
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Total time:   {total_elapsed:.1f}s ({total_elapsed/60:.1f} mins)")
    print(f"Avg speed:    {len(all_chunks)/total_elapsed:.1f} chunks/sec")

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(all_chunks, f)
    print(f"Saved to {CHUNKS_FILE}")
    return all_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Run on CPU only")
    args = parser.parse_args()
    process_all_filings(cpu_only=args.cpu)
