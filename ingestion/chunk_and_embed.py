import json
import re
import time
import argparse
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "sec-edgar-filings"
CHUNKS_FILE = BASE_DIR / "data" / "chunks.json"

embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBED_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL")
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

def extract_text(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="ignore")

def clean_text(text: str) -> str:
    # Remove SGML/HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove base64 encoded blobs (long strings of alphanumeric chars with no spaces)
    text = re.sub(r"[A-Za-z0-9+/]{100,}", " ", text)
    # Remove SEC EDGAR header metadata block
    text = re.sub(r"<SEC-HEADER>.*?</SEC-HEADER>", " ", text, flags=re.DOTALL)
    # Remove XBRL/XML blocks
    text = re.sub(r"<XBRL>.*?</XBRL>", " ", text, flags=re.DOTALL)
    # Remove lines that are just dashes, underscores or special chars (table borders)
    text = re.sub(r"^[\-_=*]{3,}$", "", text, flags=re.MULTILINE)
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
            text = extract_text(filepath)
            raw_chunks = splitter.split_text(text)
            text = clean_text(text)
            chunks = splitter.split_text(text)
            print(f"  Chunks before cleaning: {len(raw_chunks)} → after: {len(chunks)}")

            print(f"  Embedding {len(chunks)} chunks...")
            file_start = time.time()
            vectors = embeddings.embed_documents(chunks)
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