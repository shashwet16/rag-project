# ingest.py — put a PDF into the vector store.
#
# Usage:
#     python3 ingest.py data/sample.pdf
#     python3 ingest.py ~/Downloads/some_book.pdf
#
# Run this once per PDF. After that, test_rag.py can answer questions
# about it. Re-ingesting the same PDF is safe (chunks are upserted by ID).

import sys

from app.ingestion.pdf_loader import load_pdf
from app.processing.cleaner import clean_documents
from app.processing.chunker import chunk_documents
from app.vectorstore.embedder import embed_documents
from app.vectorstore.store import store_documents, get_collection_count
from app.config import COLLECTION_NAME

if len(sys.argv) != 2:
    print("Usage: python3 ingest.py <path-to-pdf>")
    sys.exit(1)

pdf_path = sys.argv[1]

print(f"Ingesting {pdf_path} into collection '{COLLECTION_NAME}'")
print(f"  Before: {get_collection_count()} chunks stored\n")

docs = load_pdf(pdf_path)
docs = clean_documents(docs)
chunks = chunk_documents(docs)
embeddings = embed_documents(chunks)
store_documents(chunks, embeddings)

print(f"\n  After: {get_collection_count()} chunks stored")
