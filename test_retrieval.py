# test_retrieval.py
from app.ingestion.pdf_loader import load_pdf
from app.processing.cleaner import clean_documents
from app.processing.chunker import chunk_documents
from app.vectorstore.embedder import embed_documents
from app.vectorstore.store import store_documents, get_collection_count
from app.retrieval.retriever import retrieve, print_results

# --- Ingest if not already done ---
if get_collection_count() == 0:
    docs = load_pdf("data/sample.pdf")
    docs = clean_documents(docs)
    chunks = chunk_documents(docs)
    embeddings = embed_documents(chunks)
    store_documents(chunks, embeddings)

# --- Test several queries — inspect quality manually ---
test_queries = [
    "What are the types of machine learning?",
    "How is reinforcement learning different from supervised learning?",
    "What programming language is best for cooking pasta?",  # should score LOW
]

for q in test_queries:
    results = retrieve(q, top_k=3)
    print_results(q, results)