# test_chromadb.py
from app.ingestion.pdf_loader import load_pdf
from app.processing.cleaner import clean_documents
from app.processing.chunker import chunk_documents
from app.vectorstore.embedder import embed_documents, embed_text
from app.vectorstore.store import store_documents, search, get_collection_count

# --- Full pipeline: PDF → clean → chunk → embed → store ---
docs   = load_pdf("/home/shashwat/Downloads/MACROANSWERS Unofficial.pdf")
docs   = clean_documents(docs)
chunks = chunk_documents(docs)
embeddings = embed_documents(chunks)
store_documents(chunks, embeddings)

print(f"\nTotal chunks stored: {get_collection_count()}")

# --- Search test ---
query = "What are the types of machine learning?"
query_vector = embed_text(query)
results = search(query_vector, n_results=3)

print(f"\nQuery: {query}\n")
for i, (text, meta, dist) in enumerate(zip(
    results["documents"][0], results["metadatas"][0], results["distances"][0]
)):
    print(f"Result {i+1} — distance {dist:.4f}")
    print(f"  Source: {meta.get('source')} page {meta.get('page')}")
    print(f"  Text:   {text[:100]}...\n")