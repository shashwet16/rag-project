"""
Multi-document RAG test.

WHAT: Ingests several different source types, then tests
      different categories of questions to verify retrieval
      works correctly across multiple documents.

WHY:  Single-document testing doesn't catch cross-source
      issues like contamination, dominance, or contradictions.

RUN:  python test_multi_doc.py
"""

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.web_loader import load_url
from app.ingestion.yt_loader import load_youtube
from app.processing.cleaner import clean_documents
from app.processing.chunker import chunk_documents
from app.vectorstore.embedder import embed_documents
from app.vectorstore.store import store_documents, get_collection_count
from app.rag_pipeline import ask, format_response


def ingest_all():
    """
    Ingest a mix of sources into ChromaDB.
    Replace these with your actual test documents.
    """
    all_docs = []

    # --- PDF ---
    # WHAT: Load your test PDF(s)
    # Replace with actual file paths you have
    try:
        pdf_docs = load_pdf("Machine-Learning-Systems-Vol1.pdf")
        all_docs.extend(pdf_docs)
        print(f"Loaded PDF: {len(pdf_docs)} pages")
    except Exception as e:
        print(f"PDF load failed: {e}")

    # --- Web URL ---
    try:
        web_docs = load_url("https://en.wikipedia.org/wiki/Machine_learning")
        all_docs.extend(web_docs)
        print(f"Loaded URL: {len(web_docs)} documents")
    except Exception as e:
        print(f"URL load failed: {e}")

    # --- YouTube ---
    try:
        yt_docs = load_youtube("https://www.youtube.com/watch?v=aircAruvnKk")
        all_docs.extend(yt_docs)
        print(f"Loaded YouTube: {len(yt_docs)} documents")
    except Exception as e:
        print(f"YouTube load failed: {e}")

    if not all_docs:
        print("No documents loaded. Check your sources.")
        return

    # Clean → chunk → embed → store
    # WHY the main collection, not a separate one → ask() only ever reads
    #     COLLECTION_NAME from config. A "test_multi" collection would be
    #     filled but never searched.
    docs = clean_documents(all_docs)
    chunks = chunk_documents(docs)
    print(f"\nTotal chunks: {len(chunks)}")

    embeddings = embed_documents(chunks)
    store_documents(chunks, embeddings)
    print(f"Collection now holds {get_collection_count()} chunks\n")


def test_queries():
    """
    Test different query categories.
    """
    test_cases = {
        "Single-document": [
            "What are the three types of machine learning?",
        ],
        "Cross-document": [
            "What do the different sources say about neural networks?",
        ],
        "Unanswerable": [
            "What is the population of Tokyo?",
            "Who won the 2024 Olympics?",
        ],
        "Ambiguous": [
            "Tell me about learning",
        ],
    }

    for category, questions in test_cases.items():
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category}")
        print(f"{'='*60}")

        for q in questions:
            result = ask(q)
            print(format_response(result))
            print(f"\n  [Source count: {len(result['sources'])}]")
            # Print source types for debugging
            types = set(s["source_type"] for s in result["sources"])
            print(f"  [Source types: {types if types else 'none'}]")
            print()


if __name__ == "__main__":
    print("=== INGESTING MULTIPLE SOURCES ===\n")
    ingest_all()

    print("\n=== TESTING QUERIES ===")
    test_queries()