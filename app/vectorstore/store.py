import chromadb
# WHAT: Vector database library — stores embeddings, runs HNSW search internally
# WHY:  We need fast similarity search over thousands of chunk vectors
# PROJECT: This is where all our chunk embeddings live after ingestion

from app.config import CHROMA_PERSIST_DIR, COLLECTION_NAME


def get_client():
    """
    PersistentClient → saves to disk, survives process restarts.
    """
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    # WHAT: path = folder where ChromaDB writes its index + data files
    # WHY:  Without persistence, all data would vanish when the script ends
    # PROJECT: CHROMA_PERSIST_DIR = "./chroma_db" from config.py


def get_collection():
    """
    Collection = like a SQL table. Holds all chunks from all sources.
    """
    client = get_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # WHAT: get_or_create → idempotent, safe to call every run
        # WHY:  First run creates it, later runs just connect to existing data

        metadata={"hnsw:space": "cosine"}
        # WHAT: Configures HNSW to use cosine similarity as distance metric
        # WHY:  Cosine matches how our embedding model measures similarity
        #       (see Phase 8 — angle between vectors, not magnitude)
        # ALTERNATIVE: "l2" (Euclidean) or "ip" (inner product/dot product)
    )


def store_documents(docs, embeddings):
    """
    Store chunks + their embeddings in ChromaDB.

    Args:
        docs:       list of Document objects (chunks)
        embeddings: list of vectors, SAME ORDER as docs
    """
    collection = get_collection()

    ids       = [doc.doc_id for doc in docs]
    texts     = [doc.text for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    # WHAT: Build parallel lists — ChromaDB's add() expects separate arrays
    # WHY:  ids[i], texts[i], embeddings[i], metadatas[i] must all refer
    #       to the SAME chunk — order matters critically here

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    # WHAT: Inserts all chunks in one batch call, overwriting any existing
    #       row that already has the same ID
    # WHY:  Batch insert is much faster than looping per chunk
    # WHY upsert not add: add() would append a second copy of every chunk
    #       each time the same source is ingested. Paired with the
    #       deterministic IDs from chunker.make_chunk_id(), upsert makes
    #       re-ingestion idempotent — run it twice, the count stays put.
    # PROJECT: This is called once per ingestion (PDF/URL/YouTube)

    print(f"  Stored {len(docs)} chunks in ChromaDB")


def search(query_embedding, n_results=5, where=None):
    """
    Find the K most similar chunks to a query embedding.

    Args:
        query_embedding: vector of the user's question
        n_results:       top-K to return
        where:           optional metadata filter, e.g. {"source_type": "pdf"}
    """
    collection = get_collection()

    return collection.query(
        query_embeddings=[query_embedding],
        # WHAT: Wrapped in a list — ChromaDB supports batched queries
        # PROJECT: We only send one query vector at a time

        n_results=n_results,
        # WHAT: How many nearest neighbors to return (our TOP_K)

        where=where,
        # WHAT: Optional metadata filter — restricts search to matching docs
        # WHY:  Lets us scope search, e.g. "only search PDF chunks"
        # PROJECT: Used later in Phase 13 (multi-document RAG)

        include=["documents", "metadatas", "distances"]
        # WHAT: What to return alongside IDs
        # WHY:  distances = how close each match is (lower = more similar
        #       for cosine distance, since ChromaDB returns 1 - cosine_sim)
    )


def get_collection_count():
    return get_collection().count()
    # WHAT: Total chunks currently stored — useful for debugging/verification