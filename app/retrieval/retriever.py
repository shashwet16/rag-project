from app.vectorstore.embedder import embed_text
from app.vectorstore.store import search
from app.config import TOP_K

# WHAT: Maximum distance allowed for a chunk to be considered relevant.
# WHY:  ChromaDB always returns results — even for completely unrelated queries.
#       This threshold filters out chunks that are "closest" but still too far.
# TUNE: Start with 0.7. Lower = stricter (fewer results, more precise).
#       Higher = looser (more results, more noise). Adjust based on testing.
RELEVANCE_THRESHOLD = 0.7


def retrieve(query, top_k=None, where=None):
    """
    Embed the query, search ChromaDB, filter by relevance threshold.

    Args:
        query:  the user's question (plain text)
        top_k:  how many chunks to fetch (defaults to config.TOP_K)
        where:  optional metadata filter, e.g. {"source_type": "pdf"}

    Returns:
        list of dicts: [{text, metadata, distance, score}, ...]
        Empty list if nothing relevant found.
    """
    top_k = top_k or TOP_K

    query_vector = embed_text(query)
    # WHAT: Embed the question using the SAME model used for chunks
    # WHY:  Query and documents must share the same vector space

    raw_results = search(query_vector, n_results=top_k, where=where)
    # WHAT: Cosine-distance HNSW search — already includes distances
    #       (see store.search), which we need for the threshold filter.

    chunks = []
    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]
    # WHAT: ChromaDB wraps results in an outer list (for batch queries)
    # WHY:  We only send one query, so we always index [0]

    for text, meta, dist in zip(documents, metadatas, distances):
        if dist > RELEVANCE_THRESHOLD:
            # WHAT: Skip chunks above the distance threshold.
            # WHY:  High distance = low similarity = not relevant.
            #       Passing irrelevant context to the LLM causes hallucination.
            continue

        chunks.append({
            "text": text,
            "metadata": meta,
            # WHY nested → rag_pipeline.extract_sources reads chunk["metadata"]
            "distance": round(dist, 4),
            "score": round(1 - dist, 4),
            # WHAT: cosine similarity = 1 - cosine distance (higher = better)
        })

    return chunks


def print_results(query, results):
    """
    Pretty-print retrieval results for manual inspection.
    THIS is how we verify retrieval quality before touching any LLM.
    """
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} chunks\n")

    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        page = meta.get("page")
        print(f"[{i}] score={r['score']}  source={meta.get('source', 'unknown')}"
              + (f" page={page}" if page else ""))
        print(f"    {r['text'][:150]}...")
        print()
