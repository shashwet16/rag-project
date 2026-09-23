from app.retrieval.retriever import retrieve
from app.generation.llm import generate_answer


def extract_sources(chunks):
    """
    Collect unique sources from retrieved chunks.
    
    WHAT: Reads metadata from each chunk, deduplicates,
          builds a human-readable label for each source.
    WHY:  Without dedup, if 3 chunks came from the same PDF page,
          we'd list that page 3 times — looks broken.
    PROJECT: Sources are extracted HERE in Python, not by the LLM.
             This guarantees correct citations — no hallucination possible.
    """
    seen = set()
    sources = []

    for chunk in chunks:
        meta = chunk["metadata"]
        # WHAT: retrieve() returns {"text", "metadata", "distance", "score"}.
        #       source / page / source_type live INSIDE metadata, not on the
        #       chunk itself. Reading chunk.get("source") silently gives
        #       "unknown" for everything.

        source = meta.get("source", "unknown")
        source_type = meta.get("source_type", "unknown")
        page = meta.get("page")

        dedup_key = (source, page)
        # WHAT: Tuple of (source, page) as a unique identifier.
        # WHY:  Two chunks from the same page = one citation, not two.

        if dedup_key not in seen:
            seen.add(dedup_key)

            # Format label based on source type
            if source_type == "pdf" and page is not None:
                label = f"{source} (page {page})"
            elif source_type == "web":
                label = source
            elif source_type == "youtube":
                label = f"YouTube: {source}"
            else:
                label = source

            sources.append({
                "source": source,
                "source_type": source_type,
                "page": page,
                "label": label,
            })

    return sources


def ask(query, top_k=5):
    """
    Complete RAG pipeline: question → retrieve → generate → cite.

    WHAT: Returns a structured dict with answer + sources,
          instead of just a plain string.
    WHY:  Separating answer from sources makes this:
          - API-ready (FastAPI can return it as JSON directly)
          - Testable (check sources independently from answer)
          - Trustworthy (user can verify where info came from)
    """
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "No relevant documents found in the knowledge base.",
            "sources": [],
            "query": query,
        }

    answer = generate_answer(query, chunks)
    sources = extract_sources(chunks)

    return {
        "answer": answer,
        "sources": sources,
        "query": query,
    }


def format_response(result):
    """
    Pretty-print for terminal/testing.
    The API (Phase 15) will return the raw dict as JSON instead.
    """
    output = []
    output.append(f"Question: {result['query']}")
    output.append(f"\nAnswer:\n{result['answer']}")

    if result["sources"]:
        output.append("\nSources:")
        for i, src in enumerate(result["sources"], 1):
            output.append(f"  {i}. {src['label']}")
    else:
        output.append("\nSources: None")

    return "\n".join(output)