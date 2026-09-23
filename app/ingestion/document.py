import uuid
# uuid → Python's built-in module for generating unique identifiers.
# uuid4() creates a random ID like "a3f1b2c4-5678-9012-abcd-ef3456789012"
# Why → every document needs a unique ID to track it through the pipeline:
#        ingestion → chunking → embedding → ChromaDB → retrieval


class Document:
    """
    Standard document format for the entire RAG pipeline.
    Every loader (PDF, web, YouTube) outputs this.
    Downstream code (chunker, embedder, store) only knows Document.
    """
    # class → a blueprint for creating objects with the same structure.
    # Why → enforces a consistent format across all source types.
    # Without this, each loader returns slightly different dicts
    # and every downstream function must handle each format separately.

    def __init__(self, text, metadata=None, doc_id=None):
        # __init__ → constructor, runs when you do Document(...)
        
        self.text = text

        self.metadata = metadata or {}
        # or {} → if metadata is None, use empty dict instead.
        # Why → avoids NoneType errors when doing self.metadata["key"] later.

        self.doc_id = doc_id or str(uuid.uuid4())
        # If no ID provided, auto-generate one.
        # str() converts UUID object to string for easier storage/comparison.
        # This ID follows the document: loader → chunker → ChromaDB → retrieval.

    def __repr__(self):
        # __repr__ → controls what you see when you print(document).
        # Without it: <__main__.Document object at 0x7f...>  (useless)
        # With it:    Document(source=sample.pdf, chars=1500)  (useful for debugging)
        source = self.metadata.get("source", "unknown")
        # .get("source", "unknown") → safely gets value, returns "unknown" if key missing.
        # Why → some documents might not have "source" in metadata.
        return f"Document(source={source}, chars={len(self.text)})"