from langchain_text_splitters import RecursiveCharacterTextSplitter# RecursiveCharacterTextSplitter → LangChain's text splitting utility.
# Why LangChain for this one thing → it implements recursive splitting
# correctly with overlap, handling edge cases we'd have to write ourselves.
# We're not using LangChain for the whole pipeline — just this utility.

import hashlib
# hashlib → for deriving a chunk ID from the chunk's own content.
# Why not uuid4 → uuid4 is random, so the SAME chunk gets a new ID every run.
# ChromaDB keys on ID, so random IDs make re-ingestion insert duplicates
# instead of updating. Hashing the content makes the ID stable across runs.


def make_chunk_id(text, metadata):
    """
    Build a deterministic ID for a chunk.

    Same source + same page + same position + same text → same ID, always.
    That is what lets store_documents() upsert instead of duplicating.
    """
    key = "|".join([
        str(metadata.get("source", "")),
        str(metadata.get("page", "")),
        str(metadata.get("chunk_index", "")),
        text,
    ])
    # Include the text itself → if a source is edited and re-ingested,
    # changed chunks get new IDs while untouched ones keep theirs.
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

from app.ingestion.document import Document
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(docs):
    """
    Split a list of Documents into smaller chunks.
    Each chunk inherits metadata from its parent document.
    
    Flow:
      [Document, Document, ...] 
      → split each into chunks
      → each chunk becomes a new Document
      → return [chunk_doc, chunk_doc, ...]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        # chunk_size → maximum characters per chunk.
        # From config.py so we can tune it in one place.

        chunk_overlap=CHUNK_OVERLAP,
        # chunk_overlap → how many chars the next chunk re-uses from previous.
        # Why → prevents losing context at chunk boundaries.

        length_function=len,
        # length_function → how to measure chunk size.
        # len() counts characters. Alternative: count tokens instead.
        # Why characters not tokens → simpler, no tokenizer needed here.
        # Trade-off → actual token count varies by model.

        separators=["\n\n", "\n", " ", ""],
        # separators → what to try splitting on, in order of preference.
        # "\n\n" → paragraph breaks (best, preserves meaning)
        # "\n"   → line breaks
        # " "    → word boundaries (never cut mid-word)
        # ""     → characters (last resort, rarely reached)
    )

    all_chunks = []

    for doc in docs:
        # Split this document's text into chunk strings
        chunks = splitter.split_text(doc.text)
        # split_text() → takes a string, returns list of strings.
        # Each string is one chunk, respecting chunk_size and overlap.

        for i, chunk_text in enumerate(chunks):
            # i → chunk index (0, 1, 2...) within this document.
            # enumerate() → gives us both index and value in the loop.

            chunk_metadata = doc.metadata.copy()
            # .copy() → creates a NEW dict with same key-value pairs.
            # Why copy() and not just assign → dicts are mutable.
            # Without copy(), all chunks share the SAME metadata dict.
            # Changing one chunk's metadata would change all others.

            chunk_metadata["chunk_index"] = i
            # Track which chunk this is within the parent document.
            # Useful for debugging: "retrieved chunk 3 of 8 from sample.pdf"

            chunk_metadata["chunk_total"] = len(chunks)
            # Total chunks from this document.

            chunk_doc = Document(
                text=chunk_text,
                metadata=chunk_metadata,
                doc_id=make_chunk_id(chunk_text, chunk_metadata)
                # Deterministic ID derived from the chunk's content+position.
                # Why → chunks are stored independently in ChromaDB, and a
                # stable ID means re-ingesting a source updates those rows
                # rather than appending a second copy of every chunk.
            )

            all_chunks.append(chunk_doc)

    print(f"  Chunked {len(docs)} documents → {len(all_chunks)} chunks")
    return all_chunks