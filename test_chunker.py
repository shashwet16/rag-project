from app.ingestion.pdf_loader import load_pdf
from app.ingestion.web_loader import load_url
from app.processing.cleaner import clean_documents
from app.processing.chunker import chunk_documents

# --- Load + clean + chunk a PDF ---
print("=== PDF Chunking ===")
docs = load_pdf("/home/shashwat/Downloads/MACROANSWERS Unofficial.pdf")
docs = clean_documents(docs)
chunks = chunk_documents(docs)

print(f"Total chunks: {len(chunks)}\n")

# Show first 3 chunks
for chunk in chunks[:3]:
    print(f"Chunk {chunk.metadata['chunk_index'] + 1} of {chunk.metadata['chunk_total']}")
    print(f"Source:  {chunk.metadata['source']}, page {chunk.metadata.get('page', 'N/A')}")
    print(f"Chars:   {len(chunk.text)}")
    print(f"Preview: {chunk.text[:100]}...")
    print(f"ID:      {chunk.doc_id}")
    print()

# --- Check overlap is working ---
print("=== Overlap Check ===")
if len(chunks) >= 2:
    chunk1_end = chunks[0].text[-100:]
    chunk2_start = chunks[1].text[:100:]
    print(f"End of chunk 1:   ...{chunk1_end}")
    print(f"Start of chunk 2: {chunk2_start}...")
    # You should see some shared text between chunk1_end and chunk2_start
    # That's the overlap working correctly