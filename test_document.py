from app.ingestion.pdf_loader import load_pdf
from app.ingestion.web_loader import load_url
from app.ingestion.yt_loader import load_youtube

# --- Test all three loaders ---
print("=== PDF ===")
pdf_docs = load_pdf("/home/shashwat/Downloads/MACROANSWERS Unofficial.pdf")
for doc in pdf_docs:
    print(f"  {doc}")
    # Should show: Document(source=sample.pdf, chars=XXX)
    print(f"  ID:   {doc.doc_id}")
    print(f"  Type: {doc.metadata['source_type']}")
    print()

print("=== WEB ===")
web_docs = load_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
for doc in web_docs:
    print(f"  {doc}")
    print(f"  ID:   {doc.doc_id}")
    print()

print("=== YOUTUBE ===")
yt_docs = load_youtube("https://www.youtube.com/watch?v=rfscVS0vtbw")
for doc in yt_docs:
    print(f"  {doc}")
    print(f"  ID:   {doc.doc_id}")
    print()

# --- The whole point: all three are now the same type ---
all_docs = pdf_docs + web_docs + yt_docs
# list + list → combines them into one list.
# This works because all loaders now return lists of Document objects.

print(f"Total documents: {len(all_docs)}")
print(f"All same type: {all(isinstance(d, type(all_docs[0])) for d in all_docs)}")
# isinstance() checks if an object is a certain type.
# This should print True — proving normalization works.
# Every doc has .text, .metadata, .doc_id regardless of source.