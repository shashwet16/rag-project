from app.ingestion.pdf_loader import load_pdf 

docs = load_pdf("/home/shashwat/Downloads/MACROANSWERS Unofficial.pdf")
print(f"\nTotal documents: {len(docs)}\n")

for doc in docs:
    print(f"--- Page {doc.metadata['page']} ---")
    print(f"Source: {doc.metadata['source']}")
    print(f"Text preview: {doc.text[:100]}...")
    print()