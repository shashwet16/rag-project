from app.ingestion.web_loader import load_url

# Test with a simple, reliable page
url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

try:
    docs = load_url(url)
    
    if docs:
        doc = docs[0]
        print(f"Title:  {doc.metadata['title']}")
        print(f"Domain: {doc.metadata['domain']}")
        print(f"Chars:  {len(doc.text)}")
        print(f"\nFirst 300 chars:\n{doc.text[:300]}")
    else:
        print("No text extracted")
        
except Exception as e:
    print(f"Error: {e}")