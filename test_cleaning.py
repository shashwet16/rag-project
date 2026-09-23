from app.processing.cleaner import clean_text, clean_documents
from app.ingestion.document import Document

# --- Test 1: Whitespace normalization ---
dirty = "Hello     world\n\n\n\n\n\nNext paragraph"
clean = clean_text(dirty)
print("=== Whitespace ===")
print(f"Before: {repr(dirty)}")
print(f"After:  {repr(clean)}")
# repr() → shows string with escape characters visible.
# "Hello\n\nWorld" instead of actual line breaks.
# Why → lets us SEE the whitespace changes clearly.
print()

# --- Test 2: Duplicate lines ---
dirty2 = "Navigation Menu\nHome\nAbout\nActual content here\nNavigation Menu\nHome\nAbout"
clean2 = clean_text(dirty2)
print("=== Duplicates ===")
print(f"Before: {dirty2}")
print(f"After:  {clean2}")
print()

# --- Test 3: Unicode ---
dirty3 = "It\u2019s a \u201csmart\u201d quote with\u00a0spaces"
clean3 = clean_text(dirty3)
print("=== Unicode ===")
print(f"Before: {dirty3}")
print(f"After:  {clean3}")
print()

# --- Test 4: Clean a list of Documents ---
docs = [
    Document(text="Good content here", metadata={"source": "doc1.pdf"}),
    Document(text="   \n\n\n   ", metadata={"source": "empty.pdf"}),
    Document(text="More  good   content\n\n\n\n\nhere", metadata={"source": "doc2.pdf"}),
]
print("=== Document list ===")
cleaned = clean_documents(docs)
for doc in cleaned:
    print(f"  {doc}")
# empty.pdf should be removed — its text is just whitespace.
# doc2.pdf should have normalized whitespace.