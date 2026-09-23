import re
# re → Python's regular expression module.
# Why → regex lets us find and replace text patterns efficiently.
# Example: replace 3+ newlines with just 2, in one operation.


def clean_text(text):
    """
    Clean raw extracted text for embedding quality.
    
    Pipeline:
      raw text → fix unicode → normalize whitespace → 
      remove duplicate lines → strip → return clean text
    """

    if not text:
        return ""
    # Guard clause → if text is None or empty string, return early.
    # Why → prevents errors in the cleaning steps below.
    # Pattern: check invalid input first, handle it, then proceed with valid input.

    # --- Step 1: Fix common Unicode issues ---
    text = text.replace("\u00a0", " ")
    # \u00a0 → non-breaking space. Looks like a normal space but isn't.
    # Why → PDFs and web pages often use these. They break text splitting
    # because Python's .split() doesn't treat them as regular spaces.

    text = text.replace("\u2019", "'")
    # \u2019 → smart/curly right single quote ('). 
    # Replace with standard ASCII apostrophe for consistency.

    text = text.replace("\u2018", "'")
    # \u2018 → smart/curly left single quote (').

    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # \u201c, \u201d → smart/curly double quotes (" ").
    # Why → standardizing quotes prevents inconsistent chunking
    # and makes text matching more reliable.

    # --- Step 2: Normalize whitespace ---
    text = re.sub(r"[ \t]+", " ", text)
    # r"[ \t]+" → matches one or more spaces or tabs in a row.
    # Replaces with single space.
    # "Hello     world\t\there" → "Hello world here"
    # Why → extra spaces waste chunk space and add noise to embeddings.

    text = re.sub(r"\n{3,}", "\n\n", text)
    # r"\n{3,}" → matches 3 or more consecutive newlines.
    # Replaces with exactly 2 newlines (one blank line).
    # Why → PDFs often have huge gaps between sections.
    # We keep paragraph breaks (double newline) but remove excessive gaps.

    # --- Step 3: Remove duplicate lines ---
    lines = text.split("\n")
    # Split text into individual lines for processing.

    seen = set()
    # set → unordered collection that only stores unique values.
    # Why → O(1) lookup time. Checking "have I seen this line before?"
    # is instant regardless of how many lines we've processed.

    unique_lines = []
    for line in lines:
        stripped = line.strip()
        # .strip() → removes leading/trailing whitespace.
        # We strip for comparison but keep original formatting.

        if stripped == "":
            unique_lines.append("")
            continue
            # Keep blank lines (they mark paragraph boundaries).
            # continue → skip to next iteration, don't add to 'seen'.

        if stripped not in seen:
            seen.add(stripped)
            unique_lines.append(line)
            # First time seeing this line → keep it.
        # If already seen → skip it (duplicate removed).
        # Why → PDFs repeat headers/footers on every page.
        # Web pages repeat navigation text.

    text = "\n".join(unique_lines)

    # --- Step 4: Final cleanup ---
    text = text.strip()
    # Remove leading/trailing whitespace from the entire text.

    return text


def clean_document(doc):
    """
    Clean a Document object's text in-place.
    Returns the same document with cleaned text.
    
    Why a separate function → keeps clean_text() pure (works on strings)
    while clean_document() works on our Document objects.
    This separation lets us test text cleaning independently.
    """
    doc.text = clean_text(doc.text)
    return doc


def clean_documents(docs):
    """
    Clean a list of Document objects.
    Removes documents that become empty after cleaning.
    
    Flow: [raw docs] → clean each → filter out empties → [clean docs]
    """
    cleaned = []
    for doc in docs:
        clean_document(doc)

        if doc.text:
            cleaned.append(doc)
        else:
            print(f"  Removed empty document: {doc.metadata.get('source', 'unknown')}")
            # Why → after cleaning, some documents may have no useful text.
            # Example: a PDF page that was only a header/footer.
            # Embedding empty text is pointless and wastes storage.

    print(f"  Cleaned {len(docs)} docs → {len(cleaned)} non-empty docs")
    return cleaned