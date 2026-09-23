import os
import pdfplumber
from app.ingestion.document import Document

def load_pdf(file_path):
    """
    Load a PDF file and extract text from each page.
    
    Args:
        file_path: path to the PDF file
        
    Returns:
        list of Document objects, one per page that contains text
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    # Check if it's actually a PDF
    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF file: {file_path}")
    
    documents = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            
            # Skip pages with no text
            if not text or text.strip() == "":
                print(f"  Warning: Page {page_num} has no text, skipping")
                continue
            doc = Document(
                text=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "source_type": "pdf",
                    "page": page_num,
                    "total_pages": len(pdf.pages),
                }
            )
            # Was a raw dict before, now a Document object.
            # Downstream code accesses doc.text, doc.metadata, doc.doc_id            
            
            documents.append(doc)
    
    if not documents:
        print(f"  Warning: No text extracted from {file_path}")
    
    print(f"  Extracted {len(documents)} pages from {os.path.basename(file_path)}")
    return documents